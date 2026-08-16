#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant constexpr int VALUES_PER_THREAD8 = 8;
constant constexpr int BYTES_PER_PACK8 = 4;
constant constexpr int BLOCK_SIZE8 = VALUES_PER_THREAD8 * 32;
constant constexpr int RESULTS_PER_SIMDGROUP8 = 4;

template <typename T>
inline float sigmoid_stable_native(float x) {
  float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
  return (x < 0.0f) ? y : 1.0f - y;
}

template <typename T>
[[kernel]] void gdn_norm_gate_stage(
    const device T* x [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* weight [[buffer(2)]],
    device T* scratch [[buffer(3)]],
    constant const int& rows [[buffer(4)]],
    constant const int& dv [[buffer(5)]],
    constant const int& hv [[buffer(6)]],
    constant const float& eps [[buffer(7)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  uint row = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  if (int(row) >= rows) {
    return;
  }

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  int m = int(row) / hv;
  int h = int(row) - m * hv;
  size_t row_offset = size_t(row) * size_t(dv);
  size_t scratch_offset = (size_t(m) * size_t(hv) + size_t(h)) * size_t(dv);

  float acc = 0.0f;
  uint base = lid * 4;
  for (int i = 0; i < 4; ++i) {
    uint idx = base + uint(i);
    if (idx < uint(dv)) {
      float xi = float(x[row_offset + idx]);
      acc += xi * xi;
    }
  }

  acc = simd_sum(acc);
  if (simdgroup_index_in_threadgroup == 0) {
    local_sums[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    local_sums[simdgroup_index_in_threadgroup] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index_in_threadgroup == 0) {
    acc = simd_sum(local_sums[lane]);
    if (lane == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / float(dv) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = 0; i < 4; ++i) {
    uint idx = base + uint(i);
    if (idx < uint(dv)) {
      T normed_t = weight[idx] *
          static_cast<T>(float(x[row_offset + idx]) * local_inv_mean[0]);
      float gate_f = float(gate[row_offset + idx]);
      float silu = gate_f * sigmoid_stable_native<T>(gate_f);
      scratch[scratch_offset + idx] = static_cast<T>(silu * float(normed_t));
    }
  }
}

template <typename T>
inline float load_vector8_exact_native(const device T* x, thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD8; ++i) {
    float xi = float(x[i]);
    sum += xi;
    x_thread[i] = xi;
  }
  return sum;
}

inline float qdot8_exact_native(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD8; ++i) {
    accum += x_thread[i] * float(w[i]);
  }
  return scale * accum + sum * bias;
}

template <typename T>
[[kernel]] void qmv8_rowwise_down(
    const device T* x [[buffer(0)]],
    const device uint8_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    device T* y [[buffer(4)]],
    constant const int& M_size [[buffer(5)]],
    constant const int& K_size [[buffer(6)]],
    constant const int& N_size [[buffer(7)]],
    constant const int& GS [[buffer(8)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  uint m_idx = threadgroup_position_in_grid.x;
  uint n_tile = threadgroup_position_in_grid.y;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  int M = int(M_size);
  int K = int(K_size);
  int N = int(N_size);
  if (int(m_idx) >= M) {
    return;
  }

  int num_simdgroups = int(threads_per_threadgroup.y);
  int bn = RESULTS_PER_SIMDGROUP8 * num_simdgroups;
  int scale_step_per_thread = GS / VALUES_PER_THREAD8;
  int out_row = int(n_tile) * bn + int(simd_gid) * RESULTS_PER_SIMDGROUP8;
  int in_vec_size_w = K * BYTES_PER_PACK8 / 4;
  int in_vec_size_g = K / GS;

  const device uint8_t* ws =
      w + out_row * in_vec_size_w + int(simd_lid) * 2 * BYTES_PER_PACK8;
  const device T* sc =
      scales + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
  const device T* bs =
      biases + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
  const device T* x_row = x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD8;

  float result[RESULTS_PER_SIMDGROUP8] = {0.0f};
  float x_thread[VALUES_PER_THREAD8];

  for (int k = 0; k < K; k += BLOCK_SIZE8) {
    float sum = load_vector8_exact_native<T>(x_row, x_thread);
    for (int row = 0; row < RESULTS_PER_SIMDGROUP8; ++row) {
      int n = out_row + row;
      if (n < N) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        result[row] += qdot8_exact_native(
            wl, x_thread, float(sl[0]), float(bl[0]), sum);
      }
    }
    ws += BLOCK_SIZE8 * BYTES_PER_PACK8 / 4;
    sc += BLOCK_SIZE8 / GS;
    bs += BLOCK_SIZE8 / GS;
    x_row += BLOCK_SIZE8;
  }

  for (int row = 0; row < RESULTS_PER_SIMDGROUP8; ++row) {
    int n = out_row + row;
    if (n < N) {
      float reduced = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[int(m_idx) * N + n] = T(reduced);
      }
    }
  }
}

// clang-format off
template [[host_name("gdn_norm_gate_stage_bfloat16")]]
[[kernel]] decltype(gdn_norm_gate_stage<bfloat16_t>)
gdn_norm_gate_stage<bfloat16_t>;

template [[host_name("gdn_norm_gate_stage_float16")]]
[[kernel]] decltype(gdn_norm_gate_stage<half>)
gdn_norm_gate_stage<half>;

template [[host_name("qmv8_rowwise_down_bfloat16")]]
[[kernel]] decltype(qmv8_rowwise_down<bfloat16_t>)
qmv8_rowwise_down<bfloat16_t>;

template [[host_name("qmv8_rowwise_down_float16")]]
[[kernel]] decltype(qmv8_rowwise_down<half>)
qmv8_rowwise_down<half>;
// clang-format on
