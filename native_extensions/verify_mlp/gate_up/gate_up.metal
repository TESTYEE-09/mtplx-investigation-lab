#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;

template <typename T>
inline T sigmoid_mlx_exact_native(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < T(0)) ? y : 1 - y;
}

template <typename T>
inline T swiglu_mlx_exact_native(T gate, T up) {
  T silu = gate * sigmoid_mlx_exact_native<T>(gate);
  return T(silu * up);
}

template <typename T>
inline float load_vector4_exact_native(const device T* x, thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline float qdot4_exact_native(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
    uint16_t packed = ws[i];
    accum +=
        x_thread[4 * i] * float(packed & 0x000f) +
        x_thread[4 * i + 1] * float(packed & 0x00f0) +
        x_thread[4 * i + 2] * float(packed & 0x0f00) +
        x_thread[4 * i + 3] * float(packed & 0xf000);
  }
  return scale * accum + sum * bias;
}

template <typename T>
[[kernel]] void gate_up_swiglu_qmv4_rowwise(
    const device T* x [[buffer(0)]],
    const device uint8_t* gate_w [[buffer(1)]],
    const device T* gate_scales [[buffer(2)]],
    const device T* gate_biases [[buffer(3)]],
    const device uint8_t* up_w [[buffer(4)]],
    const device T* up_scales [[buffer(5)]],
    const device T* up_biases [[buffer(6)]],
    device T* y [[buffer(7)]],
    constant const int& M_size [[buffer(8)]],
    constant const int& K_size [[buffer(9)]],
    constant const int& N_size [[buffer(10)]],
    constant const int& GS [[buffer(11)]],
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
  int bn = RESULTS_PER_SIMDGROUP * num_simdgroups;
  int scale_step_per_thread = GS / VALUES_PER_THREAD;
  int out_row = int(n_tile) * bn + int(simd_gid) * RESULTS_PER_SIMDGROUP;
  int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
  int in_vec_size_g = K / GS;

  const device uint8_t* gate_w_base =
      gate_w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device uint8_t* up_w_base =
      up_w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device T* gate_scales_base =
      gate_scales + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* gate_biases_base =
      gate_biases + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* up_scales_base =
      up_scales + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* up_biases_base =
      up_biases + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* x_base =
      x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

  float gate_result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float up_result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];

  const device uint8_t* gate_ws = gate_w_base;
  const device uint8_t* up_ws = up_w_base;
  const device T* gate_sc = gate_scales_base;
  const device T* gate_bs = gate_biases_base;
  const device T* up_sc = up_scales_base;
  const device T* up_bs = up_biases_base;
  const device T* x_ptr = x_base;

  for (int k = 0; k < K; k += BLOCK_SIZE) {
    float x_sum = load_vector4_exact_native<T>(x_ptr, x_thread);

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        const device uint8_t* gate_wl = gate_ws + row * in_vec_size_w;
        const device uint8_t* up_wl = up_ws + row * in_vec_size_w;
        const device T* gate_sl = gate_sc + row * in_vec_size_g;
        const device T* gate_bl = gate_bs + row * in_vec_size_g;
        const device T* up_sl = up_sc + row * in_vec_size_g;
        const device T* up_bl = up_bs + row * in_vec_size_g;
        float gate_scale = float(gate_sl[0]);
        float gate_bias = float(gate_bl[0]);
        float up_scale = float(up_sl[0]);
        float up_bias = float(up_bl[0]);
        gate_result[row] += qdot4_exact_native(
            gate_wl, x_thread, gate_scale, gate_bias, x_sum);
        up_result[row] += qdot4_exact_native(
            up_wl, x_thread, up_scale, up_bias, x_sum);
      }
    }

    gate_ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    up_ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    gate_sc += BLOCK_SIZE / GS;
    gate_bs += BLOCK_SIZE / GS;
    up_sc += BLOCK_SIZE / GS;
    up_bs += BLOCK_SIZE / GS;
    x_ptr += BLOCK_SIZE;
  }

  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    int n = out_row + row;
    if (n < N) {
      float gate_sum = simd_sum(gate_result[row]);
      float up_sum = simd_sum(up_result[row]);
      if (simd_lid == 0) {
        T gate_value = T(gate_sum);
        T up_value = T(up_sum);
        y[int(m_idx) * N + n] =
            swiglu_mlx_exact_native<T>(gate_value, up_value);
      }
    }
  }
}

template <typename T>
[[kernel]] void qmv4_rowwise_down(
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
  int bn = RESULTS_PER_SIMDGROUP * num_simdgroups;
  int scale_step_per_thread = GS / VALUES_PER_THREAD;
  int out_row = int(n_tile) * bn + int(simd_gid) * RESULTS_PER_SIMDGROUP;
  int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
  int in_vec_size_g = K / GS;

  const device uint8_t* w_base =
      w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device T* scales_base =
      scales + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* biases_base =
      biases + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* x_base =
      x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

  float result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];

  const device uint8_t* ws = w_base;
  const device T* sc = scales_base;
  const device T* bs = biases_base;
  const device T* x_ptr = x_base;

  for (int k = 0; k < K; k += BLOCK_SIZE) {
    float x_sum = load_vector4_exact_native<T>(x_ptr, x_thread);

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float scale = float(sl[0]);
        float bias = float(bl[0]);
        result[row] += qdot4_exact_native(wl, x_thread, scale, bias, x_sum);
      }
    }

    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GS;
    bs += BLOCK_SIZE / GS;
    x_ptr += BLOCK_SIZE;
  }

  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    int n = out_row + row;
    if (n < N) {
      float out = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[int(m_idx) * N + n] = static_cast<T>(out);
      }
    }
  }
}

template <typename T>
[[kernel]] void qmv4_rowwise_down_residual(
    const device T* x [[buffer(0)]],
    const device uint8_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device T* residual [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant const int& M_size [[buffer(6)]],
    constant const int& K_size [[buffer(7)]],
    constant const int& N_size [[buffer(8)]],
    constant const int& GS [[buffer(9)]],
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
  int bn = RESULTS_PER_SIMDGROUP * num_simdgroups;
  int scale_step_per_thread = GS / VALUES_PER_THREAD;
  int out_row = int(n_tile) * bn + int(simd_gid) * RESULTS_PER_SIMDGROUP;
  int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
  int in_vec_size_g = K / GS;

  const device uint8_t* w_base =
      w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device T* scales_base =
      scales + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* biases_base =
      biases + out_row * in_vec_size_g +
      int(simd_lid) / scale_step_per_thread;
  const device T* x_base =
      x + int(m_idx) * K + int(simd_lid) * VALUES_PER_THREAD;

  float result[RESULTS_PER_SIMDGROUP] = {0.0f};
  float x_thread[VALUES_PER_THREAD];

  const device uint8_t* ws = w_base;
  const device T* sc = scales_base;
  const device T* bs = biases_base;
  const device T* x_ptr = x_base;

  for (int k = 0; k < K; k += BLOCK_SIZE) {
    float x_sum = load_vector4_exact_native<T>(x_ptr, x_thread);

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float scale = float(sl[0]);
        float bias = float(bl[0]);
        result[row] += qdot4_exact_native(wl, x_thread, scale, bias, x_sum);
      }
    }

    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GS;
    bs += BLOCK_SIZE / GS;
    x_ptr += BLOCK_SIZE;
  }

  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    int n = out_row + row;
    if (n < N) {
      float out = simd_sum(result[row]);
      if (simd_lid == 0) {
        int offset = int(m_idx) * N + n;
        T mlp_value = static_cast<T>(out);
        y[offset] = static_cast<T>(residual[offset] + mlp_value);
      }
    }
  }
}

// clang-format off
template [[host_name("gate_up_swiglu_qmv4_rowwise_bfloat16")]]
[[kernel]] decltype(gate_up_swiglu_qmv4_rowwise<bfloat16_t>)
gate_up_swiglu_qmv4_rowwise<bfloat16_t>;

template [[host_name("gate_up_swiglu_qmv4_rowwise_float16")]]
[[kernel]] decltype(gate_up_swiglu_qmv4_rowwise<half>)
gate_up_swiglu_qmv4_rowwise<half>;

template [[host_name("qmv4_rowwise_down_bfloat16")]]
[[kernel]] decltype(qmv4_rowwise_down<bfloat16_t>)
qmv4_rowwise_down<bfloat16_t>;

template [[host_name("qmv4_rowwise_down_float16")]]
[[kernel]] decltype(qmv4_rowwise_down<half>)
qmv4_rowwise_down<half>;

template [[host_name("qmv4_rowwise_down_residual_bfloat16")]]
[[kernel]] decltype(qmv4_rowwise_down_residual<bfloat16_t>)
qmv4_rowwise_down_residual<bfloat16_t>;

template [[host_name("qmv4_rowwise_down_residual_float16")]]
[[kernel]] decltype(qmv4_rowwise_down_residual<half>)
qmv4_rowwise_down_residual<half>;
// clang-format on
