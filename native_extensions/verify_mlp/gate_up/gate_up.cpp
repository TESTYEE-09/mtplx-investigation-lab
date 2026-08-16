#include <dlfcn.h>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "gate_up/gate_up.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mtplx_native {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

mx::array gate_up_swiglu_qmv4_rowwise(
    const mx::array& x,
    const mx::array& gate_w,
    const mx::array& gate_scales,
    const mx::array& gate_biases,
    const mx::array& up_w,
    const mx::array& up_scales,
    const mx::array& up_biases,
    int group_size,
    int num_simdgroups,
    mx::StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::runtime_error("gate_up_swiglu_qmv4_rowwise expects a 2D x array");
  }
  if (gate_w.ndim() != 2 || up_w.ndim() != 2) {
    throw std::runtime_error("gate/up weights must be 2D");
  }
  if (gate_w.shape() != up_w.shape()) {
    throw std::runtime_error("gate/up weight shapes must match");
  }
  if (gate_scales.shape() != up_scales.shape() ||
      gate_biases.shape() != up_biases.shape()) {
    throw std::runtime_error("gate/up quant parameter shapes must match");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::runtime_error("group_size must be 32, 64, or 128");
  }
  if (num_simdgroups != 2 && num_simdgroups != 4) {
    throw std::runtime_error("num_simdgroups must be 2 or 4");
  }

  auto stream = mx::to_stream(s);

  mx::Shape out_shape{x.shape()[0], gate_w.shape()[0]};
  return mx::array(
      out_shape,
      x.dtype(),
      std::make_shared<GateUpSwiGLUQMV4Rowwise>(
          stream, group_size, num_simdgroups),
      {
          x,
          gate_w,
          gate_scales,
          gate_biases,
          up_w,
          up_scales,
          up_biases,
      });
}

mx::array gate_up_swiglu_down_qmv4_rowwise(
    const mx::array& x,
    const mx::array& gate_w,
    const mx::array& gate_scales,
    const mx::array& gate_biases,
    const mx::array& up_w,
    const mx::array& up_scales,
    const mx::array& up_biases,
    const mx::array& down_w,
    const mx::array& down_scales,
    const mx::array& down_biases,
    int group_size,
    int num_simdgroups,
    mx::StreamOrDevice s) {
  if (x.ndim() != 2) {
    throw std::runtime_error(
        "gate_up_swiglu_down_qmv4_rowwise expects a 2D x array");
  }
  if (gate_w.ndim() != 2 || up_w.ndim() != 2 || down_w.ndim() != 2) {
    throw std::runtime_error("gate/up/down weights must be 2D");
  }
  if (gate_w.shape() != up_w.shape()) {
    throw std::runtime_error("gate/up weight shapes must match");
  }
  if (gate_scales.shape() != up_scales.shape() ||
      gate_biases.shape() != up_biases.shape()) {
    throw std::runtime_error("gate/up quant parameter shapes must match");
  }
  if (gate_w.shape()[0] != down_w.shape()[1] * 8) {
    throw std::runtime_error(
        "down weight packed input dimension must match gate/up output size");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::runtime_error("group_size must be 32, 64, or 128");
  }
  if (num_simdgroups != 2 && num_simdgroups != 4) {
    throw std::runtime_error("num_simdgroups must be 2 or 4");
  }

  auto stream = mx::to_stream(s);

  mx::Shape out_shape{x.shape()[0], down_w.shape()[0]};
  return mx::array(
      out_shape,
      x.dtype(),
      std::make_shared<GateUpSwiGLUDownQMV4Rowwise>(
          stream, group_size, num_simdgroups),
      {
          x,
          gate_w,
          gate_scales,
          gate_biases,
          up_w,
          up_scales,
          up_biases,
          down_w,
          down_scales,
          down_biases,
      });
}

mx::array gate_up_swiglu_down_residual_qmv4_rowwise(
    const mx::array& x,
    const mx::array& residual,
    const mx::array& gate_w,
    const mx::array& gate_scales,
    const mx::array& gate_biases,
    const mx::array& up_w,
    const mx::array& up_scales,
    const mx::array& up_biases,
    const mx::array& down_w,
    const mx::array& down_scales,
    const mx::array& down_biases,
    int group_size,
    int num_simdgroups,
    mx::StreamOrDevice s) {
  if (x.ndim() != 2 || residual.ndim() != 2) {
    throw std::runtime_error(
        "gate_up_swiglu_down_residual_qmv4_rowwise expects 2D x/residual arrays");
  }
  if (gate_w.ndim() != 2 || up_w.ndim() != 2 || down_w.ndim() != 2) {
    throw std::runtime_error("gate/up/down weights must be 2D");
  }
  if (gate_w.shape() != up_w.shape()) {
    throw std::runtime_error("gate/up weight shapes must match");
  }
  if (gate_scales.shape() != up_scales.shape() ||
      gate_biases.shape() != up_biases.shape()) {
    throw std::runtime_error("gate/up quant parameter shapes must match");
  }
  if (gate_w.shape()[0] != down_w.shape()[1] * 8) {
    throw std::runtime_error(
        "down weight packed input dimension must match gate/up output size");
  }
  if (residual.shape()[0] != x.shape()[0] ||
      residual.shape()[1] != down_w.shape()[0]) {
    throw std::runtime_error(
        "residual shape must be [M, down output dimension]");
  }
  if (residual.dtype() != x.dtype()) {
    throw std::runtime_error("residual dtype must match x dtype");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::runtime_error("group_size must be 32, 64, or 128");
  }
  if (num_simdgroups != 2 && num_simdgroups != 4) {
    throw std::runtime_error("num_simdgroups must be 2 or 4");
  }

  auto stream = mx::to_stream(s);

  mx::Shape out_shape{x.shape()[0], down_w.shape()[0]};
  return mx::array(
      out_shape,
      x.dtype(),
      std::make_shared<GateUpSwiGLUDownResidualQMV4Rowwise>(
          stream, group_size, num_simdgroups),
      {
          x,
          residual,
          gate_w,
          gate_scales,
          gate_biases,
          up_w,
          up_scales,
          up_biases,
          down_w,
          down_scales,
          down_biases,
      });
}

void GateUpSwiGLUQMV4Rowwise::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUQMV4Rowwise has no CPU implementation.");
}

void GateUpSwiGLUDownQMV4Rowwise::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownQMV4Rowwise has no CPU implementation.");
}

void GateUpSwiGLUDownResidualQMV4Rowwise::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownResidualQMV4Rowwise has no CPU implementation.");
}

#ifdef _METAL_

void GateUpSwiGLUQMV4Rowwise::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& gate_w = inputs[1];
  const auto& gate_scales = inputs[2];
  const auto& gate_biases = inputs[3];
  const auto& up_w = inputs[4];
  const auto& up_scales = inputs[5];
  const auto& up_biases = inputs[6];
  auto& out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  std::string kname = "gate_up_swiglu_qmv4_rowwise_";
  kname += mx::type_to_name(out.dtype());
  auto lib = d.get_library("mtplx_native_mlp_ext", current_binary_dir());
  auto kernel = d.get_kernel(kname, lib);

  auto& compute_encoder = mx::metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int M = static_cast<int>(x.shape()[0]);
  int K = static_cast<int>(x.shape()[1]);
  int N = static_cast<int>(gate_w.shape()[0]);
  int rps = 4;

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(gate_w, 1);
  compute_encoder.set_input_array(gate_scales, 2);
  compute_encoder.set_input_array(gate_biases, 3);
  compute_encoder.set_input_array(up_w, 4);
  compute_encoder.set_input_array(up_scales, 5);
  compute_encoder.set_input_array(up_biases, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_bytes(M, 8);
  compute_encoder.set_bytes(K, 9);
  compute_encoder.set_bytes(N, 10);
  compute_encoder.set_bytes(group_size_, 11);

  int bn = rps * num_simdgroups_;
  size_t grid_y = static_cast<size_t>(num_simdgroups_) *
      static_cast<size_t>((N + bn - 1) / bn);
  MTL::Size group_dims =
      MTL::Size(32, static_cast<size_t>(num_simdgroups_), 1);
  MTL::Size grid_dims =
      MTL::Size(static_cast<size_t>(32 * M), grid_y, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void GateUpSwiGLUDownQMV4Rowwise::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& gate_w = inputs[1];
  const auto& gate_scales = inputs[2];
  const auto& gate_biases = inputs[3];
  const auto& up_w = inputs[4];
  const auto& up_scales = inputs[5];
  const auto& up_biases = inputs[6];
  const auto& down_w = inputs[7];
  const auto& down_scales = inputs[8];
  const auto& down_biases = inputs[9];
  auto& out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& compute_encoder = mx::metal::get_command_encoder(s);
  auto lib = d.get_library("mtplx_native_mlp_ext", current_binary_dir());

  int M = static_cast<int>(x.shape()[0]);
  int K = static_cast<int>(x.shape()[1]);
  int N_act = static_cast<int>(gate_w.shape()[0]);
  int N_out = static_cast<int>(down_w.shape()[0]);
  int rps = 4;
  int bn = rps * num_simdgroups_;

  mx::array scratch(mx::Shape{M, N_act}, x.dtype(), nullptr, {});
  scratch.set_data(mx::allocator::malloc(scratch.nbytes()));
  compute_encoder.add_temporary(scratch);

  std::string gate_kname = "gate_up_swiglu_qmv4_rowwise_";
  gate_kname += mx::type_to_name(out.dtype());
  auto gate_kernel = d.get_kernel(gate_kname, lib);
  compute_encoder.set_compute_pipeline_state(gate_kernel);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(gate_w, 1);
  compute_encoder.set_input_array(gate_scales, 2);
  compute_encoder.set_input_array(gate_biases, 3);
  compute_encoder.set_input_array(up_w, 4);
  compute_encoder.set_input_array(up_scales, 5);
  compute_encoder.set_input_array(up_biases, 6);
  compute_encoder.set_output_array(scratch, 7);
  compute_encoder.set_bytes(M, 8);
  compute_encoder.set_bytes(K, 9);
  compute_encoder.set_bytes(N_act, 10);
  compute_encoder.set_bytes(group_size_, 11);

  size_t gate_grid_y = static_cast<size_t>(num_simdgroups_) *
      static_cast<size_t>((N_act + bn - 1) / bn);
  MTL::Size group_dims =
      MTL::Size(32, static_cast<size_t>(num_simdgroups_), 1);
  MTL::Size gate_grid_dims =
      MTL::Size(static_cast<size_t>(32 * M), gate_grid_y, 1);
  compute_encoder.dispatch_threads(gate_grid_dims, group_dims);

  std::string down_kname = "qmv4_rowwise_down_";
  down_kname += mx::type_to_name(out.dtype());
  auto down_kernel = d.get_kernel(down_kname, lib);
  compute_encoder.set_compute_pipeline_state(down_kernel);

  int K_down = N_act;
  compute_encoder.set_input_array(scratch, 0);
  compute_encoder.set_input_array(down_w, 1);
  compute_encoder.set_input_array(down_scales, 2);
  compute_encoder.set_input_array(down_biases, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(M, 5);
  compute_encoder.set_bytes(K_down, 6);
  compute_encoder.set_bytes(N_out, 7);
  compute_encoder.set_bytes(group_size_, 8);

  size_t down_grid_y = static_cast<size_t>(num_simdgroups_) *
      static_cast<size_t>((N_out + bn - 1) / bn);
  MTL::Size down_grid_dims =
      MTL::Size(static_cast<size_t>(32 * M), down_grid_y, 1);
  compute_encoder.dispatch_threads(down_grid_dims, group_dims);
}

void GateUpSwiGLUDownResidualQMV4Rowwise::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& residual = inputs[1];
  const auto& gate_w = inputs[2];
  const auto& gate_scales = inputs[3];
  const auto& gate_biases = inputs[4];
  const auto& up_w = inputs[5];
  const auto& up_scales = inputs[6];
  const auto& up_biases = inputs[7];
  const auto& down_w = inputs[8];
  const auto& down_scales = inputs[9];
  const auto& down_biases = inputs[10];
  auto& out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& compute_encoder = mx::metal::get_command_encoder(s);
  auto lib = d.get_library("mtplx_native_mlp_ext", current_binary_dir());

  int M = static_cast<int>(x.shape()[0]);
  int K = static_cast<int>(x.shape()[1]);
  int N_act = static_cast<int>(gate_w.shape()[0]);
  int N_out = static_cast<int>(down_w.shape()[0]);
  int rps = 4;
  int bn = rps * num_simdgroups_;

  mx::array scratch(mx::Shape{M, N_act}, x.dtype(), nullptr, {});
  scratch.set_data(mx::allocator::malloc(scratch.nbytes()));
  compute_encoder.add_temporary(scratch);

  std::string gate_kname = "gate_up_swiglu_qmv4_rowwise_";
  gate_kname += mx::type_to_name(out.dtype());
  auto gate_kernel = d.get_kernel(gate_kname, lib);
  compute_encoder.set_compute_pipeline_state(gate_kernel);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(gate_w, 1);
  compute_encoder.set_input_array(gate_scales, 2);
  compute_encoder.set_input_array(gate_biases, 3);
  compute_encoder.set_input_array(up_w, 4);
  compute_encoder.set_input_array(up_scales, 5);
  compute_encoder.set_input_array(up_biases, 6);
  compute_encoder.set_output_array(scratch, 7);
  compute_encoder.set_bytes(M, 8);
  compute_encoder.set_bytes(K, 9);
  compute_encoder.set_bytes(N_act, 10);
  compute_encoder.set_bytes(group_size_, 11);

  size_t gate_grid_y = static_cast<size_t>(num_simdgroups_) *
      static_cast<size_t>((N_act + bn - 1) / bn);
  MTL::Size group_dims =
      MTL::Size(32, static_cast<size_t>(num_simdgroups_), 1);
  MTL::Size gate_grid_dims =
      MTL::Size(static_cast<size_t>(32 * M), gate_grid_y, 1);
  compute_encoder.dispatch_threads(gate_grid_dims, group_dims);

  std::string down_kname = "qmv4_rowwise_down_residual_";
  down_kname += mx::type_to_name(out.dtype());
  auto down_kernel = d.get_kernel(down_kname, lib);
  compute_encoder.set_compute_pipeline_state(down_kernel);

  int K_down = N_act;
  compute_encoder.set_input_array(scratch, 0);
  compute_encoder.set_input_array(down_w, 1);
  compute_encoder.set_input_array(down_scales, 2);
  compute_encoder.set_input_array(down_biases, 3);
  compute_encoder.set_input_array(residual, 4);
  compute_encoder.set_output_array(out, 5);
  compute_encoder.set_bytes(M, 6);
  compute_encoder.set_bytes(K_down, 7);
  compute_encoder.set_bytes(N_out, 8);
  compute_encoder.set_bytes(group_size_, 9);

  size_t down_grid_y = static_cast<size_t>(num_simdgroups_) *
      static_cast<size_t>((N_out + bn - 1) / bn);
  MTL::Size down_grid_dims =
      MTL::Size(static_cast<size_t>(32 * M), down_grid_y, 1);
  compute_encoder.dispatch_threads(down_grid_dims, group_dims);
}

#else

void GateUpSwiGLUQMV4Rowwise::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUQMV4Rowwise has no GPU implementation.");
}

void GateUpSwiGLUDownQMV4Rowwise::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownQMV4Rowwise has no GPU implementation.");
}

void GateUpSwiGLUDownResidualQMV4Rowwise::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownResidualQMV4Rowwise has no GPU implementation.");
}

#endif

std::vector<mx::array> GateUpSwiGLUQMV4Rowwise::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GateUpSwiGLUQMV4Rowwise has no jvp implementation.");
}

std::vector<mx::array> GateUpSwiGLUDownQMV4Rowwise::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownQMV4Rowwise has no jvp implementation.");
}

std::vector<mx::array> GateUpSwiGLUDownResidualQMV4Rowwise::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownResidualQMV4Rowwise has no jvp implementation.");
}

std::vector<mx::array> GateUpSwiGLUQMV4Rowwise::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GateUpSwiGLUQMV4Rowwise has no vjp implementation.");
}

std::vector<mx::array> GateUpSwiGLUDownQMV4Rowwise::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownQMV4Rowwise has no vjp implementation.");
}

std::vector<mx::array> GateUpSwiGLUDownResidualQMV4Rowwise::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownResidualQMV4Rowwise has no vjp implementation.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GateUpSwiGLUQMV4Rowwise::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GateUpSwiGLUQMV4Rowwise has no vmap implementation.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GateUpSwiGLUDownQMV4Rowwise::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownQMV4Rowwise has no vmap implementation.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
GateUpSwiGLUDownResidualQMV4Rowwise::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "GateUpSwiGLUDownResidualQMV4Rowwise has no vmap implementation.");
}

bool GateUpSwiGLUQMV4Rowwise::is_equivalent(
    const mx::Primitive& other) const {
  const auto& rhs = static_cast<const GateUpSwiGLUQMV4Rowwise&>(other);
  return group_size_ == rhs.group_size_ &&
      num_simdgroups_ == rhs.num_simdgroups_;
}

bool GateUpSwiGLUDownQMV4Rowwise::is_equivalent(
    const mx::Primitive& other) const {
  const auto& rhs = static_cast<const GateUpSwiGLUDownQMV4Rowwise&>(other);
  return group_size_ == rhs.group_size_ &&
      num_simdgroups_ == rhs.num_simdgroups_;
}

bool GateUpSwiGLUDownResidualQMV4Rowwise::is_equivalent(
    const mx::Primitive& other) const {
  const auto& rhs =
      static_cast<const GateUpSwiGLUDownResidualQMV4Rowwise&>(other);
  return group_size_ == rhs.group_size_ &&
      num_simdgroups_ == rhs.num_simdgroups_;
}

} // namespace mtplx_native
