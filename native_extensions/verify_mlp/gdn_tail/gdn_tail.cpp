#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "gdn_tail/gdn_tail.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mtplx_native {

std::string current_binary_dir();

mx::array gdn_norm_gate_out_qmv8(
    const mx::array& x,
    const mx::array& gate,
    const mx::array& norm_weight,
    const mx::array& out_w,
    const mx::array& out_scales,
    const mx::array& out_biases,
    int hv,
    float eps,
    int group_size,
    int num_simdgroups,
    mx::StreamOrDevice s) {
  if (x.ndim() != 2 || gate.ndim() != 2) {
    throw std::runtime_error("gdn_norm_gate_out_qmv8 expects 2D x/gate arrays");
  }
  if (x.shape() != gate.shape()) {
    throw std::runtime_error("x and gate shapes must match");
  }
  if (norm_weight.ndim() != 1 || norm_weight.shape()[0] != x.shape()[1]) {
    throw std::runtime_error("norm_weight must match x axis");
  }
  if (out_w.ndim() != 2 || out_scales.ndim() != 2 || out_biases.ndim() != 2) {
    throw std::runtime_error("out projection quant arrays must be 2D");
  }
  if (hv <= 0 || x.shape()[0] % hv != 0) {
    throw std::runtime_error("x row count must be divisible by hv");
  }
  if (out_w.shape()[1] * 4 != hv * x.shape()[1]) {
    throw std::runtime_error("packed out projection K does not match hv * axis");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::runtime_error("group_size must be 32, 64, or 128");
  }
  if (num_simdgroups != 2 && num_simdgroups != 4) {
    throw std::runtime_error("num_simdgroups must be 2 or 4");
  }

  auto stream = mx::to_stream(s);
  mx::Shape out_shape{x.shape()[0] / hv, out_w.shape()[0]};
  return mx::array(
      out_shape,
      x.dtype(),
      std::make_shared<GdnNormGateOutQMV8>(
          stream, hv, eps, group_size, num_simdgroups),
      {x, gate, norm_weight, out_w, out_scales, out_biases});
}

void GdnNormGateOutQMV8::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("GdnNormGateOutQMV8 has no CPU implementation.");
}

#ifdef _METAL_

void GdnNormGateOutQMV8::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& x = inputs[0];
  const auto& gate = inputs[1];
  const auto& norm_weight = inputs[2];
  const auto& out_w = inputs[3];
  const auto& out_scales = inputs[4];
  const auto& out_biases = inputs[5];
  auto& out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& compute_encoder = mx::metal::get_command_encoder(s);
  auto lib = d.get_library("mtplx_native_mlp_ext", current_binary_dir());

  int rows = static_cast<int>(x.shape()[0]);
  int dv = static_cast<int>(x.shape()[1]);
  int m = rows / hv_;
  int k_out = hv_ * dv;
  int n_out = static_cast<int>(out_w.shape()[0]);
  int rps = 4;
  int bn = rps * num_simdgroups_;

  mx::array scratch(mx::Shape{m, k_out}, x.dtype(), nullptr, {});
  scratch.set_data(mx::allocator::malloc(scratch.nbytes()));
  compute_encoder.add_temporary(scratch);

  std::string norm_kname = "gdn_norm_gate_stage_";
  norm_kname += mx::type_to_name(out.dtype());
  auto norm_kernel = d.get_kernel(norm_kname, lib);
  compute_encoder.set_compute_pipeline_state(norm_kernel);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(gate, 1);
  compute_encoder.set_input_array(norm_weight, 2);
  compute_encoder.set_output_array(scratch, 3);
  compute_encoder.set_bytes(rows, 4);
  compute_encoder.set_bytes(dv, 5);
  compute_encoder.set_bytes(hv_, 6);
  compute_encoder.set_bytes(eps_, 7);

  MTL::Size norm_group_dims = MTL::Size(32, 1, 1);
  MTL::Size norm_grid_dims = MTL::Size(static_cast<size_t>(32 * rows), 1, 1);
  compute_encoder.dispatch_threads(norm_grid_dims, norm_group_dims);

  std::string out_kname = "qmv8_rowwise_down_";
  out_kname += mx::type_to_name(out.dtype());
  auto out_kernel = d.get_kernel(out_kname, lib);
  compute_encoder.set_compute_pipeline_state(out_kernel);

  compute_encoder.set_input_array(scratch, 0);
  compute_encoder.set_input_array(out_w, 1);
  compute_encoder.set_input_array(out_scales, 2);
  compute_encoder.set_input_array(out_biases, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(m, 5);
  compute_encoder.set_bytes(k_out, 6);
  compute_encoder.set_bytes(n_out, 7);
  compute_encoder.set_bytes(group_size_, 8);

  MTL::Size qmv_group_dims =
      MTL::Size(32, static_cast<size_t>(num_simdgroups_), 1);
  size_t grid_y = static_cast<size_t>(num_simdgroups_) *
      static_cast<size_t>((n_out + bn - 1) / bn);
  MTL::Size qmv_grid_dims = MTL::Size(static_cast<size_t>(32 * m), grid_y, 1);
  compute_encoder.dispatch_threads(qmv_grid_dims, qmv_group_dims);
}

#else

void GdnNormGateOutQMV8::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("GdnNormGateOutQMV8 has no GPU implementation.");
}

#endif

std::vector<mx::array> GdnNormGateOutQMV8::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GdnNormGateOutQMV8 has no jvp implementation.");
}

std::vector<mx::array> GdnNormGateOutQMV8::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("GdnNormGateOutQMV8 has no vjp implementation.");
}

std::pair<std::vector<mx::array>, std::vector<int>> GdnNormGateOutQMV8::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("GdnNormGateOutQMV8 has no vmap implementation.");
}

bool GdnNormGateOutQMV8::is_equivalent(const mx::Primitive& other) const {
  const auto& rhs = static_cast<const GdnNormGateOutQMV8&>(other);
  return hv_ == rhs.hv_ && eps_ == rhs.eps_ &&
      group_size_ == rhs.group_size_ &&
      num_simdgroups_ == rhs.num_simdgroups_;
}

} // namespace mtplx_native
