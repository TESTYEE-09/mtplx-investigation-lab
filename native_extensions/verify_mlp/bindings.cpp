#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "gdn_tail/gdn_tail.h"
#include "gate_up/gate_up.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native MLX VerifyCore MLP kernel probes for MTPLX";

  m.def(
      "gate_up_swiglu_qmv4_rowwise",
      &mtplx_native::gate_up_swiglu_qmv4_rowwise,
      "x"_a,
      "gate_w"_a,
      "gate_scales"_a,
      "gate_biases"_a,
      "up_w"_a,
      "up_scales"_a,
      "up_biases"_a,
      "group_size"_a,
      "num_simdgroups"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Exact qmv-compatible fused gate+up+SwiGLU activation for small-M verify.");

  m.def(
      "gate_up_swiglu_down_qmv4_rowwise",
      &mtplx_native::gate_up_swiglu_down_qmv4_rowwise,
      "x"_a,
      "gate_w"_a,
      "gate_scales"_a,
      "gate_biases"_a,
      "up_w"_a,
      "up_scales"_a,
      "up_biases"_a,
      "down_w"_a,
      "down_scales"_a,
      "down_biases"_a,
      "group_size"_a,
      "num_simdgroups"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Managed two-dispatch exact qmv-compatible gate+up+SwiGLU+down MLP probe.");

  m.def(
      "gate_up_swiglu_down_residual_qmv4_rowwise",
      &mtplx_native::gate_up_swiglu_down_residual_qmv4_rowwise,
      "x"_a,
      "residual"_a,
      "gate_w"_a,
      "gate_scales"_a,
      "gate_biases"_a,
      "up_w"_a,
      "up_scales"_a,
      "up_biases"_a,
      "down_w"_a,
      "down_scales"_a,
      "down_biases"_a,
      "group_size"_a,
      "num_simdgroups"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Managed exact qmv-compatible MLP probe that fuses the final residual add.");

  m.def(
      "gdn_norm_gate_out_qmv8",
      &mtplx_native::gdn_norm_gate_out_qmv8,
      "x"_a,
      "gate"_a,
      "norm_weight"_a,
      "out_w"_a,
      "out_scales"_a,
      "out_biases"_a,
      "hv"_a,
      "eps"_a,
      "group_size"_a,
      "num_simdgroups"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Managed GDN RMSNormGated + int8 out_proj probe.");
}
