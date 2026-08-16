#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mtplx_native {

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
    int num_simdgroups = 2,
    mx::StreamOrDevice s = {});

class GdnNormGateOutQMV8 : public mx::Primitive {
 public:
  explicit GdnNormGateOutQMV8(
      mx::Stream stream,
      int hv,
      float eps,
      int group_size,
      int num_simdgroups)
      : mx::Primitive(stream),
        hv_(hv),
        eps_(eps),
        group_size_(group_size),
        num_simdgroups_(num_simdgroups) {};

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "GdnNormGateOutQMV8";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int hv_;
  float eps_;
  int group_size_;
  int num_simdgroups_;
};

} // namespace mtplx_native
