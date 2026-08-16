#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mtplx_native {

mx::array gate_up_swiglu_qmv4_rowwise(
    const mx::array& x,
    const mx::array& gate_w,
    const mx::array& gate_scales,
    const mx::array& gate_biases,
    const mx::array& up_w,
    const mx::array& up_scales,
    const mx::array& up_biases,
    int group_size,
    int num_simdgroups = 2,
    mx::StreamOrDevice s = {});

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
    int num_simdgroups = 2,
    mx::StreamOrDevice s = {});

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
    int num_simdgroups = 2,
    mx::StreamOrDevice s = {});

class GateUpSwiGLUQMV4Rowwise : public mx::Primitive {
 public:
  explicit GateUpSwiGLUQMV4Rowwise(
      mx::Stream stream,
      int group_size,
      int num_simdgroups)
      : mx::Primitive(stream),
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
    return "GateUpSwiGLUQMV4Rowwise";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int group_size_;
  int num_simdgroups_;
};

class GateUpSwiGLUDownQMV4Rowwise : public mx::Primitive {
 public:
  explicit GateUpSwiGLUDownQMV4Rowwise(
      mx::Stream stream,
      int group_size,
      int num_simdgroups)
      : mx::Primitive(stream),
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
    return "GateUpSwiGLUDownQMV4Rowwise";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int group_size_;
  int num_simdgroups_;
};

class GateUpSwiGLUDownResidualQMV4Rowwise : public mx::Primitive {
 public:
  explicit GateUpSwiGLUDownResidualQMV4Rowwise(
      mx::Stream stream,
      int group_size,
      int num_simdgroups)
      : mx::Primitive(stream),
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
    return "GateUpSwiGLUDownResidualQMV4Rowwise";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int group_size_;
  int num_simdgroups_;
};

} // namespace mtplx_native
