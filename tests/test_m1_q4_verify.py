"""Tests for the exact q4 M=2/M=3 verify lanes (Apple M1 Pro).

Three concerns:

* **Gating** — the lanes must resolve to "off" on any GPU that is not
  ``applegpu_g13s``, and the opt-out must win over every per-lane override.
* **Exactness** — the kernels must be *bit*-identical to stock MLX, and the
  dispatch must never hand a q8 weight to a 4-bit kernel (this model carries
  q4 and q8 MLPs at the same ``(K, N)``).
* **Graph-cache correctness** — a lane change must never replay a verify graph
  traced under a different lane set. That aliasing produced a fake A/B once;
  it must not be possible again.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import m1_q4_verify as lanes
from mtplx.verify_kernels import (
    exact_qmv_m23,
    exact_qmv_m23_dual,
    exact_qmv_m23_eligible,
    exact_qmv_m23_multi,
    exact_qmv_m23_multi_eligible,
)

ON_G13S = lanes.config().supported


@pytest.fixture(autouse=True)
def _restore_lane_config():
    yield
    lanes.reset_config()


def _force_arch(monkeypatch, architecture: str) -> None:
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": architecture})
    lanes.reset_config()


# -- gating -------------------------------------------------------------------


def test_unsupported_gpu_keeps_every_lane_off(monkeypatch):
    monkeypatch.setenv(lanes.MLP_ENV, "1")
    monkeypatch.setenv(lanes.GDN_ENV, "1")
    _force_arch(monkeypatch, "applegpu_g16p")

    cfg = lanes.config()
    assert not cfg.supported
    assert not cfg.mlp and not cfg.gdn_proj
    assert not cfg.any_enabled
    assert lanes.lane_signature() == ("m1q4", "off")


def test_supported_gpu_auto_enables_mlp_only(monkeypatch):
    monkeypatch.delenv(lanes.MLP_ENV, raising=False)
    monkeypatch.delenv(lanes.GDN_ENV, raising=False)
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)

    cfg = lanes.config()
    assert cfg.supported
    assert cfg.mlp is lanes._MLP_DEFAULT_ON
    assert cfg.gdn_proj is lanes._GDN_DEFAULT_ON


def test_master_opt_out_beats_per_lane_overrides(monkeypatch):
    monkeypatch.setenv(lanes.MLP_ENV, "1")
    monkeypatch.setenv(lanes.GDN_ENV, "1")
    monkeypatch.setenv(lanes.DISABLE_ENV, "1")
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)

    assert not lanes.config().any_enabled


def test_per_lane_tristate(monkeypatch):
    monkeypatch.setenv(lanes.MLP_ENV, "0")
    monkeypatch.setenv(lanes.GDN_ENV, "1")
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)

    cfg = lanes.config()
    assert not cfg.mlp
    assert cfg.gdn_proj


def test_lane_signature_separates_configurations(monkeypatch):
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)
    seen = set()
    for mlp, gdn in (("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")):
        monkeypatch.setenv(lanes.MLP_ENV, mlp)
        monkeypatch.setenv(lanes.GDN_ENV, gdn)
        lanes.reset_config()
        seen.add(lanes.lane_signature())
    assert len(seen) == 4


def test_lane_signature_tracks_the_validated_set(monkeypatch):
    monkeypatch.setenv(lanes.MLP_ENV, "1")
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)

    before = lanes.lane_signature()
    lanes._VALIDATED[("dual", mx.bfloat16, 2, 5120, 17408)] = True
    try:
        assert lanes.lane_signature() != before
    finally:
        lanes._VALIDATED.clear()


# -- eligibility --------------------------------------------------------------


def test_eligibility_is_narrow():
    x2 = mx.zeros((2, 5120), dtype=mx.bfloat16)
    w = mx.zeros((17408, 5120 // 8), dtype=mx.uint32)
    assert exact_qmv_m23_eligible(x2, w, bits=4, group_size=32)
    # wrong quant format
    assert not exact_qmv_m23_eligible(x2, w, bits=8, group_size=32)
    assert not exact_qmv_m23_eligible(x2, w, bits=4, group_size=64)
    # wrong row count
    x4 = mx.zeros((4, 5120), dtype=mx.bfloat16)
    assert not exact_qmv_m23_eligible(x4, w, bits=4, group_size=32)
    # wrong dtype. fp16/fp32 rows are accepted -- MLX promotes a bf16
    # residual against fp16 scales to fp32 -- but an integer row is not.
    x2i = mx.zeros((2, 5120), dtype=mx.int32)
    assert not exact_qmv_m23_eligible(x2i, w, bits=4, group_size=32)
    x2f = mx.zeros((2, 5120), dtype=mx.float32)
    assert exact_qmv_m23_eligible(x2f, w, bits=4, group_size=32)
    # unknown shape
    w_other = mx.zeros((4096, 5120 // 8), dtype=mx.uint32)
    assert not exact_qmv_m23_eligible(x2, w_other, bits=4, group_size=32)


# -- exactness ----------------------------------------------------------------


def _q4_linear(k: int, n: int, seed: int) -> nn.QuantizedLinear:
    mx.random.seed(seed)
    dense = nn.Linear(k, n, bias=False)
    dense.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    module = nn.QuantizedLinear.from_linear(dense, group_size=32, bits=4)
    mx.eval(module.parameters())
    return module


def _stock(module, x):
    return mx.quantized_matmul(
        x,
        module["weight"],
        scales=module["scales"],
        biases=module["biases"],
        transpose=True,
        group_size=int(module.group_size),
        bits=int(module.bits),
        mode=str(module.mode),
    )


def _assert_bit_identical(reference, candidate):
    import numpy as np

    mx.eval(reference, candidate)
    assert np.array_equal(
        np.asarray(reference.astype(mx.float32)),
        np.asarray(candidate.astype(mx.float32)),
    )


requires_g13s = pytest.mark.skipif(
    not ON_G13S, reason="kernels are only built for applegpu_g13s"
)


@requires_g13s
@pytest.mark.parametrize("m", (2, 3))
@pytest.mark.parametrize("shape", ((5120, 17408), (17408, 5120)))
def test_single_kernel_is_bit_identical(m, shape):
    k, n = shape
    module = _q4_linear(k, n, seed=11)
    x = lanes._probe_rows(m, k)
    _assert_bit_identical(
        _stock(module, x),
        exact_qmv_m23(
            x, module["weight"], module["scales"], module["biases"], group_size=32
        ),
    )


@requires_g13s
@pytest.mark.parametrize("m", (2, 3))
def test_fused_gate_up_is_bit_identical(m):
    k, n = 5120, 17408
    gate = _q4_linear(k, n, seed=21)
    up = _q4_linear(k, n, seed=22)
    x = lanes._probe_rows(m, k)
    gate_y, up_y = exact_qmv_m23_dual(x, gate, up)
    _assert_bit_identical(_stock(gate, x), gate_y)
    _assert_bit_identical(_stock(up, x), up_y)


@requires_g13s
@pytest.mark.parametrize("m", (2, 3))
def test_fused_gdn_family_is_bit_identical(m):
    k = 5120
    ns = (10240, 6144, 48, 48)
    family = tuple(_q4_linear(k, n, seed=31 + i) for i, n in enumerate(ns))
    x = lanes._probe_rows(m, k)
    assert exact_qmv_m23_multi_eligible(x, family, ns=ns, nsg=2)
    candidates = exact_qmv_m23_multi(x, family, nsg=2)
    for module, candidate in zip(family, candidates):
        _assert_bit_identical(_stock(module, x), candidate)


# -- dispatch safety ----------------------------------------------------------


@requires_g13s
def test_q8_module_is_declined_at_a_validated_q4_shape(monkeypatch):
    """The q4 and q8 MLPs share ``(K, N)``; only the quant format separates them.

    A shape-only key would feed 8-bit packing to a nibble-unpacking kernel.
    """
    monkeypatch.setenv(lanes.MLP_ENV, "1")
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)

    k, n = 17408, 5120
    lanes._VALIDATED[("single", mx.bfloat16, 2, k, n)] = True
    try:
        mx.random.seed(41)
        dense = nn.Linear(k, n, bias=False)
        dense.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        q8 = nn.QuantizedLinear.from_linear(dense, group_size=32, bits=8)
        mx.eval(q8.parameters())
        x = lanes._probe_rows(2, k)
        assert lanes.route_quantized_linear(q8, x) is None

        q4 = nn.QuantizedLinear.from_linear(dense, group_size=32, bits=4)
        mx.eval(q4.parameters())
        assert lanes.route_quantized_linear(q4, x) is not None
    finally:
        lanes._VALIDATED.clear()


def test_unvalidated_shape_is_declined(monkeypatch):
    monkeypatch.setenv(lanes.MLP_ENV, "1")
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)
    lanes._VALIDATED.clear()

    module = _q4_linear(5120, 17408, seed=51) if ON_G13S else None
    if module is None:
        pytest.skip("needs a Metal device to build the fixture")
    x = lanes._probe_rows(2, 5120)
    # Nothing validated -> stock, even though the shape is a supported one.
    assert lanes.route_quantized_linear(module, x) is None


@requires_g13s
@pytest.mark.parametrize("m", [2, 3])
def test_fp16_scales_against_bf16_rows_stay_bit_identical(m):
    """The shipped model's own dtype combination.

    ``Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`` stores q4 scales/biases in fp16
    while the residual stream is bf16. MLX promotes such a quantized matmul to
    fp32 -- it upcasts the rows and returns fp32 -- so a kernel that reads the
    bf16 rows directly reproduces qmv_fast's *bf16* partial sums and misses
    the fp32 reference by ~1%, far outside any ULP argument. Guarding the
    combination directly: it is the one the user actually runs, and it passed
    every bf16-scale fixture while being wrong.
    """
    k, n = 5120, 17408
    module = _q4_linear(k, n, seed=77)
    module["scales"] = module["scales"].astype(mx.float16)
    module["biases"] = module["biases"].astype(mx.float16)
    mx.eval(module.parameters())

    x = lanes._probe_rows(m, k)
    assert x.dtype == mx.bfloat16
    reference = _stock(module, x)
    mx.eval(reference)
    assert reference.dtype == mx.float32, "MLX should promote fp16 scales to fp32"

    candidate = exact_qmv_m23(
        x, module["weight"], module["scales"], module["biases"], group_size=32
    )
    mx.eval(candidate)
    assert candidate.dtype == mx.float32
    _assert_bit_identical(reference, candidate)


def test_validation_covers_every_supported_activation_dtype():
    """Validation must not derive its dtypes from an assumed residual.

    The residual dtype is a property of the checkpoint. Deriving the candidate
    dtypes from an assumed bf16 residual made an fp16 checkpoint validate bf16
    and fp32, run fp16, and decline every dispatch -- twenty passing checks and
    zero fused launches. Every supported dtype gets validated instead.
    """
    module = nn.QuantizedLinear(512, 512, bits=4, group_size=32)
    for scale_dtype in (mx.bfloat16, mx.float16):
        module["scales"] = module["scales"].astype(scale_dtype)
        assert lanes._activation_dtypes(module) == lanes._ACTIVATIONS
    assert set(lanes._ACTIVATIONS) == {mx.bfloat16, mx.float16, mx.float32}


def test_lane_signature_survives_mixed_dtype_validation_keys(monkeypatch):
    """Validated keys carry ``mx.Dtype``, which is not orderable.

    The signature feeds the compiled-verify graph cache key, so it is computed
    on every trace. Sorting the raw key tuples raises ``TypeError`` the moment
    two keys differ only in their activation dtype -- which is exactly the
    state a model with fp16 scales leaves behind.
    """
    monkeypatch.setenv(lanes.MLP_ENV, "1")
    _force_arch(monkeypatch, lanes.SUPPORTED_ARCHITECTURE)
    lanes._VALIDATED.clear()
    try:
        lanes._VALIDATED[("single", mx.bfloat16, 2, 5120, 17408)] = True
        lanes._VALIDATED[("single", mx.float32, 2, 5120, 17408)] = True
        signature = lanes.lane_signature()
        assert isinstance(hash(signature), int), "must be usable as a cache key"

        lanes._VALIDATED[("single", mx.float16, 2, 5120, 17408)] = True
        assert lanes.lane_signature() != signature
    finally:
        lanes._VALIDATED.clear()
