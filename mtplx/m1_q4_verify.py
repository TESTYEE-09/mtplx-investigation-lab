"""Exact q4 M=2/M=3 verify projections for Apple M1 Pro (``applegpu_g13s``).

Speculative verification on Qwen3.8-27B replays the accepted prefix plus the
draft rows through the decoder, so every verify forward runs the quantized
projections at ``M = speculative_depth + 1`` rows.  MLX 0.32 only has a
row-parallel verify kernel from ``M >= 4``; at ``M = 2`` and ``M = 3`` it falls
back to ``qmv_fast``, one Metal launch per projection, each re-reading the same
hidden state.

This module ships two *shape-specialized* replacements that keep ``qmv_fast``'s
arithmetic and SIMD reduction order byte for byte and only change scheduling:

``mlp``
    ``gate_proj`` and ``up_proj`` share one launch and one read of ``x``; the
    stock ``swiglu`` and an exact ``down_proj`` launch follow (56 layers).

``gdn_proj``
    ``in_proj_qkv``/``z``/``b``/``a`` share one launch instead of four
    (48 layers).

Both are guarded three ways:

1. **Device** — only ``applegpu_g13s``; every other GPU keeps stock MLX.
2. **Shape/dtype** — only affine q4, group 32, BF16, the exact Qwen3.8 matrix
   shapes, ``M in (2, 3)``, and non-prefill phases.
3. **Loaded weights** — :func:`validate` runs every candidate launch against
   stock MLX on the model's *own* weights before any verify graph is compiled,
   and only ``(kind, M, K, N, ...)`` keys that compare bit-identical are
   allowed to serve.  Anything unchecked, failing, or raising stays on stock.

The lanes are process-level: they are resolved once at load and never flip
underneath a compiled graph.  :func:`lane_signature` feeds the compiled-verify
graph cache key so a configuration change can never alias a graph traced under
a different lane set (see ``graphbank._shared_or_new_verify_step``).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# The only GPU architecture these kernels have been validated on.
SUPPORTED_ARCHITECTURE = "applegpu_g13s"

# Qwen3.8-27B Optimized-Speed q4 shapes served by the MLP lane.
_MLP_K = 5120
_MLP_N = 17408

#: Master opt-out. Set to a truthy value to keep stock MLX everywhere.
DISABLE_ENV = "MTPLX_DISABLE_M1_Q4_VERIFY"
#: Per-lane tri-state overrides: unset = auto, "0" = off, "1" = force on.
MLP_ENV = "MTPLX_M1_Q4_MLP"
GDN_ENV = "MTPLX_M1_Q4_GDN_PROJ"
#: Simdgroups per threadgroup for the fused GDN launch.
GDN_NSG_ENV = "MTPLX_M1_Q4_GDN_NSG"
#: Opt-in lane engagement counters (research instrumentation; off by default).
STATS_ENV = "MTPLX_M1_Q4_STATS"

# Auto-enable defaults on supported hardware. Both lanes are measured wins in
# the balanced cross-process bracket, and the combination is the fastest exact
# configuration at both D1 and D2 -- notably faster than either lane alone, so
# neither is shipped without the other. See docs/perf/m1-q4-verify-lanes.md.
_MLP_DEFAULT_ON = True
_GDN_DEFAULT_ON = True

_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no"}


@dataclass(frozen=True)
class LaneConfig:
    """The effective lane configuration for this process."""

    architecture: str
    supported: bool
    mlp: bool
    gdn_proj: bool
    gdn_nsg: int

    @property
    def any_enabled(self) -> bool:
        return self.mlp or self.gdn_proj

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "supported": self.supported,
            "mlp": self.mlp,
            "gdn_proj": self.gdn_proj,
            "gdn_nsg": self.gdn_nsg,
        }


def _tristate(name: str, *, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return default


@lru_cache(maxsize=1)
def _architecture() -> str:
    try:
        return str(mx.device_info().get("architecture", ""))
    except Exception:  # noqa: BLE001 - non-Metal builds, CPU-only CI
        return ""


@lru_cache(maxsize=1)
def config() -> LaneConfig:
    """Resolve the lane configuration once per process.

    These gates sit on the graph-construction path of every quantized
    projection of every layer, so the environment and device lookups are
    resolved a single time rather than per call.  Call :func:`reset_config`
    after mutating the environment (tests and lab harnesses only).
    """
    architecture = _architecture()
    supported = architecture.lower() == SUPPORTED_ARCHITECTURE
    if not supported or _tristate(DISABLE_ENV, default=False):
        return LaneConfig(architecture, supported, False, False, 2)
    try:
        nsg = int(os.environ.get(GDN_NSG_ENV, "2") or 2)
    except ValueError:
        nsg = 2
    return LaneConfig(
        architecture=architecture,
        supported=True,
        mlp=_tristate(MLP_ENV, default=_MLP_DEFAULT_ON),
        gdn_proj=_tristate(GDN_ENV, default=_GDN_DEFAULT_ON),
        gdn_nsg=max(1, min(8, nsg)),
    )


def reset_config() -> None:
    """Re-resolve the cached lane configuration and drop validation state."""
    config.cache_clear()
    _architecture.cache_clear()
    _stats_enabled.cache_clear()
    _VALIDATED.clear()


def lane_signature() -> tuple[Any, ...]:
    """Stable identity of everything that changes the emitted verify graph.

    Consumed by the compiled-verify graph cache key.  Two processes (or two
    phases of one process) that disagree on this tuple must never share a
    traced graph.
    """
    cfg = config()
    if not cfg.any_enabled:
        return ("m1q4", "off")
    # The lane flags alone are not enough: a lane only emits a fused launch for
    # shapes that cleared load-time validation, so the cleared set is part of
    # the graph's identity.
    # Rendered to text before sorting: the keys carry ``mx.Dtype`` members,
    # which are equality-comparable but not orderable, so sorting the raw
    # tuples raises as soon as two keys differ only in their dtype.
    served = tuple(sorted(str(key) for key, ok in _VALIDATED.items() if ok))
    return ("m1q4", cfg.mlp, cfg.gdn_proj, cfg.gdn_nsg, served)


# -- engagement counters ------------------------------------------------------

#: Research instrumentation. Only written when ``MTPLX_M1_Q4_STATS`` is set:
#: "the tokens matched" is not evidence that a fused launch actually served,
#: so the lab harnesses need a way to prove which path the compiled graph took.
LANE_STATS: dict[str, int] = {
    "mlp_fused": 0,
    "mlp_stock": 0,
    "qlinear_fused": 0,
    "gdn_fused": 0,
    "gdn_stock": 0,
}


@lru_cache(maxsize=1)
def _stats_enabled() -> bool:
    return _tristate(STATS_ENV, default=False)


def _tick(name: str) -> None:
    if _stats_enabled():
        LANE_STATS[name] += 1


def reset_lane_stats() -> None:
    for key in LANE_STATS:
        LANE_STATS[key] = 0


#: Optional path for the exit dump. The tuner runs each candidate in its own
#: subprocess whose stdout it parses, so a harness that only reads the parent's
#: output cannot otherwise tell whether the fused launches served. Research
#: instrumentation: inert unless ``MTPLX_M1_Q4_STATS`` is also set.
STATS_FILE_ENV = "MTPLX_M1_Q4_STATS_FILE"


def _dump_lane_stats() -> None:
    """Append this process's engagement counters at exit."""
    path = os.environ.get(STATS_FILE_ENV)
    if not path:
        return
    payload = {"pid": os.getpid(), "argv": sys.argv[1:], "stats": dict(LANE_STATS)}
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:  # never let instrumentation break a run
        pass


if _stats_enabled():
    atexit.register(_dump_lane_stats)


# -- load-time validation -----------------------------------------------------

# key -> bit-identical against stock MLX on this model's own weights. Every key
# carries the activation dtype: MLX promotes a quantized matmul across the
# activation and scale dtypes, so the same projection is a different
# computation depending on the rows it is handed.
_VALIDATED: dict[tuple, bool] = {}

# Activation dtypes the lanes will consider. Anything else declines to stock.
_ACTIVATIONS = (mx.bfloat16, mx.float16, mx.float32)

_GDN_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _stock_qmm(module: Any, x: mx.array) -> mx.array:
    """Stock MLX ``QuantizedLinear`` arithmetic, independent of any patch."""
    return mx.quantized_matmul(
        x,
        module["weight"],
        scales=module["scales"],
        biases=module.get("biases"),
        transpose=True,
        group_size=int(module.group_size),
        bits=int(module.bits),
        mode=str(module.mode),
    )


def _probe_rows(m: int, k: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Deterministic, nontrivial probe rows.

    Exercises signs, both scale and bias terms, and the exact SIMD reduction
    order; a constant or one-hot probe would pass even for a wrong kernel.
    Built in fp32 and cast, so each dtype is probed at its own precision.
    """
    base = (mx.arange(k, dtype=mx.float32) % 251 - 125) / 64
    return mx.stack(
        [(base * float(row + 1) / float(m)).astype(dtype) for row in range(m)]
    )


def _activation_dtypes(module: Any) -> tuple[mx.Dtype, ...]:
    """Activation dtypes to validate this projection under.

    Every supported dtype, deliberately: the residual dtype is a property of
    the checkpoint, not a constant, and MLX promotes a quantized matmul across
    the activation *and* scale dtypes, so the dtype a projection is called with
    depends on both. Deriving the candidates from an assumed residual is how
    this shipped inert once -- an fp16 checkpoint validated bf16 and fp32, ran
    fp16, and declined every dispatch while reporting twenty passing checks.
    Validating all three costs a few hundred milliseconds at load and removes
    the guess. A dtype that was never validated simply never dispatches.
    """
    del module  # dtype coverage is fixed; kept for call-site symmetry
    return _ACTIVATIONS


def _bit_identical(reference: mx.array, candidate: mx.array) -> bool:
    import numpy as np

    return bool(
        np.array_equal(
            np.asarray(reference.astype(mx.float32)),
            np.asarray(candidate.astype(mx.float32)),
        )
    )


def _quantized(module: Any, *, bits: int = 4) -> bool:
    return (
        module is not None
        and int(getattr(module, "bits", 0) or 0) == bits
        and int(getattr(module, "group_size", 0) or 0) == 32
        and str(getattr(module, "mode", "affine")) == "affine"
    )


def _same_scale_dtype(*modules: Any) -> bool:
    """Do these projections agree on their scale/bias dtype?

    The fused kernels carry one ``S`` template parameter for every projection
    they serve, and this model family ships fp16 scales against bf16
    activations, so the dtypes are genuinely independent and must be checked.
    """
    dtypes = {module["scales"].dtype for module in modules}
    dtypes |= {module["biases"].dtype for module in modules}
    return len(dtypes) == 1


def _gdn_family(gdn: Any) -> tuple[Any, ...] | None:
    modules = []
    for name in _GDN_PROJECTIONS:
        module = getattr(gdn, name, None)
        if module is None:
            return None
        modules.append(module)
    return tuple(modules)


def _language_layers(model: Any) -> list[Any] | None:
    try:
        return list(model.language_model.model.layers)
    except AttributeError:
        return None


def _validate_mlp(layers: list[Any], report: dict[str, Any]) -> None:
    from mlx_lm.models.qwen3_next import swiglu

    from .verify_kernels import exact_qmv_m23, exact_qmv_m23_dual, exact_qmv_m23_eligible

    # One representative per distinct shape: the weights differ per layer but
    # the kernel schedule and reduction order do not, and a 56-layer sweep
    # would add seconds to every load for no additional coverage.
    single: dict[tuple[int, int], Any] = {}
    triples: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        for module in (gate, up, down):
            if not _quantized(module):
                continue
            key = (int(module["weight"].shape[1]) * 8, int(module["weight"].shape[0]))
            single.setdefault(key, module)
        if _quantized(gate) and _quantized(up) and _quantized(down):
            if tuple(gate["weight"].shape) == tuple(up["weight"].shape) and (
                _same_scale_dtype(gate, up)
            ):
                shape = (
                    int(gate["weight"].shape[1]) * 8,
                    int(gate["weight"].shape[0]),
                )
                triples.setdefault(shape, (gate, up, down))

    for (k, n), module in sorted(single.items()):
        for dtype in _activation_dtypes(module):
            for m in (2, 3):
                item: dict[str, Any] = {
                    "lane": "mlp",
                    "kind": "single",
                    "m": m,
                    "k": k,
                    "n": n,
                    "dtype": str(dtype),
                }
                x = _probe_rows(m, k, dtype)
                if not exact_qmv_m23_eligible(
                    x, module["weight"], bits=4, group_size=int(module.group_size)
                ):
                    continue
                try:
                    reference = _stock_qmm(module, x)
                    candidate = exact_qmv_m23(
                        x,
                        module["weight"],
                        module["scales"],
                        module["biases"],
                        group_size=int(module.group_size),
                    )
                    mx.eval(reference, candidate)
                    equal = _bit_identical(reference, candidate)
                except Exception as exc:  # noqa: BLE001
                    equal = False
                    item["error"] = f"{type(exc).__name__}: {exc}"
                _VALIDATED[("single", dtype, m, k, n)] = equal
                _record(report, item, equal)

    for (k, n), (gate, up, down) in sorted(triples.items()):
        if (k, n) != (_MLP_K, _MLP_N):
            continue
        for dtype, m in ((d, r) for d in _activation_dtypes(gate) for r in (2, 3)):
            item = {
                "lane": "mlp",
                "kind": "dual",
                "m": m,
                "k": k,
                "n": n,
                "dtype": str(dtype),
            }
            try:
                x = _probe_rows(m, k, dtype)
                reference_gate = _stock_qmm(gate, x)
                reference_up = _stock_qmm(up, x)
                candidate_gate, candidate_up = exact_qmv_m23_dual(x, gate, up)
                mx.eval(reference_gate, reference_up, candidate_gate, candidate_up)
                equal = _bit_identical(reference_gate, candidate_gate) and _bit_identical(
                    reference_up, candidate_up
                )
            except Exception as exc:  # noqa: BLE001
                equal = False
                item["error"] = f"{type(exc).__name__}: {exc}"
            _VALIDATED[("dual", dtype, m, k, n)] = equal
            _record(report, item, equal)

            # End-to-end: the fused pair, stock SwiGLU, and the exact down
            # launch must reproduce the stock MLP output exactly.
            whole = {
                "lane": "mlp",
                "kind": "mlp",
                "m": m,
                "k": k,
                "n": n,
                "dtype": str(dtype),
            }
            try:
                reference = _stock_qmm(down, swiglu(reference_gate, reference_up))
                candidate = _stock_qmm(down, swiglu(candidate_gate, candidate_up))
                mx.eval(reference, candidate)
                whole_equal = equal and _bit_identical(reference, candidate)
            except Exception as exc:  # noqa: BLE001
                whole_equal = False
                whole["error"] = f"{type(exc).__name__}: {exc}"
            _VALIDATED[("mlp", dtype, m, k, n)] = whole_equal
            _record(report, whole, whole_equal)


def _validate_gdn(layers: list[Any], report: dict[str, Any], nsg: int) -> None:
    from .verify_kernels import exact_qmv_m23_multi, exact_qmv_m23_multi_eligible

    representatives: dict[tuple, tuple[Any, ...]] = {}
    families = 0
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        family = _gdn_family(gdn)
        if family is None or not all(_quantized(module) for module in family):
            continue
        families += 1
        k = int(family[0]["weight"].shape[1]) * 8
        ns = tuple(int(module["weight"].shape[0]) for module in family)
        representatives.setdefault((k, ns), family)
    report["gdn_layers"] = families

    for (k, ns), family in sorted(representatives.items()):
        for dtype, m in ((d, r) for d in _activation_dtypes(family[0]) for r in (2, 3)):
            item: dict[str, Any] = {
                "lane": "gdn_proj",
                "m": m,
                "k": k,
                "ns": list(ns),
                "nsg": nsg,
                "dtype": str(dtype),
            }
            x = _probe_rows(m, k, dtype)
            if not exact_qmv_m23_multi_eligible(x, family, ns=ns, nsg=nsg):
                item["eligible"] = False
                _VALIDATED[("gdn", dtype, m, k, ns, nsg)] = False
                _record(report, item, False)
                continue
            try:
                references = [_stock_qmm(module, x) for module in family]
                candidates = exact_qmv_m23_multi(x, family, nsg=nsg)
                mx.eval(*references, *candidates)
                equal = all(
                    _bit_identical(reference, candidate)
                    for reference, candidate in zip(references, candidates)
                )
            except Exception as exc:  # noqa: BLE001
                equal = False
                item["error"] = f"{type(exc).__name__}: {exc}"
            _VALIDATED[("gdn", dtype, m, k, ns, nsg)] = equal
            _record(report, item, equal)


def _record(report: dict[str, Any], item: dict[str, Any], passed: bool) -> None:
    item["passed"] = passed
    report["checks"].append(item)
    report["passed" if passed else "failed"] += 1


def validate(model: Any) -> dict[str, Any]:
    """Prove every enabled lane bit-identical on this model's own weights.

    Must run before the first verify compilation: self-checking inside the
    projection call is unsafe because the first invocation may be an
    ``mx.compile`` trace, where an eager comparison cannot run.  A lane whose
    checks do not all pass simply never serves — the per-key registry is the
    only thing the dispatch helpers consult.
    """
    cfg = config()
    report: dict[str, Any] = {
        "architecture": cfg.architecture,
        "supported": cfg.supported,
        "mlp": cfg.mlp,
        "gdn_proj": cfg.gdn_proj,
        "gdn_nsg": cfg.gdn_nsg,
        "passed": 0,
        "failed": 0,
        "checks": [],
    }
    _VALIDATED.clear()
    if not cfg.any_enabled:
        report["ready"] = False
        report["reason"] = "disabled" if cfg.supported else "unsupported_gpu"
        return report

    layers = _language_layers(model)
    if layers is None:
        report["ready"] = False
        report["reason"] = "qwen3_next_layers_not_found"
        return report

    if cfg.mlp:
        _validate_mlp(layers, report)
    if cfg.gdn_proj:
        _validate_gdn(layers, report, cfg.gdn_nsg)

    report["ready"] = bool(report["checks"]) and report["failed"] == 0
    if report["failed"]:
        # Loud: a failure here means the machine disagrees with the validated
        # configuration and every affected shape has silently gone stock.
        logger.warning(
            "[m1-q4-verify] %d/%d exactness checks failed; those shapes stay on "
            "stock MLX",
            report["failed"],
            report["failed"] + report["passed"],
        )
    return report


def validated_keys() -> dict[tuple, bool]:
    """Read-only view of the load-time validation registry (tests/harnesses)."""
    return dict(_VALIDATED)


# -- dispatch -----------------------------------------------------------------


def _verify_rows(x: mx.array) -> int:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    return rows


def route_quantized_linear(module: Any, x: mx.array) -> mx.array | None:
    """Exact single q4 projection for a validated verify shape, else ``None``.

    Serves ``down_proj`` inside the fused MLP and any other q4 projection whose
    exact shape was validated at load.
    """
    cfg = config()
    if not cfg.mlp or x.ndim < 2 or x.dtype not in _ACTIVATIONS:
        return None
    # Quant format before shape: this model mixes q4 and q8 projections at the
    # *same* (K, N), so a shape-only key would hand a q8 weight to a kernel
    # that unpacks 4-bit nibbles.
    if not _quantized(module) or "bias" in module:
        return None
    from .verify_kernels import exact_qmv_m23, exact_qmv_m23_eligible

    m = _verify_rows(x)
    k = int(x.shape[-1])
    n = int(module["weight"].shape[0])
    if _VALIDATED.get(("single", x.dtype, m, k, n)) is not True:
        return None
    x2 = x.reshape(m, k)
    if not exact_qmv_m23_eligible(
        x2, module["weight"], bits=int(module.bits), group_size=int(module.group_size)
    ):
        return None
    _tick("qlinear_fused")
    return exact_qmv_m23(
        x2,
        module["weight"],
        module["scales"],
        module["biases"],
        group_size=int(module.group_size),
    ).reshape(*x.shape[:-1], n)


def gdn_input_projections(
    gdn: Any, inputs: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    """One exact launch for ``qkv``/``z``/``b``/``a``, or ``None`` for stock.

    ``inputs`` is the GDN hidden state ``(B, S, K)``.
    """
    cfg = config()
    if not cfg.gdn_proj:
        return None
    if inputs.dtype not in _ACTIVATIONS or len(inputs.shape) != 3:
        return None
    b, s, k = (int(value) for value in inputs.shape)
    m = b * s
    family = _gdn_family(gdn)
    if (
        family is None
        or not all(_quantized(module) for module in family)
        or any("bias" in module for module in family)
    ):
        _tick("gdn_stock")
        return None
    ns = tuple(int(module["weight"].shape[0]) for module in family)
    if _VALIDATED.get(("gdn", inputs.dtype, m, k, ns, cfg.gdn_nsg)) is not True:
        _tick("gdn_stock")
        return None

    from .verify_kernels import exact_qmv_m23_multi

    outputs = exact_qmv_m23_multi(inputs.reshape(m, k), family, nsg=cfg.gdn_nsg)
    _tick("gdn_fused")
    return tuple(output.reshape(b, s, n) for output, n in zip(outputs, ns))


# -- monkeypatches ------------------------------------------------------------

_QLINEAR_PATCH: dict[str, Any] = {"installed": False, "original": None}
_MLP_PATCH: dict[str, Any] = {"installed": False, "original": None}


def install_qlinear_patch() -> dict[str, Any]:
    """Wrap ``nn.QuantizedLinear.__call__`` with the exact M=2/M=3 q4 route.

    Installed *after* the NAX verify patch when both are active, so declining
    here falls through to whatever route was already in place (NAX covers
    ``M`` in 4..16 for q4; these lanes cover 2 and 3, so they never contend).
    """
    cfg = config()
    if not cfg.mlp:
        return {"installed": False, "reason": "lane_disabled"}
    if _QLINEAR_PATCH["installed"]:
        return {"installed": True, "already": True}

    import mlx.nn as nn

    from .attention_context import current_attention_phase

    original = nn.QuantizedLinear.__call__

    def patched(self, x: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
        if int(getattr(self, "bits", 0) or 0) == 4 and x.ndim >= 2:
            if current_attention_phase() != "prefill":
                routed = route_quantized_linear(self, x)
                if routed is not None:
                    return routed
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _QLINEAR_PATCH["installed"] = True
    _QLINEAR_PATCH["original"] = original
    return {"installed": True, "already": False}


def uninstall_qlinear_patch() -> None:
    if _QLINEAR_PATCH["installed"]:
        import mlx.nn as nn

        nn.QuantizedLinear.__call__ = _QLINEAR_PATCH["original"]
        _QLINEAR_PATCH["installed"] = False
        _QLINEAR_PATCH["original"] = None


def install_mlp_patch() -> dict[str, Any]:
    """Route validated verify-shaped ``Qwen3NextMLP`` calls through one launch.

    ``mlx_lm``'s MLP has no hook, so the fused gate+up launch is installed as a
    class-level patch that declines to the captured original for anything the
    load-time validation did not clear.
    """
    cfg = config()
    if not cfg.mlp:
        return {"installed": False, "reason": "lane_disabled"}
    if _MLP_PATCH["installed"]:
        return {"installed": True, "already": True}

    from mlx_lm.models.qwen3_next import Qwen3NextMLP, swiglu

    from .attention_context import current_attention_phase
    from .verify_kernels import exact_qmv_m23_dual

    original = Qwen3NextMLP.__call__

    def patched(self, x: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
        if x.dtype in _ACTIVATIONS and x.ndim >= 2:
            m = _verify_rows(x)
            k = int(x.shape[-1])
            gate = self.gate_proj
            up = self.up_proj
            # Eight of this model's 64 MLPs are q8 at the *same* (K, N) as the
            # 56 q4 ones. Validate the quant format, not just the shape.
            if not (_quantized(gate) and _quantized(up)):
                _tick("mlp_stock")
                return original(self, x)
            n = int(gate["weight"].shape[0])
            if not _same_scale_dtype(gate, up):
                _tick("mlp_stock")
                return original(self, x)
            if (
                _VALIDATED.get(("dual", x.dtype, m, k, n)) is True
                and _VALIDATED.get(("mlp", x.dtype, m, k, n)) is True
                and current_attention_phase() != "prefill"
            ):
                down = self.down_proj
                gate_y, up_y = exact_qmv_m23_dual(x.reshape(m, k), gate, up)
                out = down(swiglu(gate_y, up_y))
                _tick("mlp_fused")
                return out.reshape(*x.shape[:-1], int(down["weight"].shape[0]))
        _tick("mlp_stock")
        return original(self, x)

    Qwen3NextMLP.__call__ = patched
    _MLP_PATCH["installed"] = True
    _MLP_PATCH["original"] = original
    return {"installed": True, "already": False}


def uninstall_mlp_patch() -> None:
    if _MLP_PATCH["installed"]:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        Qwen3NextMLP.__call__ = _MLP_PATCH["original"]
        _MLP_PATCH["installed"] = False
        _MLP_PATCH["original"] = None
