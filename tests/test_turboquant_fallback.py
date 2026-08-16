"""Regression tests for the TurboQuant graceful-fallback path.

Background: the user's fork hit a fatal mid-stream crash when serving a
TurboQuant-quantized model on a host that did not have the vllm-metal
external ops installed (``nanobind`` missing from the venv,
``vllm_metal/paged_ops.cpp`` never JIT-built, no ``MTPLX_VLLM_METAL_REPO``).
``_load_vllm_metal_ops`` raised :class:`RuntimeError` during a write into
the TurboQuant paged cache (``_write_tail`` calling ``ops.tq_encode``),
which surfaced to the OpenAI-compatible streaming endpoint as a 500 and
killed the in-flight generation.

These tests pin the new behavior:

* ``install_vllm_metal_paged_attention_kv_cache`` must NOT raise when a
  TurboQuant config is requested but external ops are unavailable; it must
  downgrade to the plain paged layout and record the reason in stats.
* ``VllmMetalPagedKVCache._write_tail`` must NOT raise on the same
  condition; it must reroute through the non-TurboQuant write path and
  produce a usable cache snapshot.
* ``VllmMetalPagedKVCache.paged_attention`` must NOT raise on the same
  condition; it must bail out cleanly so the caller can take its dense
  fallback.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import mtplx.cache_state as cache_state
from mtplx.cache_state import (
    VllmMetalPagedKVCache,
    configure_tail_owned_attention_kv_cache,
    install_vllm_metal_paged_attention_kv_cache,
)
from mtplx.turboquant import TurboQuantConfig


# The exact RuntimeError text observed in production. Reproducing it
# verbatim here keeps the test honest about the scenario it is guarding
# against.
_OBSERVED_OPS_FAILURE = RuntimeError(
    "vllm-metal paged-attention ops are unavailable; "
    "vendored vllm_metal.metal: No module named 'nanobind'; "
    "reference checkout missing: /tmp/REFERENCES:TOOLS/vllm-metal. "
    "Install MTPLX with its Darwin/arm64 dependencies or set "
    "MTPLX_VLLM_METAL_REPO to a working vllm-metal checkout."
)


def _force_ops_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``_load_vllm_metal_ops`` raise the observed RuntimeError."""

    # Reset the module-level "we already warned" latch so each test sees
    # the warning fire (and so order between tests doesn't matter).
    monkeypatch.setattr(cache_state, "_VLLM_METAL_OPS_UNAVAILABLE_WARNED", False)

    def _raise() -> None:
        raise _OBSERVED_OPS_FAILURE

    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", _raise)


def test_install_downgrades_turboquant_when_external_ops_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The crash trigger: TurboQuant install with no external ops on host."""

    from mlx_lm.models.cache import KVCache

    _force_ops_failure(monkeypatch)
    cache = [KVCache()]
    config = TurboQuantConfig(key_quant="q8_0", value_quant="q3_0")

    # Must not raise: the install used to crash here with the observed
    # RuntimeError, taking the in-flight request down with it.
    stats = install_vllm_metal_paged_attention_kv_cache(
        cache,
        block_size=16,
        num_blocks=64,
        turboquant_config=config,
    )

    assert stats["entries"] == 1
    # Cache must have been downgraded to the plain paged mode.
    assert stats["mode"] == "vllm_metal_paged"
    assert stats["turboquant"] == 0
    assert stats["turboquant_disabled_reason"] == "vllm_metal_ops_unavailable"
    assert "turboquant_k_quant" not in stats
    paged = cache[0]
    assert isinstance(paged, VllmMetalPagedKVCache)
    assert paged.turboquant is False
    assert paged.turboquant_config is None

    # The operator-facing warning must have fired exactly once on stderr.
    captured = capsys.readouterr()
    assert "vllm-metal external ops unavailable" in captured.err
    assert captured.err.count("vllm-metal external ops unavailable") == 1


def test_configure_tail_owned_via_env_downgrades_turboquant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the env-driven entry point must also degrade gracefully."""

    from mlx_lm.models.cache import KVCache

    _force_ops_failure(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT", "q8_0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT", "q3_0")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged"
    assert stats["turboquant"] == 0
    assert stats["turboquant_disabled_reason"] == "vllm_metal_ops_unavailable"
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].turboquant is False


def test_configure_tail_owned_via_env_falls_back_to_plain_q8_when_ops_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_lm.models.cache import KVCache

    _force_ops_failure(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT", "q8_0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT", "q8_0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged_kv_q8"
    assert stats["turboquant"] == 0
    assert stats["kv_quant"] == 1
    assert stats["kv_quant_mode"] == "q8"
    assert stats["turboquant_disabled_reason"] == "vllm_metal_ops_unavailable"
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].turboquant is False
    assert cache[0].kv_quant is True


def test_write_tail_falls_back_to_plain_paged_when_ops_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TurboQuant cache reaching ``_write_tail`` without ops must not crash.

    This guards the in-stream branch the user actually hit: the install
    layer above already drops TurboQuant, but a snapshot-restored cache or
    a directly-constructed one must still survive the write.
    """

    _force_ops_failure(monkeypatch)
    config = TurboQuantConfig(key_quant="q8_0", value_quant="q3_0")
    paged = VllmMetalPagedKVCache(
        block_size=16,
        num_blocks=4,
        turboquant_config=config,
    )
    # The constructor doesn't allocate the packed layout (no keys/values
    # passed), so we go through ``_ensure_allocated``-driven downgrade on
    # first write. Use head_dim=128 so the TurboQuant validator would have
    # been happy if ops had loaded.
    keys = mx.zeros((1, 2, 7, 128), dtype=mx.float16)
    values = mx.zeros((1, 2, 7, 128), dtype=mx.float16)

    # Must not raise.
    paged.update_without_fetch(keys, values)

    # The cache produced a usable snapshot, not None.
    assert paged.offset == 7
    assert paged.turboquant is False
    assert paged.turboquant_config is None
    assert paged.key_cache is not None
    assert paged.value_cache is not None
    # Plain paged layout: shape ``[num_blocks, block_size, n_kv_heads, head_dim]``.
    assert tuple(paged.key_cache.shape) == (4, 16, 2, 128)
    assert tuple(paged.value_cache.shape) == (4, 16, 2, 128)
    stats = paged.paged_stats()
    assert stats["mode"] == "vllm_metal_paged"
    assert int(stats["updates"]) == 1


@pytest.mark.parametrize(
    ("value_quant", "expected_packed_dim", "expected_centroids"),
    [("q4_0", 64, 16), ("q8_0", 128, 256)],
)
def test_write_tail_supports_q4_q8_turboquant_values(
    monkeypatch: pytest.MonkeyPatch,
    value_quant: str,
    expected_packed_dim: int,
    expected_centroids: int,
) -> None:
    """q4/q8 V modes are real TurboQuant candidates, not rejected configs."""

    class _StubOps:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, bool]] = []

        def tq_encode(self, *_args, **_kwargs):
            (
                _k_3d,
                _v_3d,
                key_cache,
                value_cache,
                key_scale,
                value_scale,
                key_zero,
                _slot,
                _centroids,
                v_bits,
                k_bits,
                k_signed,
            ) = _args
            self.calls.append((int(v_bits), int(k_bits), bool(k_signed)))
            return key_cache, value_cache, key_scale, value_scale, key_zero

    stub = _StubOps()
    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", lambda: stub)
    config = TurboQuantConfig(key_quant="q8_0", value_quant=value_quant)
    paged = VllmMetalPagedKVCache(
        block_size=16,
        num_blocks=4,
        turboquant_config=config,
    )
    keys = mx.zeros((1, 2, 7, 128), dtype=mx.float16)
    values = mx.zeros((1, 2, 7, 128), dtype=mx.float16)

    paged.update_without_fetch(keys, values)

    assert paged.turboquant is True
    assert tuple(paged.key_cache.shape) == (4, 16, 2, 128)
    assert tuple(paged.value_cache.shape) == (4, 16, 2, expected_packed_dim)
    assert tuple(paged._turboquant_v_centroids.shape) == (expected_centroids,)
    assert stub.calls == [(config.value_bits, config.key_bits, True)]


def test_paged_attention_bails_out_for_turboquant_when_ops_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A residual TurboQuant ``paged_attention`` call must bail out cleanly.

    We force a cache to keep ``turboquant=True`` after allocation by
    monkeypatching ``_load_vllm_metal_ops_optional`` to return a stub at
    allocation time and ``None`` at attention time. This simulates the
    pathological case where install/allocation succeeded but the runtime
    ops module disappeared (e.g. the Python session reloaded).
    """

    monkeypatch.setattr(
        cache_state, "_VLLM_METAL_OPS_UNAVAILABLE_WARNED", False
    )
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "")  # default path

    config = TurboQuantConfig(key_quant="q8_0", value_quant="q3_0")
    # Build the cache with a stub ops object that *does* expose tq_encode
    # so allocation and write succeed.
    class _StubOps:
        def tq_encode(self, *_args, **_kwargs):
            # Return the same caches unchanged; we only need write_tail to
            # finish, not produce real quantized data for this test.
            (
                _k_3d,
                _v_3d,
                key_cache,
                value_cache,
                key_scale,
                value_scale,
                key_zero,
                _slot,
                _centroids,
                _vbits,
                _kbits,
                _ksigned,
            ) = _args
            return key_cache, value_cache, key_scale, value_scale, key_zero

    stub = _StubOps()
    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", lambda: stub)

    paged = VllmMetalPagedKVCache(
        block_size=16,
        num_blocks=4,
        turboquant_config=config,
    )
    keys = mx.zeros((1, 2, 7, 128), dtype=mx.float16)
    values = mx.zeros((1, 2, 7, 128), dtype=mx.float16)
    paged.update_without_fetch(keys, values)
    assert paged.turboquant is True  # alloc/write took the TQ path

    # Now make the next ops load fail (simulating the runtime miss).
    def _raise() -> None:
        raise _OBSERVED_OPS_FAILURE

    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", _raise)

    queries = mx.zeros((1, 8, 1, 128), dtype=mx.float16)
    # Must not raise; bailout returns None so the caller takes its dense
    # fallback instead of crashing mid-stream.
    out = paged.paged_attention(queries, scale=128**-0.5, mask="causal")
    assert out is None
    stats = paged.paged_stats()
    bailouts = stats["paged_attention_bailouts_by_phase_reason"]
    assert any("turboquant_unsupported" in key for key in bailouts), bailouts
