#!/usr/bin/env python3
"""Compare MTPLX paged-tail SDPA with DFlash-style GQA SDPA routes.

This is a synthetic kernel-shape probe for the Qwen3.6-style verify path:
``B=1, Hq=48, Hkv=8, D=256``.  It does not prove model quality; it answers a
narrow implementation question: whether the GQA routing idea beats the current
in-tree paged-tail attention kernel on the exact long-context shape.

The grouped/per-head helpers are adapted from dflash-mlx's GQA SDPA shim.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mtplx.kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail


_GQA_STREAMS: dict[int, list[Any]] = {}


def tail_causal_mask(q_len: int, kv_len: int) -> mx.array:
    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = mx.arange(kv_len)[None, :]
    return k_pos <= q_pos


def repeat_gqa_mask(mask: Any, *, q_len: int, kv_len: int, gqa: int) -> Any:
    if mask is None:
        return None
    if isinstance(mask, str) and mask == "causal":
        mask = tail_causal_mask(q_len, kv_len)
        reps = [1] * mask.ndim
        reps[-2] = int(gqa)
        return mx.tile(mask, tuple(reps))
    if not isinstance(mask, mx.array):
        return mask
    if int(mask.shape[-2]) != q_len:
        return mask
    reps = [1] * mask.ndim
    reps[-2] = int(gqa)
    return mx.tile(mask, tuple(reps))


def grouped_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Any,
) -> mx.array:
    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    if kv_heads <= 0 or query_heads == kv_heads or query_heads % kv_heads != 0:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=scale,
            mask=mask,
        )
    gqa = query_heads // kv_heads
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
    output = scaled_dot_product_attention(
        grouped_queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask=grouped_mask,
    )
    return output.reshape(batch_size, kv_heads, gqa, q_len, head_dim).reshape(
        batch_size,
        query_heads,
        q_len,
        head_dim,
    )


def per_head_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Any,
) -> mx.array:
    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    gqa = query_heads // kv_heads
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
    outputs = [
        scaled_dot_product_attention(
            grouped_queries[:, head : head + 1, :, :],
            keys[:, head : head + 1, :, :],
            values[:, head : head + 1, :, :],
            cache=None,
            scale=scale,
            mask=grouped_mask,
        )
        for head in range(kv_heads)
    ]
    output = mx.concatenate(outputs, axis=1)
    return output.reshape(batch_size, kv_heads, gqa, q_len, head_dim).reshape(
        batch_size,
        query_heads,
        q_len,
        head_dim,
    )


def _streams_for(kv_heads: int) -> list[Any]:
    if kv_heads not in _GQA_STREAMS:
        _GQA_STREAMS[kv_heads] = [mx.new_stream(mx.gpu) for _ in range(kv_heads)]
    return _GQA_STREAMS[kv_heads]


def async_per_head_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Any,
) -> mx.array:
    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    gqa = query_heads // kv_heads
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
    outputs = []
    for head, stream in enumerate(_streams_for(kv_heads)):
        with mx.stream(stream):
            output = scaled_dot_product_attention(
                grouped_queries[:, head : head + 1, :, :],
                keys[:, head : head + 1, :, :],
                values[:, head : head + 1, :, :],
                cache=None,
                scale=scale,
                mask=grouped_mask,
            )
            mx.async_eval(output)
            outputs.append(output)
    output = mx.concatenate(outputs, axis=1)
    return output.reshape(batch_size, kv_heads, gqa, q_len, head_dim).reshape(
        batch_size,
        query_heads,
        q_len,
        head_dim,
    )


def median_ms(fn: Callable[[], mx.array | None], *, warmups: int, iters: int) -> tuple[float | None, mx.array | None]:
    out = None
    for _ in range(warmups):
        out = fn()
        if out is None:
            return None, None
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        if out is None:
            return None, None
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), out


def build_case(
    *,
    q_len: int,
    kv_len: int,
    hq: int,
    hkv: int,
    d: int,
    block_size: int,
    dtype: Any,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, float]:
    blocks = math.ceil(kv_len / block_size)
    padded = blocks * block_size
    queries = mx.random.normal((1, hq, q_len, d)).astype(dtype)
    flat_k = mx.random.normal((padded, hkv, d)).astype(dtype)
    flat_v = mx.random.normal((padded, hkv, d)).astype(dtype)
    key_cache = flat_k.reshape(blocks, block_size, hkv, d)
    value_cache = flat_v.reshape(blocks, block_size, hkv, d)
    keys = flat_k[:kv_len].transpose(1, 0, 2)[None, ...]
    values = flat_v[:kv_len].transpose(1, 0, 2)[None, ...]
    mx.eval(queries, keys, values, key_cache, value_cache)
    mx.synchronize()
    return queries, keys, values, key_cache, value_cache, 1.0 / math.sqrt(d)


def run_case(args: argparse.Namespace, *, q_len: int, kv_len: int) -> dict[str, Any]:
    queries, keys, values, key_cache, value_cache, scale = build_case(
        q_len=q_len,
        kv_len=kv_len,
        hq=args.hq,
        hkv=args.hkv,
        d=args.d,
        block_size=args.block_size,
        dtype=mx.float16,
    )
    routes: dict[str, Callable[[], mx.array | None]] = {
        "dense_stock": lambda: scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=scale,
            mask="causal",
        ),
        "paged_2pass_tail": lambda: sdpa_2pass_paged_tail(
            queries=queries,
            key_cache=key_cache,
            value_cache=value_cache,
            offset=kv_len,
            block_size=args.block_size,
            scale=scale,
            mask="causal",
            max_q_len=args.max_q_len,
        ),
        "grouped_gqa": lambda: grouped_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask="causal",
        ),
        "per_head_gqa": lambda: per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask="causal",
        ),
        "async_per_head_gqa": lambda: async_per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask="causal",
        ),
    }
    timings: dict[str, float | None] = {}
    outputs: dict[str, mx.array] = {}
    for name, fn in routes.items():
        ms, out = median_ms(fn, warmups=args.warmups, iters=args.iters)
        timings[name] = ms
        if out is not None:
            outputs[name] = out
    baseline = outputs.get("dense_stock")
    errors = {}
    if baseline is not None:
        for name, out in outputs.items():
            if name == "dense_stock":
                continue
            errors[name] = float(mx.max(mx.abs(out - baseline)).item())
    result: dict[str, Any] = {
        "q_len": q_len,
        "kv_len": kv_len,
        "shape": {"B": 1, "Hq": args.hq, "Hkv": args.hkv, "D": args.d},
        "timings_ms": timings,
        "max_abs_error_vs_dense": errors,
    }
    dense_ms = timings.get("dense_stock")
    if dense_ms:
        result["speedup_vs_dense"] = {
            name: (dense_ms / ms if ms else None)
            for name, ms in timings.items()
            if name != "dense_stock"
        }
    paged_ms = timings.get("paged_2pass_tail")
    if paged_ms:
        result["speedup_vs_paged_2pass_tail"] = {
            name: (paged_ms / ms if ms else None)
            for name, ms in timings.items()
            if name != "paged_2pass_tail"
        }
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return result


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-lens", default="1,4,5,8")
    parser.add_argument("--kv-lens", default="16384,65536,100416")
    parser.add_argument("--hq", type=int, default=48)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-q-len", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not mx.metal.is_available():
        raise SystemExit("MLX Metal is not available")
    mx.random.seed(1234)
    for q_len in parse_csv_ints(args.q_lens):
        for kv_len in parse_csv_ints(args.kv_lens):
            print(json.dumps(run_case(args, q_len=q_len, kv_len=kv_len), sort_keys=True))


if __name__ == "__main__":
    main()
