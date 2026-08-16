#!/usr/bin/env python3
"""Probe MLX PR #3026 quantized SDPA on MTPLX-like verify shapes.

This script intentionally depends only on ``mlx.core`` so it can be run with
``PYTHONPATH`` pointed at an isolated MLX source build. It measures dense SDPA
against quantized SDPA for the long-context vector-attention shape that matters
for MTPLX verify: small query length, D=256, grouped-query attention.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx


@dataclass
class ProbeResult:
    mode: str
    bits: int
    group_size: int | None
    key_length: int
    query_length: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dense_ms_mean: float
    dense_ms_median: float
    quantized_ms_mean: float
    quantized_ms_median: float
    speedup_median: float
    max_abs_error: float


def _time_call(fn, *, warmups: int, iters: int) -> list[float]:
    for _ in range(warmups):
        y = fn()
        mx.eval(y)
        mx.synchronize()
    times: list[float] = []
    for _ in range(iters):
        started = time.perf_counter()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def _quantize_kv(
    k: mx.array,
    v: mx.array,
    *,
    mode: str,
    bits: int,
    group_size: int | None,
) -> tuple[mx.array, mx.array, mx.array | None, mx.array, mx.array, mx.array | None]:
    kwargs: dict[str, Any] = {"mode": mode}
    if mode == "affine":
        kwargs["bits"] = bits
        if group_size is not None:
            kwargs["group_size"] = group_size
    elif group_size is not None:
        kwargs["group_size"] = group_size
    k_parts = mx.quantize(k, **kwargs)
    v_parts = mx.quantize(v, **kwargs)
    if mode == "affine":
        k_q, k_scales, k_biases = k_parts
        v_q, v_scales, v_biases = v_parts
    else:
        k_q, k_scales = k_parts
        v_q, v_scales = v_parts
        k_biases = None
        v_biases = None
    mx.eval(k_q, k_scales, v_q, v_scales)
    if k_biases is not None and v_biases is not None:
        mx.eval(k_biases, v_biases)
    mx.synchronize()
    return k_q, k_scales, k_biases, v_q, v_scales, v_biases


def run_one(args: argparse.Namespace, *, mode: str, bits: int, group_size: int | None) -> ProbeResult:
    mx.random.seed(args.seed)
    q = (args.input_scale * mx.random.normal(
        shape=(1, args.query_heads, args.query_length, args.head_dim),
        dtype=mx.float16,
    ))
    k = (args.input_scale * mx.random.normal(
        shape=(1, args.kv_heads, args.key_length, args.head_dim),
        dtype=mx.float16,
    ))
    v = (args.input_scale * mx.random.normal(
        shape=(1, args.kv_heads, args.key_length, args.head_dim),
        dtype=mx.float16,
    ))
    mx.eval(q, k, v)
    mx.synchronize()

    scale = 1.0 / math.sqrt(args.head_dim)
    dense_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(dense_ref)
    mx.synchronize()

    k_q, k_scales, k_biases, v_q, v_scales, v_biases = _quantize_kv(
        k,
        v,
        mode=mode,
        bits=bits,
        group_size=group_size,
    )

    def dense_call() -> mx.array:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

    def quantized_call() -> mx.array:
        if mode == "affine":
            return mx.fast.quantized_scaled_dot_product_attention(
                q,
                k_q,
                k_scales,
                k_biases,
                v_q,
                v_scales,
                v_biases,
                scale=scale,
                mode=mode,
                bits=bits,
                group_size=group_size,
                causal=True,
            )
        return mx.fast.quantized_scaled_dot_product_attention(
            q,
            k_q,
            k_scales,
            v_q,
            v_scales,
            scale=scale,
            mode=mode,
            bits=bits,
            group_size=group_size,
            causal=True,
        )

    q_out = quantized_call()
    mx.eval(q_out)
    mx.synchronize()
    max_abs_error = float(mx.max(mx.abs(q_out - dense_ref)).item())

    dense_times = _time_call(dense_call, warmups=args.warmups, iters=args.iters)
    quant_times = _time_call(quantized_call, warmups=args.warmups, iters=args.iters)

    dense_median = statistics.median(dense_times)
    quant_median = statistics.median(quant_times)
    result = ProbeResult(
        mode=mode,
        bits=bits,
        group_size=group_size,
        key_length=args.key_length,
        query_length=args.query_length,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dense_ms_mean=statistics.mean(dense_times),
        dense_ms_median=dense_median,
        quantized_ms_mean=statistics.mean(quant_times),
        quantized_ms_median=quant_median,
        speedup_median=dense_median / quant_median if quant_median > 0 else 0.0,
        max_abs_error=max_abs_error,
    )

    mx.synchronize()
    return result


def _mode_spec(raw: str) -> tuple[str, int, int | None]:
    parts = raw.split(":")
    mode = parts[0]
    bits = int(parts[1]) if len(parts) > 1 and parts[1] else (8 if mode == "mxfp8" else 4)
    group_size = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return mode, bits, group_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-length", type=int, default=16384)
    parser.add_argument("--query-length", type=int, default=4)
    parser.add_argument("--query-heads", type=int, default=48)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--mode", action="append", default=[])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.1)
    args = parser.parse_args()

    modes = args.mode or ["affine:8:64", "affine:4:64", "mxfp8:8", "mxfp4:4"]
    results = []
    for spec in modes:
        mode, bits, group_size = _mode_spec(spec)
        result = run_one(args, mode=mode, bits=bits, group_size=group_size)
        row = asdict(result)
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    print(json.dumps({"results": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
