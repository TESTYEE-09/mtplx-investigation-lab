#!/usr/bin/env python3
"""Isolated Optimized-Speed q4 M=2/M=3 verification-kernel race."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np


def _stock(x: mx.array, module: Any) -> mx.array:
    return mx.quantized_matmul(
        x,
        module.weight,
        scales=module.scales,
        biases=module.biases,
        transpose=True,
        group_size=int(module.group_size),
        bits=int(module.bits),
        mode=str(module.mode),
    )


def _stock_rowwise(x: mx.array, module: Any) -> mx.array:
    return mx.concatenate([_stock(x[row : row + 1], module) for row in range(x.shape[0])])


def _vk_exact_m(x: mx.array, module: Any) -> mx.array:
    from mtplx.verify_kernels import _run_ksplit

    return _run_ksplit(
        int(x.shape[0]),
        x,
        module.weight,
        module.scales,
        module.biases,
        bits=4,
        group_size=int(module.group_size),
        dual=False,
        kconst=True,
    )


def _vk_pad_m4(x: mx.array, module: Any) -> mx.array:
    from mtplx.verify_kernels import vk_qmm_m4_impl

    return vk_qmm_m4_impl(
        "vk_k",
        x,
        module.weight,
        module.scales,
        module.biases,
        bits=4,
        group_size=int(module.group_size),
    )


@lru_cache(maxsize=None)
def _exact_qmv_kernel(
    m: int, k: int, n: int, group_size: int, nsg: int, dtype: mx.Dtype
):
    """Shape-specialized copy of MLX 0.32 affine_qmv_fast's arithmetic order."""
    source = f"""
        constexpr int M = {m};
        constexpr int K = {k};
        constexpr int N = {n};
        constexpr int GS = {group_size};
        constexpr int NSG = {nsg};
        constexpr int VALUES = 16;
        constexpr int BLOCK = 512;

        uint row = threadgroup_position_in_grid.x;
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        int n0 = int(tile) * (NSG * 4) + int(sg) * 4;

        const device uint8_t* ws = (const device uint8_t*)wq;
        int K_bytes = K / 2;
        int K_groups = K / GS;
        ws += n0 * K_bytes + int(lane) * 8;
        const device T* sp = scales + n0 * K_groups + int(lane) / 2;
        const device T* bp = biases + n0 * K_groups + int(lane) / 2;
        const device T* xp = x + int(row) * K + int(lane) * VALUES;

        float result[4] = {{0.0f}};
        float xt[VALUES];
        for (int kb = 0; kb < K; kb += BLOCK) {{
            float sum = 0.0f;
            for (int i = 0; i < VALUES; i += 4) {{
                sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
                xt[i] = xp[i];
                xt[i + 1] = xp[i + 1] / 16.0f;
                xt[i + 2] = xp[i + 2] / 256.0f;
                xt[i + 3] = xp[i + 3] / 4096.0f;
            }}
            for (int j = 0; j < 4; ++j) {{
                const device uint16_t* wp =
                    (const device uint16_t*)(ws + j * K_bytes);
                float accum = 0.0f;
                for (int i = 0; i < VALUES / 4; ++i) {{
                    uint16_t packed = wp[i];
                    accum +=
                        xt[4 * i] * (packed & 0x000f) +
                        xt[4 * i + 1] * (packed & 0x00f0) +
                        xt[4 * i + 2] * (packed & 0x0f00) +
                        xt[4 * i + 3] * (packed & 0xf000);
                }}
                float scale = float(sp[j * K_groups]);
                float bias = float(bp[j * K_groups]);
                result[j] += scale * accum + sum * bias;
            }}
            ws += BLOCK / 2;
            sp += BLOCK / GS;
            bp += BLOCK / GS;
            xp += BLOCK;
        }}
        for (int j = 0; j < 4; ++j) {{
            result[j] = simd_sum(result[j]);
            if (lane == 0) {{ y[int(row) * N + n0 + j] = T(result[j]); }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_exp_exact_qmv_m{m}_k{k}_n{n}_gs{group_size}_sg{nsg}",
        input_names=["x", "wq", "scales", "biases"],
        output_names=["y"],
        source=source,
    )


def _exact_qmv(x: mx.array, module: Any, *, nsg: int) -> mx.array:
    m, k = map(int, x.shape)
    n = int(module.weight.shape[0])
    kernel = _exact_qmv_kernel(m, k, n, int(module.group_size), nsg, x.dtype)
    (y,) = kernel(
        inputs=[x, module.weight, module.scales, module.biases],
        template=[("T", x.dtype)],
        grid=(32 * nsg * m, n // (4 * nsg), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y


def _time(fn: Callable[[], mx.array], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - started)
    return samples


def _eval_value(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)


def _compare(reference: mx.array, candidate: mx.array) -> dict[str, Any]:
    mx.eval(reference, candidate)
    ref = np.asarray(reference.astype(mx.float32))
    cand = np.asarray(candidate.astype(mx.float32))
    diff = np.abs(ref - cand)
    return {
        "array_equal": bool(np.array_equal(ref, cand)),
        "mismatched": int(np.count_nonzero(ref != cand)),
        "elements": int(ref.size),
        "max_abs": float(diff.max(initial=0.0)),
        "mean_abs": float(diff.mean()),
    }


def _paired_time(
    stock: Callable[[], mx.array],
    candidate: Callable[[], mx.array],
    *,
    epochs: int,
    inner: int,
) -> dict[str, Any]:
    """Adjacent A/B/B/A timing to suppress frequency and thermal drift."""
    for _ in range(5):
        _eval_value(stock())
        _eval_value(candidate())
    samples: dict[str, list[float]] = {"stock": [], "candidate": []}
    functions = {"stock": stock, "candidate": candidate}
    for epoch in range(epochs):
        order = ("stock", "candidate") if epoch % 2 == 0 else ("candidate", "stock")
        for name in order:
            started = time.perf_counter()
            for _ in range(inner):
                _eval_value(functions[name]())
            samples[name].append(1000.0 * (time.perf_counter() - started) / inner)
    stock_ms = statistics.median(samples["stock"])
    candidate_ms = statistics.median(samples["candidate"])
    return {
        "stock_median_ms": stock_ms,
        "candidate_median_ms": candidate_ms,
        "speedup_vs_stock": stock_ms / candidate_ms,
        "stock_samples_ms": samples["stock"],
        "candidate_samples_ms": samples["candidate"],
    }


def _round_robin_time(
    functions: dict[str, Callable[[], mx.array]],
    *,
    epochs: int,
    inner: int,
) -> dict[str, Any]:
    """Rotate all schedulers through every timing position in each epoch."""
    names = list(functions)
    for function in functions.values():
        for _ in range(3):
            _eval_value(function())
    samples = {name: [] for name in names}
    for epoch in range(epochs):
        offset = epoch % len(names)
        order = names[offset:] + names[:offset]
        if (epoch // len(names)) % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            for _ in range(inner):
                _eval_value(functions[name]())
            samples[name].append(1000.0 * (time.perf_counter() - started) / inner)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "medians_ms": medians,
        "speedup_vs_stock": {
            name: medians["stock"] / value for name, value in medians.items()
        },
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--paired-epochs", type=int, default=30)
    parser.add_argument("--paired-inner", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mtplx.runtime import load

    rt = load(args.model, mtp=True)
    layer = rt.model.language_model.model.layers[0]
    modules = {
        "mlp.gate_proj": layer.mlp.gate_proj,
        "mlp.up_proj": layer.mlp.up_proj,
        "mlp.down_proj": layer.mlp.down_proj,
    }
    payload: dict[str, Any] = {
        "model": str(args.model),
        "device": mx.device_info(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": [],
    }
    rng = np.random.default_rng(20260816)
    for name, module in modules.items():
        k = int(module.weight.shape[1]) * 8
        n = int(module.weight.shape[0])
        for m in (2, 3, 4):
            x = mx.array(rng.standard_normal((m, k), dtype=np.float32), dtype=mx.bfloat16)
            reference = _stock(x, module)
            variants = {
                "stock": lambda x=x, module=module: _stock(x, module),
                "stock_rowwise": lambda x=x, module=module: _stock_rowwise(x, module),
                **{
                    f"exact_qmv_sg{nsg}": (
                        lambda nsg=nsg, x=x, module=module: _exact_qmv(
                            x, module, nsg=nsg
                        )
                    )
                    for nsg in (1, 2, 4, 8)
                },
                "vk_exact_m": lambda x=x, module=module: _vk_exact_m(x, module),
                "vk_pad_m4": lambda x=x, module=module: _vk_pad_m4(x, module),
            }
            variant_rows = {}
            for variant, fn in variants.items():
                mx.reset_peak_memory()
                samples = _time(fn, args.warmup, args.repeats)
                candidate = fn()
                variant_rows[variant] = {
                    "median_ms": 1000.0 * statistics.median(samples),
                    "samples_ms": [1000.0 * value for value in samples],
                    "active_memory_bytes": int(mx.get_active_memory()),
                    "peak_memory_bytes": int(mx.get_peak_memory()),
                    "comparison": _compare(reference, candidate),
                }
            stock_ms = variant_rows["stock"]["median_ms"]
            for row in variant_rows.values():
                row["speedup_vs_stock"] = stock_ms / row["median_ms"]
            paired_production = None
            paired_dual = None
            scheduler_round_robin = None
            if m in (2, 3):
                from mtplx.verify_kernels import exact_qmv_m23

                def production(x=x, module=module):  # type: ignore[no-untyped-def]
                    return exact_qmv_m23(
                        x,
                        module.weight,
                        module.scales,
                        module.biases,
                        group_size=int(module.group_size),
                    )
                paired_production = _paired_time(
                    variants["stock"],
                    production,
                    epochs=args.paired_epochs,
                    inner=args.paired_inner,
                )
                paired_production["comparison"] = _compare(reference, production())
                scheduler_round_robin = _round_robin_time(
                    {
                        "stock": variants["stock"],
                        **{
                            f"exact_qmv_sg{nsg}": variants[f"exact_qmv_sg{nsg}"]
                            for nsg in (1, 2, 4, 8)
                        },
                    },
                    epochs=args.paired_epochs,
                    inner=args.paired_inner,
                )
                if name == "mlp.gate_proj":
                    from mtplx.verify_kernels import exact_qmv_m23_dual

                    up_module = modules["mlp.up_proj"]

                    def stock_dual(x=x, module=module, up_module=up_module):  # type: ignore[no-untyped-def]
                        return _stock(x, module), _stock(x, up_module)

                    def candidate_dual(x=x, module=module, up_module=up_module):  # type: ignore[no-untyped-def]
                        return exact_qmv_m23_dual(x, module, up_module)

                    paired_dual = _paired_time(
                        stock_dual,
                        candidate_dual,
                        epochs=args.paired_epochs,
                        inner=args.paired_inner,
                    )
                    ref_gate, ref_up = stock_dual()
                    cand_gate, cand_up = candidate_dual()
                    paired_dual["comparison"] = {
                        "gate": _compare(ref_gate, cand_gate),
                        "up": _compare(ref_up, cand_up),
                    }
            payload["rows"].append(
                {
                    "module": name,
                    "M": m,
                    "K": k,
                    "N": n,
                    "bits": int(module.bits),
                    "group_size": int(module.group_size),
                    "mode": str(module.mode),
                    "variants": variant_rows,
                    "paired_production": paired_production,
                    "paired_dual": paired_dual,
                    "scheduler_round_robin": scheduler_round_robin,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
