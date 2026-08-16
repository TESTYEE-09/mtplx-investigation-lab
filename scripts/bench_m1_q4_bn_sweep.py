#!/usr/bin/env python3
"""Column-tile width sweep for the exact M=2/M=3 q4 verify kernels.

MLX 0.32 ``qmv_fast`` gives each simdgroup BN=4 output columns and has each
lane accumulate its own K stride sequentially, then ``simd_sum``. The *width*
of that column tile is a pure scheduling choice: every output column keeps its
own accumulator, its own lane->K mapping and its own reduction tree, so
widening BN changes which simdgroup computes a column, never how it is
computed. It is therefore a candidate for extracting occupancy/reuse without
the bit-exactness loss that sank the split-K prototypes (split-K reassociates
the K reduction itself, which is what changes BF16 output).

This sweeps BN x NSG against stock for the real MLP and GDN q4 shapes and
reports both timing and exactness for every point.
"""

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
        module["weight"],
        scales=module["scales"],
        biases=module["biases"],
        transpose=True,
        group_size=int(module.group_size),
        bits=int(module.bits),
        mode=str(module.mode),
    )


@lru_cache(maxsize=None)
def _single_kernel(k: int, n: int, gs: int, nsg: int, bn: int, dtype: mx.Dtype):
    source = f"""
        constexpr int K = {k};
        constexpr int N = {n};
        constexpr int GS = {gs};
        constexpr int NSG = {nsg};
        constexpr int BN = {bn};
        constexpr int VALUES = 16;
        constexpr int BLOCK = 512;
        constexpr int K_bytes = K / 2;
        constexpr int K_groups = K / GS;

        uint row = threadgroup_position_in_grid.x;
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        int n0 = int(tile) * (NSG * BN) + int(sg) * BN;

        const device uint8_t* ws = (const device uint8_t*)wq;
        ws += n0 * K_bytes + int(lane) * 8;
        const device T* sp = scales + n0 * K_groups + int(lane) / 2;
        const device T* bp = biases + n0 * K_groups + int(lane) / 2;
        const device T* xp = x + int(row) * K + int(lane) * VALUES;

        float result[BN] = {{0.0f}};
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
            for (int j = 0; j < BN; ++j) {{
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
                result[j] += float(sp[j * K_groups]) * accum
                    + sum * float(bp[j * K_groups]);
            }}
            ws += BLOCK / 2;
            sp += BLOCK / GS;
            bp += BLOCK / GS;
            xp += BLOCK;
        }}
        for (int j = 0; j < BN; ++j) {{
            result[j] = simd_sum(result[j]);
            if (lane == 0) {{ y[int(row) * N + n0 + j] = T(result[j]); }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"lab_qmv_k{k}_n{n}_gs{gs}_sg{nsg}_bn{bn}",
        input_names=["x", "wq", "scales", "biases"],
        output_names=["y"],
        source=source,
    )


def _single(x: mx.array, module: Any, *, nsg: int, bn: int) -> mx.array:
    m, k = map(int, x.shape)
    n = int(module["weight"].shape[0])
    kernel = _single_kernel(k, n, int(module.group_size), nsg, bn, x.dtype)
    (y,) = kernel(
        inputs=[x, module["weight"], module["scales"], module["biases"]],
        template=[("T", x.dtype)],
        grid=(32 * nsg * m, n // (bn * nsg), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y


@lru_cache(maxsize=None)
def _dual_kernel(k: int, n: int, gs: int, nsg: int, bn: int, dtype: mx.Dtype):
    source = f"""
        constexpr int K = {k};
        constexpr int N = {n};
        constexpr int GS = {gs};
        constexpr int NSG = {nsg};
        constexpr int BN = {bn};
        constexpr int VALUES = 16;
        constexpr int BLOCK = 512;
        constexpr int K_bytes = K / 2;
        constexpr int K_groups = K / GS;

        uint row = threadgroup_position_in_grid.x;
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        int n0 = int(tile) * (NSG * BN) + int(sg) * BN;

        const device uint8_t* wg = (const device uint8_t*)wq_a;
        const device uint8_t* wu = (const device uint8_t*)wq_b;
        wg += n0 * K_bytes + int(lane) * 8;
        wu += n0 * K_bytes + int(lane) * 8;
        const device T* sap = scales_a + n0 * K_groups + int(lane) / 2;
        const device T* sbp = scales_b + n0 * K_groups + int(lane) / 2;
        const device T* bap = biases_a + n0 * K_groups + int(lane) / 2;
        const device T* bbp = biases_b + n0 * K_groups + int(lane) / 2;
        const device T* xp = x + int(row) * K + int(lane) * VALUES;

        float result_a[BN] = {{0.0f}};
        float result_b[BN] = {{0.0f}};
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
            for (int j = 0; j < BN; ++j) {{
                const device uint16_t* wap =
                    (const device uint16_t*)(wg + j * K_bytes);
                const device uint16_t* wbp =
                    (const device uint16_t*)(wu + j * K_bytes);
                float accum_a = 0.0f;
                float accum_b = 0.0f;
                for (int i = 0; i < VALUES / 4; ++i) {{
                    uint16_t pa = wap[i];
                    uint16_t pb = wbp[i];
                    accum_a +=
                        xt[4 * i] * (pa & 0x000f) +
                        xt[4 * i + 1] * (pa & 0x00f0) +
                        xt[4 * i + 2] * (pa & 0x0f00) +
                        xt[4 * i + 3] * (pa & 0xf000);
                    accum_b +=
                        xt[4 * i] * (pb & 0x000f) +
                        xt[4 * i + 1] * (pb & 0x00f0) +
                        xt[4 * i + 2] * (pb & 0x0f00) +
                        xt[4 * i + 3] * (pb & 0xf000);
                }}
                result_a[j] += float(sap[j * K_groups]) * accum_a
                    + sum * float(bap[j * K_groups]);
                result_b[j] += float(sbp[j * K_groups]) * accum_b
                    + sum * float(bbp[j * K_groups]);
            }}
            wg += BLOCK / 2;
            wu += BLOCK / 2;
            sap += BLOCK / GS;
            sbp += BLOCK / GS;
            bap += BLOCK / GS;
            bbp += BLOCK / GS;
            xp += BLOCK;
        }}
        for (int j = 0; j < BN; ++j) {{
            result_a[j] = simd_sum(result_a[j]);
            result_b[j] = simd_sum(result_b[j]);
            if (lane == 0) {{
                y_a[int(row) * N + n0 + j] = T(result_a[j]);
                y_b[int(row) * N + n0 + j] = T(result_b[j]);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"lab_qmv_dual_k{k}_n{n}_gs{gs}_sg{nsg}_bn{bn}",
        input_names=[
            "x", "wq_a", "scales_a", "biases_a", "wq_b", "scales_b", "biases_b",
        ],
        output_names=["y_a", "y_b"],
        source=source,
    )


def _dual(x: mx.array, a: Any, b: Any, *, nsg: int, bn: int):
    m, k = map(int, x.shape)
    n = int(a["weight"].shape[0])
    kernel = _dual_kernel(k, n, int(a.group_size), nsg, bn, x.dtype)
    return tuple(
        kernel(
            inputs=[
                x,
                a["weight"], a["scales"], a["biases"],
                b["weight"], b["scales"], b["biases"],
            ],
            template=[("T", x.dtype)],
            grid=(32 * nsg * m, n // (bn * nsg), 1),
            threadgroup=(32 * nsg, 1, 1),
            output_shapes=[(m, n), (m, n)],
            output_dtypes=[x.dtype, x.dtype],
        )
    )


def _eval_value(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)


def _equal(reference: Any, candidate: Any) -> dict[str, Any]:
    refs = reference if isinstance(reference, tuple) else (reference,)
    cands = candidate if isinstance(candidate, tuple) else (candidate,)
    mx.eval(*refs, *cands)
    equal = True
    max_abs = 0.0
    for r, c in zip(refs, cands):
        ra = np.asarray(r.astype(mx.float32))
        ca = np.asarray(c.astype(mx.float32))
        equal = equal and bool(np.array_equal(ra, ca))
        max_abs = max(max_abs, float(np.abs(ra - ca).max(initial=0.0)))
    return {"array_equal": equal, "max_abs": max_abs}


def _paired(stock: Callable, candidate: Callable, *, epochs: int, inner: int):
    for _ in range(5):
        _eval_value(stock())
        _eval_value(candidate())
    samples: dict[str, list[float]] = {"stock": [], "candidate": []}
    fns = {"stock": stock, "candidate": candidate}
    for epoch in range(epochs):
        order = ("stock", "candidate") if epoch % 2 == 0 else ("candidate", "stock")
        for name in order:
            started = time.perf_counter()
            for _ in range(inner):
                _eval_value(fns[name]())
            samples[name].append(1000.0 * (time.perf_counter() - started) / inner)
    s = statistics.median(samples["stock"])
    c = statistics.median(samples["candidate"])
    return {"stock_ms": s, "candidate_ms": c, "speedup": s / c}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mtplx.runtime import load

    rt = load(args.model, mtp=True)
    layers = rt.model.language_model.model.layers
    gdn = None
    for layer in layers:
        candidate = getattr(layer, "linear_attn", None)
        if candidate is not None and getattr(candidate, "in_proj_qkv", None) is not None:
            gdn = candidate
            break
    mlp = layers[0].mlp

    singles = {
        "mlp.gate_proj": mlp.gate_proj,
        "mlp.down_proj": mlp.down_proj,
        "gdn.in_proj_qkv": gdn.in_proj_qkv,
        "gdn.in_proj_z": gdn.in_proj_z,
    }
    payload: dict[str, Any] = {
        "model": str(args.model),
        "device": mx.device_info(),
        "rows": [],
    }

    for m in (2, 3):
        base = (mx.arange(5120, dtype=mx.float32) % 251 - 125) / 64
        for name, module in singles.items():
            k = int(module["weight"].shape[1]) * 8
            n = int(module["weight"].shape[0])
            kbase = (mx.arange(k, dtype=mx.float32) % 251 - 125) / 64
            x = mx.stack(
                [(kbase * float(r + 1) / m).astype(mx.bfloat16) for r in range(m)]
            )
            reference = _stock(x, module)
            mx.eval(reference)
            for nsg in (1, 2, 4):
                for bn in (4, 8, 16):
                    if n % (bn * nsg):
                        continue
                    row = {"M": m, "target": name, "nsg": nsg, "bn": bn,
                           "K": k, "N": n, "kind": "single"}
                    try:
                        cand = lambda x=x, module=module, nsg=nsg, bn=bn: _single(
                            x, module, nsg=nsg, bn=bn
                        )
                        row.update(_equal(reference, cand()))
                        row["paired"] = _paired(
                            lambda x=x, module=module: _stock(x, module),
                            cand,
                            epochs=args.epochs,
                            inner=args.inner,
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    payload["rows"].append(row)
                    print(json.dumps({q: row.get(q) for q in
                                      ("M", "target", "nsg", "bn", "array_equal")}),
                          flush=True)

        pairs = {
            "mlp.gate+up": (mlp.gate_proj, mlp.up_proj),
        }
        for name, (a, b) in pairs.items():
            k = int(a["weight"].shape[1]) * 8
            n = int(a["weight"].shape[0])
            x = mx.stack(
                [(base * float(r + 1) / m).astype(mx.bfloat16) for r in range(m)]
            )
            reference = (_stock(x, a), _stock(x, b))
            mx.eval(*reference)
            for nsg in (1, 2, 4):
                for bn in (4, 8, 16):
                    if n % (bn * nsg):
                        continue
                    row = {"M": m, "target": name, "nsg": nsg, "bn": bn,
                           "K": k, "N": n, "kind": "dual"}
                    try:
                        cand = lambda x=x, a=a, b=b, nsg=nsg, bn=bn: _dual(
                            x, a, b, nsg=nsg, bn=bn
                        )
                        row.update(_equal(reference, cand()))
                        row["paired"] = _paired(
                            lambda x=x, a=a, b=b: (_stock(x, a), _stock(x, b)),
                            cand,
                            epochs=args.epochs,
                            inner=args.inner,
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    payload["rows"].append(row)
                    print(json.dumps({q: row.get(q) for q in
                                      ("M", "target", "nsg", "bn", "array_equal")}),
                          flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
