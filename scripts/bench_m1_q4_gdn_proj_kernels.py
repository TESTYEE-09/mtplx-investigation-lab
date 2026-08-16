#!/usr/bin/env python3
"""Isolated Optimized-Speed q4 M=2/M=3 GDN input-projection kernel race.

Extends the accepted MLP gate/up dual-launch morphology to the GDN q4
row-independent input projections (``in_proj_qkv``, ``in_proj_z``,
``in_proj_b``, ``in_proj_a``). Every candidate must be bit-identical to stock
MLX before its timing is worth reading.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


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
    }


def _paired_time(
    stock: Callable[[], Any],
    candidate: Callable[[], Any],
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


def _deterministic_rows(m: int, k: int) -> mx.array:
    base = (mx.arange(k, dtype=mx.float32) % 251 - 125) / 64
    return mx.stack(
        [(base * float(row + 1) / float(m)).astype(mx.bfloat16) for row in range(m)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--paired-epochs", type=int, default=25)
    parser.add_argument("--paired-inner", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mtplx.runtime import load
    from mtplx.verify_kernels import (
        exact_qmv_m23_multi,
        exact_qmv_m23_multi_eligible,
    )

    rt = load(args.model, mtp=True)
    layers = rt.model.language_model.model.layers
    gdn = None
    gdn_index = -1
    for index, layer in enumerate(layers):
        candidate = getattr(layer, "linear_attn", None)
        if candidate is not None and getattr(candidate, "in_proj_qkv", None) is not None:
            gdn = candidate
            gdn_index = index
            break
    if gdn is None:
        raise SystemExit("no GDN layer with split in_proj_* projections found")

    modules = {name: getattr(gdn, name) for name in PROJECTIONS}
    payload: dict[str, Any] = {
        "model": str(args.model),
        "device": mx.device_info(),
        "gdn_layer_index": gdn_index,
        "inventory": {
            name: {
                "packed_shape": [int(v) for v in module["weight"].shape],
                "logical_K": int(module["weight"].shape[1]) * 8,
                "logical_N": int(module["weight"].shape[0]),
                "bits": int(module.bits),
                "group_size": int(module.group_size),
                "mode": str(module.mode),
                "has_bias": "bias" in module,
            }
            for name, module in modules.items()
        },
        "rows": [],
    }

    k = int(modules["in_proj_qkv"]["weight"].shape[1]) * 8

    combos: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = [
        ("qkv", ("in_proj_qkv",), (1, 2, 4)),
        ("z", ("in_proj_z",), (1, 2, 4)),
        ("b", ("in_proj_b",), (1,)),
        ("a", ("in_proj_a",), (1,)),
        ("qkv+z", ("in_proj_qkv", "in_proj_z"), (1, 2, 4, 8)),
        ("b+a", ("in_proj_b", "in_proj_a"), (1, 2)),
        (
            "qkv+z+b+a",
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
            (1, 2, 4),
        ),
    ]

    for m in (2, 3):
        x = _deterministic_rows(m, k)
        references = {name: _stock(x, module) for name, module in modules.items()}
        mx.eval(*references.values())
        for label, names, nsgs in combos:
            group = [modules[name] for name in names]

            def stock_group(group=group, x=x):  # type: ignore[no-untyped-def]
                return tuple(_stock(x, module) for module in group)

            for nsg in nsgs:
                row: dict[str, Any] = {
                    "M": m,
                    "combo": label,
                    "nsg": nsg,
                    "N": [int(module["weight"].shape[0]) for module in group],
                }
                ns = tuple(int(module["weight"].shape[0]) for module in group)
                if not exact_qmv_m23_multi_eligible(x, group, ns=ns, nsg=nsg):
                    row["eligible"] = False
                    payload["rows"].append(row)
                    continue
                row["eligible"] = True
                try:

                    def candidate_group(group=group, x=x, nsg=nsg):  # type: ignore[no-untyped-def]
                        return exact_qmv_m23_multi(x, group, nsg=nsg)

                    outs = candidate_group()
                    row["comparison"] = {
                        name: _compare(references[name], out)
                        for name, out in zip(names, outs)
                    }
                    row["array_equal"] = all(
                        item["array_equal"] for item in row["comparison"].values()
                    )
                    row["paired"] = _paired_time(
                        stock_group,
                        candidate_group,
                        epochs=args.paired_epochs,
                        inner=args.paired_inner,
                    )
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                payload["rows"].append(row)
                print(
                    json.dumps(
                        {
                            key: row.get(key)
                            for key in ("M", "combo", "nsg", "array_equal")
                        }
                    ),
                    flush=True,
                )

        # Whole-family comparison: the four stock launches versus one fused
        # launch, which is the shape the production hook actually replaces.
        family = [modules[name] for name in PROJECTIONS]
        ns_family = tuple(int(module["weight"].shape[0]) for module in family)

        def stock_family(family=family, x=x):  # type: ignore[no-untyped-def]
            return tuple(_stock(x, module) for module in family)

        for nsg in (1, 2, 4):
            if not exact_qmv_m23_multi_eligible(x, family, ns=ns_family, nsg=nsg):
                continue

            def fused_family(family=family, x=x, nsg=nsg):  # type: ignore[no-untyped-def]
                return exact_qmv_m23_multi(x, family, nsg=nsg)

            payload["rows"].append(
                {
                    "M": m,
                    "combo": "family_vs_four_stock_launches",
                    "nsg": nsg,
                    "eligible": True,
                    "paired": _paired_time(
                        stock_family,
                        fused_family,
                        epochs=args.paired_epochs,
                        inner=args.paired_inner,
                    ),
                    "array_equal": all(
                        _compare(references[name], out)["array_equal"]
                        for name, out in zip(PROJECTIONS, fused_family())
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["inventory"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
