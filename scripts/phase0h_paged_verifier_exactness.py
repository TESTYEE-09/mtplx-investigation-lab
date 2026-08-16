#!/usr/bin/env python3
"""Phase 0H paged-verifier exactness gate.

This compares the active vLLM-Metal paged target path against the stock MLX
target path on the same model, prompt, and sampler. It is intentionally focused
on target-verifier distribution safety before any decay-speed optimization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PROMPT = (
    "Create a single-file HTML5 Canvas flappy bird game. All visuals drawn "
    "procedurally. Animated bird with distinct up-stroke and down-stroke wing "
    "shapes, body tilt, squash-and-stretch on flap, feather particles from "
    "wing tips. Pipes with gradient shading, cap/lip, cylindrical highlight. "
    "Three-layer parallax background: sky with day/night colour cycle and "
    "stars, clouds with bobbing, rolling hills. Death explosion, +1 score pop, "
    "ambient floating motes. Start screen, death screen with best score in "
    "localStorage. Delta-time physics. Make it gorgeous."
)


@dataclass(frozen=True)
class SamplerConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


PAGED_ENV_KEYS = (
    "MTPLX_VLLM_METAL_PAGED_ATTN",
    "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE",
    "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE",
    "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N",
    "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES",
    "MTPLX_SPLIT_FULL_ATTN",
    "MTPLX_BLOCKWISE_ATTN",
    "MTPLX_SDPA_2PASS",
)


@contextmanager
def patched_env(updates: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _csv_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item < 1 for item in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def _repeat_tokens(token_ids: list[int], needed: int) -> tuple[list[int], bool]:
    if not token_ids:
        raise ValueError("prompt encoded to no tokens")
    out = list(token_ids)
    while len(out) < needed:
        out.extend(token_ids)
    return out[:needed], len(token_ids) < needed


def _profile_env(args: argparse.Namespace, *, enabled: bool) -> dict[str, str | None]:
    env = {key: None for key in PAGED_ENV_KEYS}
    if not enabled:
        return env
    env.update(
        {
            "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
            "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": str(args.block_size),
            "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": str(args.num_blocks),
            "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": args.attention_impl,
        }
    )
    if args.partitioned:
        env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] = "1"
        env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] = str(args.partition_threshold)
        env["MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE"] = str(args.partition_size)
    if args.exact_gather_last_n:
        env["MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N"] = str(
            args.exact_gather_last_n
        )
    if args.exact_gather_indices:
        env["MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES"] = (
            args.exact_gather_indices
        )
    return env


def _eval_result(value: Any) -> None:
    import mlx.core as mx

    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)


def _last_logits_np(logits: Any) -> np.ndarray:
    import mlx.core as mx

    final_logits = logits[:, -1, :].astype(mx.float32)
    mx.eval(final_logits)
    return np.asarray(final_logits, dtype=np.float32).reshape(-1)


def _forward_full_prefix_last_logits(rt: Any, token_ids: list[int]) -> tuple[np.ndarray, dict[str, Any], float]:
    import mlx.core as mx

    from mtplx.cache_state import tail_owned_attention_kv_stats

    cache = rt.make_cache()
    input_ids = mx.array([token_ids], dtype=mx.int32)
    started = time.perf_counter()
    logits = rt.forward_ar(input_ids, cache=cache, return_hidden=False)
    row = _last_logits_np(logits)
    elapsed = time.perf_counter() - started
    stats = tail_owned_attention_kv_stats(cache)
    return row, stats, elapsed


def _forward_decode_from_stock_prefix_last_logits(
    rt: Any,
    prefix_ids: list[int],
    verify_ids: list[int],
    args: argparse.Namespace,
    *,
    paged: bool,
) -> tuple[np.ndarray, dict[str, Any], float]:
    import mlx.core as mx

    from mtplx.attention_split import configure_split_full_attention
    from mtplx.cache_state import (
        install_vllm_metal_paged_attention_kv_cache,
        tail_owned_attention_kv_stats,
    )

    with patched_env(_profile_env(args, enabled=False)):
        configure_split_full_attention(rt.model)
        cache = rt.make_cache()
        prefill_ids = mx.array([prefix_ids], dtype=mx.int32)
        prefill = rt.forward_ar(prefill_ids, cache=cache, return_hidden=False)
        _eval_result(prefill)
    if paged:
        with patched_env(_profile_env(args, enabled=True)):
            install_vllm_metal_paged_attention_kv_cache(
                cache,
                block_size=args.block_size,
                num_blocks=args.num_blocks,
            )
    input_ids = mx.array([verify_ids], dtype=mx.int32)
    with patched_env(_profile_env(args, enabled=paged)):
        configure_split_full_attention(rt.model)
        started = time.perf_counter()
        logits = rt.forward_ar(input_ids, cache=cache, return_hidden=False)
        row = _last_logits_np(logits)
        elapsed = time.perf_counter() - started
    stats = tail_owned_attention_kv_stats(cache)
    return row, stats, elapsed


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        out = np.zeros_like(logits, dtype=np.float64)
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


def _distribution_from_logits(logits: np.ndarray, sampler: SamplerConfig) -> np.ndarray:
    probs = _softmax(logits, sampler.temperature)
    mask = np.ones(probs.shape[0], dtype=bool)
    if 0 < sampler.top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_sorted = cumulative <= float(sampler.top_p)
        if keep_sorted.size:
            keep_sorted[0] = True
            first_over = np.argmax(cumulative >= float(sampler.top_p))
            keep_sorted[: first_over + 1] = True
        nucleus_mask = np.zeros_like(mask)
        nucleus_mask[order[keep_sorted]] = True
        mask &= nucleus_mask
    if sampler.top_k and 0 < sampler.top_k < probs.shape[0]:
        scoped = np.where(mask, probs, 0.0)
        keep = np.argpartition(-scoped, int(sampler.top_k) - 1)[: int(sampler.top_k)]
        top_mask = np.zeros_like(mask)
        top_mask[keep] = True
        mask &= top_mask
    filtered = np.where(mask, probs, 0.0)
    total = filtered.sum()
    if total <= 0:
        filtered[int(np.argmax(probs))] = 1.0
        total = 1.0
    return filtered / total


def _top_ids(logits: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), int(logits.shape[0]))
    ids = np.argpartition(-logits, k - 1)[:k]
    order = np.argsort(-logits[ids])
    return ids[order].astype(np.int64)


def _sample_with_shared_uniforms(
    p: np.ndarray,
    q: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    p_cdf = np.cumsum(p)
    q_cdf = np.cumsum(q)
    # Guard final CDF entry against roundoff below 1.0.
    p_cdf[-1] = 1.0
    q_cdf[-1] = 1.0
    matches = 0
    first_mismatch: dict[str, Any] | None = None
    for idx, u in enumerate(rng.random(draws)):
        p_tok = int(np.searchsorted(p_cdf, u, side="left"))
        q_tok = int(np.searchsorted(q_cdf, u, side="left"))
        if p_tok == q_tok:
            matches += 1
        elif first_mismatch is None:
            first_mismatch = {
                "draw_index": idx,
                "u": float(u),
                "stock_token": p_tok,
                "paged_token": q_tok,
                "stock_prob": float(p[p_tok]),
                "paged_prob": float(q[q_tok]),
            }
    return {
        "draws": int(draws),
        "matches": int(matches),
        "agreement": float(matches / max(1, draws)),
        "first_mismatch": first_mismatch,
    }


def _distribution_metrics(
    stock_logits: np.ndarray,
    paged_logits: np.ndarray,
    sampler: SamplerConfig,
    *,
    top_k_compare: int,
    sample_seed: int,
    sample_draws: int,
) -> dict[str, Any]:
    diff = paged_logits.astype(np.float32) - stock_logits.astype(np.float32)
    stock_argmax = int(np.argmax(stock_logits))
    paged_argmax = int(np.argmax(paged_logits))
    stock_top = _top_ids(stock_logits, top_k_compare)
    paged_top = _top_ids(paged_logits, top_k_compare)
    overlap = len(set(stock_top.tolist()) & set(paged_top.tolist()))

    p = _distribution_from_logits(stock_logits, sampler)
    q = _distribution_from_logits(paged_logits, sampler)
    p_support = p > 0
    q_support = q > 0
    support_union = p_support | q_support
    support_intersection = p_support & q_support
    support_jaccard = float(
        support_intersection.sum() / max(1, support_union.sum())
    )
    eps = 1e-300
    kl_stock_to_paged = float(np.sum(p[p_support] * np.log(p[p_support] / np.maximum(q[p_support], eps))))
    kl_paged_to_stock = float(np.sum(q[q_support] * np.log(q[q_support] / np.maximum(p[q_support], eps))))
    tv = float(0.5 * np.sum(np.abs(p - q)))
    pseudo_n = 10000.0
    chi_square = float(
        np.sum(((q[support_union] - p[support_union]) * pseudo_n) ** 2 / np.maximum(p[support_union] * pseudo_n, eps))
    )
    sample = _sample_with_shared_uniforms(
        p,
        q,
        seed=sample_seed,
        draws=sample_draws,
    )
    return {
        "logits": {
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "rms_diff": float(math.sqrt(float(np.mean(diff.astype(np.float64) ** 2)))),
            "stock_argmax": stock_argmax,
            "paged_argmax": paged_argmax,
            "argmax_match": bool(stock_argmax == paged_argmax),
        },
        "topk": {
            "k": int(top_k_compare),
            "stock": stock_top.tolist(),
            "paged": paged_top.tolist(),
            "overlap": int(overlap),
            "overlap_ratio": float(overlap / max(1, min(top_k_compare, stock_logits.shape[0]))),
        },
        "distribution": {
            "stock_support_size": int(p_support.sum()),
            "paged_support_size": int(q_support.sum()),
            "support_intersection": int(support_intersection.sum()),
            "support_union": int(support_union.sum()),
            "support_jaccard": support_jaccard,
            "kl_stock_to_paged": kl_stock_to_paged,
            "kl_paged_to_stock": kl_paged_to_stock,
            "total_variation": tv,
            "chi_square_pseudo_n_10000": chi_square,
            "support_equal": bool(np.array_equal(p_support, q_support)),
        },
        "controlled_rng_sample": sample,
    }


def _row_passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    metrics = row["metrics"]
    return bool(
        metrics["logits"]["max_abs_diff"] <= args.max_logit_diff
        and metrics["topk"]["overlap_ratio"] >= args.min_topk_overlap
        and metrics["distribution"]["total_variation"] <= args.max_total_variation
        and metrics["distribution"]["support_equal"]
        and metrics["controlled_rng_sample"]["agreement"] >= args.min_sample_agreement
        and metrics["logits"]["argmax_match"]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.attention_split import configure_split_full_attention
    from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
    from mtplx.runtime import load

    rt = load(args.model, mtp=not args.no_mtp)
    if args.prompt_suite:
        case = load_prompt_suite(args.prompt_suite)[args.prompt_index]
        base_ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=not args.raw_prompt_suite,
            enable_thinking=False if args.disable_thinking else None,
        )
        prompt_source = {
            "prompt_suite": str(args.prompt_suite),
            "prompt_index": int(args.prompt_index),
            "prompt_id": case.id,
            "prompt_sha256": case.prompt_sha256,
        }
    else:
        base_ids = list(rt.tokenizer.encode(args.prompt))
        prompt_source = {"prompt": args.prompt}
    max_context = max(args.contexts)
    needed = max_context + (args.verify_tokens if args.mode == "decode-from-stock-prefix" else 0)
    token_ids, synthetic_repeat = _repeat_tokens(base_ids, needed)
    sampler = SamplerConfig(args.temperature, args.top_p, args.top_k)

    rows: list[dict[str, Any]] = []
    for context_len in args.contexts:
        prefix = token_ids[:context_len]
        verify_ids = token_ids[context_len : context_len + args.verify_tokens]
        if args.mode == "full-prefix":
            with patched_env(_profile_env(args, enabled=False)):
                configure_split_full_attention(rt.model)
                stock_logits, stock_cache_stats, stock_elapsed = _forward_full_prefix_last_logits(rt, prefix)
            mx.clear_cache()
            with patched_env(_profile_env(args, enabled=True)):
                configure_split_full_attention(rt.model)
                paged_logits, paged_cache_stats, paged_elapsed = _forward_full_prefix_last_logits(rt, prefix)
        else:
            stock_logits, stock_cache_stats, stock_elapsed = _forward_decode_from_stock_prefix_last_logits(
                rt,
                prefix,
                verify_ids,
                args,
                paged=False,
            )
            mx.clear_cache()
            paged_logits, paged_cache_stats, paged_elapsed = _forward_decode_from_stock_prefix_last_logits(
                rt,
                prefix,
                verify_ids,
                args,
                paged=True,
            )
        mx.clear_cache()
        row = {
            "mode": args.mode,
            "context_len": int(context_len),
            "verify_tokens": int(args.verify_tokens if args.mode == "decode-from-stock-prefix" else context_len),
            "stock_elapsed_s": stock_elapsed,
            "paged_elapsed_s": paged_elapsed,
            "stock_cache_stats": stock_cache_stats,
            "paged_cache_stats": paged_cache_stats,
            "metrics": _distribution_metrics(
                stock_logits,
                paged_logits,
                sampler,
                top_k_compare=args.top_k_compare,
                sample_seed=args.seed + int(context_len),
                sample_draws=args.sample_draws,
            ),
        }
        row["passed"] = _row_passes(row, args)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    passed = all(row["passed"] for row in rows)
    return {
        "run_id": f"phase0h-paged-verifier-exactness-{time.strftime('%Y%m%d-%H%M%S')}",
        "model": str(args.model),
        "mtp_enabled": not args.no_mtp,
        "prompt_source": prompt_source,
        "prompt_tokens": len(base_ids),
        "synthetic_repeat": synthetic_repeat,
        "contexts": args.contexts,
        "sampler": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "profile": {
            "paged": True,
            "partitioned": args.partitioned,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "partition_threshold": args.partition_threshold,
            "partition_size": args.partition_size,
            "attention_impl": args.attention_impl,
            "exact_gather_last_n": args.exact_gather_last_n,
            "exact_gather_indices": args.exact_gather_indices,
        },
        "thresholds": {
            "max_logit_diff": args.max_logit_diff,
            "max_total_variation": args.max_total_variation,
            "min_topk_overlap": args.min_topk_overlap,
            "min_sample_agreement": args.min_sample_agreement,
        },
        "passed": passed,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.6-27B-MLXCommunity-4bit-CyanKiwiMTP"))
    parser.add_argument("--contexts", type=_csv_ints, default=_csv_ints("128,512,2048"))
    parser.add_argument("--mode", choices=["full-prefix", "decode-from-stock-prefix"], default="decode-from-stock-prefix")
    parser.add_argument("--verify-tokens", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-suite", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--raw-prompt-suite", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-k-compare", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-draws", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--partitioned", action="store_true", default=True)
    parser.add_argument("--no-partitioned", action="store_false", dest="partitioned")
    parser.add_argument("--partition-threshold", type=int, default=2048)
    parser.add_argument("--partition-size", type=int, default=512)
    parser.add_argument("--attention-impl", default="")
    parser.add_argument("--exact-gather-last-n", type=int, default=0)
    parser.add_argument("--exact-gather-indices", default="")
    parser.add_argument("--max-logit-diff", type=float, default=3e-2)
    parser.add_argument("--max-total-variation", type=float, default=5e-3)
    parser.add_argument("--min-topk-overlap", type=float, default=0.95)
    parser.add_argument("--min-sample-agreement", type=float, default=0.995)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output) if args.output else None}, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
