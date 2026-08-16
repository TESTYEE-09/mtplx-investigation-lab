#!/usr/bin/env python3
"""Repeated production-style Qwen3.8 capture-commit benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import mlx.core as mx

from mtplx import m1_q4_verify
from mtplx.generation import generate_mtpk
from mtplx.profiles import apply_profile_env
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    try:
        values = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        values = tokenizer.encode(prompt)
    return [int(value) for value in values]


def _one_run(
    rt: Any,
    prompt_ids: list[int],
    *,
    depth: int,
    max_tokens: int,
    seed: int,
    sampler: SamplerConfig,
) -> dict[str, Any]:
    out = generate_mtpk(
        rt,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=depth,
        seed=seed,
        stop_token_ids=set(),
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )
    stats = out.stats
    calls = max(1, int(stats.verify_calls))
    acceptance = [
        (float(accepted) / float(drafted) if drafted else None)
        for accepted, drafted in zip(stats.accepted_by_depth, stats.drafted_by_depth)
    ]
    return {
        "depth": depth,
        "generated_tokens": int(stats.generated_tokens),
        "tok_s": float(stats.tok_s),
        "decode_tok_s": float(stats.decode_tok_s),
        "elapsed_s": float(stats.elapsed_s),
        "verify_calls": int(stats.verify_calls),
        "verify_eval_ms_per_call": 1000.0 * float(stats.verify_eval_time_s) / calls,
        "verify_forward_ms_per_call": 1000.0
        * float(stats.verify_forward_time_s)
        / calls,
        "verify_ms_per_call": 1000.0 * float(stats.verify_time_s) / calls,
        "draft_ms_per_call": 1000.0 * float(stats.draft_time_s) / calls,
        "acceptance_by_position": acceptance,
        "mean_accept_probability_by_position": list(
            stats.mean_accept_probability_by_depth
        ),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "peak_memory_bytes": int(stats.peak_memory_bytes),
        "active_memory_bytes": int(mx.get_active_memory()),
        "tokens": [int(value) for value in out.tokens],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Write a Python function that merges two sorted integer lists. "
            "Include type hints, a concise docstring, and two examples."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        choices=(1, 2, 3),
    )
    parser.add_argument("--runtime-profile", default="turbo")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Do not inherit unrelated experimental lanes from the invoking shell.
    for name in (
        "MTPLX_EXPERIMENTAL_M1_GDN_VERIFY",
        "MTPLX_EXPERIMENTAL_MTP_LM_HEAD_TOPK",
    ):
        os.environ.pop(name, None)
    apply_profile_env(args.runtime_profile)
    os.environ["MTPLX_MTP_HISTORY_POLICY"] = "committed"

    rt = load(args.model, mtp=True)
    prompt_ids = _encode_prompt(rt.tokenizer, args.prompt)
    sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    results = []
    for depth in args.depths:
        _one_run(
            rt,
            prompt_ids,
            depth=depth,
            max_tokens=args.warmup_tokens,
            seed=args.seed,
            sampler=sampler,
        )
        rows = [
            _one_run(
                rt,
                prompt_ids,
                depth=depth,
                max_tokens=args.max_tokens,
                seed=args.seed,
                sampler=sampler,
            )
            for _ in range(args.repeats)
        ]
        results.append(
            {
                "depth": depth,
                "samples": rows,
                "median": {
                    key: _median(rows, key)
                    for key in (
                        "tok_s",
                        "decode_tok_s",
                        "verify_eval_ms_per_call",
                        "verify_forward_ms_per_call",
                        "verify_ms_per_call",
                        "draft_ms_per_call",
                        "peak_memory_bytes",
                        "active_memory_bytes",
                    )
                },
                "acceptance_by_position": rows[0]["acceptance_by_position"],
                "mean_accept_probability_by_position": rows[0][
                    "mean_accept_probability_by_position"
                ],
                "repeat_tokens_identical": all(
                    row["tokens"] == rows[0]["tokens"] for row in rows[1:]
                ),
            }
        )
    payload = {
        "model": str(args.model),
        "prompt_tokens": len(prompt_ids),
        "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "mtp_history_policy": "committed",
        "runtime_profile": args.runtime_profile,
        "m1_q4_lanes": m1_q4_verify.config().as_dict(),
        "m1_q4_lane_signature": repr(m1_q4_verify.lane_signature()),
        "m1_q4_lane_stats": dict(m1_q4_verify.LANE_STATS),
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("lanes:", payload["m1_q4_lanes"])
    print("lane stats:", payload["m1_q4_lane_stats"])
    for result in results:
        median = result["median"]
        print(
            f"depth={result['depth']} tok_s={median['tok_s']:.3f} "
            f"verify={median['verify_ms_per_call']:.1f}ms "
            f"forward={median['verify_forward_ms_per_call']:.1f}ms"
        )


if __name__ == "__main__":
    main()
