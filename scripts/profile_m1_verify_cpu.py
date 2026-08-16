#!/usr/bin/env python3
"""Where the M1 Pro verify forward spends CPU time, and which lane served.

``verify_forward`` is pure graph construction (no ``mx.eval``), and on this
machine it dwarfs the final eval wait, so decode throughput is limited by the
Python/graph side rather than by GPU bandwidth. This profiles that side and
reports the experimental-lane engagement counters, so a "tokens matched"
result can be distinguished from "the experimental launch never ran".
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
from pathlib import Path
from typing import Any

import mlx.core as mx

from mtplx.generation import generate_mtpk
from mtplx.profiles import apply_profile_env
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


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
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--runtime-profile", default="turbo")
    parser.add_argument("--mlp-lane", default="1")
    parser.add_argument("--gdn-lane", default="1")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--top", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    apply_profile_env(args.runtime_profile)
    os.environ["MTPLX_MTP_HISTORY_POLICY"] = "committed"
    os.environ["MTPLX_M1_Q4_MLP"] = args.mlp_lane
    os.environ["MTPLX_M1_Q4_GDN_PROJ"] = args.gdn_lane
    os.environ["MTPLX_M1_Q4_STATS"] = "1"

    rt = load(args.model, mtp=True)
    from mtplx.m1_q4_verify import LANE_STATS as M1_LANE_STATS

    prompt_ids = _encode_prompt(rt.tokenizer, args.prompt)
    sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)

    def run(max_tokens: int):
        return generate_mtpk(
            rt,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_sampler=sampler,
            speculative_depth=args.depth,
            seed=args.seed,
            stop_token_ids=set(),
            mtp_history_policy="committed",
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    # The verify step is traced once and cached process-wide
    # (graphbank._SHARED_VERIFY_STEPS), so the Python hooks only execute
    # during tracing. Counting the warmup separately from the timed run is
    # the only way to tell "the fused launch is baked into the graph" from
    # "the lane never engaged".
    for key in M1_LANE_STATS:
        M1_LANE_STATS[key] = 0
    run(args.warmup_tokens)
    trace_phase_stats = dict(M1_LANE_STATS)
    for key in M1_LANE_STATS:
        M1_LANE_STATS[key] = 0

    profiler = cProfile.Profile()
    profiler.enable()
    out = run(args.max_tokens)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(args.top)
    cumulative = stream.getvalue()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(args.top)
    tottime = stream.getvalue()

    stats = out.stats
    calls = max(1, int(stats.verify_calls))
    payload = {
        "device": mx.device_info(),
        "depth": args.depth,
        "mlp_lane": args.mlp_lane,
        "gdn_lane": args.gdn_lane,
        "lane_stats_trace_phase": trace_phase_stats,
        "lane_stats": dict(M1_LANE_STATS),
        "lane_calls_per_verify": {
            key: value / calls for key, value in M1_LANE_STATS.items()
        },
        "verify_calls": calls,
        "tok_s": float(stats.tok_s),
        "verify_ms_per_call": 1000.0 * float(stats.verify_time_s) / calls,
        "verify_forward_ms_per_call": 1000.0
        * float(stats.verify_forward_time_s)
        / calls,
        "verify_eval_ms_per_call": 1000.0 * float(stats.verify_eval_time_s) / calls,
        "graphbank": stats.graphbank,
        "profile_cumulative": cumulative,
        "profile_tottime": tottime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "lane_stats_trace_phase",
        "lane_stats", "lane_calls_per_verify", "verify_calls", "tok_s",
        "verify_forward_ms_per_call", "verify_eval_ms_per_call",
    )}, indent=2))
    print(json.dumps(payload["graphbank"], indent=2, default=str)[:3000])
    print(cumulative[:6000])


if __name__ == "__main__":
    main()
