#!/usr/bin/env python3
"""Collect MTP linear input activation stats for AWQ-style calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from collect_mtp_hidden_calib import (  # noqa: E402
    _input_pair,
    _target_trace,
    _target_trace_sampled,
)
from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite, now_run_id  # noqa: E402
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR  # noqa: E402
from mtplx.mtp_activation_stats import (  # noqa: E402
    activation_stats_to_npz_payload,
    install_mtp_activation_recorders,
)
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.runtime import load  # noqa: E402
from mtplx.sampling import SamplerConfig  # noqa: E402


def _parse_targets(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    targets = [part.strip() for part in value.split(",") if part.strip()]
    if not targets:
        raise ValueError("targets CSV did not contain any targets")
    return targets


def _argmax_token(logits: mx.array) -> int:
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    return int(token.reshape(-1).tolist()[0])


def collect_mtp_activation_stats(
    model_path: Path | str,
    prompt_suite: Path | str,
    output_dir: Path | str,
    *,
    limit: int | None = 2,
    windows: int = 32,
    stride: int = 1,
    depth: int = 2,
    max_prompt_tokens: int = 256,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
    base_hidden_variant: str = "pre_norm",
    mtp_hidden_variant: str = "pre_norm",
    cache_policy: str = "persistent",
    anchor: str = "prompt_boundary",
    concat_order: str = "embedding_hidden",
    target_sampler: str = "stochastic",
    sampler_temperature: float = 0.6,
    seed: int = 0,
    targets: list[str] | None = None,
    summary_top_n: int = 16,
    summary_group_size: int = 64,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    mtp_quant_bits: int | None = None,
    mtp_quant_group_size: int = 64,
    mtp_quant_mode: str = "affine",
) -> dict[str, Any]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if windows < 1:
        raise ValueError("windows must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if cache_policy not in {"fresh", "persistent"}:
        raise ValueError("cache_policy must be 'fresh' or 'persistent'")
    if anchor not in {"prompt_boundary", "after_one_target"}:
        raise ValueError("anchor must be 'prompt_boundary' or 'after_one_target'")
    if target_sampler not in {"greedy", "stochastic"}:
        raise ValueError("target_sampler must be 'greedy' or 'stochastic'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rt = load(
        model_path,
        mtp=True,
        contract=MTPContract(
            mtp_quant_bits=mtp_quant_bits,
            mtp_quant_group_size=mtp_quant_group_size,
            mtp_quant_mode=mtp_quant_mode,
        ),
        mtp_adapter=mtp_adapter,
        merge_mtp_adapter=merge_mtp_adapter,
    )
    recorders = install_mtp_activation_recorders(rt.model, targets=targets)

    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]

    target_token_count = ((windows - 1) * stride) + depth + 4
    draft_calls = 0
    for prompt_index, case in enumerate(prompts):
        ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
        )[-max_prompt_tokens:]
        if target_sampler == "stochastic":
            trace = _target_trace_sampled(
                rt,
                ids,
                target_token_count=target_token_count,
                base_hidden_variant=base_hidden_variant,
                sampler=SamplerConfig(
                    temperature=sampler_temperature,
                    top_p=0.95,
                    top_k=20,
                ),
                seed=seed + prompt_index,
            )
        else:
            trace = _target_trace(
                rt,
                ids,
                target_token_count=target_token_count,
                base_hidden_variant=base_hidden_variant,
            )

        for window_index in range(windows):
            window_start = window_index * stride
            anchor_offset = window_start + (0 if anchor == "prompt_boundary" else 1)
            rec_hidden, rec_next_token = _input_pair(trace, anchor_offset)
            rec_cache = rt.make_mtp_cache() if cache_policy == "persistent" else None

            for depth_index in range(depth):
                current_depth = depth_index + 1
                step_cache = rec_cache if cache_policy == "persistent" else rt.make_mtp_cache()
                logits, rec_next_hidden = rt.draft_mtp(
                    rec_hidden,
                    mx.array([[int(rec_next_token)]]),
                    mtp_cache=step_cache,
                    concat_order=concat_order,
                    return_hidden=True,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_depth=current_depth,
                )
                mx.eval(logits, rec_next_hidden)
                rec_next_token = _argmax_token(logits)
                rec_hidden = rec_next_hidden[:, -1:, :]
                draft_calls += 1

    stats = [recorder.stats() for recorder in recorders]
    summaries = [
        item.summary(top_n=summary_top_n, group_size=summary_group_size)
        for item in stats
    ]

    npz_path = output_dir / "activation_stats.npz"
    metadata = {
        "run_id": output_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "limit": limit,
        "windows": windows,
        "stride": stride,
        "depth": depth,
        "draft_calls": int(draft_calls),
        "max_prompt_tokens": max_prompt_tokens,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "base_hidden_variant": base_hidden_variant,
        "mtp_hidden_variant": mtp_hidden_variant,
        "cache_policy": cache_policy,
        "anchor": anchor,
        "concat_order": concat_order,
        "target_sampler": target_sampler,
        "sampler_temperature": sampler_temperature,
        "seed": seed,
        "targets": [item.target for item in stats],
        "target_count": len(stats),
        "summary_top_n": summary_top_n,
        "summary_group_size": summary_group_size,
        "mtp_adapter_path": str(mtp_adapter) if mtp_adapter is not None else None,
        "mtp_adapter_merged": bool(merge_mtp_adapter),
        "mtp_adapter_metadata": rt.mtp_adapter_metadata,
        "mtp_adapter_merge_report": rt.mtp_adapter_merge_report,
        "mtp_quant_bits": rt.contract.mtp_quant_bits,
        "mtp_quant_group_size": rt.contract.mtp_quant_group_size,
        "mtp_quant_mode": rt.contract.mtp_quant_mode,
        "mtp_quant_policy": rt.contract.mtp_quant_policy,
        "mtp_prequantized": rt.contract.mtp_prequantized,
        "elapsed_s": time.perf_counter() - started,
        "npz_path": str(npz_path),
    }
    payload = activation_stats_to_npz_payload(stats)
    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(npz_path, **payload)

    report = {
        **metadata,
        "summaries": summaries,
    }
    report_path = output_dir / "activation_stats.json"
    report["json_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_RUNTIME_MODEL_DIR)
    parser.add_argument("--prompts", type=Path, default=Path("mtplx/benchmarks/prompts/default.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--base-hidden-variant", choices=["pre_norm", "post_norm"], default="pre_norm")
    parser.add_argument("--mtp-hidden-variant", choices=["pre_norm", "post_norm"], default="pre_norm")
    parser.add_argument("--cache-policy", choices=["fresh", "persistent"], default="persistent")
    parser.add_argument("--anchor", choices=["prompt_boundary", "after_one_target"], default="prompt_boundary")
    parser.add_argument("--concat-order", choices=["embedding_hidden", "hidden_embedding"], default="embedding_hidden")
    parser.add_argument("--target-sampler", choices=["greedy", "stochastic"], default="stochastic")
    parser.add_argument("--sampler-temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--summary-top-n", type=int, default=16)
    parser.add_argument("--summary-group-size", type=int, default=64)
    parser.add_argument("--mtp-adapter", type=Path, default=None)
    parser.add_argument("--merge-mtp-adapter", action="store_true")
    parser.add_argument("--mtp-quant-bits", type=int, default=None)
    parser.add_argument("--mtp-quant-group-size", type=int, default=64)
    parser.add_argument("--mtp-quant-mode", choices=["affine", "symmetric"], default="affine")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--print-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or Path("outputs") / now_run_id("mtp-activation-stats")
    report = collect_mtp_activation_stats(
        args.model,
        args.prompts,
        output_dir,
        limit=args.limit,
        windows=args.windows,
        stride=args.stride,
        depth=args.depth,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.no_chat_template,
        enable_thinking=args.enable_thinking,
        base_hidden_variant=args.base_hidden_variant,
        mtp_hidden_variant=args.mtp_hidden_variant,
        cache_policy=args.cache_policy,
        anchor=args.anchor,
        concat_order=args.concat_order,
        target_sampler=args.target_sampler,
        sampler_temperature=args.sampler_temperature,
        seed=args.seed,
        targets=_parse_targets(args.targets),
        summary_top_n=args.summary_top_n,
        summary_group_size=args.summary_group_size,
        mtp_adapter=args.mtp_adapter,
        merge_mtp_adapter=args.merge_mtp_adapter,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.print_full:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    concise_summaries = []
    for item in report["summaries"]:
        if int(item.get("calls") or 0) <= 0:
            continue
        concise_summaries.append(
            {
                "target": item["target"],
                "calls": item["calls"],
                "rows": item["rows"],
                "mean_abs_avg": item["mean_abs_avg"],
                "mean_abs_p95": item["mean_abs_p95"],
                "mean_abs_max": item["mean_abs_max"],
                "top_1pct_mean_abs_share": item["top_1pct_mean_abs_share"],
                "top_5pct_mean_abs_share": item["top_5pct_mean_abs_share"],
                "top_channels": item["top_channels"][:5],
            }
        )
    print(
        json.dumps(
            {
                "json_path": report["json_path"],
                "npz_path": report["npz_path"],
                "draft_calls": report["draft_calls"],
                "target_count": report["target_count"],
                "active_target_count": len(concise_summaries),
                "elapsed_s": report["elapsed_s"],
                "summaries": concise_summaries[:12],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
