#!/usr/bin/env python3
"""Step-3.7-Flash MTP acceptance gate (Phase 3/4).

Runs the real speculative MTP loop on the converted Step model via the project's
depth-sweep runner, with the Step contract (pre-norm hidden, batched MoE verify),
and reports per-position acceptance + tok/s vs the AR baseline. One model load,
sweeps the requested depths.

Usage: step_acceptance.py [variant=pre_norm] [depths=1,2,3] [limit=2] [max_tokens=64]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mtplx.benchmarks.runners.mtp_adaptive import run_mtp_adaptive  # noqa: E402
from mtplx.benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep  # noqa: E402

DEFAULT_MODEL = "models/Step-3.7-Flash-MTPLX-step3p5"
DEFAULT_SUITE = "mtplx/benchmarks/prompts/calibration_coding.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", default=os.environ.get("STEP_VARIANT", "pre_norm"))
    parser.add_argument("depths", nargs="?", default=os.environ.get("STEP_DEPTHS", "1,2,3"))
    parser.add_argument("limit", nargs="?", type=int, default=int(os.environ.get("STEP_LIMIT", "2")))
    parser.add_argument(
        "max_tokens",
        nargs="?",
        type=int,
        default=int(os.environ.get("STEP_MAX_TOKENS", "64")),
    )
    parser.add_argument(
        "temperature",
        nargs="?",
        type=float,
        default=float(os.environ.get("STEP_TEMPERATURE", "0.6")),
    )
    parser.add_argument("tag", nargs="?", default=None)
    parser.add_argument("--model", default=os.environ.get("STEP_MODEL", DEFAULT_MODEL))
    parser.add_argument("--suite", default=os.environ.get("STEP_SUITE", DEFAULT_SUITE))
    parser.add_argument("--history-policy", default=os.environ.get("STEP_HISTORY_POLICY", "cycle"))
    parser.add_argument("--verify-strategy", default=os.environ.get("STEP_VERIFY_STRATEGY", "batched"))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("STEP_TOP_P", "0.95")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("STEP_TOP_K", "20")))
    parser.add_argument("--draft-temperature", type=float, default=None)
    parser.add_argument("--draft-top-p", type=float, default=None)
    parser.add_argument("--draft-top-k", type=int, default=None)
    parser.add_argument("--draft-margin-threshold", type=float, default=None)
    parser.add_argument("--min-speculative-depth", type=int, default=1)
    parser.add_argument(
        "--adaptive-policy",
        choices=("none", "streak", "expected_value"),
        default=os.environ.get("STEP_ADAPTIVE_POLICY", "none"),
    )
    parser.add_argument("--adaptive-min-depth", type=int, default=1)
    parser.add_argument("--adaptive-start-depth", type=int, default=1)
    parser.add_argument("--adaptive-increase-after", type=int, default=4)
    parser.add_argument("--adaptive-decrease-after", type=int, default=1)
    parser.add_argument("--adaptive-ev-base-depth", type=int, default=1)
    parser.add_argument(
        "--adaptive-ev-accept-priors",
        default=os.environ.get("STEP_ADAPTIVE_EV_ACCEPT_PRIORS", "0.72,0.42,0.18"),
    )
    parser.add_argument("--adaptive-ev-draft-cost-s", type=float, default=0.0048)
    parser.add_argument("--adaptive-ev-extra-verify-cost-s", type=float, default=0.006)
    parser.add_argument("--adaptive-ev-baseline-tok-s", type=float, default=55.0)
    parser.add_argument("--adaptive-ev-safety-margin", type=float, default=0.10)
    parser.add_argument("--adaptive-ev-margin-center", type=float, default=1.0)
    parser.add_argument("--adaptive-ev-margin-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-ev-confidence-weight", type=float, default=0.35)
    parser.add_argument("--adaptive-ev-min-extra-accept-probability", type=float, default=0.18)
    parser.add_argument("--adaptive-ev-warmup-full-depth-cycles", type=int, default=4)
    parser.add_argument("--adaptive-ev-exploration-interval", type=int, default=32)
    parser.add_argument("--mtp-adapter", type=Path, default=None)
    parser.add_argument("--merge-mtp-adapter", action="store_true")
    parser.add_argument("--mtp-quant-bits", type=int, default=None)
    parser.add_argument("--mtp-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--mtp-quant-mode",
        choices=("affine", "symmetric"),
        default="affine",
    )
    parser.add_argument("--adapter-ensemble-q", action="store_true")
    parser.add_argument("--adapter-ensemble-epsilon", type=float, default=0.5)
    parser.add_argument("--adapter-ensemble-min-depth", type=int, default=2)
    parser.add_argument("--online-hidden-alpha", type=float, default=0.0)
    parser.add_argument("--online-hidden-decay", type=float, default=0.8)
    parser.add_argument("--online-hidden-warmup", type=int, default=1)
    parser.add_argument("--online-hidden-max-feed-depth", type=int, default=None)
    parser.add_argument(
        "--online-hidden-key",
        choices=("global", "token"),
        default="global",
    )
    parser.add_argument("--online-correction-cache", action="store_true")
    parser.add_argument("--prompt-correction-cache", action="store_true")
    parser.add_argument("--mtp-topk-reranker-calib", type=Path, default=None)
    parser.add_argument("--mtp-topk-reranker-depths", default="4")
    parser.add_argument("--mtp-topk-reranker-topk", type=int, default=32)
    parser.add_argument("--mtp-topk-reranker-q-weight", type=float, default=0.5)
    parser.add_argument("--mtp-topk-reranker-token-weight", type=float, default=1.0)
    parser.add_argument("--mtp-topk-reranker-rank-weight", type=float, default=0.0)
    parser.add_argument("--mtp-topk-reranker-all-rows", action="store_true")
    parser.add_argument("--draft-lm-head-bits", type=int, default=None)
    parser.add_argument("--draft-lm-head-group-size", type=int, default=64)
    parser.add_argument(
        "--draft-lm-head-mode",
        choices=("affine", "symmetric"),
        default="affine",
    )
    parser.add_argument(
        "--fast-verify-profile",
        action="store_true",
        help=(
            "Set the Step verifier fast-profile env explicitly: "
            "MTPLX_LAZY_BONUS_VERIFY=1, MTPLX_BATCH_TARGET_ARRAYS=1, "
            "and MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("STEP_OUTPUT_DIR", str(REPO / "outputs")),
    )
    return parser.parse_args()


args = _parse_args()
variant = args.variant
depths = [int(x) for x in str(args.depths).split(",") if x.strip()]
limit = args.limit
max_tokens = args.max_tokens
temperature = args.temperature
tag = args.tag or variant
model = str(Path(args.model).expanduser()) if str(args.model).startswith("~") else args.model
suite = str(Path(args.suite).expanduser()) if str(args.suite).startswith("~") else args.suite


def _float_tuple_csv(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise ValueError("expected at least one float in CSV")
    return parsed


def _runtime_env_snapshot() -> dict[str, str | None]:
    keys = (
        "MTPLX_LAZY_BONUS_VERIFY",
        "MTPLX_BATCH_TARGET_ARRAYS",
        "MTPLX_BATCH_TARGET_DISTS",
        "MTPLX_DEFER_VERIFY_HIDDEN_EVAL",
        "MTPLX_VERIFY_HIDDEN_MODE",
        "MTPLX_LAZY_VERIFY_LOGITS",
        "MTPLX_SPLIT_VERIFY_EVAL",
    )
    return {key: os.environ.get(key) for key in keys}


if args.fast_verify_profile:
    os.environ["MTPLX_LAZY_BONUS_VERIFY"] = "1"
    os.environ["MTPLX_BATCH_TARGET_ARRAYS"] = "1"
    os.environ["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] = "1"

runtime_env = _runtime_env_snapshot()

print(
    "STEP_ACCEPTANCE "
    f"model={model} variant={variant} depths={depths} "
    f"limit={limit} max_tokens={max_tokens} temperature={temperature} "
    f"top_p={args.top_p} top_k={args.top_k} "
    f"history={args.history_policy} verify={args.verify_strategy} "
    f"adaptive_policy={args.adaptive_policy} "
    f"draft_temperature={args.draft_temperature} "
    f"draft_margin_threshold={args.draft_margin_threshold} "
    f"mtp_adapter={args.mtp_adapter} "
    f"merge_mtp_adapter={args.merge_mtp_adapter} "
    f"mtp_quant_bits={args.mtp_quant_bits} "
    f"adapter_ensemble_q={args.adapter_ensemble_q} "
    f"online_hidden_alpha={args.online_hidden_alpha} "
    f"draft_lm_head_bits={args.draft_lm_head_bits} "
    f"fast_verify_profile={args.fast_verify_profile} "
    f"runtime_env={runtime_env}",
    flush=True,
)

if args.adaptive_policy != "none":
    if args.draft_lm_head_bits is not None:
        raise SystemExit("--draft-lm-head-bits is only supported by fixed-depth sweeps")
    res = run_mtp_adaptive(
        model,
        suite,
        max_depth=max(depths),
        min_depth=args.adaptive_min_depth,
        start_depth=args.adaptive_start_depth,
        increase_after=args.adaptive_increase_after,
        decrease_after=args.adaptive_decrease_after,
        policy_kind=args.adaptive_policy,
        ev_base_depth=args.adaptive_ev_base_depth,
        ev_accept_priors=_float_tuple_csv(args.adaptive_ev_accept_priors),
        ev_draft_cost_s=args.adaptive_ev_draft_cost_s,
        ev_extra_verify_cost_s=args.adaptive_ev_extra_verify_cost_s,
        ev_baseline_tok_s=args.adaptive_ev_baseline_tok_s,
        ev_safety_margin=args.adaptive_ev_safety_margin,
        ev_margin_center=args.adaptive_ev_margin_center,
        ev_margin_scale=args.adaptive_ev_margin_scale,
        ev_confidence_weight=args.adaptive_ev_confidence_weight,
        ev_min_extra_accept_probability=args.adaptive_ev_min_extra_accept_probability,
        ev_warmup_full_depth_cycles=args.adaptive_ev_warmup_full_depth_cycles,
        ev_exploration_interval=args.adaptive_ev_exploration_interval,
        max_tokens=max_tokens,
        limit=limit,
        compare_ar=True,
        mtp_hidden_variant=variant,
        mtp_history_policy=args.history_policy,
        verify_strategy=args.verify_strategy,
        enable_thinking=False,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        mtp_adapter_path=args.mtp_adapter,
        merge_mtp_adapter=args.merge_mtp_adapter,
        seed=0,
    )
else:
    res = run_mtp_depth_sweep(
        model,
        suite,
        depths=depths,
        max_tokens=max_tokens,
        limit=limit,
        compare_ar=True,
        mtp_hidden_variant=variant,
        mtp_history_policy=args.history_policy,
        verify_strategy=args.verify_strategy,
        enable_thinking=False,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        draft_margin_threshold=args.draft_margin_threshold,
        min_speculative_depth=args.min_speculative_depth,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        mtp_adapter_path=args.mtp_adapter,
        merge_mtp_adapter=args.merge_mtp_adapter,
        adapter_ensemble_q=args.adapter_ensemble_q,
        adapter_ensemble_epsilon=args.adapter_ensemble_epsilon,
        adapter_ensemble_min_depth=args.adapter_ensemble_min_depth,
        online_hidden_corrector_alpha=args.online_hidden_alpha,
        online_hidden_corrector_decay=args.online_hidden_decay,
        online_hidden_corrector_warmup=args.online_hidden_warmup,
        online_hidden_corrector_max_feed_depth=args.online_hidden_max_feed_depth,
        online_hidden_corrector_key=args.online_hidden_key,
        online_correction_cache=args.online_correction_cache,
        prompt_correction_cache=args.prompt_correction_cache,
        mtp_topk_reranker_calib=args.mtp_topk_reranker_calib,
        mtp_topk_reranker_depths=args.mtp_topk_reranker_depths,
        mtp_topk_reranker_topk=args.mtp_topk_reranker_topk,
        mtp_topk_reranker_q_weight=args.mtp_topk_reranker_q_weight,
        mtp_topk_reranker_token_weight=args.mtp_topk_reranker_token_weight,
        mtp_topk_reranker_rank_weight=args.mtp_topk_reranker_rank_weight,
        mtp_topk_reranker_prefix_active_only=not args.mtp_topk_reranker_all_rows,
        draft_lm_head_bits=args.draft_lm_head_bits,
        draft_lm_head_group_size=args.draft_lm_head_group_size,
        draft_lm_head_mode=args.draft_lm_head_mode,
        seed=0,
    )

if isinstance(res, dict):
    res["step_acceptance_runtime_env"] = runtime_env
    res["step_acceptance_fast_verify_profile"] = bool(args.fast_verify_profile)

out = Path(args.output_dir).expanduser() / f"step_acceptance_{tag}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2, default=str))
print("WROTE", out, flush=True)


def _g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


# Summarize regardless of exact schema.
print("--- SUMMARY ---", flush=True)
print("top-level keys:", list(res.keys()) if isinstance(res, dict) else type(res), flush=True)
rows = res.get("depths") or res.get("results") or res.get("runs") or []
if isinstance(rows, dict):
    rows = list(rows.values())
for row in rows if isinstance(rows, list) else []:
    if not isinstance(row, dict):
        continue
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else row
    d = row.get("depth") or row.get("mtp_depth")
    acc = (
        summary.get("per_position_acceptance")
        or summary.get("acceptance_by_position")
        or summary.get("acceptance")
        or summary.get("acceptance_by_depth")
        or summary.get("accept_rate_by_depth")
        or summary.get("mean_acceptance_by_depth")
    )
    accepted = summary.get("accepted_by_depth")
    drafted = summary.get("drafted_by_depth")
    toks = summary.get("mean_tok_s") or summary.get("tok_s") or summary.get("tokens_per_s")
    decode = summary.get("mean_decode_tok_s") or summary.get("decode_tok_s")
    ar_toks = summary.get("mean_ar_tok_s") or summary.get("ar_tok_s")
    mult = summary.get("mean_speedup_vs_ar") or summary.get("speedup_vs_ar") or summary.get("speedup")
    target_rows = summary.get("target_distribution_materialized_rows")
    target_ms_per_row = summary.get("verify_target_distribution_ms_per_row")
    lazy_bonus_ms = summary.get("lazy_bonus_commit_ms_per_call")
    print(
        f"  D{d}: acceptance={acc} accepted={accepted} drafted={drafted} "
        f"tok_s={toks} decode_tok_s={decode} ar_tok_s={ar_toks} mult={mult} "
        f"target_rows={target_rows} target_ms_per_row={target_ms_per_row} "
        f"lazy_bonus_ms_per_call={lazy_bonus_ms}",
        flush=True,
    )
ar = res.get("ar") or res.get("ar_baseline") or {}
print("  AR baseline tok_s:", ar.get("tok_s") or ar.get("tokens_per_s") or _g(res, "ar_tok_s"), flush=True)
if isinstance(res.get("summary"), dict):
    summary = res["summary"]
    print(
        "  adaptive: "
        f"acceptance={summary.get('acceptance_by_depth')} "
        f"accepted={summary.get('accepted_by_depth')} "
        f"drafted={summary.get('drafted_by_depth')} "
        f"tok_s={summary.get('mean_tok_s')} "
        f"ar_tok_s={summary.get('mean_ar_tok_s')} "
        f"mult={summary.get('mean_speedup_vs_ar')}",
        flush=True,
    )
print("STEP_ACCEPTANCE_DONE", flush=True)
