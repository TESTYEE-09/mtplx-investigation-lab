#!/usr/bin/env python3
"""Train C4/FastMTP-style LoRA adapters inside the native MTP path.

C3 corrects hidden states after MTP has already produced them.  C4 changes the
proposer itself while keeping the target trunk frozen and the base model
untouched.  The adapter artifact is a sidecar NPZ loaded via ``--mtp-adapter``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

from mtplx.artifacts import expected_mtp_file, load_config  # noqa: E402
from mtplx.benchmarks.schema import now_run_id  # noqa: E402
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR  # noqa: E402
from mtplx.correctors import deterministic_train_mask, ensure_nonempty_split  # noqa: E402
from mtplx.mtp_adapters import (  # noqa: E402
    DEFAULT_C4_LORA_TARGETS,
    install_mtp_lora_adapters,
    install_saved_mtp_lora_adapter,
    iter_mtp_lora_modules,
    load_mtp_lora_adapter,
    save_mtp_lora_adapter,
)
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.runtime import load  # noqa: E402
from scripts.eval_mtp_corrector import (  # noqa: E402
    _build_persistent_replay_lookup,
    _persistent_replay_indices,
)


def _load_source_metadata(calib_npz: Path) -> dict[str, Any]:
    metadata_path = calib_npz.with_name("metadata.json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("at least one depth is required")
    if any(item < 1 for item in parsed):
        raise ValueError("depths must be >= 1")
    return sorted(set(parsed))


def _parse_target_csv(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("at least one adapter target is required")
    return parsed


def _default_adapter_targets_for_model(model_path: Path | str) -> list[str]:
    config = load_config(model_path)
    model_type = str(config.get("model_type") or "").lower()
    if model_type not in {"step3p5", "step3p7"}:
        return list(DEFAULT_C4_LORA_TARGETS)

    depth_count = int(config.get("num_nextn_predict_layers") or config.get("mtp_num_hidden_layers") or 3)
    targets: list[str] = []
    for depth_index in range(depth_count):
        prefix = f"layers.{depth_index}"
        targets.extend(
            [
                f"{prefix}.eh_proj",
                f"{prefix}.mtp_block.self_attn.q_proj",
                f"{prefix}.mtp_block.self_attn.k_proj",
                f"{prefix}.mtp_block.self_attn.v_proj",
                f"{prefix}.mtp_block.self_attn.o_proj",
                f"{prefix}.mtp_block.mlp.gate_proj",
                f"{prefix}.mtp_block.mlp.up_proj",
                f"{prefix}.mtp_block.mlp.down_proj",
                f"{prefix}.shared_head_head",
            ]
        )
        if bool(config.get("use_head_wise_attn_gate", False)):
            targets.append(f"{prefix}.mtp_block.self_attn.g_proj")
    return targets


def _model_identity_hash(model_path: Path | str) -> str:
    path = Path(model_path)
    h = hashlib.sha256()
    for rel in ("config.json", "tokenizer_config.json"):
        item = path / rel
        if item.exists():
            h.update(rel.encode("utf-8"))
            h.update(item.read_bytes())
    config = load_config(path)
    mtp_path = expected_mtp_file(path, config)
    if mtp_path.exists():
        stat = mtp_path.stat()
        h.update(str(mtp_path.resolve()).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    for item in sorted(path.glob("*.safetensors")):
        stat = item.stat()
        h.update(item.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return h.hexdigest()


def _nll(logits: mx.array, targets: mx.array) -> mx.array:
    target_logits = mx.take_along_axis(logits, targets[:, None], axis=1).reshape(-1)
    return mx.logsumexp(logits, axis=1) - target_logits


def _soft_topk_loss(
    logits: mx.array,
    target_top_indices: mx.array,
    target_top_probs: mx.array,
) -> mx.array:
    gathered = mx.take_along_axis(logits, target_top_indices, axis=1)
    log_denom = mx.logsumexp(logits, axis=1, keepdims=True)
    log_probs = gathered - log_denom
    weights = target_top_probs / mx.maximum(mx.sum(target_top_probs, axis=1, keepdims=True), 1e-12)
    return -mx.sum(weights * log_probs, axis=1)


def _target_margin_loss(logits: mx.array, targets: mx.array, margin: float) -> mx.array:
    target_logits = mx.take_along_axis(logits, targets[:, None], axis=1).reshape(-1)
    target_mask = mx.equal(mx.arange(logits.shape[1])[None, :], targets[:, None])
    best_other = mx.max(mx.where(target_mask, -mx.inf, logits), axis=1)
    return mx.maximum(0.0, float(margin) - (target_logits - best_other))


def _weighted_mean(values: mx.array, weights: mx.array) -> mx.array:
    weights = weights.astype(mx.float32)
    weights = weights / mx.maximum(mx.mean(weights), 1e-6)
    return mx.mean(values * weights)


def _greedy_acceptance_weights(
    target_p_recursive_draft: mx.array,
    recursive_matches: mx.array,
    *,
    power: float,
    min_weight: float,
    focus: str,
) -> mx.array:
    """Weight rows by live-greedy pain instead of offline top-k aesthetics."""
    p = mx.clip(target_p_recursive_draft.astype(mx.float32), 0.0, 1.0)
    low_p = mx.power(1.0 - p, float(power))
    mismatch = (1.0 - recursive_matches.astype(mx.float32))
    if focus == "low-p":
        raw = low_p
    elif focus == "mismatch":
        raw = mismatch
    elif focus == "low-p-or-mismatch":
        raw = mx.maximum(low_p, mismatch)
    else:
        raise ValueError(f"unknown greedy acceptance focus: {focus}")
    return mx.maximum(float(min_weight), raw)


def _acceptance_overlap_terms(
    logits: mx.array,
    target_top_indices: mx.array,
    target_top_probs: mx.array,
    *,
    temperature: float,
    outside_weight: float,
) -> tuple[mx.array, mx.array, mx.array]:
    scaled = logits / float(temperature)
    gathered = mx.take_along_axis(scaled, target_top_indices, axis=1)
    log_denom = mx.logsumexp(scaled, axis=1, keepdims=True)
    q_probs = mx.exp(gathered - log_denom)
    p_probs = target_top_probs.astype(mx.float32)
    overlap = mx.sum(mx.minimum(q_probs, p_probs), axis=1)
    q_top_mass = mx.sum(q_probs, axis=1)
    outside_q_mass = mx.maximum(0.0, 1.0 - q_top_mass)
    loss = -overlap + float(outside_weight) * outside_q_mass
    return loss, overlap, outside_q_mass


def _acceptance_overlap_loss(
    logits: mx.array,
    target_top_indices: mx.array,
    target_top_probs: mx.array,
    *,
    temperature: float,
    outside_weight: float,
) -> tuple[mx.array, mx.array, mx.array]:
    loss, overlap, outside_q_mass = _acceptance_overlap_terms(
        logits,
        target_top_indices,
        target_top_probs,
        temperature=temperature,
        outside_weight=outside_weight,
    )
    return mx.mean(loss), mx.mean(overlap), mx.mean(outside_q_mass)


def _find_token_offset(tokens: Sequence[int], token: Any, start: int) -> int | None:
    try:
        needle = int(token)
    except (TypeError, ValueError):
        return None
    for offset in range(max(0, int(start)), len(tokens)):
        if int(tokens[offset]) == needle:
            return offset
    return None


def _load_live_cycle_row_weights(
    paths: Sequence[Path | str],
    *,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    loss_weight: float,
    accepted_boost: float,
    rejected_boost: float,
    unverified_boost: float,
    repair_s_boost: float,
    max_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turn live speculative-cycle outcomes into per-calibration-row loss weights."""
    row_lookup = {
        (str(prompt_id), int(window_index), int(depth)): int(index)
        for index, (prompt_id, window_index, depth) in enumerate(
            zip(prompt_ids, window_indices, depths, strict=True)
        )
    }
    boosts = np.zeros(len(depths), dtype=np.float32)
    matched_by_depth: dict[str, int] = {}
    accepted_by_depth: dict[str, int] = {}
    rejected_by_depth: dict[str, int] = {}
    unverified_by_depth: dict[str, int] = {}
    summary: dict[str, Any] = {
        "enabled": bool(paths),
        "paths": [str(Path(path)) for path in paths],
        "loss_weight": float(loss_weight),
        "accepted_boost": float(accepted_boost),
        "rejected_boost": float(rejected_boost),
        "unverified_boost": float(unverified_boost),
        "repair_s_boost": float(repair_s_boost),
        "max_weight": float(max_weight),
        "live_rows": 0,
        "live_events": 0,
        "matched_rows": 0,
        "unmapped_rows": 0,
        "primary_alignment_misses": 0,
        "event_rows_without_tokens": 0,
    }

    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        depth_sections = payload.get("depths") or payload.get("results") or payload.get("runs") or []
        if isinstance(depth_sections, dict):
            depth_sections = list(depth_sections.values())
        if not isinstance(depth_sections, list):
            continue
        for depth_section in depth_sections:
            if not isinstance(depth_section, dict):
                continue
            rows = depth_section.get("rows") or []
            if not isinstance(rows, list):
                continue
            for live_row in rows:
                if not isinstance(live_row, dict):
                    continue
                prompt_id = str(live_row.get("prompt_id") or "")
                tokens = [int(token) for token in (live_row.get("tokens") or [])]
                events = live_row.get("events") or []
                if not tokens or not isinstance(events, list):
                    summary["event_rows_without_tokens"] = int(summary["event_rows_without_tokens"]) + 1
                    continue
                summary["live_rows"] = int(summary["live_rows"]) + 1
                offset = 0
                for event_index, event in enumerate(events):
                    if not isinstance(event, dict):
                        continue
                    summary["live_events"] = int(summary["live_events"]) + 1
                    primary = event.get("primary")
                    expected_offset = _find_token_offset(tokens, primary, offset)
                    if expected_offset is None:
                        summary["primary_alignment_misses"] = int(
                            summary["primary_alignment_misses"]
                        ) + 1
                        continue
                    if expected_offset != offset:
                        offset = expected_offset

                    timing = event.get("timing_s") if isinstance(event.get("timing_s"), dict) else {}
                    repair_s = float(timing.get("repair_forward") or 0.0)
                    for draft in event.get("drafts") or []:
                        if not isinstance(draft, dict):
                            continue
                        try:
                            depth = int(draft.get("depth"))
                        except (TypeError, ValueError):
                            continue
                        key = (prompt_id, int(offset), depth)
                        row_index = row_lookup.get(key)
                        if row_index is None:
                            summary["unmapped_rows"] = int(summary["unmapped_rows"]) + 1
                            continue
                        accepted = draft.get("accepted")
                        if accepted is True:
                            boost = float(accepted_boost)
                            accepted_by_depth[str(depth)] = accepted_by_depth.get(str(depth), 0) + 1
                        elif accepted is False:
                            boost = float(rejected_boost) + float(repair_s_boost) * repair_s
                            rejected_by_depth[str(depth)] = rejected_by_depth.get(str(depth), 0) + 1
                        else:
                            boost = float(unverified_boost)
                            unverified_by_depth[str(depth)] = unverified_by_depth.get(str(depth), 0) + 1
                        if boost <= 0.0:
                            continue
                        boosts[row_index] += np.float32(boost)
                        matched_by_depth[str(depth)] = matched_by_depth.get(str(depth), 0) + 1
                        summary["matched_rows"] = int(summary["matched_rows"]) + 1

                    if event_index + 1 < len(events):
                        next_primary = events[event_index + 1].get("primary")
                        aligned_next = _find_token_offset(tokens, next_primary, offset + 1)
                        if aligned_next is not None:
                            offset = aligned_next
                            continue

                    accepted_depths = int(event.get("accepted_depths") or 0)
                    offset = int(offset) + 1 + max(0, accepted_depths)
                    if event.get("bonus_token") is not None or event.get("rejected_at_depth") is not None:
                        offset += 1

    weights = 1.0 + float(loss_weight) * boosts
    if max_weight > 0:
        weights = np.minimum(weights, float(max_weight))
    summary["weighted_rows"] = int(np.count_nonzero(weights > 1.0))
    summary["mean_weight"] = float(np.mean(weights)) if len(weights) else 1.0
    summary["max_observed_weight"] = float(np.max(weights)) if len(weights) else 1.0
    summary["matched_by_depth"] = dict(sorted(matched_by_depth.items()))
    summary["accepted_by_depth"] = dict(sorted(accepted_by_depth.items()))
    summary["rejected_by_depth"] = dict(sorted(rejected_by_depth.items()))
    summary["unverified_by_depth"] = dict(sorted(unverified_by_depth.items()))
    return weights.astype(np.float32), summary


def _iter_batches(rows: np.ndarray, *, batch_size: int, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(rows) == 0:
        raise ValueError("cannot iterate empty rows")
    for _ in range(steps):
        if len(rows) >= batch_size:
            yield rng.choice(rows, size=batch_size, replace=False)
        else:
            yield rng.choice(rows, size=batch_size, replace=True)


def _draft_logits_for_rows(
    rt,
    row_indices: np.ndarray,
    *,
    input_hidden: np.ndarray,
    input_tokens: np.ndarray,
    mtp_hidden_variant: str,
    cache_policy: str,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    row_lookup: dict[tuple[str, int, int], int] | None,
    return_hidden: bool = False,
    depth_gated: bool = False,
    persistent_replay_scope: str = "through-current",
) -> tuple[mx.array, mx.array | None]:
    if cache_policy == "fresh" and not depth_gated:
        result = rt.draft_mtp(
            mx.array(input_hidden[row_indices, None, :].astype(np.float32)),
            mx.array(input_tokens[row_indices, None].astype(np.int64)),
            mtp_cache=rt.make_mtp_cache(),
            return_hidden=return_hidden,
            mtp_hidden_variant=mtp_hidden_variant,
        )
        if return_hidden:
            logits, hidden = result
            return logits[:, -1, :].astype(mx.float32), hidden[:, -1, :].astype(mx.float32)
        return result[:, -1, :].astype(mx.float32), None

    logits_rows: list[mx.array] = []
    hidden_rows: list[mx.array] = []
    for row_index in row_indices:
        mtp_cache = rt.make_mtp_cache()
        if cache_policy == "persistent":
            if row_lookup is None:
                raise ValueError("persistent cache training requires row_lookup")
            for replay_index in _persistent_replay_indices(
                int(row_index),
                prompt_ids=prompt_ids,
                window_indices=window_indices,
                depths=depths,
                row_lookup=row_lookup,
                include_current=persistent_replay_scope == "through-current",
            ):
                replay = rt.draft_mtp(
                    mx.array(input_hidden[replay_index : replay_index + 1, None, :].astype(np.float32)),
                    mx.array([[int(input_tokens[replay_index])]]),
                    mtp_cache=mtp_cache,
                    return_hidden=False,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_depth=int(depths[replay_index]),
                )
                mx.eval(replay)
        else:
            mtp_cache = rt.make_mtp_cache()
        result = rt.draft_mtp(
            mx.array(input_hidden[int(row_index) : int(row_index) + 1, None, :].astype(np.float32)),
            mx.array([[int(input_tokens[int(row_index)])]]),
            mtp_cache=mtp_cache,
            return_hidden=return_hidden,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=int(depths[int(row_index)]),
        )
        if return_hidden:
            logits, hidden = result
            logits_rows.append(logits[:, -1, :].astype(mx.float32))
            hidden_rows.append(hidden[:, -1, :].astype(mx.float32))
        else:
            logits_rows.append(result[:, -1, :].astype(mx.float32))
    logits = mx.concatenate(logits_rows, axis=0)
    hidden = mx.concatenate(hidden_rows, axis=0) if return_hidden else None
    return logits, hidden


def _rollout_d2_logits_for_rows(
    rt,
    row_indices: np.ndarray,
    *,
    input_hidden: np.ndarray,
    input_tokens: np.ndarray,
    mtp_hidden_variant: str,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    row_lookup: dict[tuple[str, int, int], int] | None,
) -> tuple[mx.array | None, np.ndarray]:
    """Draft D2 from the current adapter's D1 hidden/token, not stored D2 rows."""
    if row_lookup is None:
        raise ValueError("rollout D2 training requires row_lookup")
    logits_rows: list[mx.array] = []
    target_rows: list[int] = []
    for row_index in row_indices:
        row_index = int(row_index)
        if int(depths[row_index]) != 2:
            continue
        key = (str(prompt_ids[row_index]), int(window_indices[row_index]), 1)
        d1_index = row_lookup.get(key)
        if d1_index is None:
            continue
        mtp_cache = rt.make_mtp_cache()
        d1_logits, d1_hidden = rt.draft_mtp(
            mx.array(input_hidden[d1_index : d1_index + 1, None, :].astype(np.float32)),
            mx.array([[int(input_tokens[d1_index])]]),
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=1,
        )
        # D2 calibration rows already store the sampled D1 token as their input.
        # Using argmax here makes offline rollout look cleaner than the live
        # stochastic sampler and hides the recursive acceptance failure.
        d1_token = mx.array([int(input_tokens[row_index])], dtype=mx.int64)
        d2_logits = rt.draft_mtp(
            d1_hidden[:, -1:, :],
            d1_token.reshape(1, 1),
            mtp_cache=mtp_cache,
            return_hidden=False,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=2,
        )
        logits_rows.append(d2_logits[:, -1, :].astype(mx.float32))
        target_rows.append(row_index)
    if not logits_rows:
        return None, np.asarray([], dtype=np.int64)
    return mx.concatenate(logits_rows, axis=0), np.asarray(target_rows, dtype=np.int64)


def _topk_metrics(
    logits_np: np.ndarray,
    targets: np.ndarray,
    *,
    target_top_probs: np.ndarray,
) -> dict[str, float]:
    order = np.argpartition(-logits_np, kth=min(7, logits_np.shape[1] - 1), axis=1)[:, :8]
    rows = np.arange(len(targets))[:, None]
    local_values = logits_np[rows, order]
    sorted_local = np.argsort(-local_values, axis=1)
    top = np.take_along_axis(order, sorted_local, axis=1)
    top1 = top[:, 0] == targets
    return {
        "rows": int(len(targets)),
        "top1": float(np.mean(top1)) if len(targets) else 0.0,
        "top4": float(np.mean(np.any(top[:, :4] == targets[:, None], axis=1))) if len(targets) else 0.0,
        "top8": float(np.mean(np.any(top[:, :8] == targets[:, None], axis=1))) if len(targets) else 0.0,
        "mean_target_top_prob_mass": float(np.mean(np.sum(target_top_probs[:, :8], axis=1))) if len(targets) else 0.0,
    }


def _target_top_overlap_metrics(
    logits_np: np.ndarray,
    target_top_indices: np.ndarray,
    target_top_probs: np.ndarray,
    *,
    temperature: float,
) -> dict[str, float]:
    if len(logits_np) == 0:
        return {
            "mean_target_top_q_mass": 0.0,
            "mean_target_top_overlap_mass": 0.0,
            "mean_target_top_excess_q_mass": 0.0,
        }
    scaled = logits_np.astype(np.float64) / float(temperature)
    max_values = np.max(scaled, axis=1, keepdims=True)
    log_denoms = max_values + np.log(
        np.sum(np.exp(scaled - max_values), axis=1, keepdims=True)
    )
    rows = np.arange(len(logits_np))[:, None]
    q_probs = np.exp(scaled[rows, target_top_indices] - log_denoms)
    p_probs = target_top_probs.astype(np.float64)
    return {
        "mean_target_top_q_mass": float(np.mean(np.sum(q_probs, axis=1))),
        "mean_target_top_overlap_mass": float(np.mean(np.sum(np.minimum(q_probs, p_probs), axis=1))),
        "mean_target_top_excess_q_mass": float(np.mean(np.sum(np.maximum(q_probs - p_probs, 0.0), axis=1))),
    }


def _greedy_acceptance_lb_for_tokens(
    draft_tokens: np.ndarray,
    target_top_indices: np.ndarray,
    target_top_probs: np.ndarray,
) -> np.ndarray:
    """Recorded-top-k lower bound for exact greedy-q acceptance.

    In the temp0.6 live path the draft proposal is one-hot, so exact
    speculative acceptance is the target probability of that draft token.  The
    calibration shard records target probabilities only for target top-k; tokens
    outside that set receive a conservative lower bound of zero.
    """
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).reshape(-1)
    indices = np.asarray(target_top_indices, dtype=np.int64)
    probs = np.asarray(target_top_probs, dtype=np.float64)
    matches = indices == draft_tokens[:, None]
    return np.sum(np.where(matches, probs, 0.0), axis=1)


def _greedy_acceptance_lb_metrics(
    logits_np: np.ndarray,
    target_top_indices: np.ndarray,
    target_top_probs: np.ndarray,
) -> dict[str, float]:
    if len(logits_np) == 0:
        return {
            "mean_greedy_acceptance_lb": 0.0,
            "greedy_target_topk_hit_rate": 0.0,
        }
    draft_tokens = np.argmax(logits_np, axis=1).astype(np.int64)
    lbs = _greedy_acceptance_lb_for_tokens(
        draft_tokens,
        target_top_indices,
        target_top_probs,
    )
    return {
        "mean_greedy_acceptance_lb": float(np.mean(lbs)),
        "greedy_target_topk_hit_rate": float(np.mean(lbs > 0.0)),
    }


def _evaluate_adapter(
    rt,
    rows: np.ndarray,
    *,
    input_hidden: np.ndarray,
    input_tokens: np.ndarray,
    target_tokens: np.ndarray,
    target_top_indices: np.ndarray,
    target_top_probs: np.ndarray,
    mtp_hidden_variant: str,
    cache_policy: str,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    row_lookup: dict[tuple[str, int, int], int] | None,
    batch_size: int,
    depth_gated: bool,
    sampler_temperature: float,
    persistent_replay_scope: str,
) -> dict[str, Any]:
    by_depth: dict[str, dict[str, float]] = {}
    all_logits: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        logits, _hidden = _draft_logits_for_rows(
            rt,
            batch,
            input_hidden=input_hidden,
            input_tokens=input_tokens,
            mtp_hidden_variant=mtp_hidden_variant,
            cache_policy=cache_policy,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths,
            row_lookup=row_lookup,
            return_hidden=False,
            depth_gated=depth_gated,
            persistent_replay_scope=persistent_replay_scope,
        )
        mx.eval(logits)
        all_logits.append(np.asarray(logits, dtype=np.float32))
        all_rows.append(batch)
    if not all_logits:
        return {"rows": 0, "by_depth": {}}
    logits_np = np.concatenate(all_logits, axis=0)
    row_np = np.concatenate(all_rows, axis=0)
    overall = _topk_metrics(
        logits_np,
        target_tokens[row_np],
        target_top_probs=target_top_probs[row_np],
    )
    overall.update(
        _target_top_overlap_metrics(
            logits_np,
            target_top_indices[row_np],
            target_top_probs[row_np],
            temperature=sampler_temperature,
        )
    )
    overall.update(
        _greedy_acceptance_lb_metrics(
            logits_np,
            target_top_indices[row_np],
            target_top_probs[row_np],
        )
    )
    for depth in sorted(set(int(depths[row]) for row in row_np)):
        mask = depths[row_np] == depth
        local = _topk_metrics(
            logits_np[mask],
            target_tokens[row_np][mask],
            target_top_probs=target_top_probs[row_np][mask],
        )
        local.update(
            _target_top_overlap_metrics(
                logits_np[mask],
                target_top_indices[row_np][mask],
                target_top_probs[row_np][mask],
                temperature=sampler_temperature,
            )
        )
        local.update(
            _greedy_acceptance_lb_metrics(
                logits_np[mask],
                target_top_indices[row_np][mask],
                target_top_probs[row_np][mask],
            )
        )
        by_depth[str(depth)] = local
    return {**overall, "by_depth": by_depth}


def _evaluate_rollout_d2_adapter(
    rt,
    rows: np.ndarray,
    *,
    input_hidden: np.ndarray,
    input_tokens: np.ndarray,
    target_tokens: np.ndarray,
    target_top_indices: np.ndarray,
    target_top_probs: np.ndarray,
    mtp_hidden_variant: str,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    row_lookup: dict[tuple[str, int, int], int] | None,
    batch_size: int,
    sampler_temperature: float,
) -> dict[str, Any]:
    if row_lookup is None:
        return {
            "rows": 0,
            "requested_rows": 0,
            "dropped_rows": 0,
            "reason": "requires_persistent_replay",
        }
    rows = np.asarray(rows, dtype=np.int64)
    d2_rows = rows[depths[rows] == 2]
    if len(d2_rows) == 0:
        return {"rows": 0, "requested_rows": 0, "dropped_rows": 0}

    all_logits: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    for start in range(0, len(d2_rows), batch_size):
        batch = d2_rows[start : start + batch_size]
        logits, rollout_rows = _rollout_d2_logits_for_rows(
            rt,
            batch,
            input_hidden=input_hidden,
            input_tokens=input_tokens,
            mtp_hidden_variant=mtp_hidden_variant,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths,
            row_lookup=row_lookup,
        )
        if logits is None or len(rollout_rows) == 0:
            continue
        mx.eval(logits)
        all_logits.append(np.asarray(logits, dtype=np.float32))
        all_rows.append(rollout_rows)

    if not all_logits:
        return {
            "rows": 0,
            "requested_rows": int(len(d2_rows)),
            "dropped_rows": int(len(d2_rows)),
        }
    logits_np = np.concatenate(all_logits, axis=0)
    row_np = np.concatenate(all_rows, axis=0)
    metrics = _topk_metrics(
        logits_np,
        target_tokens[row_np],
        target_top_probs=target_top_probs[row_np],
    )
    metrics.update(
        _target_top_overlap_metrics(
            logits_np,
            target_top_indices[row_np],
            target_top_probs[row_np],
            temperature=sampler_temperature,
        )
    )
    metrics.update(
        _greedy_acceptance_lb_metrics(
            logits_np,
            target_top_indices[row_np],
            target_top_probs[row_np],
        )
    )
    metrics["requested_rows"] = int(len(d2_rows))
    metrics["dropped_rows"] = int(len(d2_rows) - len(row_np))
    metrics["by_depth"] = {"2": dict(metrics)}
    metrics["by_depth"]["2"].pop("by_depth", None)
    return metrics


def _baseline_metrics(
    rows: np.ndarray,
    *,
    recursive_top_indices: np.ndarray,
    target_top_indices: np.ndarray,
    target_tokens: np.ndarray,
    target_top_probs: np.ndarray,
    depths: np.ndarray,
) -> dict[str, Any]:
    top = recursive_top_indices[rows, :8]
    targets = target_tokens[rows]
    greedy_acceptance_lb = _greedy_acceptance_lb_for_tokens(
        recursive_top_indices[rows, 0],
        target_top_indices[rows],
        target_top_probs[rows],
    )
    overall = {
        "rows": int(len(rows)),
        "top1": float(np.mean(top[:, 0] == targets)) if len(rows) else 0.0,
        "top4": float(np.mean(np.any(top[:, :4] == targets[:, None], axis=1))) if len(rows) else 0.0,
        "top8": float(np.mean(np.any(top[:, :8] == targets[:, None], axis=1))) if len(rows) else 0.0,
        "mean_target_top_prob_mass": float(np.mean(np.sum(target_top_probs[rows, :8], axis=1))) if len(rows) else 0.0,
        "mean_greedy_acceptance_lb": float(np.mean(greedy_acceptance_lb)) if len(rows) else 0.0,
        "greedy_target_topk_hit_rate": float(np.mean(greedy_acceptance_lb > 0.0)) if len(rows) else 0.0,
    }
    by_depth: dict[str, Any] = {}
    for depth in sorted(set(int(depths[row]) for row in rows)):
        local = rows[depths[rows] == depth]
        local_top = recursive_top_indices[local, :8]
        local_targets = target_tokens[local]
        local_greedy_acceptance_lb = _greedy_acceptance_lb_for_tokens(
            recursive_top_indices[local, 0],
            target_top_indices[local],
            target_top_probs[local],
        )
        by_depth[str(depth)] = {
            "rows": int(len(local)),
            "top1": float(np.mean(local_top[:, 0] == local_targets)) if len(local) else 0.0,
            "top4": float(np.mean(np.any(local_top[:, :4] == local_targets[:, None], axis=1))) if len(local) else 0.0,
            "top8": float(np.mean(np.any(local_top[:, :8] == local_targets[:, None], axis=1))) if len(local) else 0.0,
            "mean_target_top_prob_mass": float(np.mean(np.sum(target_top_probs[local, :8], axis=1))) if len(local) else 0.0,
            "mean_greedy_acceptance_lb": float(np.mean(local_greedy_acceptance_lb)) if len(local) else 0.0,
            "greedy_target_topk_hit_rate": float(np.mean(local_greedy_acceptance_lb > 0.0)) if len(local) else 0.0,
        }
    return {**overall, "by_depth": by_depth}


def _greedy_lb(metrics: dict[str, Any] | None) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    return float(metrics.get("mean_greedy_acceptance_lb") or 0.0)


def _live_cycle_gate_reasons(
    live_eval: dict[str, Any] | None,
    *,
    tolerance: float = 1e-6,
) -> dict[str, bool | None]:
    if not isinstance(live_eval, dict):
        return {
            "live_cycle_baseline_improved": None,
            "live_cycle_pretrain_preserved": None,
        }
    adapter_lb = _greedy_lb(live_eval.get("c4_adapter"))
    baseline_lb = _greedy_lb(live_eval.get("baseline"))
    pretrain = live_eval.get("pretrain_adapter")
    pretrain_lb = _greedy_lb(pretrain) if isinstance(pretrain, dict) else None
    return {
        "live_cycle_baseline_improved": bool(adapter_lb > baseline_lb + tolerance),
        "live_cycle_pretrain_preserved": (
            bool(adapter_lb + tolerance >= pretrain_lb)
            if pretrain_lb is not None
            else None
        ),
    }


def train_mtp_adapter_c4(
    calib_npz: Path | str,
    *,
    model_path: Path | str | None = None,
    artifact_dir: Path | str = Path("outputs/adapters"),
    report_dir: Path | str = Path("outputs/reports"),
    train_fraction: float = 0.75,
    depths: list[int] | None = None,
    rank: int = 16,
    alpha: float | None = None,
    targets: list[str] | None = None,
    steps: int = 80,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    seed: int = 0,
    ce_weight: float = 1.0,
    soft_kl_weight: float = 0.25,
    margin_weight: float = 0.1,
    margin: float = 1.0,
    acceptance_overlap_weight: float = 0.0,
    acceptance_outside_weight: float = 0.25,
    greedy_acceptance_ce_weight: float = 0.0,
    greedy_acceptance_margin_weight: float = 0.0,
    greedy_acceptance_power: float = 1.0,
    greedy_acceptance_min_weight: float = 0.05,
    greedy_acceptance_focus: str = "low-p-or-mismatch",
    hidden_mse_weight: float = 0.0,
    rollout_d2_weight: float = 0.0,
    rollout_d2_acceptance_overlap_weight: float = 1.0,
    live_cycle_json: Sequence[Path | str] | None = None,
    live_cycle_loss_weight: float = 0.0,
    live_cycle_accepted_boost: float = 0.5,
    live_cycle_rejected_boost: float = 2.0,
    live_cycle_unverified_boost: float = 0.0,
    live_cycle_repair_s_boost: float = 10.0,
    live_cycle_max_weight: float = 6.0,
    eval_batch_size: int = 2,
    depth_gate: str = "all",
    persistent_replay_scope: str = "through-current",
    init_mtp_adapter: Path | str | None = None,
    context_mtp_adapter: Path | str | None = None,
    mtp_quant_bits: int | None = None,
    mtp_quant_group_size: int = 64,
    mtp_quant_mode: str = "affine",
) -> dict[str, Any]:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    calib_npz = Path(calib_npz)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = _load_source_metadata(calib_npz)

    with np.load(calib_npz, allow_pickle=False) as data:
        recursive_input_hidden = np.asarray(data["recursive_input_hidden"], dtype=np.float32)
        recursive_input_tokens = np.asarray(data["recursive_input_tokens"], dtype=np.int64)
        target_next_hidden = np.asarray(data["target_next_hidden"], dtype=np.float32)
        depths_arr = np.asarray(data["depths"], dtype=np.int64)
        prompt_ids = np.asarray(data["prompt_ids"])
        window_indices = np.asarray(data["window_indices"], dtype=np.int64)
        target_tokens = np.asarray(data["target_tokens"], dtype=np.int64)
        recursive_matches = np.asarray(data["recursive_matches"], dtype=bool)
        target_ar_p_recursive_draft = np.asarray(data["target_ar_p_recursive_draft"], dtype=np.float32)
        recursive_prefix_active = np.asarray(data["recursive_prefix_active"], dtype=bool)
        recursive_top_indices = np.asarray(data["recursive_top_indices"], dtype=np.int64)
        target_ar_top_indices = np.asarray(data["target_ar_top_indices"], dtype=np.int64)
        target_ar_top_probs = np.asarray(data["target_ar_top_probs"], dtype=np.float32)

    depth_count = int(np.max(depths_arr))
    train_depths = depths or [1, 2]
    if any(depth > depth_count for depth in train_depths):
        raise ValueError(f"depths must be <= {depth_count}")
    depth_mask = np.isin(depths_arr, np.asarray(train_depths, dtype=np.int64))
    scope_mask = recursive_prefix_active & depth_mask
    split_train_mask = deterministic_train_mask(
        prompt_ids,
        window_indices,
        train_fraction=train_fraction,
    )
    split_train_mask = ensure_nonempty_split(split_train_mask, prompt_ids, window_indices)
    train_mask = scope_mask & split_train_mask
    heldout_mask = scope_mask & (~split_train_mask)
    if not train_mask.any() or not heldout_mask.any():
        raise ValueError("C4 adapter split produced empty train or held-out rows")

    model_path = Path(model_path or source_metadata.get("model_path") or DEFAULT_RUNTIME_MODEL_DIR)
    hidden_variant = str(source_metadata.get("mtp_hidden_variant", "pre_norm"))
    cache_policy = str(source_metadata.get("cache_policy", "fresh"))
    sampler_temperature = float(source_metadata.get("sampler_temperature", 0.6))
    if cache_policy not in {"fresh", "persistent"}:
        raise ValueError("cache_policy must be 'fresh' or 'persistent'")
    if cache_policy == "persistent" and "recursive_input_hidden" not in source_metadata.get(
        "persistent_replay_fields",
        ["recursive_input_hidden"],
    ):
        raise ValueError("persistent-cache C4 training requires schema v2 replay fields")
    row_lookup = (
        _build_persistent_replay_lookup(prompt_ids, window_indices, depths_arr)
        if cache_policy == "persistent"
        else None
    )
    init_adapter_metadata: dict[str, Any] | None = None
    init_targets: list[str] = []
    if init_mtp_adapter is not None:
        init_state = load_mtp_lora_adapter(init_mtp_adapter)
        init_targets = [
            str(entry["target"])
            for entry in init_state.metadata.get("targets", [])
        ]
    adapter_targets = targets or init_targets or _default_adapter_targets_for_model(model_path)
    context_adapter_metadata: dict[str, Any] | None = None
    context_targets: list[str] = []
    if context_mtp_adapter is not None:
        context_state = load_mtp_lora_adapter(context_mtp_adapter)
        context_targets = sorted(
            str(entry["target"])
            for entry in context_state.metadata.get("targets", [])
        )
        overlap = sorted(set(context_targets) & set(adapter_targets))
        if overlap:
            raise ValueError(
                "context adapter targets overlap trainable targets: "
                + ", ".join(overlap)
            )
    depth_scales = None
    if depth_gate not in {"all", "train-depths", "d2plus"}:
        raise ValueError("depth_gate must be 'all', 'train-depths', or 'd2plus'")
    if persistent_replay_scope not in {"through-current", "before-current"}:
        raise ValueError(
            "persistent_replay_scope must be 'through-current' or 'before-current'"
        )
    if greedy_acceptance_focus not in {"low-p", "mismatch", "low-p-or-mismatch"}:
        raise ValueError(
            "greedy_acceptance_focus must be 'low-p', 'mismatch', or 'low-p-or-mismatch'"
        )
    live_cycle_row_weights = np.ones(len(depths_arr), dtype=np.float32)
    live_cycle_summary: dict[str, Any] = {"enabled": False}
    if live_cycle_json:
        live_cycle_row_weights, live_cycle_summary = _load_live_cycle_row_weights(
            list(live_cycle_json),
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths_arr,
            loss_weight=live_cycle_loss_weight,
            accepted_boost=live_cycle_accepted_boost,
            rejected_boost=live_cycle_rejected_boost,
            unverified_boost=live_cycle_unverified_boost,
            repair_s_boost=live_cycle_repair_s_boost,
            max_weight=live_cycle_max_weight,
        )
    if depth_gate == "train-depths":
        depth_scales = [1.0 if depth in train_depths else 0.0 for depth in range(1, depth_count + 1)]
    elif depth_gate == "d2plus":
        depth_scales = [0.0 if depth == 1 else 1.0 for depth in range(1, depth_count + 1)]

    mx.random.seed(seed)
    rt = load(
        model_path,
        mtp=True,
        contract=MTPContract(
            mtp_quant_bits=mtp_quant_bits,
            mtp_quant_group_size=mtp_quant_group_size,
            mtp_quant_mode=mtp_quant_mode,
        ),
    )
    if context_mtp_adapter is not None:
        context_adapter_metadata = install_saved_mtp_lora_adapter(
            rt.model,
            context_mtp_adapter,
            trainable=False,
        )
    if init_mtp_adapter is not None:
        init_adapter_metadata = install_saved_mtp_lora_adapter(
            rt.model,
            init_mtp_adapter,
            trainable=True,
        )
    installed_targets = install_mtp_lora_adapters(
        rt.model,
        rank=rank,
        alpha=alpha,
        targets=adapter_targets,
        depth_scales=depth_scales,
        trainable=True,
    )
    frozen_context_targets: list[str] = []
    if context_targets:
        context_target_set = set(context_targets)
        for target, module in iter_mtp_lora_modules(rt.model):
            if target in context_target_set:
                module.freeze()
                module.base.freeze()
                frozen_context_targets.append(target)
    optimizer = optim.Adam(learning_rate=learning_rate)
    train_rows = np.flatnonzero(train_mask)
    heldout_rows = np.flatnonzero(heldout_mask)
    live_cycle_rows = np.flatnonzero((live_cycle_row_weights > 1.0) & scope_mask)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    print(
        "C4_TRAIN_START "
        f"calib={calib_npz} depths={train_depths} "
        f"scope_rows={int(scope_mask.sum())} train_rows={int(train_mask.sum())} "
        f"heldout_rows={int(heldout_mask.sum())} steps={steps} "
        f"batch_size={batch_size} targets={len(adapter_targets)} "
        f"init_adapter={init_mtp_adapter} "
        f"context_adapter={context_mtp_adapter} "
        f"cache_policy={cache_policy} "
        f"mtp_quant_bits={rt.contract.mtp_quant_bits} "
        f"persistent_replay_scope={persistent_replay_scope} "
        f"rollout_d2_weight={rollout_d2_weight} "
        f"live_cycle_weighted_rows={live_cycle_summary.get('weighted_rows', 0)}",
        flush=True,
    )
    pretrain_live_cycle_adapter: dict[str, Any] | None = None
    if len(live_cycle_rows) and init_mtp_adapter is not None:
        print(
            "C4_PRETRAIN_LIVE_CYCLE_EVAL_START "
            f"rows={len(live_cycle_rows)} init_adapter={init_mtp_adapter}",
            flush=True,
        )
        pretrain_live_cycle_adapter = _evaluate_adapter(
            rt,
            live_cycle_rows,
            input_hidden=recursive_input_hidden,
            input_tokens=recursive_input_tokens,
            target_tokens=target_tokens,
            target_top_indices=target_ar_top_indices,
            target_top_probs=target_ar_top_probs,
            mtp_hidden_variant=hidden_variant,
            cache_policy=cache_policy,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths_arr,
            row_lookup=row_lookup,
            batch_size=eval_batch_size,
            depth_gated=depth_gate != "all",
            sampler_temperature=sampler_temperature,
            persistent_replay_scope=persistent_replay_scope,
        )
        print(
            "C4_PRETRAIN_LIVE_CYCLE_EVAL_DONE "
            f"rows={int(pretrain_live_cycle_adapter.get('rows') or 0)} "
            "greedy_lb="
            f"{pretrain_live_cycle_adapter.get('mean_greedy_acceptance_lb', 0.0):.4f}",
            flush=True,
        )

    def loss_fn(model, batch_rows_np: np.ndarray):
        logits, hidden = _draft_logits_for_rows(
            rt,
            batch_rows_np,
            input_hidden=recursive_input_hidden,
            input_tokens=recursive_input_tokens,
            mtp_hidden_variant=hidden_variant,
            cache_policy=cache_policy,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths_arr,
            row_lookup=row_lookup,
            return_hidden=hidden_mse_weight > 0.0,
            depth_gated=depth_gate != "all",
            persistent_replay_scope=persistent_replay_scope,
        )
        targets_mx = mx.array(target_tokens[batch_rows_np])
        target_top = mx.array(target_ar_top_indices[batch_rows_np, :16])
        target_probs = mx.array(target_ar_top_probs[batch_rows_np, :16])
        per_row_nll = _nll(logits, targets_mx)
        row_loss_weights = mx.array(live_cycle_row_weights[batch_rows_np])
        ce = _weighted_mean(per_row_nll, row_loss_weights)
        soft_kl = _weighted_mean(
            _soft_topk_loss(logits, target_top, target_probs),
            row_loss_weights,
        )
        per_row_margin = _target_margin_loss(logits, targets_mx, margin)
        margin_loss = _weighted_mean(per_row_margin, row_loss_weights)
        greedy_weights = _greedy_acceptance_weights(
            mx.array(target_ar_p_recursive_draft[batch_rows_np]),
            mx.array(recursive_matches[batch_rows_np]),
            power=greedy_acceptance_power,
            min_weight=greedy_acceptance_min_weight,
            focus=greedy_acceptance_focus,
        )
        greedy_weighted_ce = _weighted_mean(per_row_nll, greedy_weights)
        greedy_weighted_margin = _weighted_mean(per_row_margin, greedy_weights)
        (
            acceptance_loss_rows,
            acceptance_overlap_rows,
            acceptance_outside_q_rows,
        ) = _acceptance_overlap_terms(
            logits,
            target_top,
            target_probs,
            temperature=sampler_temperature,
            outside_weight=acceptance_outside_weight,
        )
        acceptance_loss = _weighted_mean(acceptance_loss_rows, row_loss_weights)
        acceptance_overlap = mx.mean(acceptance_overlap_rows)
        acceptance_outside_q = mx.mean(acceptance_outside_q_rows)
        hidden_mse = mx.array(0.0, dtype=mx.float32)
        if hidden_mse_weight > 0.0 and hidden is not None:
            target_hidden = mx.array(target_next_hidden[batch_rows_np])
            diff = hidden - target_hidden
            hidden_mse = _weighted_mean(mx.mean(diff * diff, axis=1), row_loss_weights)
        rollout_d2_ce = mx.array(0.0, dtype=mx.float32)
        rollout_d2_soft_kl = mx.array(0.0, dtype=mx.float32)
        rollout_d2_acceptance_loss = mx.array(0.0, dtype=mx.float32)
        rollout_d2_acceptance_overlap = mx.array(0.0, dtype=mx.float32)
        rollout_d2_rows = mx.array(0.0, dtype=mx.float32)
        if rollout_d2_weight > 0.0:
            rollout_logits, rollout_rows_np = _rollout_d2_logits_for_rows(
                rt,
                batch_rows_np,
                input_hidden=recursive_input_hidden,
                input_tokens=recursive_input_tokens,
                mtp_hidden_variant=hidden_variant,
                prompt_ids=prompt_ids,
                window_indices=window_indices,
                depths=depths_arr,
                row_lookup=row_lookup,
            )
            if rollout_logits is not None and len(rollout_rows_np):
                rollout_targets = mx.array(target_tokens[rollout_rows_np])
                rollout_target_top = mx.array(target_ar_top_indices[rollout_rows_np, :16])
                rollout_target_probs = mx.array(target_ar_top_probs[rollout_rows_np, :16])
                rollout_row_weights = mx.array(live_cycle_row_weights[rollout_rows_np])
                rollout_d2_ce = _weighted_mean(
                    _nll(rollout_logits, rollout_targets),
                    rollout_row_weights,
                )
                rollout_d2_soft_kl = _weighted_mean(
                    _soft_topk_loss(
                        rollout_logits,
                        rollout_target_top,
                        rollout_target_probs,
                    ),
                    rollout_row_weights,
                )
                (
                    rollout_d2_acceptance_loss_rows,
                    rollout_d2_acceptance_overlap_rows,
                    _rollout_outside_q_rows,
                ) = _acceptance_overlap_terms(
                    rollout_logits,
                    rollout_target_top,
                    rollout_target_probs,
                    temperature=sampler_temperature,
                    outside_weight=acceptance_outside_weight,
                )
                rollout_d2_acceptance_loss = _weighted_mean(
                    rollout_d2_acceptance_loss_rows,
                    rollout_row_weights,
                )
                rollout_d2_acceptance_overlap = mx.mean(rollout_d2_acceptance_overlap_rows)
                rollout_d2_rows = mx.array(float(len(rollout_rows_np)), dtype=mx.float32)
        rollout_d2_loss = (
            rollout_d2_ce
            + float(soft_kl_weight) * rollout_d2_soft_kl
            + float(rollout_d2_acceptance_overlap_weight) * rollout_d2_acceptance_loss
        )
        loss = (
            float(ce_weight) * ce
            + float(soft_kl_weight) * soft_kl
            + float(margin_weight) * margin_loss
            + float(acceptance_overlap_weight) * acceptance_loss
            + float(greedy_acceptance_ce_weight) * greedy_weighted_ce
            + float(greedy_acceptance_margin_weight) * greedy_weighted_margin
            + float(hidden_mse_weight) * hidden_mse
            + float(rollout_d2_weight) * rollout_d2_loss
        )
        return loss, (
            ce,
            soft_kl,
            margin_loss,
            acceptance_loss,
            acceptance_overlap,
            acceptance_outside_q,
            greedy_weighted_ce,
            greedy_weighted_margin,
            mx.mean(greedy_weights),
            mx.mean(row_loss_weights),
            hidden_mse,
            rollout_d2_ce,
            rollout_d2_soft_kl,
            rollout_d2_acceptance_overlap,
            rollout_d2_rows,
        )

    grad_fn = nn.value_and_grad(rt.model, loss_fn)
    for step_index, batch_rows in enumerate(
        _iter_batches(train_rows, batch_size=batch_size, steps=steps, seed=seed),
        start=1,
    ):
        (loss, parts), grads = grad_fn(rt.model, batch_rows)
        optimizer.update(rt.model, grads)
        if step_index == 1 or step_index == steps or step_index % max(1, steps // 5) == 0:
            mx.eval(loss, *parts, rt.model.trainable_parameters(), optimizer.state)
            (
                ce,
                soft_kl,
                margin_loss,
                acceptance_loss,
                acceptance_overlap,
                acceptance_outside_q,
                greedy_weighted_ce,
                greedy_weighted_margin,
                greedy_weight_mean,
                live_cycle_weight_mean,
                hidden_mse,
                rollout_d2_ce,
                rollout_d2_soft_kl,
                rollout_d2_acceptance_overlap,
                rollout_d2_rows,
            ) = parts
            history_row = {
                "step": float(step_index),
                "loss": float(loss.item()),
                "ce": float(ce.item()),
                "soft_topk_ce": float(soft_kl.item()),
                "margin": float(margin_loss.item()),
                "acceptance_overlap_loss": float(acceptance_loss.item()),
                "acceptance_overlap": float(acceptance_overlap.item()),
                "acceptance_outside_q_mass": float(acceptance_outside_q.item()),
                "greedy_acceptance_weighted_ce": float(greedy_weighted_ce.item()),
                "greedy_acceptance_weighted_margin": float(greedy_weighted_margin.item()),
                "greedy_acceptance_weight_mean": float(greedy_weight_mean.item()),
                "live_cycle_weight_mean": float(live_cycle_weight_mean.item()),
                "hidden_mse": float(hidden_mse.item()),
                "rollout_d2_ce": float(rollout_d2_ce.item()),
                "rollout_d2_soft_topk_ce": float(rollout_d2_soft_kl.item()),
                "rollout_d2_acceptance_overlap": float(rollout_d2_acceptance_overlap.item()),
                "rollout_d2_rows": float(rollout_d2_rows.item()),
            }
            history.append(history_row)
            print(
                "C4_TRAIN_PROGRESS "
                f"step={step_index}/{steps} "
                f"loss={history_row['loss']:.4f} ce={history_row['ce']:.4f} "
                f"acceptance_overlap={history_row['acceptance_overlap']:.4f} "
                f"rollout_d2_overlap={history_row['rollout_d2_acceptance_overlap']:.4f} "
                f"outside_q={history_row['acceptance_outside_q_mass']:.4f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    mx.eval(rt.model.parameters())
    run_id = now_run_id("c4-mtp-adapter")
    model_hash = _model_identity_hash(model_path)
    adapter_path = artifact_dir / f"{run_id}-r{rank}.npz"
    adapter_metadata = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calib_npz": str(calib_npz),
        "model_path": str(model_path),
        "model_hash": model_hash,
        "train_depths": train_depths,
        "rank": rank,
        "alpha": float(alpha if alpha is not None else rank),
        "targets_requested": adapter_targets,
        "targets_installed": installed_targets,
        "trainable_targets_installed": installed_targets,
        "init_mtp_adapter_path": str(init_mtp_adapter) if init_mtp_adapter else None,
        "init_mtp_adapter_metadata": init_adapter_metadata,
        "init_targets_installed": init_targets,
        "context_targets_installed": context_targets,
        "context_targets_frozen": frozen_context_targets,
        "saved_targets_includes_context": bool(context_targets),
        "context_mtp_adapter_path": str(context_mtp_adapter) if context_mtp_adapter else None,
        "context_mtp_adapter_metadata": context_adapter_metadata,
        "mtp_quant_bits": rt.contract.mtp_quant_bits,
        "mtp_quant_group_size": rt.contract.mtp_quant_group_size,
        "mtp_quant_mode": rt.contract.mtp_quant_mode,
        "mtp_quant_policy": rt.contract.mtp_quant_policy,
        "mtp_prequantized": rt.contract.mtp_prequantized,
        "depth_gate": depth_gate,
        "depth_scales": depth_scales,
        "persistent_replay_scope": persistent_replay_scope,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss_weights": {
            "ce": ce_weight,
            "soft_topk_kl": soft_kl_weight,
            "margin": margin_weight,
            "acceptance_overlap": acceptance_overlap_weight,
            "greedy_acceptance_ce": greedy_acceptance_ce_weight,
            "greedy_acceptance_margin": greedy_acceptance_margin_weight,
            "hidden_mse": hidden_mse_weight,
            "rollout_d2": rollout_d2_weight,
            "rollout_d2_acceptance_overlap": rollout_d2_acceptance_overlap_weight,
            "live_cycle": live_cycle_loss_weight,
        },
        "live_cycle": live_cycle_summary,
        "margin": margin,
        "acceptance_outside_weight": acceptance_outside_weight,
        "greedy_acceptance": {
            "power": greedy_acceptance_power,
            "min_weight": greedy_acceptance_min_weight,
            "focus": greedy_acceptance_focus,
        },
        "hidden_variant": hidden_variant,
        "cache_policy": cache_policy,
        "calibration_schema_version": source_metadata.get("calibration_schema_version"),
        "sampler": {
            "temperature": sampler_temperature,
            "top_k_recorded": source_metadata.get("top_k"),
        },
        "base_model_overwritten": False,
        "target_trunk_frozen": True,
    }
    save_mtp_lora_adapter(adapter_path, rt.model, metadata=adapter_metadata)
    print(f"C4_ADAPTER_SAVED path={adapter_path}", flush=True)

    eval_started = time.perf_counter()
    print(
        "C4_EVAL_START "
        f"train_rows={len(train_rows)} heldout_rows={len(heldout_rows)} "
        f"eval_batch_size={eval_batch_size}",
        flush=True,
    )
    baseline_train = _baseline_metrics(
        train_rows,
        recursive_top_indices=recursive_top_indices,
        target_top_indices=target_ar_top_indices,
        target_tokens=target_tokens,
        target_top_probs=target_ar_top_probs,
        depths=depths_arr,
    )
    baseline_heldout = _baseline_metrics(
        heldout_rows,
        recursive_top_indices=recursive_top_indices,
        target_top_indices=target_ar_top_indices,
        target_tokens=target_tokens,
        target_top_probs=target_ar_top_probs,
        depths=depths_arr,
    )
    adapter_train = _evaluate_adapter(
        rt,
        train_rows,
        input_hidden=recursive_input_hidden,
        input_tokens=recursive_input_tokens,
        target_tokens=target_tokens,
        target_top_indices=target_ar_top_indices,
        target_top_probs=target_ar_top_probs,
        mtp_hidden_variant=hidden_variant,
        cache_policy=cache_policy,
        prompt_ids=prompt_ids,
        window_indices=window_indices,
        depths=depths_arr,
        row_lookup=row_lookup,
        batch_size=eval_batch_size,
        depth_gated=depth_gate != "all",
        sampler_temperature=sampler_temperature,
        persistent_replay_scope=persistent_replay_scope,
    )
    adapter_heldout = _evaluate_adapter(
        rt,
        heldout_rows,
        input_hidden=recursive_input_hidden,
        input_tokens=recursive_input_tokens,
        target_tokens=target_tokens,
        target_top_indices=target_ar_top_indices,
        target_top_probs=target_ar_top_probs,
        mtp_hidden_variant=hidden_variant,
        cache_policy=cache_policy,
        prompt_ids=prompt_ids,
        window_indices=window_indices,
        depths=depths_arr,
        row_lookup=row_lookup,
        batch_size=eval_batch_size,
        depth_gated=depth_gate != "all",
        sampler_temperature=sampler_temperature,
        persistent_replay_scope=persistent_replay_scope,
    )
    rollout_d2_train: dict[str, Any] | None = None
    rollout_d2_heldout: dict[str, Any] | None = None
    if 2 in train_depths:
        rollout_d2_train = _evaluate_rollout_d2_adapter(
            rt,
            train_rows,
            input_hidden=recursive_input_hidden,
            input_tokens=recursive_input_tokens,
            target_tokens=target_tokens,
            target_top_indices=target_ar_top_indices,
            target_top_probs=target_ar_top_probs,
            mtp_hidden_variant=hidden_variant,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths_arr,
            row_lookup=row_lookup,
            batch_size=eval_batch_size,
            sampler_temperature=sampler_temperature,
        )
        rollout_d2_heldout = _evaluate_rollout_d2_adapter(
            rt,
            heldout_rows,
            input_hidden=recursive_input_hidden,
            input_tokens=recursive_input_tokens,
            target_tokens=target_tokens,
            target_top_indices=target_ar_top_indices,
            target_top_probs=target_ar_top_probs,
            mtp_hidden_variant=hidden_variant,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths_arr,
            row_lookup=row_lookup,
            batch_size=eval_batch_size,
            sampler_temperature=sampler_temperature,
        )
    live_cycle_eval: dict[str, Any] | None = None
    if len(live_cycle_rows):
        live_cycle_eval = {
            "baseline": _baseline_metrics(
                live_cycle_rows,
                recursive_top_indices=recursive_top_indices,
                target_top_indices=target_ar_top_indices,
                target_tokens=target_tokens,
                target_top_probs=target_ar_top_probs,
                depths=depths_arr,
            ),
        }
        if pretrain_live_cycle_adapter is not None:
            live_cycle_eval["pretrain_adapter"] = pretrain_live_cycle_adapter
        live_cycle_eval["c4_adapter"] = _evaluate_adapter(
            rt,
            live_cycle_rows,
            input_hidden=recursive_input_hidden,
            input_tokens=recursive_input_tokens,
            target_tokens=target_tokens,
            target_top_indices=target_ar_top_indices,
            target_top_probs=target_ar_top_probs,
            mtp_hidden_variant=hidden_variant,
            cache_policy=cache_policy,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths_arr,
            row_lookup=row_lookup,
            batch_size=eval_batch_size,
            depth_gated=depth_gate != "all",
            sampler_temperature=sampler_temperature,
            persistent_replay_scope=persistent_replay_scope,
        )
    one_step_gate = bool(
        adapter_heldout.get("mean_greedy_acceptance_lb", 0.0)
        > baseline_heldout.get("mean_greedy_acceptance_lb", 0.0)
    )
    rollout_d2_gate: bool | None = None
    if rollout_d2_heldout is not None and int(rollout_d2_heldout.get("rows") or 0) > 0:
        baseline_d2 = (baseline_heldout.get("by_depth") or {}).get("2", {})
        rollout_d2_gate = bool(
            rollout_d2_heldout.get("mean_greedy_acceptance_lb", 0.0)
            > baseline_d2.get("mean_greedy_acceptance_lb", 0.0)
        )
    live_cycle_gate_reasons = _live_cycle_gate_reasons(live_cycle_eval)
    live_cycle_baseline_gate = live_cycle_gate_reasons["live_cycle_baseline_improved"]
    live_cycle_pretrain_gate = live_cycle_gate_reasons["live_cycle_pretrain_preserved"]
    metrics: dict[str, Any] = {
        **adapter_metadata,
        "adapter_path": str(adapter_path),
        "scope_rows": int(scope_mask.sum()),
        "train_rows": int(train_mask.sum()),
        "heldout_rows": int(heldout_mask.sum()),
        "train_history": history,
        "evaluations": {
            "baseline": {
                "train": baseline_train,
                "heldout": baseline_heldout,
            },
            "c4_adapter": {
                "train": adapter_train,
                "heldout": adapter_heldout,
            },
        },
        "elapsed_s": time.perf_counter() - started,
        "eval_elapsed_s": time.perf_counter() - eval_started,
        "runtime_integration_gate": bool(
            one_step_gate and rollout_d2_gate is not False
            and live_cycle_baseline_gate is not False
            and live_cycle_pretrain_gate is not False
        ),
        "runtime_integration_gate_reasons": {
            "one_step_heldout_improved": one_step_gate,
            "rollout_d2_heldout_improved": rollout_d2_gate,
            **live_cycle_gate_reasons,
        },
    }
    if rollout_d2_train is not None and rollout_d2_heldout is not None:
        metrics["evaluations"]["c4_adapter_rollout_d2"] = {
            "train": rollout_d2_train,
            "heldout": rollout_d2_heldout,
        }
    if live_cycle_eval is not None:
        metrics["evaluations"]["live_cycle"] = live_cycle_eval
    metrics_path = report_dir / f"{run_id}.json"
    report_path = report_dir / f"{run_id}.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    report_path.write_text(_render_report(metrics))
    print(
        "C4_EVAL_DONE "
        f"report={metrics_path} "
        f"heldout_top1={adapter_heldout.get('top1', 0.0):.4f} "
        f"heldout_greedy_lb={adapter_heldout.get('mean_greedy_acceptance_lb', 0.0):.4f} "
        f"eval_elapsed_s={metrics['eval_elapsed_s']:.1f}",
        flush=True,
    )
    return metrics


def _render_report(metrics: dict[str, Any]) -> str:
    baseline = metrics["evaluations"]["baseline"]["heldout"]
    adapter = metrics["evaluations"]["c4_adapter"]["heldout"]
    lines = [
        f"# C4 MTP Adapter Report: {metrics['run_id']}",
        "",
        f"- adapter: `{metrics['adapter_path']}`",
        f"- model: `{metrics['model_path']}`",
        f"- model identity hash: `{metrics['model_hash']}`",
        f"- cache policy: `{metrics['cache_policy']}`",
        f"- hidden variant: `{metrics['hidden_variant']}`",
        (
            f"- MTP quant: bits={metrics.get('mtp_quant_bits')}, "
            f"group={metrics.get('mtp_quant_group_size')}, "
            f"mode={metrics.get('mtp_quant_mode')}, "
            f"prequantized={metrics.get('mtp_prequantized')}"
        ),
        f"- depths: {metrics['train_depths']}",
        f"- rank/alpha: {metrics['rank']} / {metrics['alpha']}",
        f"- installed targets: {metrics['targets_installed']}",
        f"- depth gate: `{metrics['depth_gate']}`",
        f"- train/heldout rows: {metrics['train_rows']} / {metrics['heldout_rows']}",
        "",
        "## Heldout",
        "",
        "| model | top1 | top4 | top8 | greedy accept LB | target-topk hit |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| baseline | {baseline['top1']:.4f} | {baseline['top4']:.4f} | "
            f"{baseline['top8']:.4f} | {baseline.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
            f"{baseline.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
        ),
        (
            f"| C4 adapter | {adapter['top1']:.4f} | {adapter['top4']:.4f} | "
            f"{adapter['top8']:.4f} | {adapter.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
            f"{adapter.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
        ),
        "",
        "## Proposal Overlap",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| target top-k q mass | {adapter.get('mean_target_top_q_mass', 0.0):.4f} |",
        f"| target top-k overlap mass | {adapter.get('mean_target_top_overlap_mass', 0.0):.4f} |",
        f"| target top-k excess q mass | {adapter.get('mean_target_top_excess_q_mass', 0.0):.4f} |",
        "",
    ]
    rollout_eval = metrics.get("evaluations", {}).get("c4_adapter_rollout_d2")
    if isinstance(rollout_eval, dict):
        baseline_d2 = (baseline.get("by_depth") or {}).get("2", {})
        rollout = rollout_eval.get("heldout", {})
        lines.extend(
            [
                "## Adapter-Conditioned D2 Rollout",
                "",
                "| model | rows | top1 | top4 | top8 | greedy accept LB | target-topk hit |",
                "|---|---:|---:|---:|---:|---:|---:|",
                (
                    f"| baseline D2 | {int(baseline_d2.get('rows') or 0)} | "
                    f"{baseline_d2.get('top1', 0.0):.4f} | "
                    f"{baseline_d2.get('top4', 0.0):.4f} | "
                    f"{baseline_d2.get('top8', 0.0):.4f} | "
                    f"{baseline_d2.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
                    f"{baseline_d2.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
                ),
                (
                    f"| C4 rollout D2 | {int(rollout.get('rows') or 0)} | "
                    f"{rollout.get('top1', 0.0):.4f} | "
                    f"{rollout.get('top4', 0.0):.4f} | "
                    f"{rollout.get('top8', 0.0):.4f} | "
                    f"{rollout.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
                    f"{rollout.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
                ),
                "",
            ]
        )
    live_cycle = metrics.get("live_cycle")
    live_eval = metrics.get("evaluations", {}).get("live_cycle")
    if isinstance(live_cycle, dict) and live_cycle.get("enabled"):
        lines.extend(
            [
                "## Live-Cycle Weighting",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| matched rows | {int(live_cycle.get('matched_rows') or 0)} |",
                f"| weighted rows | {int(live_cycle.get('weighted_rows') or 0)} |",
                f"| mean row weight | {float(live_cycle.get('mean_weight') or 1.0):.4f} |",
                f"| max row weight | {float(live_cycle.get('max_observed_weight') or 1.0):.4f} |",
                "",
            ]
        )
        if isinstance(live_eval, dict):
            live_base = live_eval.get("baseline", {})
            live_pretrain = live_eval.get("pretrain_adapter")
            live_adapter = live_eval.get("c4_adapter", {})
            lines.extend(
                [
                    "| live subset | top1 | top4 | top8 | greedy accept LB | target-topk hit |",
                    "|---|---:|---:|---:|---:|---:|",
                    (
                        f"| baseline | {live_base.get('top1', 0.0):.4f} | "
                        f"{live_base.get('top4', 0.0):.4f} | "
                        f"{live_base.get('top8', 0.0):.4f} | "
                        f"{live_base.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
                        f"{live_base.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
                    ),
                ]
            )
            if isinstance(live_pretrain, dict):
                lines.append(
                    f"| pretrain adapter | {live_pretrain.get('top1', 0.0):.4f} | "
                    f"{live_pretrain.get('top4', 0.0):.4f} | "
                    f"{live_pretrain.get('top8', 0.0):.4f} | "
                    f"{live_pretrain.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
                    f"{live_pretrain.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
                )
            lines.extend(
                [
                    (
                        f"| C4 adapter | {live_adapter.get('top1', 0.0):.4f} | "
                        f"{live_adapter.get('top4', 0.0):.4f} | "
                        f"{live_adapter.get('top8', 0.0):.4f} | "
                        f"{live_adapter.get('mean_greedy_acceptance_lb', 0.0):.4f} | "
                        f"{live_adapter.get('greedy_target_topk_hit_rate', 0.0):.4f} |"
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## Decision",
            "",
        ]
    )
    if metrics["runtime_integration_gate"]:
        lines.append("- offline gate: pass; run live D2/D3 with `--mtp-adapter` before promotion.")
    else:
        lines.append("- offline gate: fail; do not promote this adapter without a better shard/objective.")
    gate_reasons = metrics.get("runtime_integration_gate_reasons")
    if isinstance(gate_reasons, dict):
        lines.extend(
            [
                (
                    "- one-step heldout improved: "
                    f"{bool(gate_reasons.get('one_step_heldout_improved'))}"
                ),
                (
                    "- rollout D2 heldout improved: "
                    f"{gate_reasons.get('rollout_d2_heldout_improved')}"
                ),
                (
                    "- live-cycle baseline improved: "
                    f"{gate_reasons.get('live_cycle_baseline_improved')}"
                ),
                (
                    "- live-cycle pretrain preserved: "
                    f"{gate_reasons.get('live_cycle_pretrain_preserved')}"
                ),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calib_npz", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=Path("outputs/adapters"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--depths", default="1,2")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--targets",
        default=None,
        help=(
            "Comma-separated MTP LoRA target paths. Defaults to Qwen C4 targets "
            "for Qwen-like models and Step-specific per-depth targets for Step."
        ),
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--soft-kl-weight", type=float, default=0.25)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--acceptance-overlap-weight", type=float, default=0.0)
    parser.add_argument("--acceptance-outside-weight", type=float, default=0.25)
    parser.add_argument("--greedy-acceptance-ce-weight", type=float, default=0.0)
    parser.add_argument("--greedy-acceptance-margin-weight", type=float, default=0.0)
    parser.add_argument("--greedy-acceptance-power", type=float, default=1.0)
    parser.add_argument("--greedy-acceptance-min-weight", type=float, default=0.05)
    parser.add_argument(
        "--greedy-acceptance-focus",
        choices=["low-p", "mismatch", "low-p-or-mismatch"],
        default="low-p-or-mismatch",
    )
    parser.add_argument("--hidden-mse-weight", type=float, default=0.0)
    parser.add_argument("--rollout-d2-weight", type=float, default=0.0)
    parser.add_argument("--rollout-d2-acceptance-overlap-weight", type=float, default=1.0)
    parser.add_argument(
        "--live-cycle-json",
        type=Path,
        action="append",
        default=[],
        help="Live Step acceptance JSON to convert into per-row cycle outcome weights.",
    )
    parser.add_argument("--live-cycle-loss-weight", type=float, default=0.0)
    parser.add_argument("--live-cycle-accepted-boost", type=float, default=0.5)
    parser.add_argument("--live-cycle-rejected-boost", type=float, default=2.0)
    parser.add_argument("--live-cycle-unverified-boost", type=float, default=0.0)
    parser.add_argument("--live-cycle-repair-s-boost", type=float, default=10.0)
    parser.add_argument("--live-cycle-max-weight", type=float, default=6.0)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--depth-gate", choices=["all", "train-depths", "d2plus"], default="all")
    parser.add_argument(
        "--persistent-replay-scope",
        choices=["through-current", "before-current"],
        default="through-current",
    )
    parser.add_argument("--init-mtp-adapter", type=Path, default=None)
    parser.add_argument("--context-mtp-adapter", type=Path, default=None)
    parser.add_argument("--mtp-quant-bits", type=int, default=None)
    parser.add_argument("--mtp-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--mtp-quant-mode",
        choices=["affine", "symmetric"],
        default="affine",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = train_mtp_adapter_c4(
        args.calib_npz,
        model_path=args.model,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        train_fraction=args.train_fraction,
        depths=_parse_int_csv(args.depths),
        rank=args.rank,
        alpha=args.alpha,
        targets=_parse_target_csv(args.targets) if args.targets else None,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        ce_weight=args.ce_weight,
        soft_kl_weight=args.soft_kl_weight,
        margin_weight=args.margin_weight,
        margin=args.margin,
        acceptance_overlap_weight=args.acceptance_overlap_weight,
        acceptance_outside_weight=args.acceptance_outside_weight,
        greedy_acceptance_ce_weight=args.greedy_acceptance_ce_weight,
        greedy_acceptance_margin_weight=args.greedy_acceptance_margin_weight,
        greedy_acceptance_power=args.greedy_acceptance_power,
        greedy_acceptance_min_weight=args.greedy_acceptance_min_weight,
        greedy_acceptance_focus=args.greedy_acceptance_focus,
        hidden_mse_weight=args.hidden_mse_weight,
        rollout_d2_weight=args.rollout_d2_weight,
        rollout_d2_acceptance_overlap_weight=args.rollout_d2_acceptance_overlap_weight,
        live_cycle_json=args.live_cycle_json,
        live_cycle_loss_weight=args.live_cycle_loss_weight,
        live_cycle_accepted_boost=args.live_cycle_accepted_boost,
        live_cycle_rejected_boost=args.live_cycle_rejected_boost,
        live_cycle_unverified_boost=args.live_cycle_unverified_boost,
        live_cycle_repair_s_boost=args.live_cycle_repair_s_boost,
        live_cycle_max_weight=args.live_cycle_max_weight,
        eval_batch_size=args.eval_batch_size,
        depth_gate=args.depth_gate,
        persistent_replay_scope=args.persistent_replay_scope,
        init_mtp_adapter=args.init_mtp_adapter,
        context_mtp_adapter=args.context_mtp_adapter,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
