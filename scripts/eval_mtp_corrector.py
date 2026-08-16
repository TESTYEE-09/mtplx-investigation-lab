#!/usr/bin/env python3
"""Fit and evaluate offline hidden correctors on MTP calibration shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mtplx.benchmarks.schema import now_run_id  # noqa: E402
from mtplx.correctors import (  # noqa: E402
    NoOpCorrector,
    blend_with_identity,
    deterministic_train_mask,
    ensure_nonempty_split,
    fit_c0_stat,
    fit_c1_diagonal,
    fit_c2_low_rank,
)
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR  # noqa: E402
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.runtime import load  # noqa: E402


def _load_source_metadata(calib_npz: Path) -> dict[str, Any]:
    metadata_path = calib_npz.with_name("metadata.json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _parse_float_csv(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("at least one blend strength is required")
    for item in parsed:
        if not 0.0 <= item <= 1.0:
            raise ValueError("blend strengths must be in [0, 1]")
    return sorted(set(parsed))


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item < 1 for item in parsed):
        raise ValueError("integer CSV values must be >= 1")
    return sorted(set(parsed))


def _reconstruct_prefix_active(
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    recursive_matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_active = np.zeros(len(depths), dtype=bool)
    prefix_len_before = np.zeros(len(depths), dtype=np.int16)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, (prompt_id, window_index) in enumerate(zip(prompt_ids, window_indices, strict=True)):
        groups[(str(prompt_id), int(window_index))].append(idx)

    for indices in groups.values():
        active = True
        prefix_len = 0
        for idx in sorted(indices, key=lambda row: int(depths[row])):
            prefix_active[idx] = active
            prefix_len_before[idx] = prefix_len
            if active and bool(recursive_matches[idx]):
                prefix_len = int(depths[idx])
            else:
                active = False
    return prefix_active, prefix_len_before


def _build_next_step_arrays(
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    target_tokens: np.ndarray,
    target_forced_top_indices: np.ndarray,
    target_ar_top_indices: np.ndarray,
    target_ar_top_probs: np.ndarray,
) -> dict[str, np.ndarray]:
    next_row = np.full(len(depths), -1, dtype=np.int64)
    by_key = {
        (str(prompt_id), int(window_index), int(depth)): idx
        for idx, (prompt_id, window_index, depth) in enumerate(
            zip(prompt_ids, window_indices, depths, strict=True)
        )
    }
    for idx, (prompt_id, window_index, depth) in enumerate(zip(prompt_ids, window_indices, depths, strict=True)):
        next_row[idx] = by_key.get((str(prompt_id), int(window_index), int(depth) + 1), -1)

    has_next = next_row >= 0
    safe_next = np.where(has_next, next_row, 0)
    return {
        "has_next": has_next,
        "proposal_depths": depths + 1,
        "target_tokens": target_tokens[safe_next],
        "target_forced_top_indices": target_forced_top_indices[safe_next],
        "target_ar_top_indices": target_ar_top_indices[safe_next],
        "target_ar_top_probs": target_ar_top_probs[safe_next],
    }


def _apply_corrector_numpy(corrector: Any, hidden: np.ndarray, depths: np.ndarray) -> np.ndarray:
    if isinstance(corrector, NoOpCorrector):
        return hidden
    corrected = np.empty_like(hidden, dtype=np.float32)
    for depth in sorted(set(int(value) for value in depths)):
        rows = depths == depth
        corrected[rows] = corrector.apply_numpy(hidden[rows], depth=depth)
    return corrected


def _hidden_metrics(corrected: np.ndarray, target: np.ndarray) -> dict[str, float]:
    corrected = np.asarray(corrected, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    diff = corrected - target
    denom = float(np.mean(target * target) + 1e-8)
    nmse = float(np.mean(diff * diff) / denom)
    numerator = np.sum(corrected * target, axis=1)
    corrected_norm = np.linalg.norm(corrected, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = numerator / np.maximum(corrected_norm * target_norm, 1e-8)
    return {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "normalized_mse": nmse,
    }


def _build_persistent_replay_lookup(
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
) -> dict[tuple[str, int, int], int]:
    lookup: dict[tuple[str, int, int], int] = {}
    for idx, (prompt_id, window_index, depth) in enumerate(
        zip(prompt_ids, window_indices, depths, strict=True)
    ):
        key = (str(prompt_id), int(window_index), int(depth))
        if key in lookup:
            raise ValueError(f"duplicate persistent replay row key: {key}")
        lookup[key] = idx
    return lookup


def _persistent_replay_indices(
    row_index: int,
    *,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    row_lookup: dict[tuple[str, int, int], int],
    include_current: bool = True,
) -> list[int]:
    prompt_id = str(prompt_ids[row_index])
    window_index = int(window_indices[row_index])
    depth = int(depths[row_index])
    replay: list[int] = []
    stop_depth = depth if include_current else depth - 1
    for replay_depth in range(1, stop_depth + 1):
        key = (prompt_id, window_index, replay_depth)
        if key not in row_lookup:
            raise ValueError(f"missing persistent replay row for key {key}")
        replay.append(row_lookup[key])
    return replay


def _require_persistent_replay_context(
    *,
    recursive_input_hidden: np.ndarray | None,
    recursive_input_tokens: np.ndarray | None,
    prompt_ids: np.ndarray | None,
    window_indices: np.ndarray | None,
    depths: np.ndarray | None,
) -> None:
    missing = []
    if recursive_input_hidden is None:
        missing.append("recursive_input_hidden")
    if recursive_input_tokens is None:
        missing.append("recursive_input_tokens")
    if prompt_ids is None:
        missing.append("prompt_ids")
    if window_indices is None:
        missing.append("window_indices")
    if depths is None:
        missing.append("depths")
    if missing:
        raise ValueError(
            "persistent-cache offline evaluation requires calibration shards "
            f"with replay fields; missing {missing}. Recollect with the updated "
            "scripts/collect_mtp_hidden_calib.py --cache-policy persistent."
        )


def _draft_next_logits_persistent(
    rt,
    hidden: np.ndarray,
    input_tokens: np.ndarray,
    *,
    row_indices: np.ndarray,
    mtp_hidden_variant: str,
    recursive_input_hidden: np.ndarray,
    recursive_input_tokens: np.ndarray,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    depths: np.ndarray,
    row_lookup: dict[tuple[str, int, int], int],
) -> np.ndarray:
    logits_rows: list[np.ndarray] = []
    for local_idx, row_index in enumerate(row_indices):
        mtp_cache = rt.make_mtp_cache()
        for replay_index in _persistent_replay_indices(
            int(row_index),
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths,
            row_lookup=row_lookup,
        ):
            replay_hidden = mx.array(
                recursive_input_hidden[replay_index : replay_index + 1, None, :].astype(np.float32)
            )
            replay_token = mx.array([[int(recursive_input_tokens[replay_index])]])
            replay_logits = rt.draft_mtp(
                replay_hidden,
                replay_token,
                mtp_cache=mtp_cache,
                return_hidden=False,
                mtp_hidden_variant=mtp_hidden_variant,
            )
            mx.eval(replay_logits)
        row_hidden = mx.array(hidden[local_idx : local_idx + 1, None, :].astype(np.float32))
        row_token = mx.array([[int(input_tokens[local_idx])]])
        row_logits = rt.draft_mtp(
            row_hidden,
            row_token,
            mtp_cache=mtp_cache,
            return_hidden=False,
            mtp_hidden_variant=mtp_hidden_variant,
        )
        mx.eval(row_logits)
        logits_rows.append(np.asarray(row_logits[:, -1, :], dtype=np.float32))
    return np.concatenate(logits_rows, axis=0)


def _draft_next_logits(
    rt,
    hidden: np.ndarray,
    input_tokens: np.ndarray,
    *,
    mtp_hidden_variant: str,
    cache_policy: str,
    row_indices: np.ndarray | None = None,
    recursive_input_hidden: np.ndarray | None = None,
    recursive_input_tokens: np.ndarray | None = None,
    prompt_ids: np.ndarray | None = None,
    window_indices: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    row_lookup: dict[tuple[str, int, int], int] | None = None,
) -> np.ndarray:
    if cache_policy not in {"fresh", "persistent"}:
        raise ValueError("cache_policy must be 'fresh' or 'persistent'")
    if cache_policy == "persistent":
        _require_persistent_replay_context(
            recursive_input_hidden=recursive_input_hidden,
            recursive_input_tokens=recursive_input_tokens,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths,
        )
        if row_indices is None:
            raise ValueError("persistent-cache evaluation requires row_indices")
        if row_lookup is None:
            row_lookup = _build_persistent_replay_lookup(prompt_ids, window_indices, depths)
        return _draft_next_logits_persistent(
            rt,
            hidden,
            input_tokens,
            row_indices=row_indices,
            mtp_hidden_variant=mtp_hidden_variant,
            recursive_input_hidden=recursive_input_hidden,
            recursive_input_tokens=recursive_input_tokens,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=depths,
            row_lookup=row_lookup,
        )
    hidden_mx = mx.array(hidden[:, None, :].astype(np.float32))
    token_mx = mx.array(np.asarray(input_tokens, dtype=np.int64)[:, None])
    logits = rt.draft_mtp(
        hidden_mx,
        token_mx,
        mtp_cache=rt.make_mtp_cache(),
        return_hidden=False,
        mtp_hidden_variant=mtp_hidden_variant,
    )
    mx.eval(logits)
    return np.asarray(logits[:, -1, :], dtype=np.float32)


def _topk_indices(logits: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray]:
    actual_k = min(k, logits.shape[1])
    unsorted = np.argpartition(-logits, kth=actual_k - 1, axis=1)[:, :actual_k]
    values = np.take_along_axis(logits, unsorted, axis=1)
    order = np.argsort(-values, axis=1)
    indices = np.take_along_axis(unsorted, order, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    return indices.astype(np.int64), sorted_values.astype(np.float32)


def _softmax_probs_for_tokens(logits: np.ndarray, tokens: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    denom = np.sum(np.exp(shifted), axis=1)
    token_values = shifted[np.arange(len(tokens)), tokens]
    return (np.exp(token_values) / denom).astype(np.float32)


def _evaluate_model(
    rt,
    *,
    name: str,
    corrector: Any,
    recursive_hidden: np.ndarray,
    target_hidden: np.ndarray,
    corrector_depths: np.ndarray,
    proposal_depths: np.ndarray,
    input_tokens: np.ndarray,
    target_tokens: np.ndarray,
    prompt_ids: np.ndarray,
    window_indices: np.ndarray,
    target_forced_top_indices: np.ndarray,
    target_ar_top_indices: np.ndarray,
    target_ar_top_probs: np.ndarray,
    split_mask: np.ndarray,
    mtp_hidden_variant: str,
    cache_policy: str,
    split_name: str,
    batch_size: int,
    sampler_temperature: float,
    recursive_input_hidden: np.ndarray | None = None,
    recursive_input_tokens: np.ndarray | None = None,
) -> dict[str, Any]:
    row_indices = np.flatnonzero(split_mask)
    if len(row_indices) == 0:
        return {
            "name": name,
            "split": split_name,
            "rows": 0,
            "prompt_window_groups": 0,
            "by_depth": {},
        }
    corrected = _apply_corrector_numpy(
        corrector,
        recursive_hidden[row_indices],
        corrector_depths[row_indices],
    )
    hidden_by_depth: dict[str, Any] = {}
    for proposal_depth in sorted(set(int(value) for value in proposal_depths[row_indices])):
        local_rows = proposal_depths[row_indices] == proposal_depth
        hidden_by_depth[str(proposal_depth)] = _hidden_metrics(
            corrected[local_rows],
            target_hidden[row_indices][local_rows],
        )

    buckets: dict[int, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    max_k = 8
    row_lookup = (
        _build_persistent_replay_lookup(prompt_ids, window_indices, corrector_depths)
        if cache_policy == "persistent"
        else None
    )
    for start in range(0, len(row_indices), batch_size):
        batch_rows = row_indices[start : start + batch_size]
        local = slice(start, start + len(batch_rows))
        batch_hidden = corrected[local]
        logits = _draft_next_logits(
            rt,
            batch_hidden,
            input_tokens[batch_rows],
            mtp_hidden_variant=mtp_hidden_variant,
            cache_policy=cache_policy,
            row_indices=batch_rows,
            recursive_input_hidden=recursive_input_hidden,
            recursive_input_tokens=recursive_input_tokens,
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            depths=corrector_depths,
            row_lookup=row_lookup,
        )
        top_indices, _ = _topk_indices(logits, k=max_k)
        top1 = top_indices[:, 0]
        targets = target_tokens[batch_rows]
        target_values = logits[np.arange(len(batch_rows)), targets]
        ranks = (np.sum(logits > target_values[:, None], axis=1) + 1).astype(np.int64)
        q_probs = _softmax_probs_for_tokens(logits, top1, temperature=sampler_temperature)

        for local_idx, row in enumerate(batch_rows):
            depth = int(proposal_depths[row])
            row_top = top_indices[local_idx]
            target = int(targets[local_idx])
            bucket = buckets[depth]
            bucket["top1_match"].append(bool(row_top[0] == target))
            bucket["rank"].append(int(ranks[local_idx]))
            for k in (2, 4, 8):
                bucket[f"top{k}_contains"].append(bool(target in set(map(int, row_top[:k]))))
            forced_set = set(map(int, target_forced_top_indices[row, :max_k]))
            bucket["top8_overlap_target_forced"].append(
                len(set(map(int, row_top[:max_k])) & forced_set) / float(max_k)
            )

            candidate = int(row_top[0])
            target_candidates = target_ar_top_indices[row]
            matches = np.where(target_candidates == candidate)[0]
            bucket["proxy_available"].append(bool(matches.size))
            if matches.size:
                p_prob = float(target_ar_top_probs[row, int(matches[0])])
                q_prob = float(q_probs[local_idx])
                bucket["proxy_acceptance"].append(min(1.0, p_prob / max(q_prob, 1e-30)))
            bucket["prompt_id"].append(str(prompt_ids[row]))
            bucket["window_index"].append(int(window_indices[row]))

    by_depth: dict[str, dict[str, Any]] = {}
    for depth, bucket in sorted(buckets.items()):
        ranks = np.asarray(bucket["rank"], dtype=np.float64)
        proxy_values = np.asarray(bucket.get("proxy_acceptance", []), dtype=np.float64)
        by_depth[str(depth)] = {
            "rows": int(len(ranks)),
            "top1": float(np.mean(bucket["top1_match"])),
            "target_rank_mean": float(np.mean(ranks)),
            "target_rank_median": float(np.median(ranks)),
            "target_rank_p90": float(np.quantile(ranks, 0.90)),
            "top2_contains": float(np.mean(bucket["top2_contains"])),
            "top4_contains": float(np.mean(bucket["top4_contains"])),
            "top8_contains": float(np.mean(bucket["top8_contains"])),
            "top8_overlap_target_forced": float(np.mean(bucket["top8_overlap_target_forced"])),
            "hidden": hidden_by_depth[str(depth)],
            "proxy_temp_acceptance_mean": float(np.mean(proxy_values)) if len(proxy_values) else None,
            "proxy_temp_acceptance_coverage": float(np.mean(bucket["proxy_available"])),
        }

    return {
        "name": name,
        "split": split_name,
        "rows": int(len(row_indices)),
        "prompt_window_groups": int(len({(str(pid), int(win)) for pid, win in zip(prompt_ids[row_indices], window_indices[row_indices], strict=True)})),
        "by_depth": by_depth,
    }


def _table(metrics: dict[str, Any], *, split: str, depth: str) -> str:
    lines = [
        "| model | rows | top1 | top4 | top8 | mean rank | median rank | cosine | nMSE | proxy acc | proxy coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in metrics["model_order"]:
        item = metrics["models"][name][split]["by_depth"].get(depth)
        if item is None:
            continue
        proxy = item["proxy_temp_acceptance_mean"]
        lines.append(
            "| {name} | {rows} | {top1:.4f} | {top4:.4f} | {top8:.4f} | "
            "{rank_mean:.1f} | {rank_median:.1f} | {cos:.6f} | {nmse:.6f} | "
            "{proxy} | {coverage:.4f} |".format(
                name=name,
                rows=item["rows"],
                top1=item["top1"],
                top4=item["top4_contains"],
                top8=item["top8_contains"],
                rank_mean=item["target_rank_mean"],
                rank_median=item["target_rank_median"],
                cos=item["hidden"]["cosine_mean"],
                nmse=item["hidden"]["normalized_mse"],
                proxy="n/a" if proxy is None else f"{proxy:.4f}",
                coverage=item["proxy_temp_acceptance_coverage"],
            )
        )
    return "\n".join(lines)


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    candidate_name = metrics.get("best_candidate", {}).get("name", "c1_diagonal")
    heldout = metrics["models"][candidate_name]["heldout"]["by_depth"]
    baseline = metrics["models"]["baseline"]["heldout"]["by_depth"]
    d2_candidate = heldout.get("2")
    d2_base = baseline.get("2")
    if d2_candidate is None or d2_base is None:
        return {
            "runtime_integration_gate": False,
            "reason": "no held-out depth-2 rows",
        }
    top1_delta = d2_candidate["top1"] - d2_base["top1"]
    top4_delta = d2_candidate["top4_contains"] - d2_base["top4_contains"]
    top8_delta = d2_candidate["top8_contains"] - d2_base["top8_contains"]
    proxy = d2_candidate["proxy_temp_acceptance_mean"]
    baseline_proxy = d2_base["proxy_temp_acceptance_mean"]
    projected_correction_rate = None if proxy is None else 1.0 - proxy
    proxy_delta = None if proxy is None or baseline_proxy is None else proxy - baseline_proxy
    candidate_strength = metrics.get("best_candidate", {}).get("blend_strength")
    non_noop_candidate = candidate_strength is None or float(candidate_strength) > 0.0
    top1_gate = d2_candidate["top1"] >= 0.70 and top1_delta >= 0.03
    topk_gate = top4_delta >= 0.10 or top8_delta >= 0.10
    proxy_gate = (
        projected_correction_rate is not None
        and projected_correction_rate <= 0.12
        and proxy_delta is not None
        and proxy_delta > 0.0
    )
    gate = (
        non_noop_candidate
        and (top1_gate or topk_gate or proxy_gate)
    )
    return {
        "runtime_integration_gate": bool(gate),
        "candidate_name": candidate_name,
        "candidate_blend_strength": candidate_strength,
        "d2_candidate_top1": d2_candidate["top1"],
        "d2_baseline_top1": d2_base["top1"],
        "d2_top1_delta": top1_delta,
        "d2_top4_delta": top4_delta,
        "d2_top8_delta": top8_delta,
        "d2_candidate_proxy_temp_acceptance_mean": proxy,
        "d2_baseline_proxy_temp_acceptance_mean": baseline_proxy,
        "d2_proxy_delta": proxy_delta,
        "projected_correction_rate": projected_correction_rate,
        "top1_gate": top1_gate,
        "topk_gate": topk_gate,
        "proxy_gate": proxy_gate,
    }


def _candidate_score(evaluation: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    d2 = evaluation["by_depth"].get("2")
    if d2 is None:
        return (-1.0, -1.0, -1.0, -1e12, -1e12, -1.0)
    proxy = d2["proxy_temp_acceptance_mean"]
    return (
        float(d2["top1"]),
        float(d2["top4_contains"]),
        float(d2["top8_contains"]),
        -float(d2["target_rank_median"]),
        -float(d2["target_rank_mean"]),
        float(proxy) if proxy is not None else -1.0,
    )


def fit_and_evaluate(
    calib_npz: Path | str,
    *,
    model_path: Path | str | None = None,
    output_dir: Path | str = Path("outputs/reports"),
    artifact_dir: Path | str = Path("outputs/correctors"),
    train_fraction: float = 0.75,
    batch_size: int = 8,
    sampler_temperature: float | None = None,
    prefix_active_only: bool = True,
    blend_strengths: list[float] | None = None,
    c2_ranks: list[int] | None = None,
    c2_ridge: float = 1e-3,
) -> dict[str, Any]:
    calib_npz = Path(calib_npz)
    source_metadata = _load_source_metadata(calib_npz)
    output_dir = Path(output_dir)
    artifact_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with np.load(calib_npz, allow_pickle=False) as data:
        recursive_hidden = np.asarray(data["recursive_hidden"], dtype=np.float32)
        recursive_input_hidden = (
            np.asarray(data["recursive_input_hidden"], dtype=np.float32)
            if "recursive_input_hidden" in data
            else None
        )
        if "target_next_hidden" not in data:
            raise ValueError(
                "calibration shard lacks target_next_hidden; recollect with the updated collector"
            )
        target_hidden = np.asarray(data["target_next_hidden"], dtype=np.float32)
        depths = np.asarray(data["depths"], dtype=np.int64)
        prompt_ids = np.asarray(data["prompt_ids"])
        window_indices = np.asarray(data["window_indices"], dtype=np.int64)
        target_tokens = np.asarray(data["target_tokens"], dtype=np.int64)
        recursive_input_tokens = np.asarray(data["recursive_input_tokens"], dtype=np.int64)
        recursive_draft_tokens = np.asarray(data["recursive_draft_tokens"], dtype=np.int64)
        recursive_matches = np.asarray(data["recursive_matches"], dtype=bool)
        if "recursive_prefix_active" in data:
            recursive_prefix_active = np.asarray(data["recursive_prefix_active"], dtype=bool)
            recursive_prefix_len_before = np.asarray(data["recursive_prefix_len_before"], dtype=np.int16)
        else:
            recursive_prefix_active, recursive_prefix_len_before = _reconstruct_prefix_active(
                prompt_ids,
                window_indices,
                depths,
                recursive_matches,
            )
        if "recursive_accepted_through_depth" in data:
            recursive_accepted_through_depth = np.asarray(data["recursive_accepted_through_depth"], dtype=bool)
        else:
            recursive_accepted_through_depth = recursive_prefix_active & recursive_matches
        target_forced_top_indices = np.asarray(data["target_forced_top_indices"], dtype=np.int64)
        target_ar_top_indices = np.asarray(data["target_ar_top_indices"], dtype=np.int64)
        target_ar_top_probs = np.asarray(data["target_ar_top_probs"], dtype=np.float32)
    next_step = _build_next_step_arrays(
        prompt_ids,
        window_indices,
        depths,
        target_tokens,
        target_forced_top_indices,
        target_ar_top_indices,
        target_ar_top_probs,
    )
    row_scope_mask = (
        recursive_accepted_through_depth & next_step["has_next"]
        if prefix_active_only
        else next_step["has_next"]
    )

    split_train_mask = deterministic_train_mask(
        prompt_ids,
        window_indices,
        train_fraction=train_fraction,
    )
    split_train_mask = ensure_nonempty_split(split_train_mask, prompt_ids, window_indices)
    train_mask = split_train_mask & row_scope_mask
    heldout_mask = (~split_train_mask) & row_scope_mask
    if not train_mask.any() or not heldout_mask.any():
        raise ValueError("prefix-active split produced empty train or held-out rows")
    depth_count = int(np.max(depths))
    hidden_variant = str(source_metadata.get("mtp_hidden_variant", "pre_norm"))
    cache_policy = str(source_metadata.get("cache_policy", "fresh"))
    model_path = model_path or source_metadata.get("model_path") or DEFAULT_RUNTIME_MODEL_DIR
    sampler_temperature = float(
        sampler_temperature
        if sampler_temperature is not None
        else source_metadata.get("sampler_temperature", 0.6)
    )

    fit_meta = {
        "calib_npz": str(calib_npz),
        "train_fraction": train_fraction,
        "prefix_active_only": prefix_active_only,
        "train_rows": int(train_mask.sum()),
        "heldout_rows": int(heldout_mask.sum()),
        "source_metadata": source_metadata,
    }
    c0 = fit_c0_stat(
        recursive_hidden[train_mask],
        target_hidden[train_mask],
        depths[train_mask],
        depth_count=depth_count,
        hidden_variant=hidden_variant,
        metadata=fit_meta,
    )
    c1 = fit_c1_diagonal(
        recursive_hidden[train_mask],
        target_hidden[train_mask],
        depths[train_mask],
        depth_count=depth_count,
        hidden_variant=hidden_variant,
        metadata=fit_meta,
    )

    run_id = now_run_id("corrector-c0-c1")
    c0_path = artifact_dir / f"{run_id}-c0-stat.npz"
    c1_path = artifact_dir / f"{run_id}-c1-diagonal.npz"
    c0.save(c0_path)
    c1.save(c1_path)
    c2_correctors = []
    c2_paths: dict[str, str] = {}
    for rank in c2_ranks or []:
        c2 = fit_c2_low_rank(
            recursive_hidden[train_mask],
            target_hidden[train_mask],
            depths[train_mask],
            depth_count=depth_count,
            rank=rank,
            ridge=c2_ridge,
            hidden_variant=hidden_variant,
            metadata={
                **fit_meta,
                "rank": rank,
                "ridge": c2_ridge,
            },
        )
        c2_path = artifact_dir / f"{run_id}-c2-low-rank-r{rank}.npz"
        c2.save(c2_path)
        c2_correctors.append((f"c2_rank_{rank}", c2))
        c2_paths[f"c2_rank_{rank}"] = str(c2_path)

    rt = load(model_path, mtp=True, contract=MTPContract())
    started = time.perf_counter()
    blend_strengths = blend_strengths or [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    candidate_evals = []
    for strength in blend_strengths:
        candidate = blend_with_identity(c1, strength, kind="c1_diagonal_blend")
        heldout_eval = _evaluate_model(
            rt,
            name=f"c1_blend_{strength:g}",
            corrector=candidate,
            recursive_hidden=recursive_hidden,
            target_hidden=target_hidden,
            corrector_depths=depths,
            proposal_depths=next_step["proposal_depths"],
            input_tokens=recursive_draft_tokens,
            target_tokens=next_step["target_tokens"],
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            target_forced_top_indices=next_step["target_forced_top_indices"],
            target_ar_top_indices=next_step["target_ar_top_indices"],
            target_ar_top_probs=next_step["target_ar_top_probs"],
            recursive_input_hidden=recursive_input_hidden,
            recursive_input_tokens=recursive_input_tokens,
            split_mask=heldout_mask,
            mtp_hidden_variant=hidden_variant,
            cache_policy=cache_policy,
            split_name="heldout",
            batch_size=batch_size,
            sampler_temperature=sampler_temperature,
        )
        candidate_evals.append(
            {
                "name": f"c1_blend_{strength:g}",
                "kind": "c1_blend",
                "blend_strength": float(strength),
                "score": list(_candidate_score(heldout_eval)),
                "heldout": heldout_eval,
                "corrector": candidate,
            }
        )
    for c2_name, c2 in c2_correctors:
        heldout_eval = _evaluate_model(
            rt,
            name=c2_name,
            corrector=c2,
            recursive_hidden=recursive_hidden,
            target_hidden=target_hidden,
            corrector_depths=depths,
            proposal_depths=next_step["proposal_depths"],
            input_tokens=recursive_draft_tokens,
            target_tokens=next_step["target_tokens"],
            prompt_ids=prompt_ids,
            window_indices=window_indices,
            target_forced_top_indices=next_step["target_forced_top_indices"],
            target_ar_top_indices=next_step["target_ar_top_indices"],
            target_ar_top_probs=next_step["target_ar_top_probs"],
            recursive_input_hidden=recursive_input_hidden,
            recursive_input_tokens=recursive_input_tokens,
            split_mask=heldout_mask,
            mtp_hidden_variant=hidden_variant,
            cache_policy=cache_policy,
            split_name="heldout",
            batch_size=batch_size,
            sampler_temperature=sampler_temperature,
        )
        candidate_evals.append(
            {
                "name": c2_name,
                "kind": "c2_low_rank",
                "rank": int(c2.rank),
                "score": list(_candidate_score(heldout_eval)),
                "heldout": heldout_eval,
                "corrector": c2,
            }
        )
    best_candidate = max(candidate_evals, key=lambda item: tuple(item["score"]))
    c1_candidates = [item for item in candidate_evals if item["kind"] == "c1_blend"]
    best_c1_candidate = max(c1_candidates, key=lambda item: tuple(item["score"]))
    c1_blend_best = blend_with_identity(
        c1,
        float(best_c1_candidate["blend_strength"]),
        kind="c1_diagonal_blend_best",
    )
    c1_blend_best_path = artifact_dir / f"{run_id}-c1-diagonal-blend-best.npz"
    c1_blend_best.save(c1_blend_best_path)
    best_corrector = best_candidate["corrector"]
    best_path = artifact_dir / f"{run_id}-best-candidate.npz"
    best_corrector.save(best_path)

    models = {
        "baseline": NoOpCorrector(),
        "c0_stat": c0,
        "c1_diagonal": c1,
        "c1_blend_best": c1_blend_best,
        "best_candidate": best_corrector,
    }
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calib_npz": str(calib_npz),
        "model_path": str(model_path),
        "hidden_variant": hidden_variant,
        "cache_policy": cache_policy,
        "evaluation_contract": "input_corrector_next_mtp_step",
        "sampler_temperature": sampler_temperature,
        "train_fraction": train_fraction,
        "prefix_active_only": prefix_active_only,
        "row_scope_rows": int(row_scope_mask.sum()),
        "train_rows": int(train_mask.sum()),
        "heldout_rows": int(heldout_mask.sum()),
        "train_prompt_window_groups": int(len({(str(pid), int(win)) for pid, win in zip(prompt_ids[train_mask], window_indices[train_mask], strict=True)})),
        "heldout_prompt_window_groups": int(len({(str(pid), int(win)) for pid, win in zip(prompt_ids[heldout_mask], window_indices[heldout_mask], strict=True)})),
        "corrector_paths": {
            "c0_stat": str(c0_path),
            "c1_diagonal": str(c1_path),
            "c1_blend_best": str(c1_blend_best_path),
            "best_candidate": str(best_path),
            **c2_paths,
        },
        "model_order": list(models.keys()),
        "blend_candidates": [
            {
                "name": item["name"],
                "kind": item["kind"],
                **({"blend_strength": item["blend_strength"]} if "blend_strength" in item else {}),
                **({"rank": item["rank"]} if "rank" in item else {}),
                "score": item["score"],
                "heldout_by_depth": item["heldout"]["by_depth"],
            }
            for item in candidate_evals
        ],
        "best_candidate": {
            "name": "best_candidate",
            "source_name": best_candidate["name"],
            "kind": best_candidate["kind"],
            **({"rank": best_candidate["rank"]} if "rank" in best_candidate else {}),
            "blend_strength": (
                float(best_candidate["blend_strength"])
                if "blend_strength" in best_candidate
                else None
            ),
            "score": best_candidate["score"],
        },
        "best_c1_blend_strength": float(best_c1_candidate["blend_strength"]),
        "source_metadata": source_metadata,
        "models": {},
    }

    for name, corrector in models.items():
        metrics["models"][name] = {
            "train": _evaluate_model(
                rt,
                name=name,
                corrector=corrector,
                recursive_hidden=recursive_hidden,
                target_hidden=target_hidden,
                corrector_depths=depths,
                proposal_depths=next_step["proposal_depths"],
                input_tokens=recursive_draft_tokens,
                target_tokens=next_step["target_tokens"],
                prompt_ids=prompt_ids,
                window_indices=window_indices,
                target_forced_top_indices=next_step["target_forced_top_indices"],
                target_ar_top_indices=next_step["target_ar_top_indices"],
                target_ar_top_probs=next_step["target_ar_top_probs"],
                recursive_input_hidden=recursive_input_hidden,
                recursive_input_tokens=recursive_input_tokens,
                split_mask=train_mask,
                mtp_hidden_variant=hidden_variant,
                cache_policy=cache_policy,
                split_name="train",
                batch_size=batch_size,
                sampler_temperature=sampler_temperature,
            ),
            "heldout": _evaluate_model(
                rt,
                name=name,
                corrector=corrector,
                recursive_hidden=recursive_hidden,
                target_hidden=target_hidden,
                corrector_depths=depths,
                proposal_depths=next_step["proposal_depths"],
                input_tokens=recursive_draft_tokens,
                target_tokens=next_step["target_tokens"],
                prompt_ids=prompt_ids,
                window_indices=window_indices,
                target_forced_top_indices=next_step["target_forced_top_indices"],
                target_ar_top_indices=next_step["target_ar_top_indices"],
                target_ar_top_probs=next_step["target_ar_top_probs"],
                recursive_input_hidden=recursive_input_hidden,
                recursive_input_tokens=recursive_input_tokens,
                split_mask=heldout_mask,
                mtp_hidden_variant=hidden_variant,
                cache_policy=cache_policy,
                split_name="heldout",
                batch_size=batch_size,
                sampler_temperature=sampler_temperature,
            ),
        }

    metrics["elapsed_s"] = time.perf_counter() - started
    metrics["decision"] = _decision(metrics)

    metrics_path = output_dir / f"{run_id}.json"
    report_path = output_dir / f"{run_id}.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    report_path.write_text(_render_report(metrics, report_path=report_path, metrics_path=metrics_path))
    metrics["metrics_path"] = str(metrics_path)
    metrics["report_path"] = str(report_path)
    return metrics


def _render_report(metrics: dict[str, Any], *, report_path: Path, metrics_path: Path) -> str:
    decision = metrics["decision"]
    lines = [
        "# MTP Hidden Corrector Report",
        "",
        f"- report: `{report_path}`",
        f"- metrics: `{metrics_path}`",
        f"- calibration shard: `{metrics['calib_npz']}`",
        f"- model: `{metrics['model_path']}`",
        f"- hidden variant: `{metrics['hidden_variant']}`",
        f"- cache policy: `{metrics['cache_policy']}`",
        f"- evaluation contract: `{metrics['evaluation_contract']}`",
        f"- prefix-active only: `{metrics['prefix_active_only']}`",
        f"- scoped rows: `{metrics['row_scope_rows']}`",
        f"- train rows: `{metrics['train_rows']}` across `{metrics['train_prompt_window_groups']}` prompt/window groups",
        f"- held-out rows: `{metrics['heldout_rows']}` across `{metrics['heldout_prompt_window_groups']}` prompt/window groups",
        f"- C0 artifact: `{metrics['corrector_paths']['c0_stat']}`",
        f"- C1 artifact: `{metrics['corrector_paths']['c1_diagonal']}`",
        f"- best blended C1 artifact: `{metrics['corrector_paths']['c1_blend_best']}`",
        f"- best blended C1 strength: `{metrics['best_c1_blend_strength']}`",
        f"- best overall artifact: `{metrics['corrector_paths']['best_candidate']}`",
        f"- best overall source: `{metrics['best_candidate']['source_name']}` / `{metrics['best_candidate']['kind']}`",
        "",
        "## Held-Out Metrics",
        "",
    ]
    for depth in sorted(metrics["models"]["baseline"]["heldout"]["by_depth"], key=lambda value: int(value)):
        lines.extend([f"### Depth {depth}", "", _table(metrics, split="heldout", depth=depth), ""])

    lines.extend(
        [
            "## Train Metrics",
            "",
        ]
    )
    for depth in sorted(metrics["models"]["baseline"]["train"]["by_depth"], key=lambda value: int(value)):
        lines.extend([f"### Depth {depth}", "", _table(metrics, split="train", depth=depth), ""])

    lines.extend(
        [
            "## Blend Sweep",
            "",
            "| candidate | kind | D2 top1 | D2 top4 | D2 top8 | D2 median rank | D2 proxy acc |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["blend_candidates"]:
        d2 = item["heldout_by_depth"].get("2", {})
        proxy = d2.get("proxy_temp_acceptance_mean")
        if item["kind"] == "c1_blend":
            label = f"blend={item['blend_strength']:.3f}"
        else:
            label = f"rank={item['rank']}"
        lines.append(
            "| {label} | {kind} | {top1} | {top4} | {top8} | {rank} | {proxy} |".format(
                label=label,
                kind=item["kind"],
                top1="n/a" if "top1" not in d2 else f"{d2['top1']:.4f}",
                top4="n/a" if "top4_contains" not in d2 else f"{d2['top4_contains']:.4f}",
                top8="n/a" if "top8_contains" not in d2 else f"{d2['top8_contains']:.4f}",
                rank="n/a" if "target_rank_median" not in d2 else f"{d2['target_rank_median']:.1f}",
                proxy="n/a" if proxy is None else f"{proxy:.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- runtime integration gate: `{decision['runtime_integration_gate']}`",
        ]
    )
    for key, value in decision.items():
        if key == "runtime_integration_gate":
            continue
        lines.append(f"- {key}: `{value}`")
    if decision["runtime_integration_gate"]:
        lines.append("- next step: integrate the winning blended C1 behind an explicit runtime flag and test D2/D3 graphbank_capture_commit.")
    else:
        lines.append("- next step: do not integrate the offline corrector yet; move objective toward CE/KL/margin on next-step logits.")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calib_npz", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("outputs/correctors"))
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampler-temperature", type=float, default=None)
    parser.add_argument("--all-rows", action="store_true", help="fit/evaluate every row instead of prefix-active rows only")
    parser.add_argument("--blend-strengths", default="0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1")
    parser.add_argument("--c2-ranks", default="", help="comma-separated low-rank C2 residual ranks to evaluate")
    parser.add_argument("--c2-ridge", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = fit_and_evaluate(
        args.calib_npz,
        model_path=args.model,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
        sampler_temperature=args.sampler_temperature,
        prefix_active_only=not args.all_rows,
        blend_strengths=_parse_float_csv(args.blend_strengths),
        c2_ranks=_parse_int_csv(args.c2_ranks) if args.c2_ranks else [],
        c2_ridge=args.c2_ridge,
    )
    print(
        json.dumps(
            {
                "run_id": metrics["run_id"],
                "decision": metrics["decision"],
                "metrics_path": metrics["metrics_path"],
                "report_path": metrics["report_path"],
                "corrector_paths": metrics["corrector_paths"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
