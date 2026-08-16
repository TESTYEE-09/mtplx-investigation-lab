#!/usr/bin/env python3
"""Collect recursive-vs-target-forced native MTP hidden calibration shards."""

from __future__ import annotations

import argparse
import hashlib
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

from mtplx.benchmarks.runners.mtp_chain_probe import (  # noqa: E402
    _input_pair,
    _prefill_prompt,
    _target_trace,
)
from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite, now_run_id  # noqa: E402
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR  # noqa: E402
from mtplx.generation import _sample_from_logits  # noqa: E402
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.runtime import load  # noqa: E402
from mtplx.sampling import SamplerConfig  # noqa: E402


def _topk(logits: mx.array, k: int) -> tuple[np.ndarray, np.ndarray]:
    flat = logits.astype(mx.float32).reshape(-1)
    actual_k = min(int(k), int(flat.shape[0]))
    indices = mx.argpartition(-flat, kth=actual_k - 1, axis=-1)[:actual_k]
    values = flat[indices]
    order = mx.argsort(-values)
    indices = indices[order]
    values = values[order]
    mx.eval(indices, values)
    return (
        np.asarray(indices, dtype=np.int64).reshape(-1),
        np.asarray(values, dtype=np.float32).reshape(-1),
    )


def _rank(logits: mx.array, token: int) -> int:
    flat = logits.astype(mx.float32).reshape(-1)
    value = flat[int(token)]
    rank = mx.sum(flat > value) + 1
    mx.eval(rank)
    return int(rank.item())


def _entropy(logits: mx.array, *, temperature: float) -> float:
    flat = logits.astype(mx.float32).reshape(-1) / float(temperature)
    max_value = mx.max(flat)
    shifted = flat - max_value
    exp_shifted = mx.exp(shifted)
    denom = mx.sum(exp_shifted)
    probs = exp_shifted / denom
    log_probs = shifted - mx.log(denom)
    entropy = -mx.sum(probs * log_probs)
    mx.eval(entropy)
    return float(entropy.item())


def _probs_for_tokens(logits: mx.array, tokens: list[int], *, temperature: float) -> dict[int, float]:
    unique_tokens = sorted({int(token) for token in tokens})
    if not unique_tokens:
        return {}
    flat = logits.astype(mx.float32).reshape(-1) / float(temperature)
    max_value = mx.max(flat)
    denom = max_value + mx.log(mx.sum(mx.exp(flat - max_value)))
    mx.eval(denom)
    denom_value = float(denom.item())
    probs: dict[int, float] = {}
    for token in unique_tokens:
        value = float(flat[token].item())
        probs[token] = float(np.exp(value - denom_value))
    return probs


def _hidden_row(hidden: mx.array) -> np.ndarray:
    mx.eval(hidden)
    arr = np.asarray(hidden.astype(mx.float32), dtype=np.float32)
    return arr.reshape(-1, arr.shape[-1])[-1]


def _cache_offset(mtp_cache) -> int:
    if not mtp_cache:
        return 0
    return int(getattr(mtp_cache[0], "offset", 0))


def _array_signature(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        return None
    return {
        "shape": [int(item) for item in shape],
        "dtype": str(dtype),
    }


def _cache_signature_hash(mtp_cache) -> str:
    entries: list[dict[str, Any]] = []
    for item in mtp_cache or []:
        cache_signatures = []
        for value in getattr(item, "cache", []):
            signature = _array_signature(value)
            if signature is not None:
                cache_signatures.append(signature)
        entries.append(
            {
                "type": type(item).__name__,
                "offset": int(getattr(item, "offset", 0)),
                "keys": _array_signature(getattr(item, "keys", None)),
                "values": _array_signature(getattr(item, "values", None)),
                "cache": cache_signatures,
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_trace_sampled(
    rt,
    ids: list[int],
    *,
    target_token_count: int,
    base_hidden_variant: str,
    include_logits: bool = False,
    sampler: SamplerConfig,
    seed: int,
) -> dict[str, Any]:
    cache, logits, prompt_hidden = _prefill_prompt(
        rt,
        ids,
        base_hidden_variant=base_hidden_variant,
    )
    rng = np.random.default_rng(seed)
    tokens: list[int] = []
    hiddens: list[mx.array] = []
    target_logits: list[mx.array] = []
    for _ in range(target_token_count):
        if include_logits:
            target_logits.append(logits)
        token, _dist = _sample_from_logits(logits[0], sampler, rng)
        tokens.append(int(token))
        logits_next, hidden_next = rt.forward_ar(
            mx.array([[int(token)]]),
            cache=cache,
            return_hidden=True,
            hidden_variant=base_hidden_variant,
        )
        mx.eval(logits_next, hidden_next)
        hiddens.append(hidden_next[:, -1:, :])
        logits = logits_next[:, -1, :]
    return {
        "prompt_len": len(ids),
        "prompt_hidden": prompt_hidden,
        "target_tokens": tokens,
        "target_hiddens": hiddens,
        "target_logits": target_logits,
    }


def _append_topk(target: list[np.ndarray], values: np.ndarray, k: int, *, dtype) -> None:
    padded = np.zeros(k, dtype=dtype)
    padded[: len(values)] = values.astype(dtype, copy=False)
    target.append(padded)


def collect_hidden_calibration(
    model_path: Path | str,
    prompt_suite: Path | str,
    output_dir: Path | str,
    *,
    limit: int | None = 2,
    windows: int = 64,
    stride: int = 1,
    depth: int = 5,
    max_prompt_tokens: int = 256,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
    base_hidden_variant: str = "pre_norm",
    mtp_hidden_variant: str = "pre_norm",
    cache_policy: str = "fresh",
    anchor: str = "prompt_boundary",
    concat_order: str = "embedding_hidden",
    top_k: int = 16,
    sampler_temperature: float = 0.6,
    target_sampler: str = "greedy",
    seed: int = 0,
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
    if mtp_hidden_variant not in {"pre_norm", "post_norm"}:
        raise ValueError("collector C0/C1 calibration expects pre_norm or post_norm MTP hidden")
    if target_sampler not in {"greedy", "stochastic"}:
        raise ValueError("target_sampler must be 'greedy' or 'stochastic'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]

    started = time.perf_counter()
    target_token_count = ((windows - 1) * stride) + depth + 4

    recursive_hidden: list[np.ndarray] = []
    recursive_input_hidden: list[np.ndarray] = []
    target_forced_hidden: list[np.ndarray] = []
    target_next_hidden: list[np.ndarray] = []
    depths: list[int] = []
    prompt_ids: list[str] = []
    prompt_indices: list[int] = []
    window_indices: list[int] = []
    window_starts: list[int] = []
    target_tokens: list[int] = []
    recursive_input_tokens: list[int] = []
    recursive_draft_tokens: list[int] = []
    target_forced_input_tokens: list[int] = []
    target_forced_draft_tokens: list[int] = []
    recursive_target_ranks: list[int] = []
    target_forced_target_ranks: list[int] = []
    recursive_matches: list[bool] = []
    target_forced_matches: list[bool] = []
    recursive_prefix_active: list[bool] = []
    recursive_accepted_through_depth: list[bool] = []
    recursive_prefix_len_before: list[int] = []
    recursive_entropy: list[float] = []
    recursive_margin: list[float] = []
    topk_overlap: list[float] = []
    recursive_q_prob: list[float] = []
    target_ar_p_recursive_draft: list[float] = []
    recursive_mtp_cache_offset_before: list[int] = []
    recursive_mtp_cache_offset_after: list[int] = []
    recursive_mtp_cache_hash_before: list[str] = []
    recursive_mtp_cache_hash_after: list[str] = []
    recursive_top_indices: list[np.ndarray] = []
    recursive_top_values: list[np.ndarray] = []
    target_forced_top_indices: list[np.ndarray] = []
    target_forced_top_values: list[np.ndarray] = []
    target_ar_top_indices: list[np.ndarray] = []
    target_ar_top_values: list[np.ndarray] = []
    target_ar_top_probs: list[np.ndarray] = []
    jsonl_rows: list[dict[str, Any]] = []

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
                include_logits=True,
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
                include_logits=True,
            )

        for window_index in range(windows):
            window_start = window_index * stride
            anchor_offset = window_start + (0 if anchor == "prompt_boundary" else 1)
            rec_hidden, rec_next_token = _input_pair(trace, anchor_offset)
            forced_hidden, forced_next_token = _input_pair(trace, anchor_offset)
            rec_cache = rt.make_mtp_cache() if cache_policy == "persistent" else None
            forced_cache = rt.make_mtp_cache() if cache_policy == "persistent" else None
            prefix_active = True
            prefix_len = 0

            for depth_index in range(depth):
                current_depth = depth_index + 1
                target_index = anchor_offset + depth_index + 1
                target_token = int(trace["target_tokens"][target_index])

                rec_step_cache = rec_cache if cache_policy == "persistent" else rt.make_mtp_cache()
                rec_cache_offset_before = _cache_offset(rec_step_cache)
                rec_cache_hash_before = _cache_signature_hash(rec_step_cache)
                rec_input_hidden_row = _hidden_row(rec_hidden)
                rec_logits, rec_next_hidden = rt.draft_mtp(
                    rec_hidden,
                    mx.array([[rec_next_token]]),
                    mtp_cache=rec_step_cache,
                    concat_order=concat_order,
                    return_hidden=True,
                    mtp_hidden_variant=mtp_hidden_variant,
                )
                forced_step_cache = forced_cache if cache_policy == "persistent" else rt.make_mtp_cache()
                forced_logits, forced_next_hidden = rt.draft_mtp(
                    forced_hidden,
                    mx.array([[forced_next_token]]),
                    mtp_cache=forced_step_cache,
                    concat_order=concat_order,
                    return_hidden=True,
                    mtp_hidden_variant=mtp_hidden_variant,
                )
                mx.eval(rec_logits, rec_next_hidden, forced_logits, forced_next_hidden)
                rec_cache_offset_after = _cache_offset(rec_step_cache)
                rec_cache_hash_after = _cache_signature_hash(rec_step_cache)

                rec_row_logits = rec_logits[:, -1, :][0]
                forced_row_logits = forced_logits[:, -1, :][0]
                target_row_logits = trace["target_logits"][target_index][0]

                rec_top_idx, rec_top_val = _topk(rec_row_logits, top_k)
                forced_top_idx, forced_top_val = _topk(forced_row_logits, top_k)
                target_top_idx, target_top_val = _topk(target_row_logits, top_k)
                target_top_prob_map = _probs_for_tokens(
                    target_row_logits,
                    [int(token) for token in target_top_idx],
                    temperature=sampler_temperature,
                )
                target_top_probs = np.array(
                    [target_top_prob_map[int(token)] for token in target_top_idx],
                    dtype=np.float32,
                )

                rec_draft_token = int(rec_top_idx[0])
                forced_draft_token = int(forced_top_idx[0])
                rec_probs = _probs_for_tokens(
                    rec_row_logits,
                    [rec_draft_token],
                    temperature=sampler_temperature,
                )
                target_probs = _probs_for_tokens(
                    target_row_logits,
                    [rec_draft_token],
                    temperature=sampler_temperature,
                )
                margin = float(rec_top_val[0] - rec_top_val[1]) if len(rec_top_val) > 1 else float("inf")
                overlap = len(set(map(int, rec_top_idx)) & set(map(int, forced_top_idx))) / float(top_k)

                row_index = len(depths)
                recursive_hidden.append(_hidden_row(rec_next_hidden[:, -1:, :]))
                recursive_input_hidden.append(rec_input_hidden_row)
                target_forced_hidden.append(_hidden_row(forced_next_hidden[:, -1:, :]))
                target_next_hidden.append(_hidden_row(trace["target_hiddens"][target_index]))
                depths.append(current_depth)
                prompt_ids.append(case.id)
                prompt_indices.append(prompt_index)
                window_indices.append(window_index)
                window_starts.append(window_start)
                target_tokens.append(target_token)
                recursive_input_tokens.append(int(rec_next_token))
                recursive_draft_tokens.append(rec_draft_token)
                target_forced_input_tokens.append(int(forced_next_token))
                target_forced_draft_tokens.append(forced_draft_token)
                recursive_target_ranks.append(_rank(rec_row_logits, target_token))
                target_forced_target_ranks.append(_rank(forced_row_logits, target_token))
                recursive_match = rec_draft_token == target_token
                recursive_matches.append(recursive_match)
                target_forced_matches.append(forced_draft_token == target_token)
                recursive_prefix_active.append(prefix_active)
                recursive_accepted_through_depth.append(prefix_active and recursive_match)
                recursive_prefix_len_before.append(prefix_len)
                recursive_entropy.append(_entropy(rec_row_logits, temperature=sampler_temperature))
                recursive_margin.append(margin)
                topk_overlap.append(overlap)
                recursive_q_prob.append(rec_probs[rec_draft_token])
                target_ar_p_recursive_draft.append(target_probs[rec_draft_token])
                recursive_mtp_cache_offset_before.append(rec_cache_offset_before)
                recursive_mtp_cache_offset_after.append(rec_cache_offset_after)
                recursive_mtp_cache_hash_before.append(rec_cache_hash_before)
                recursive_mtp_cache_hash_after.append(rec_cache_hash_after)
                _append_topk(recursive_top_indices, rec_top_idx, top_k, dtype=np.int64)
                _append_topk(recursive_top_values, rec_top_val, top_k, dtype=np.float32)
                _append_topk(target_forced_top_indices, forced_top_idx, top_k, dtype=np.int64)
                _append_topk(target_forced_top_values, forced_top_val, top_k, dtype=np.float32)
                _append_topk(target_ar_top_indices, target_top_idx, top_k, dtype=np.int64)
                _append_topk(target_ar_top_values, target_top_val, top_k, dtype=np.float32)
                _append_topk(target_ar_top_probs, target_top_probs, top_k, dtype=np.float32)

                jsonl_rows.append(
                    {
                        "row_index": row_index,
                        "prompt_id": case.id,
                        "prompt_index": prompt_index,
                        "category": case.category,
                        "prompt_tokens": len(ids),
                        "window_index": window_index,
                        "window_start": window_start,
                        "anchor": anchor,
                        "anchor_offset": anchor_offset,
                        "depth": current_depth,
                        "base_hidden_variant": base_hidden_variant,
                        "mtp_hidden_variant": mtp_hidden_variant,
                        "cache_policy": cache_policy,
                        "concat_order": concat_order,
                        "recursive_input_token": int(rec_next_token),
                        "target_forced_input_token": int(forced_next_token),
                        "recursive_draft_token": rec_draft_token,
                        "target_forced_draft_token": forced_draft_token,
                        "target_token": target_token,
                        "recursive_target_rank": recursive_target_ranks[-1],
                        "target_forced_target_rank": target_forced_target_ranks[-1],
                        "recursive_match": recursive_matches[-1],
                        "target_forced_match": target_forced_matches[-1],
                        "recursive_prefix_active": recursive_prefix_active[-1],
                        "recursive_accepted_through_depth": recursive_accepted_through_depth[-1],
                        "recursive_prefix_len_before": recursive_prefix_len_before[-1],
                        "recursive_margin": margin,
                        "recursive_entropy_temp": sampler_temperature,
                        "recursive_entropy": recursive_entropy[-1],
                        "topk_overlap_with_target_forced": overlap,
                        "recursive_q_prob_temp": sampler_temperature,
                        "recursive_q_prob": recursive_q_prob[-1],
                        "target_ar_p_recursive_draft": target_ar_p_recursive_draft[-1],
                        "recursive_mtp_cache_offset_before": rec_cache_offset_before,
                        "recursive_mtp_cache_offset_after": rec_cache_offset_after,
                        "recursive_mtp_cache_hash_before": rec_cache_hash_before,
                        "recursive_mtp_cache_hash_after": rec_cache_hash_after,
                    }
                )

                rec_hidden = rec_next_hidden[:, -1:, :]
                rec_next_token = rec_draft_token
                if prefix_active and recursive_match:
                    prefix_len = current_depth
                else:
                    prefix_active = False
                if depth_index + 1 < depth:
                    forced_hidden, forced_next_token = _input_pair(trace, anchor_offset + depth_index + 1)

    if not depths:
        raise RuntimeError("collector produced no rows")

    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    shard_path = output_dir / "hidden_calib.npz"
    np.savez_compressed(
        shard_path,
        recursive_hidden=np.stack(recursive_hidden).astype(np.float32),
        recursive_input_hidden=np.stack(recursive_input_hidden).astype(np.float32),
        target_forced_hidden=np.stack(target_forced_hidden).astype(np.float32),
        target_next_hidden=np.stack(target_next_hidden).astype(np.float32),
        depths=np.asarray(depths, dtype=np.int16),
        prompt_ids=np.asarray(prompt_ids),
        prompt_indices=np.asarray(prompt_indices, dtype=np.int32),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        window_starts=np.asarray(window_starts, dtype=np.int32),
        target_tokens=np.asarray(target_tokens, dtype=np.int64),
        recursive_input_tokens=np.asarray(recursive_input_tokens, dtype=np.int64),
        recursive_draft_tokens=np.asarray(recursive_draft_tokens, dtype=np.int64),
        target_forced_input_tokens=np.asarray(target_forced_input_tokens, dtype=np.int64),
        target_forced_draft_tokens=np.asarray(target_forced_draft_tokens, dtype=np.int64),
        recursive_target_ranks=np.asarray(recursive_target_ranks, dtype=np.int32),
        target_forced_target_ranks=np.asarray(target_forced_target_ranks, dtype=np.int32),
        recursive_matches=np.asarray(recursive_matches, dtype=bool),
        target_forced_matches=np.asarray(target_forced_matches, dtype=bool),
        recursive_prefix_active=np.asarray(recursive_prefix_active, dtype=bool),
        recursive_accepted_through_depth=np.asarray(recursive_accepted_through_depth, dtype=bool),
        recursive_prefix_len_before=np.asarray(recursive_prefix_len_before, dtype=np.int16),
        recursive_entropy=np.asarray(recursive_entropy, dtype=np.float32),
        recursive_margin=np.asarray(recursive_margin, dtype=np.float32),
        topk_overlap=np.asarray(topk_overlap, dtype=np.float32),
        recursive_q_prob=np.asarray(recursive_q_prob, dtype=np.float32),
        target_ar_p_recursive_draft=np.asarray(target_ar_p_recursive_draft, dtype=np.float32),
        recursive_mtp_cache_offset_before=np.asarray(recursive_mtp_cache_offset_before, dtype=np.int32),
        recursive_mtp_cache_offset_after=np.asarray(recursive_mtp_cache_offset_after, dtype=np.int32),
        recursive_mtp_cache_hash_before=np.asarray(recursive_mtp_cache_hash_before),
        recursive_mtp_cache_hash_after=np.asarray(recursive_mtp_cache_hash_after),
        recursive_top_indices=np.stack(recursive_top_indices).astype(np.int64),
        recursive_top_values=np.stack(recursive_top_values).astype(np.float32),
        target_forced_top_indices=np.stack(target_forced_top_indices).astype(np.int64),
        target_forced_top_values=np.stack(target_forced_top_values).astype(np.float32),
        target_ar_top_indices=np.stack(target_ar_top_indices).astype(np.int64),
        target_ar_top_values=np.stack(target_ar_top_values).astype(np.float32),
        target_ar_top_probs=np.stack(target_ar_top_probs).astype(np.float32),
    )

    rows_by_depth = {
        str(d): int(sum(1 for value in depths if value == d))
        for d in range(1, depth + 1)
    }
    recursive_agreement = {
        str(d): float(np.mean(np.asarray(recursive_matches)[np.asarray(depths) == d]))
        for d in range(1, depth + 1)
    }
    target_forced_agreement = {
        str(d): float(np.mean(np.asarray(target_forced_matches)[np.asarray(depths) == d]))
        for d in range(1, depth + 1)
    }
    prefix_active_rows_by_depth = {
        str(d): int(np.sum(np.asarray(recursive_prefix_active)[np.asarray(depths) == d]))
        for d in range(1, depth + 1)
    }
    accepted_through_rows_by_depth = {
        str(d): int(np.sum(np.asarray(recursive_accepted_through_depth)[np.asarray(depths) == d]))
        for d in range(1, depth + 1)
    }
    metadata = {
        "run_id": output_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "limit": limit,
        "windows": windows,
        "stride": stride,
        "depth": depth,
        "max_prompt_tokens": max_prompt_tokens,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "base_hidden_variant": base_hidden_variant,
        "mtp_hidden_variant": mtp_hidden_variant,
        "cache_policy": cache_policy,
        "anchor": anchor,
        "concat_order": concat_order,
        "top_k": top_k,
        "sampler_temperature": sampler_temperature,
        "target_sampler": target_sampler,
        "target_sampler_seed": seed,
        "mtp_adapter_path": str(mtp_adapter) if mtp_adapter is not None else None,
        "mtp_adapter_merged": bool(merge_mtp_adapter),
        "mtp_adapter_metadata": rt.mtp_adapter_metadata,
        "mtp_adapter_merge_report": rt.mtp_adapter_merge_report,
        "mtp_quant_bits": rt.contract.mtp_quant_bits,
        "mtp_quant_group_size": rt.contract.mtp_quant_group_size,
        "mtp_quant_mode": rt.contract.mtp_quant_mode,
        "mtp_quant_policy": rt.contract.mtp_quant_policy,
        "mtp_prequantized": rt.contract.mtp_prequantized,
        "calibration_schema_version": 2,
        "persistent_replay_fields": [
            "recursive_input_hidden",
            "recursive_input_tokens",
            "recursive_mtp_cache_offset_before",
            "recursive_mtp_cache_offset_after",
            "recursive_mtp_cache_hash_before",
            "recursive_mtp_cache_hash_after",
        ],
        "rows": len(depths),
        "rows_by_depth": rows_by_depth,
        "prefix_active_rows_by_depth": prefix_active_rows_by_depth,
        "accepted_through_rows_by_depth": accepted_through_rows_by_depth,
        "recursive_agreement_by_depth": recursive_agreement,
        "target_forced_agreement_by_depth": target_forced_agreement,
        "elapsed_s": time.perf_counter() - started,
        "npz_path": str(shard_path),
        "rows_jsonl_path": str(rows_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_RUNTIME_MODEL_DIR)
    parser.add_argument("--prompts", type=Path, default=Path("mtplx/benchmarks/prompts/default.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--base-hidden-variant", choices=["pre_norm", "post_norm"], default="pre_norm")
    parser.add_argument("--mtp-hidden-variant", choices=["pre_norm", "post_norm"], default="pre_norm")
    parser.add_argument("--cache-policy", choices=["fresh", "persistent"], default="fresh")
    parser.add_argument("--anchor", choices=["prompt_boundary", "after_one_target"], default="prompt_boundary")
    parser.add_argument("--concat-order", choices=["embedding_hidden", "hidden_embedding"], default="embedding_hidden")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--sampler-temperature", type=float, default=0.6)
    parser.add_argument("--target-sampler", choices=["greedy", "stochastic"], default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mtp-adapter", type=Path, default=None)
    parser.add_argument("--merge-mtp-adapter", action="store_true")
    parser.add_argument("--mtp-quant-bits", type=int, default=None)
    parser.add_argument("--mtp-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--mtp-quant-mode",
        choices=["affine", "symmetric"],
        default="affine",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or Path("outputs/hidden-calib") / now_run_id("hidden-calib")
    metadata = collect_hidden_calibration(
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
        top_k=args.top_k,
        sampler_temperature=args.sampler_temperature,
        target_sampler=args.target_sampler,
        seed=args.seed,
        mtp_adapter=args.mtp_adapter,
        merge_mtp_adapter=args.merge_mtp_adapter,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
