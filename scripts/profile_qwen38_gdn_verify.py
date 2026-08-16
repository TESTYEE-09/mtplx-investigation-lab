#!/usr/bin/env python3
"""Fine-grained Qwen3.8 capture-commit verification profiler.

This intentionally has two lanes:

* ``unsplit`` runs the production ``forward_ar_capture`` graph with its single
  output realization.  Those numbers are suitable for absolute verify latency.
* ``synchronized`` rebuilds the same forward with explicit realization at
  operation boundaries.  It perturbs scheduling, so its numbers are suitable
  only for attribution.  The exact production kernels and cache mutations are
  retained.

The script never writes into the model directory.  All output paths are chosen
by the caller and are expected to live in the investigation repository.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx


VERIFY_BACKEND = "linear_gdn_from_conv_tape"


def _eval_timed(*values: Any) -> float:
    started = time.perf_counter()
    mx.eval(*values)
    return time.perf_counter() - started


def _prefill(rt: Any, prompt_ids: list[int]) -> Any:
    cache = rt.make_cache()
    logits = rt.forward_ar(
        mx.array([prompt_ids], dtype=mx.int32), cache=cache, return_hidden=False
    )
    mx.eval(logits)
    return cache


def _candidate_ids(tokenizer: Any, count: int) -> list[int]:
    ids = list(
        tokenizer.encode(
            "Profile the recurrent verification path precisely and report evidence."
        )
    )
    if not ids:
        raise RuntimeError("tokenizer produced no candidate ids")
    while len(ids) < count:
        ids.extend(ids)
    return [int(value) for value in ids[:count]]


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return [int(value) for value in rendered]
    except Exception:
        return [int(value) for value in tokenizer.encode(prompt)]


class Attribution:
    def __init__(self) -> None:
        self.sections: dict[str, float] = defaultdict(float)
        self.layers: list[dict[str, Any]] = []
        self.eval_calls = 0
        self.eval_wall_s = 0.0

    def realize(self, section: str, *values: Any) -> float:
        elapsed = _eval_timed(*values)
        self.sections[section] += elapsed
        self.eval_calls += 1
        self.eval_wall_s += elapsed
        return elapsed

    def operation(
        self,
        section: str,
        layer: dict[str, Any],
        fn: Callable[[], Any],
    ) -> Any:
        build_started = time.perf_counter()
        value = fn()
        build_elapsed = time.perf_counter() - build_started
        values = value if isinstance(value, tuple) else (value,)
        eval_elapsed = self.realize(section, *values)
        layer[section] = eval_elapsed
        layer[f"{section}_build_s"] = build_elapsed
        return value


def _synchronized_capture_forward(
    model: Any,
    inputs: mx.array,
    cache: Any,
) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]], Attribution]:
    """Same capture graph as production, split only by diagnostic mx.eval calls."""
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.gated_delta import compute_g

    from mtplx.gdn_capture import (
        _gdn_input_projections,
        _linear_gated_delta_from_conv_tape_capture,
        _maybe_contiguous_authoritative_gdn_leaf,
        _stock_conv1d_capture,
    )

    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    acc = Attribution()
    hidden_states = inner.embed_tokens(inputs)
    acc.realize("embedding_s", hidden_states)
    fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
    captures: dict[int, dict[str, mx.array]] = {}

    for layer_idx, (decoder, layer_cache) in enumerate(zip(inner.layers, cache)):
        row: dict[str, Any] = {
            "layer": layer_idx,
            "kind": "gdn" if decoder.is_linear else "attention",
        }
        layer_started = time.perf_counter()
        normed = acc.operation(
            "input_norm_s", row, lambda decoder=decoder: decoder.input_layernorm(hidden_states)
        )
        if decoder.is_linear:
            gdn = decoder.linear_attn
            B, S, _ = normed.shape
            qkv, z, b, a = acc.operation(
                "gdn_input_projections_s",
                row,
                lambda gdn=gdn: _gdn_input_projections(gdn, normed),
            )
            z = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
            conv_state = layer_cache[0]
            state = layer_cache[1]
            # Both leaves are already realized after prefill.  This measures
            # host synchronization/materialization only; device state reads
            # themselves occur in the recurrence kernel and are counted there.
            acc.realize("gdn_recurrent_state_ready_sync_s", conv_state, state)
            row["gdn_recurrent_state_ready_sync_s"] = acc.sections[
                "gdn_recurrent_state_ready_sync_s"
            ] - sum(
                item.get("gdn_recurrent_state_ready_sync_s", 0.0)
                for item in acc.layers
            )
            conv_out, conv_states = acc.operation(
                "gdn_convolution_s",
                row,
                lambda gdn=gdn: _stock_conv1d_capture(qkv, conv_state, gdn),
            )
            g, beta = acc.operation(
                "gdn_gates_s",
                row,
                lambda gdn=gdn: (
                    compute_g(gdn.A_log, a, gdn.dt_bias),
                    mx.sigmoid(b),
                ),
            )
            out, final_state, tape = acc.operation(
                "gdn_recurrence_s",
                row,
                lambda gdn=gdn: _linear_gated_delta_from_conv_tape_capture(
                    conv_out, g, beta, state, gdn
                ),
            )
            update_started = time.perf_counter()
            new_conv = mx.contiguous(conv_states[:, -1, :, :])
            new_state = _maybe_contiguous_authoritative_gdn_leaf(final_state)
            layer_cache[0] = new_conv
            layer_cache[1] = new_state
            layer_cache.advance(S)
            row["gdn_recurrent_state_update_build_s"] = (
                time.perf_counter() - update_started
            )
            row["gdn_recurrent_state_update_s"] = acc.realize(
                "gdn_recurrent_state_update_s", new_conv, new_state
            )
            out = acc.operation(
                "gdn_norm_gate_s", row, lambda gdn=gdn: gdn.norm(out, z)
            )
            r = acc.operation(
                "gdn_output_projection_s",
                row,
                lambda gdn=gdn: gdn.out_proj(out.reshape(B, S, -1)),
            )
            captures[layer_idx] = {
                "conv_states": conv_states,
                "conv_out": conv_out,
                "g": g,
                "state_in": state,
                "tape": tape,
                "gdn_meta": {
                    "head_k_dim": int(gdn.head_k_dim),
                    "head_v_dim": int(gdn.head_v_dim),
                    "num_k_heads": int(gdn.num_k_heads),
                    "num_v_heads": int(gdn.num_v_heads),
                    "key_dim": int(gdn.key_dim),
                    "conv_dim": int(gdn.conv_dim),
                },
            }
        else:
            r = acc.operation(
                "attention_work_s",
                row,
                lambda decoder=decoder: decoder.self_attn(
                    normed, mask=fa_mask, cache=layer_cache
                ),
            )
        h = acc.operation("residual_s", row, lambda: hidden_states + r)
        mlp_input = acc.operation(
            "post_attention_norm_s",
            row,
            lambda decoder=decoder: decoder.post_attention_layernorm(h),
        )
        mlp_out = acc.operation("mlp_s", row, lambda decoder=decoder: decoder.mlp(mlp_input))
        hidden_states = acc.operation("mlp_residual_s", row, lambda: h + mlp_out)
        row["diagnostic_layer_wall_s"] = time.perf_counter() - layer_started
        acc.layers.append(row)

    post_norm = inner.norm(hidden_states)
    acc.realize("final_norm_s", post_norm)
    logits = (
        inner.embed_tokens.as_linear(post_norm)
        if text_model.args.tie_word_embeddings
        else text_model.lm_head(post_norm)
    )
    acc.realize("lm_head_s", logits)
    return logits, post_norm, captures, acc


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _summarize_attributions(runs: list[Attribution]) -> dict[str, Any]:
    section_names = sorted({name for run in runs for name in run.sections})
    sections = {
        name: _median([run.sections.get(name, 0.0) for run in runs])
        for name in section_names
    }
    layer_ids = sorted({int(row["layer"]) for run in runs for row in run.layers})
    layers = []
    for layer_id in layer_ids:
        rows = [
            next(row for row in run.layers if int(row["layer"]) == layer_id)
            for run in runs
        ]
        keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if key.endswith("_s") and isinstance(value, (int, float))
            }
        )
        layers.append(
            {
                "layer": layer_id,
                "kind": rows[0]["kind"],
                **{key: _median([float(row.get(key, 0.0)) for row in rows]) for key in keys},
            }
        )
    return {
        "sections_median_s": sections,
        "layers_median": layers,
        "eval_calls": int(_median([float(run.eval_calls) for run in runs])),
        "eval_wall_median_s": _median([run.eval_wall_s for run in runs]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.profiles import apply_profile_env
    from mtplx.runtime import load

    if args.runtime_profile != "none":
        apply_profile_env(args.runtime_profile)
    os.environ["MTPLX_MTP_HISTORY_POLICY"] = "committed"
    rt = load(args.model, mtp=True)
    prompt_ids = _prompt_ids(rt.tokenizer, args.prompt)
    candidates = _candidate_ids(rt.tokenizer, 4)
    inner = rt.model.language_model.model
    topology = ["gdn" if layer.is_linear else "attention" for layer in inner.layers]
    result: dict[str, Any] = {
        "model": str(args.model),
        "runtime_profile": args.runtime_profile,
        "device": mx.device_info(),
        "mlx_backend": VERIFY_BACKEND,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "mtp_history_policy": "committed",
        "prompt_tokens": len(prompt_ids),
        "topology": {
            "layers": len(topology),
            "gdn_layers": topology.count("gdn"),
            "attention_layers": topology.count("attention"),
            "pattern": topology,
        },
        "rows": [],
    }

    for depth in (1, 2, 3):
        M = depth + 1
        verify_ids = mx.array([candidates[:M]], dtype=mx.int32)
        # Compile/warm the exact production graph and its fixed M specialization.
        for _ in range(args.warmup):
            cache = _prefill(rt, prompt_ids)
            logits, hidden, captures = rt.forward_ar_capture(
                verify_ids,
                cache=cache,
                return_hidden=True,
                hidden_variant="post_norm",
                capture_backend=VERIFY_BACKEND,
            )
            mx.eval(logits, hidden, *[v for c in captures.values() if isinstance(c, dict) for v in c.values() if isinstance(v, mx.array)])

        unsplit = []
        hidden_only = []
        build_times = []
        peak_bytes = []
        for _ in range(args.repeats):
            cache = _prefill(rt, prompt_ids)
            mx.reset_peak_memory()
            build_started = time.perf_counter()
            logits, hidden, captures = rt.forward_ar_capture(
                verify_ids,
                cache=cache,
                return_hidden=True,
                hidden_variant="post_norm",
                capture_backend=VERIFY_BACKEND,
            )
            build_times.append(time.perf_counter() - build_started)
            materialized = [logits, hidden]
            for capture in captures.values():
                if isinstance(capture, dict):
                    materialized.extend(
                        value for value in capture.values() if isinstance(value, mx.array)
                    )
            unsplit.append(_eval_timed(*materialized))
            peak_bytes.append(int(mx.get_peak_memory()))

            # Product profiles set MTPLX_LAZY_VERIFY_LOGITS=1.  The named
            # verify_eval boundary therefore realizes hidden only; captures
            # remain lazy until the selected prefix is committed and logits
            # are realized later by target-distribution sampling.
            cache = _prefill(rt, prompt_ids)
            _, hidden_lazy, _ = rt.forward_ar_capture(
                verify_ids,
                cache=cache,
                return_hidden=True,
                hidden_variant="post_norm",
                capture_backend=VERIFY_BACKEND,
            )
            hidden_only.append(_eval_timed(hidden_lazy))

        attributions = []
        for _ in range(args.attribution_repeats):
            cache = _prefill(rt, prompt_ids)
            _, _, _, acc = _synchronized_capture_forward(rt.model, verify_ids, cache)
            attributions.append(acc)

        result["rows"].append(
            {
                "depth": depth,
                "M": M,
                "unsplit_graph_build_median_s": _median(build_times),
                "unsplit_mx_eval_median_s": _median(unsplit),
                "unsplit_mx_eval_samples_s": unsplit,
                "hidden_only_mx_eval_median_s": _median(hidden_only),
                "hidden_only_mx_eval_samples_s": hidden_only,
                "active_memory_bytes": int(mx.get_active_memory()),
                "peak_memory_median_bytes": int(_median([float(v) for v in peak_bytes])),
                "attribution": _summarize_attributions(attributions),
            }
        )

    # Incremental costs are the cleanest per-speculative-row estimate because
    # each production kernel keeps T sequential inside one fixed-M launch.
    for previous, current in zip(result["rows"], result["rows"][1:]):
        current["incremental_vs_previous"] = {
            "extra_rows": int(current["M"] - previous["M"]),
            "unsplit_mx_eval_s": float(current["unsplit_mx_eval_median_s"])
            - float(previous["unsplit_mx_eval_median_s"]),
            "sections_s": {
                name: float(current["attribution"]["sections_median_s"].get(name, 0.0))
                - float(previous["attribution"]["sections_median_s"].get(name, 0.0))
                for name in sorted(
                    set(current["attribution"]["sections_median_s"])
                    | set(previous["attribution"]["sections_median_s"])
                )
            },
        }
    if args.trace_output is not None:
        trace_path = args.trace_output.resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        verify_ids = mx.array([candidates[:4]], dtype=mx.int32)
        cache = _prefill(rt, prompt_ids)
        mx.metal.start_capture(str(trace_path))
        try:
            logits, hidden, _ = rt.forward_ar_capture(
                verify_ids,
                cache=cache,
                return_hidden=True,
                hidden_variant="post_norm",
                capture_backend=VERIFY_BACKEND,
            )
            mx.eval(logits, hidden)
        finally:
            mx.metal.stop_capture()
        result["metal_trace"] = str(trace_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Write a compact Python function that merges two sorted lists.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attribution-repeats", type=int, default=2)
    parser.add_argument(
        "--runtime-profile",
        default="none",
        help="Optional MTPLX profile overlay; 'none' preserves the requested core alone.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
