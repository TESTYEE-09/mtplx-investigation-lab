from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


M5_PP_GATES = {32768: 500.0, 65536: 325.0, 131072: 240.0}
M3_ULTRA_PP_GATES = {32768: 375.0, 65536: 300.0, 131072: 220.0}
M5_MEMORY_GATES_GB = {32768: 35.0, 65536: 50.0, 131072: 75.0}


def _load_artifact() -> dict:
    path = os.environ.get("MTPLX_PREFILL_TPS_ARTIFACT")
    if not path:
        pytest.skip("set MTPLX_PREFILL_TPS_ARTIFACT to validate a prefill ladder artifact")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _machine_gates(payload: dict) -> dict[int, float]:
    chip = str((payload.get("hardware") or {}).get("chip") or "").lower()
    if "m3 ultra" in chip:
        return M3_ULTRA_PP_GATES
    return M5_PP_GATES


def test_prefill_ladder_artifact_schema():
    payload = _load_artifact()

    assert payload["kind"] == "prefill_ladder"
    assert isinstance(payload.get("rows"), list)
    assert payload.get("hardware")
    assert payload.get("prompt", {}).get("policy") == "coding_agent_tail_v2"
    assert payload.get("prompt", {}).get("format") == "chat"
    assert payload.get("prompt", {}).get("enable_thinking") is False
    assert payload.get("prompt", {}).get("tail_sha256")
    assert payload.get("prompt", {}).get("release_valid") is True
    required = {
        "context_tokens",
        "prompt_tps",
        "ttft_s",
        "decode_tok_s",
        "generated_tokens",
        "accepted_drafts",
        "drafted_tokens",
        "draft_acceptance_rate",
        "verify_calls",
        "verify_time_s",
        "draft_time_s",
        "peak_memory_gb",
        "prompt_target_prefill_time_s",
        "prompt_mtp_history_time_s",
        "mtp_history_policy",
        "mtp_history_window_tokens",
        "prompt_policy",
        "prompt_style",
        "prompt_format",
        "prompt_enable_thinking",
        "prompt_tail_sha256",
        "prompt_tail_preserved",
        "prompt_release_valid",
        "large_q_split_sdpa_fallback_calls",
        "partitioned_paged_calls",
    }
    for row in payload["rows"]:
        assert required <= set(row)
        assert row["prompt_policy"] == "coding_agent_tail_v2"
        assert row["prompt_style"] == "coding-agent"
        assert row["prompt_format"] == "chat"
        assert row["prompt_enable_thinking"] is False
        assert row["prompt_tail_sha256"] == payload["prompt"]["tail_sha256"]
        assert row["prompt_tail_preserved"] is True
        assert row["prompt_release_valid"] is True


def test_prefill_ladder_artifact_release_gates():
    payload = _load_artifact()
    rows = {int(row["context_tokens"]): row for row in payload.get("rows") or []}
    gates = _machine_gates(payload)
    missing = [context for context in gates if context not in rows]
    if missing and os.environ.get("MTPLX_PREFILL_TPS_REQUIRE_FULL") == "1":
        pytest.fail(f"artifact does not include release contexts: {missing}")
    contexts_to_validate = [context for context in gates if context in rows]
    if not contexts_to_validate:
        pytest.skip(f"artifact does not include release contexts: {missing}")

    for context in contexts_to_validate:
        min_pp = gates[context]
        row = rows[context]
        route = str(row.get("prefill_route") or "")
        assert int(row["generated_tokens"]) == int(payload["max_tokens"])
        assert float(row["prompt_tps"]) >= min_pp
        assert int(row.get("large_q_split_sdpa_fallback_calls") or 0) == 0
        assert int(row.get("prefill_large_q_split_sdpa_fallback_calls") or 0) == 0
        partitioned_calls = int(row.get("partitioned_paged_calls") or 0)
        if route in {"contiguous_then_repage", "contiguous_dense_decode"}:
            assert partitioned_calls == 0
        else:
            assert partitioned_calls > 0
        if context in M5_MEMORY_GATES_GB:
            assert float(row.get("peak_memory_gb") or 0.0) <= M5_MEMORY_GATES_GB[context]
