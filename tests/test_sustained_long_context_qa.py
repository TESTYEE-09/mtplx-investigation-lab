from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest


def _require_long_context_qa() -> tuple[Path, Path]:
    if os.environ.get("MTPLX_RUN_LONG_CONTEXT_QA") not in {"1", "true", "yes", "on"}:
        pytest.skip("set MTPLX_RUN_LONG_CONTEXT_QA=1 to run Apple Silicon long-context QA")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("long-context QA is only defined for Apple Silicon")
    artifact = Path(os.environ.get("MTPLX_LONG_CONTEXT_QA_ARTIFACT", "")).expanduser()
    if not artifact.exists():
        pytest.skip("set MTPLX_LONG_CONTEXT_QA_ARTIFACT to the real 16k/32k QA JSON")
    baseline = Path(os.environ.get("MTPLX_QA_BURST_BASELINE", "")).expanduser()
    if not baseline.exists():
        pytest.skip("set MTPLX_QA_BURST_BASELINE to a local Burst baseline JSON")
    return artifact, baseline


def _rows_by_context(summary: dict) -> dict[int, dict]:
    candidate_rows = summary.get("rows")
    if not isinstance(candidate_rows, list):
        candidate_rows = []
        for key, value in summary.items():
            if key.endswith("_row") and isinstance(value, dict):
                candidate_rows.append(value)
            if key.startswith("sustained_") and isinstance(value, dict):
                candidate_rows.append(value)
    rows: dict[int, dict] = {}
    for row in candidate_rows:
        prompt_tokens = int(row.get("prompt_tokens") or row.get("context") or 0)
        if prompt_tokens:
            rows[prompt_tokens] = row
    return rows


def test_sustained_32k_memory_and_16k_speed_qa():
    artifact_path, baseline_path = _require_long_context_qa()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    burst_16k_decode = float(baseline["burst_16k_decode_tok_s"])
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = _rows_by_context(summary)
    row_16k = min(rows.values(), key=lambda row: abs(int(row.get("prompt_tokens", 0)) - 16384))
    row_32k = min(rows.values(), key=lambda row: abs(int(row.get("prompt_tokens", 0)) - 32768))

    assert float(row_32k["peak_memory_gb"]) < 35.0
    decode_16k = float(row_16k.get("decode_tok_s") or row_16k.get("generation_tps"))
    assert decode_16k >= 0.85 * burst_16k_decode
    assert int(row_16k.get("decode_dense_fallback_calls") or 0) == 0
    assert int(row_32k.get("decode_dense_fallback_calls") or 0) == 0
    assert int(row_32k.get("full_logits_tokens_emitted") or 0) == 0
