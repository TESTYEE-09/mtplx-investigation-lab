from __future__ import annotations

import numpy as np

from mtplx.trace_parity import compare_traces, safe_array_key


def _trace(prompt_tokens, key, value):
    return (
        {
            "source": "unit",
            "prompt_token_ids": prompt_tokens,
            "records": [
                {
                    "prompt_id": "p",
                    "window_index": 0,
                    "mode": "recursive",
                    "depth": 1,
                    "arrays": {"hidden_in": key},
                }
            ],
        },
        {key: np.asarray(value, dtype=np.float32)},
    )


def test_compare_traces_detects_token_mismatch() -> None:
    left_meta, left_arrays = _trace([1, 2, 3], "a", [1.0])
    right_meta, right_arrays = _trace([1, 2, 4], "b", [1.0])
    result = compare_traces(left_meta, left_arrays, right_meta, right_arrays)
    assert result["token_ids_match"] is False


def test_compare_traces_reports_first_array_divergence() -> None:
    left_meta, left_arrays = _trace([1, 2, 3], "a", [1.0, 2.0])
    right_meta, right_arrays = _trace([1, 2, 3], "b", [1.0, 2.5])
    result = compare_traces(left_meta, left_arrays, right_meta, right_arrays, atol=1e-6)
    assert result["token_ids_match"] is True
    assert result["first_divergence"]["boundary"] == "hidden_in"
    assert result["first_divergence"]["max_abs_error"] == 0.5


def test_safe_array_key_is_npz_friendly() -> None:
    assert safe_array_key("p/1", "window 0", "recursive", 1, "hidden:in") == "p_1.window_0.recursive.1.hidden_in"
