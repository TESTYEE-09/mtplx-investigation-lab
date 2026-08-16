"""Schema + sanity tests for the AIME 2026 benchmark dataset.

Verifies that ``mtplx/benchmarks/prompts/aime_2026.jsonl`` is well-formed:
exactly 30 rows, 15 per set, unique ids, integer answers in [0, 999], and
non-empty problem text. Cheap to run on every CI invocation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "mtplx"
    / "benchmarks"
    / "prompts"
    / "aime_2026.jsonl"
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    assert DATASET_PATH.is_file(), f"missing dataset: {DATASET_PATH}"
    out: list[dict] = []
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                pytest.fail(f"{DATASET_PATH}:{line_no} not valid JSON: {exc}")
    return out


def test_row_count(rows: list[dict]) -> None:
    assert len(rows) == 30, f"expected 30 rows, got {len(rows)}"


def test_set_split(rows: list[dict]) -> None:
    by_set: dict[str, list[dict]] = {"AIME I": [], "AIME II": []}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r)
    assert sorted(by_set) == ["AIME I", "AIME II"], (
        f"unexpected set names: {sorted(by_set)}"
    )
    assert len(by_set["AIME I"]) == 15, len(by_set["AIME I"])
    assert len(by_set["AIME II"]) == 15, len(by_set["AIME II"])


def test_unique_ids(rows: list[dict]) -> None:
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 30, f"duplicate ids: {len(ids) - len(set(ids))}"


def test_id_format(rows: list[dict]) -> None:
    for r in rows:
        tag = "I" if r["set"] == "AIME I" else "II"
        expected = f"2026-{tag}-{r['index']}"
        assert r["id"] == expected, f"id mismatch: {r['id']} vs {expected}"


def test_index_ranges(rows: list[dict]) -> None:
    aime1 = sorted([r["index"] for r in rows if r["set"] == "AIME I"])
    aime2 = sorted([r["index"] for r in rows if r["set"] == "AIME II"])
    assert aime1 == list(range(1, 16))
    assert aime2 == list(range(1, 16))


def test_year(rows: list[dict]) -> None:
    for r in rows:
        assert r["year"] == 2026, f"year != 2026 for {r['id']}"


def test_answer_type_and_range(rows: list[dict]) -> None:
    for r in rows:
        ans = r["answer"]
        assert isinstance(ans, int), f"{r['id']} answer not int: {type(ans)}"
        assert 0 <= ans <= 999, f"{r['id']} answer out of [0,999]: {ans}"


def test_problem_non_empty(rows: list[dict]) -> None:
    for r in rows:
        text = r["problem"].strip()
        assert text, f"{r['id']} has empty problem"
        # AIME problems are dense; bare-minimum sanity check.
        assert len(text) >= 40, f"{r['id']} problem suspiciously short: {text!r}"


def test_source_present(rows: list[dict]) -> None:
    for r in rows:
        assert r["source"].startswith("https://"), f"{r['id']} bad source url"


def test_required_keys(rows: list[dict]) -> None:
    required = {"id", "set", "year", "index", "problem", "answer", "source"}
    for r in rows:
        missing = required - set(r)
        assert not missing, f"{r.get('id', '<unknown>')} missing keys: {missing}"


# Canonical answer key pinned to the MathArena AIME 2026 release (the
# academic NeurIPS '25 benchmark dataset). Differs from the AoPS Wiki US
# AIME answer key in exactly two cells:
#
#   - AIME II Q1: 178 (MathArena, international wording "sum of 10th terms
#     of arithmetic sequences ...") vs AoPS 196 (US wording).
#   - AIME II Q10: 850 (MathArena, international wording "sum of all
#     possible values of BC") vs AoPS 340 (US wording "greatest possible
#     value of BC"). AoPS Solution 2 of Problem 10 explicitly notes
#     the international variant gives 85 * (1+2+3+4) = 850.
#
# The dataset's `problem` field uses MathArena's wording, so this is
# the correct canonical answer key for THIS dataset. Any future drift
# in the JSONL must update this list in lock step.
EXPECTED_MATHARENA_2026 = [
    # AIME I (1..15)
    277, 62, 79, 70, 65, 441, 396, 244, 29, 156, 896, 161, 39, 681, 83,
    # AIME II (1..15)
    178, 243, 503, 279, 190, 50, 754, 245, 669, 850, 132, 223, 107, 157, 393,
]


def test_pinned_canonical_answers(rows: list[dict]) -> None:
    """The 30 answers must exactly match the MathArena canonical list.

    Fails loudly on any silent drift in `aime_2026.jsonl`.
    """
    assert len(rows) == 30
    actual = [r["answer"] for r in rows]
    assert actual == EXPECTED_MATHARENA_2026, (
        "AIME 2026 answers drifted from the MathArena canonical key. "
        "Diff (idx: actual vs expected):\n"
        + "\n".join(
            f"  {i}: {a} vs {e}"
            for i, (a, e) in enumerate(zip(actual, EXPECTED_MATHARENA_2026))
            if a != e
        )
    )
