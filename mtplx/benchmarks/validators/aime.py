"""AIME answer extractor and grader.

Pure-stdlib module. Given a model's full response text for an AIME problem,
extract the integer answer the model committed to (preferring ``\\boxed{N}``,
falling back to natural-language ``final answer is N`` phrasing) and grade
it against the gold integer.

AIME answers are integers in ``[0, 999]`` per the MAA contest spec. Anything
outside that range is preserved as an int (so the runner can distinguish
"the model said 1487" from "the model said nothing parseable") but graded as
wrong by ``grade()``.

Conventions
-----------
- ``extract_boxed(text)`` returns the LAST ``\\boxed{...}`` payload that
  parses as an integer. We take the LAST match because reasoning models
  often box several scratch values before committing to the final answer.
- LaTeX-spaced variants ``\\boxed{\\,247\\,}`` and ``\\boxed{ 247 }`` work.
- Fallback: scan the tail of the text for ``final answer is X`` /
  ``answer: X`` / ``answer is X`` (case-insensitive).
- Failure returns ``None`` (the runner reports ``status: "abstain"``).
"""

from __future__ import annotations

import re
from typing import Literal


__all__ = [
    "GradeStatus",
    "extract_boxed",
    "grade",
]


GradeStatus = Literal["correct", "wrong", "abstain"]


# ``\boxed{ ... }`` body: optional LaTeX thin-space ``\,``, optional whitespace,
# optional sign, 1-4 digits, optional whitespace, optional ``\,``. We use a
# non-greedy permissive body capture then re-parse to int so we don't match
# things like ``\boxed{x+1}`` as 1.
_BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*\\?,?\s*(-?\d{1,4})\s*\\?,?\s*\}",
    re.IGNORECASE,
)

# Fallback when no ``\boxed{...}`` exists. Scans the tail only (last 400 chars)
# so we don't accidentally grab a number from the middle of the reasoning.
_FALLBACK_RE = re.compile(
    r"(?:final\s+answer\s+is|the\s+answer\s+is|answer\s+is|answer\s*[:=])"
    r"\s*\$?\s*(-?\d{1,4})",
    re.IGNORECASE,
)

_FALLBACK_TAIL_CHARS = 400


def extract_boxed(text: str) -> int | None:
    """Extract the model's committed integer answer.

    Prefers ``\\boxed{N}``, falls back to a natural-language tail scan.
    Returns the integer (possibly outside ``[0, 999]``) or ``None``.

    The integer is intentionally NOT clamped here so the runner can record
    extracted values for diagnostics. Grade-correctness happens in
    :func:`grade`.
    """
    if not isinstance(text, str) or not text:
        return None

    matches = _BOXED_RE.findall(text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass

    tail = text[-_FALLBACK_TAIL_CHARS:]
    tail_matches = _FALLBACK_RE.findall(tail)
    if tail_matches:
        try:
            return int(tail_matches[-1])
        except ValueError:
            pass

    return None


def grade(extracted: int | None, expected: int) -> GradeStatus:
    """Grade a single AIME problem.

    - ``extracted is None``     -> ``"abstain"`` (no parseable answer)
    - extracted == expected     -> ``"correct"``
    - else                      -> ``"wrong"`` (includes out-of-range ints)
    """
    if extracted is None:
        return "abstain"
    if extracted == expected:
        return "correct"
    return "wrong"
