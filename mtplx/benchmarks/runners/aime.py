"""AIME 2026 benchmark runner.

Owns a single in-process state machine that:

1. Loads the 30 AIME 2026 problems from
   ``mtplx/benchmarks/prompts/aime_2026.jsonl``.
2. For each problem, opens a streaming OpenAI-compatible chat completion
   against the local MTPLX server (``POST /v1/chat/completions``,
   ``stream=true``) with a benchmark-owned answer contract.
3. Forwards ``reasoning_content`` and ``content`` deltas as event-stream
   events on an ``asyncio.Queue`` consumed by ``GET .../stream``.
4. Treats only visible ``content`` as the model's submitted answer.
   ``reasoning_content`` is display/debug output and is never a scoring
   channel.
5. Grades the captured answer text via
   :mod:`mtplx.benchmarks.validators.aime` and persists per-question rows
   to ``~/.mtplx/benchmarks/aime/<run_id>.jsonl``.

State machine
-------------

::

    idle -> running -> paused -> running -> done
    running -> cancelled
    running -> error

Concurrency contract: ``MAX_PARALLEL = 1``. ``start`` returns ``409`` if
any other run is in ``running`` or ``paused`` state. Finished runs stay
in the registry for ``RETENTION_S`` seconds for reconnect/replay before
being GC'd.

Pause semantics: pause aborts the in-flight chat completion, leaves the
current question unscored, and waits on that same question until resume.
Resume starts a fresh attempt with a fresh two-message prompt. Cancel
cancels the runner's outer asyncio task, which propagates
``CancelledError`` into the streaming response, freeing the underlying
decode.

Decoupling for tests
--------------------

The runner takes an optional ``chat_stream_factory`` callable so tests
can inject a fake event stream without spinning up a real chat
completion. In production, the factory defaults to an httpx-backed
streamer pointed at the local server.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import datetime as dt
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Any

from mtplx.benchmarks.validators.aime import GradeStatus, extract_boxed, grade


__all__ = [
    "MAX_PARALLEL",
    "RETENTION_S",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_PERSIST_DIR",
    "SYSTEM_PROMPT",
    "USER_PROMPT_SUFFIX",
    "RunState",
    "AIMEProblem",
    "QuestionResult",
    "AIMEAnswerVerification",
    "AIMECapRecovery",
    "AIMEVisibleSubmission",
    "AIMEQuestionRuntime",
    "AIMERunner",
    "ConcurrentRunError",
    "extract_candidate_answer",
    "extract_verified_answer",
    "extract_boxed_submission",
    "extract_declared_answer",
    "aime_verifier_messages",
    "aime_visible_submission_messages",
    "load_dataset",
    "default_chat_stream_factory",
    "list_active_run_id",
    "list_runs",
    "get_run",
    "start_run",
    "stop_runs",
]


logger = logging.getLogger(__name__)


# --- Public configuration -------------------------------------------------

MAX_PARALLEL: int = 1
RETENTION_S: float = 300.0  # 5 minutes
TAIL_BUFFER_EVENTS: int = 256  # events kept for SSE reconnect replay
COMMIT_SCAN_TAIL_CHARS: int = 4096
VISIBLE_SUBMISSION_MAX_TOKENS: int = 0
PROGRESS_TRACE_MILESTONES: tuple[tuple[str, int], ...] = (
    ("first_token", 1),
    ("first_32", 32),
    ("first_64", 64),
    ("first_128", 128),
    ("first_256", 256),
)

DEFAULT_DATASET_PATH: Path = (
    Path(__file__).resolve().parents[1] / "prompts" / "aime_2026.jsonl"
)
DEFAULT_PERSIST_DIR: Path = Path.home() / ".mtplx" / "benchmarks" / "aime"
ANSWER_COMMIT_MARKER: str = "candidate_answer"
VERIFIED_ANSWER_MARKERS: tuple[str, ...] = (
    "verified_answer",
    "corrected_answer",
    "independent_answer",
)
_CANDIDATE_ANSWER_RE = re.compile(
    rf"\b{ANSWER_COMMIT_MARKER}\b\s*[:=]\s*(-?\d{{1,4}})\b",
    re.IGNORECASE,
)
_VERIFIED_ANSWER_RE = re.compile(
    r"\b(?:verified_answer|corrected_answer|independent_answer)\b"
    r"\s*[:=]\s*(-?\d{1,4})\b",
    re.IGNORECASE,
)
_BOXED_SUBMISSION_RE = re.compile(
    r"\\boxed\s*\{\s*\\?,?\s*(-?\d{1,4})\s*\\?,?\s*\}",
    re.IGNORECASE,
)
_STREAM_STABLE_ANSWER_BOUNDARY_RE = re.compile(
    r"\s*(?:[$}]|[.,;:)\]\n\r]|</think>|\\boxed\b)",
    re.IGNORECASE,
)
_STREAM_STABLE_ANSWER_TRAILING_CHARS = frozenset(".$},;:)]\n\r")
_DECLARED_ANSWER_RE = re.compile(
    r"(?:\bfinal\s+answer\b\s*(?:is|=|:)\s*"
    r"|\b(?:the\s+)?answer\b\s*(?:is|=|:)\s*)"
    r"(?:\\boxed\s*\{\s*)?(-?\d{1,4})\b",
    re.IGNORECASE,
)
_VARIABLE_SUM_ANSWER_RE = re.compile(
    r"\b(?:m\s*\+\s*n|p\s*\+\s*q)\s*=\s*\$?\s*"
    r"(?:-?\d{1,4}\s*\+\s*)+-?\d{1,4}\s*=\s*(-?\d{1,4})\b",
    re.IGNORECASE,
)
_TOTAL_COUNT_ARITHMETIC_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:the\s+)?total\s+count\s*(?:is|=|:)\s*\$?\s*"
    r"(?:-?\d{1,4}\s*\+\s*)+-?\d{1,4}\s*=\s*(-?\d{1,4})"
    r"\s*(?:\$|\.)"
)
_TOTAL_NUMBER_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?"
    r"(?:final\s+)?total\s+number\s+of\s+[^.\n]{1,120}?"
    r"\s+(?:is|=|:)\s*\$?\s*(-?\d{1,4})\s*(?:\$|\.)\s*$"
)
_TOTAL_DESIRED_ARITHMETIC_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?"
    r"total\s+(?:good|valid|desired|wanted|acceptable|possible)"
    r"(?:\s+(?:values|integers|numbers|cases|choices|solutions))?"
    r"\s*(?:is|=|:)\s*\$?\s*"
    r"(?P<expr>-?\d{1,4}(?:\s*[+-]\s*-?\d{1,4})+)"
    r"\s*=\s*(?P<result>-?\d{1,4})\s*(?:\$|\.)\s*$"
)
_DIVISOR_COUNT_SCALAR_RE = re.compile(
    r"(?im)^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?"
    r"number\s+of\s+(?:positive\s+)?divisors(?:\s+of\s+[^.\n]{1,140})?"
    r"\s*(?:is|=|:)\s*\$?\s*(?P<result>-?\d{1,4})\s*\$?\.?\s*$"
)
_DIVISOR_COUNT_EXPRESSION_RE = re.compile(
    r"(?im)^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?"
    r"number\s+of\s+(?:positive\s+)?divisors(?:\s+of\s+[^.\n]{1,140})?"
    r"\s+\$?[^.\n]{1,160}=\s*(?P<result>-?\d{1,4})\s*\$?\.?\s*$"
)
_DIVISOR_FINAL_CONTEXT_RE = re.compile(
    r"(?:\bproduct\s+of\s+all\s+possible\s+positive\s+values\b|"
    r"\bproduct\s+[A-Z]\s*=|\bproduct\s*=|"
    r"\ball\s+possible\s+positive\s+values\b|"
    r"\bpositive\s+values\s+of\s+x\b)",
    re.IGNORECASE,
)
_SET_SIZE_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?"
    r"size\s+of\s+(?:this|the)\s+set\s*(?:is|=|:)\s*"
    r"\$?\s*(-?\d{1,4})\s*(?:\$|\.)\s*$"
)
_SET_CARDINALITY_CONTEXT_RE = re.compile(
    r"\b(?:set\s+of\s+(?:values|integers|numbers)"
    r"|valid\s+(?:values|integers|numbers)"
    r"|possible\s+(?:values|integers|numbers)"
    r"|such\s+(?:values|integers|numbers))\b",
    re.IGNORECASE,
)
_COUNT_SUMMARY_HEADER_RE = re.compile(
    r"^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?(?:final\s+)?"
    r"(?:(?:counts?|cases?|contributions?)\s+(?:are|is)\s*:"
    r"|total\s+count\s+is\s+sum\s+over\b[^.\n]{0,160}\.?)\s*$",
    re.IGNORECASE,
)
_COUNT_SUMMARY_ITEM_RE = re.compile(
    r"^\s*[^:\n]{1,80}:\s*\$?\s*-?\d{1,4}\s*\$?\.?\s*$",
    re.IGNORECASE,
)
_COUNT_SUMMARY_TOTAL_RE = re.compile(
    r"^\s*(?:sum|total)\s*(?:=|:|is)\s*\$?\s*"
    r"(?:(?:-?\d{1,4}\s*\+\s*)+-?\d{1,4}\s*=\s*)?"
    r"(-?\d{1,4})\s*\$?\.?\s*$",
    re.IGNORECASE,
)
_SUM_COUNTS_CONTEXT_RE = re.compile(
    r"\b(?:sum|add)\s+(?:the\s+)?counts\b",
    re.IGNORECASE,
)
_TOTAL_SUM_EXPRESSION_RE = re.compile(
    r"^\s*total(?:\s+(?:sum|count))?\s*(?:=|:|is)\s*\$?\s*"
    r"(?P<expr>-?\d{1,4}(?:\s*\+\s*-?\d{1,4})+)\s*\$?\.?\s*$",
    re.IGNORECASE,
)
_ARITHMETIC_STEP_RE = re.compile(
    r"^\s*\$?\s*(?:-?\d{1,4}\s*\+\s*)+-?\d{1,4}\s*=\s*"
    r"(?P<result>-?\d{1,4})\s*\$?\.?\s*$"
)
_COUNT_ARITHMETIC_CONFIRM_RE = re.compile(
    r"^\s*(?:(?:so|therefore|hence)\s+)?(?:(?:the\s+)?"
    r"calculation\s+(?:is\s+)?|sum\s+(?:is|gives)\s+)"
    r"\$?\s*(?P<expr>[0-9+\-()\s]{3,160})=\s*"
    r"(?P<result>-?\d{1,4})\s*\$?"
    r"(?:\s+(?:is\s+)?correct)?\.?\s*$",
    re.IGNORECASE,
)
_COUNT_ARITHMETIC_CONTEXT_RE = re.compile(
    r"\b(?:counts?|cases?|solutions?|total\s+count|sum\s+the\s+counts)\b",
    re.IGNORECASE,
)
_FINAL_COUNT_SCALAR_RE = re.compile(
    r"(?im)^\s*(?:(?:so|therefore|hence)\s+)?(?:the\s+)?(?:final\s+)?"
    r"(?:count|number\s+of\s+"
    r"(?:such|valid|possible|desired|wanted|acceptable)\s+"
    r"(?:values|integers|numbers))"
    r"\s*(?:is|=|:)\s*\$?\s*(?P<result>-?\d{1,4})\s*\$?\.?\s*$"
)
_FINAL_COUNT_CARDINALITY_CONTEXT_RE = re.compile(
    r"(?:\bthe\s+set\s+of\s+valid\b"
    r"|\bset\s+of\s+(?:values|integers|numbers)\b"
    r"|\b(?:valid|possible|such|desired|wanted|acceptable)\s+"
    r"(?:values|integers|numbers)\b"
    r"|\bvalid\s+N\b"
    r"|\bcomposites?\s+minus\s+prime\s+squares?\b"
    r"|\bprime\s+squares?\b)",
    re.IGNORECASE,
)
_FINAL_COUNT_FINALITY_CONTEXT_RE = re.compile(
    r"(?:\bexactly\b"
    r"|\bcovers\s+all\s+cases\b"
    r"|\blogic\s+covers\b"
    r"|\blogic\s+seems\s+solid\b"
    r"|\bno\s+(?:other|more)\b"
    r"|\btherefore\b"
    r"|\bhence\b"
    r"|\bso\s+the\s+set\b"
    r"|\bfinal\s+count\b)",
    re.IGNORECASE,
)
_INTERMEDIATE_COUNT_CONTEXT_RE = re.compile(
    r"\b(?:for|case|subcase|when)\s+"
    r"(?:k|m|n|N|x|this|that|the\s+current)\b",
    re.IGNORECASE,
)

# The solver prompts are a FORMAT CONTRACT only: they tell the model where
# the runner looks for the answer (visible content after </think>) and the
# exact submission lines it parses. They deliberately contain no solution
# strategy, length, style, or verification coaching — the score must
# measure the model, not the prompt. Any rescue passes that can override
# an extracted answer are governed separately by answer_verification /
# cap_recovery, which default to "off" for every model except Gemma-4 in
# non-thinking mode; runs disclose both the exact prompts and that policy
# in their summary payload.
SYSTEM_PROMPT: str = (
    "You are solving AIME problems. The runner scores only visible content "
    "after </think>; it never scores hidden reasoning. Close the reasoning "
    "block with </think> and present your solution in visible content. Do "
    "not put candidate_answer or a boxed final answer inside hidden "
    "reasoning. End visible content with exactly two lines: "
    "candidate_answer=N and \\boxed{N}, where N is the requested integer. "
    "Do not repeat these instructions."
)
USER_PROMPT_SUFFIX: str = (
    "\n\nClose hidden reasoning with </think> and present your solution in "
    "visible content. End with exactly candidate_answer=N and \\boxed{N}."
)
FAST_SYSTEM_PROMPT: str = (
    "You are solving AIME problems. Hidden reasoning is disabled: write "
    "your solution directly in visible content. End with exactly two lines: "
    "candidate_answer=N and \\boxed{N}, where N is the requested integer. "
    "Do not repeat these instructions."
)
FAST_USER_PROMPT_SUFFIX: str = (
    "\n\nWhen the requested integer N is determined, end with exactly "
    "candidate_answer=N and \\boxed{N}."
)
VERIFIER_ANSWER_ONLY_SYSTEM_PROMPT: str = (
    "You are an independent AIME answer checker for a live local benchmark. "
    "You are not given the official answer key or the first solver's answer. "
    "Solve visibly but very compactly. Prefer one general formula, invariant, "
    "or recurrence over exhaustive case enumeration when possible. Use at most "
    "six short lines of math, then exactly verified_answer=N and \\boxed{N}. "
    "Do not write candidate_answer, audit text, examples, or prose after the "
    "boxed answer."
)
VERIFIER_ANSWER_ONLY_USER_PROMPT_SUFFIX: str = (
    "\n\nGive the shortest reliable derivation you can, then write exactly "
    "verified_answer=N and \\boxed{N}."
)
VERIFIER_TIEBREAKER_SYSTEM_PROMPT: str = (
    "You are an independent AIME answer checker for a live local benchmark. "
    "You are not given the official answer key or the first solver's answer. "
    "Solve the problem independently and compactly with no hidden reasoning. "
    "At the end write exactly verified_answer=N on its own line followed by "
    "\\boxed{N}. Do not write candidate_answer. Do not audit another solution, "
    "do not mention confidence, and do not include prose after the boxed answer."
)
VERIFIER_TIEBREAKER_USER_PROMPT_SUFFIX: str = (
    "\n\nIndependently solve this AIME problem. Use a compact visible derivation, "
    "then write exactly verified_answer=N and \\boxed{N}."
)
VERIFIER_ADJUDICATOR_SYSTEM_PROMPT: str = (
    "You are an independent AIME adjudicator for a live local benchmark. "
    "You are not given the official answer key. You are given only the problem "
    "and candidate answers produced by local solver passes. Do not trust any "
    "candidate. Check enough to choose the correct requested AIME integer, or "
    "produce the corrected integer if all candidates are wrong. Prefer one "
    "general formula, invariant, or recurrence over exhaustive case enumeration "
    "when possible. Use at most six short lines of math, then exactly "
    "verified_answer=N and \\boxed{N}. Do not write candidate_answer, "
    "confidence, or prose after the boxed answer."
)
CAP_RECOVERY_FINALIZER_SYSTEM_PROMPT: str = (
    "You are finalizing a capped AIME solver attempt for a live local "
    "benchmark. You are not given the official answer key. The prior scratch "
    "may be incomplete or wrong. Use it only as untrusted working notes. "
    "Continue or correct the solution in compact visible math, then write "
    "exactly candidate_answer=N and \\boxed{N}. Use no hidden reasoning, no "
    "source notes, no audits, no examples list, and no prose after the boxed "
    "answer."
)
CAP_RECOVERY_FRESH_SYSTEM_PROMPT: str = (
    "You are a fresh visible AIME recovery solver for a live local benchmark. "
    "You are not given the official answer key or another solver's scratch. "
    "Solve independently with compact visible math. Prefer one reliable "
    "formula, invariant, recurrence, or counted table over a long narrative. "
    "When the requested AIME integer is determined, write exactly "
    "candidate_answer=N and \\boxed{N}. Use no hidden reasoning, no source "
    "notes, no audits, no examples list, and no prose after the boxed answer."
)
VISIBLE_SUBMISSION_SYSTEM_PROMPT: str = (
    "You are continuing a reasoning-enabled AIME attempt for a live local "
    "benchmark. You are not given the official answer key. The previous "
    "assistant response produced hidden scratch but did not submit visible "
    "answer content. Keep reasoning enabled. Continue naturally from the prior "
    "scratch if it is useful, correct it if it is wrong, and submit the "
    "answer visibly as candidate_answer=N and \\boxed{N}. The runner scores "
    "only visible content from this pass, never hidden reasoning."
)
GEMMA_FAST_AIME_MAX_TOKENS: int = 2048
GEMMA_THINKING_RECOVERY_MAX_TOKENS: int = 2048
GEMMA_FRESH_FINALIZER_MAX_TOKENS: int = 1280
GEMMA_VISIBLE_FINALIZER_MAX_TOKENS: int = 768
CAP_RECOVERY_SCRATCH_MAX_CHARS: int = 8000
VERIFIER_MAX_TOKENS: int = 1024
VERIFIER_ADJUDICATOR_MAX_TOKENS: int = 768


def default_enable_thinking_for_model(model_id: str | None) -> bool:
    """Return the AIME reasoning-channel default for a model id."""

    return True


def default_answer_verification_for_model(
    model_id: str | None, *, enable_thinking: bool
) -> str:
    """Return the default no-key answer verification policy."""

    lowered = str(model_id or "").lower()
    if not enable_thinking and ("gemma4" in lowered or "gemma-4" in lowered):
        return "fast_majority"
    return "off"


def default_cap_recovery_for_model(
    model_id: str | None, *, enable_thinking: bool
) -> str:
    """Return the answer-recovery policy for capped no-answer rows."""

    lowered = str(model_id or "").lower()
    if not enable_thinking and ("gemma4" in lowered or "gemma-4" in lowered):
        return "fresh_finalizer"
    return "off"


def default_max_tokens_for_model(
    model_id: str | None, *, enable_thinking: bool
) -> int | None:
    """Return the AIME response cap for the default model policy."""

    lowered = str(model_id or "").lower()
    if not enable_thinking and ("gemma4" in lowered or "gemma-4" in lowered):
        return GEMMA_FAST_AIME_MAX_TOKENS
    return None


def aime_prompt_messages(
    problem: "AIMEProblem", *, enable_thinking: bool
) -> list[dict[str, str]]:
    """Build the request messages for the selected AIME reasoning mode."""

    if enable_thinking:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": problem.problem + USER_PROMPT_SUFFIX,
            },
        ]
    return [
        {"role": "system", "content": FAST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": problem.problem + FAST_USER_PROMPT_SUFFIX,
        },
    ]


def aime_verifier_messages(
    problem: "AIMEProblem",
    *,
    style: str = "answer_only",
    candidate_answers: Sequence[int] | None = None,
) -> list[dict[str, str]]:
    """Build the independent answer-check prompt.

    The verifier deliberately receives only the problem statement. It does not
    receive the official answer key or the first pass's proposed answer, so a
    correction is self-consistency evidence rather than benchmark-key leakage.
    """

    if style == "adjudicator":
        candidates = ", ".join(str(int(answer)) for answer in candidate_answers or ())
        if not candidates:
            candidates = "none"
        return [
            {"role": "system", "content": VERIFIER_ADJUDICATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    problem.problem
                    + "\n\nCandidate answers from prior local solver passes: "
                    + candidates
                    + "\n\nCheck which candidate is correct, or correct them "
                    "if needed. Use the shortest reliable derivation you can, "
                    "then write exactly verified_answer=N and \\boxed{N}."
                ),
            },
        ]
    if style == "tiebreaker":
        return [
            {"role": "system", "content": VERIFIER_TIEBREAKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": problem.problem + VERIFIER_TIEBREAKER_USER_PROMPT_SUFFIX,
            },
        ]
    return [
        {"role": "system", "content": VERIFIER_ANSWER_ONLY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": problem.problem + VERIFIER_ANSWER_ONLY_USER_PROMPT_SUFFIX,
        },
    ]


def aime_cap_recovery_messages(
    problem: "AIMEProblem", first_answer_text: str
) -> list[dict[str, str]]:
    scratch = str(first_answer_text or "")
    if len(scratch) > CAP_RECOVERY_SCRATCH_MAX_CHARS:
        scratch = scratch[-CAP_RECOVERY_SCRATCH_MAX_CHARS:]
    return [
        {"role": "system", "content": CAP_RECOVERY_FINALIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                problem.problem
                + "\n\nPrior capped scratch, untrusted:\n"
                + scratch
                + "\n\nFinalize the requested AIME integer now. If the scratch "
                "is incomplete, continue from the most reliable point; if it is "
                "wrong, correct it. End with exactly candidate_answer=N and "
                "\\boxed{N}."
            ),
        },
    ]


def aime_fresh_cap_recovery_messages(problem: "AIMEProblem") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CAP_RECOVERY_FRESH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                problem.problem
                + "\n\nSolve this independently and compactly. End with exactly "
                "candidate_answer=N and \\boxed{N}."
            ),
        },
    ]


def aime_visible_submission_messages(
    problem: "AIMEProblem",
    *,
    first_reasoning_text: str,
    first_answer_text: str = "",
) -> list[dict[str, str]]:
    """Build the visible-submission prompt after no visible answer.

    This deliberately passes the full prior scratch. The runner does not parse
    or shorten that scratch into an answer; the model has to make a new visible
    submission for the row to score.
    """

    prior_visible = str(first_answer_text or "")
    prior_scratch = str(first_reasoning_text or "")
    prior_visible_section = (
        "\n\nPrevious visible content, untrusted:\n" + prior_visible
        if prior_visible
        else ""
    )
    return [
        {"role": "system", "content": VISIBLE_SUBMISSION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                problem.problem
                + "\n\nPrevious hidden scratch, untrusted:\n"
                + prior_scratch
                + prior_visible_section
                + "\n\nSubmit the requested AIME integer visibly now. Do not "
                "use the official answer key. Do not restart the solution. "
                "Do not enumerate cases again. If the scratch is reliable, "
                "submit from it; if it is incomplete, finish only the missing "
                "arithmetic. Keep reasoning enabled and end with exactly "
                "candidate_answer=N and \\boxed{N}."
            ),
        },
    ]


def _has_stable_stream_answer_boundary(text: str, end: int) -> bool:
    """Return true only when a streamed answer is visibly complete."""

    if end <= 0 or end > len(text):
        return False
    if text[end - 1] in _STREAM_STABLE_ANSWER_TRAILING_CHARS:
        return True
    return _STREAM_STABLE_ANSWER_BOUNDARY_RE.match(text[end:]) is not None


def extract_candidate_answer(
    text: str, *, require_stable_boundary: bool = False
) -> int | None:
    """Extract the explicit AIME answer marker from model text.

    This is intentionally narrower than the grader's prose fallback. It is a
    model-submitted answer marker, not permission for the runner to stop or
    synthesize answer text on the model's behalf.
    """

    if not isinstance(text, str) or not text:
        return None
    matches = [
        match
        for match in _CANDIDATE_ANSWER_RE.finditer(text)
        if not require_stable_boundary
        or _has_stable_stream_answer_boundary(text, match.end())
    ]
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except ValueError:
        return None


def extract_verified_answer(
    text: str, *, require_stable_boundary: bool = False
) -> int | None:
    """Extract the independent verifier's explicit answer marker."""

    if not isinstance(text, str) or not text:
        return None
    matches = [
        match
        for match in _VERIFIED_ANSWER_RE.finditer(text)
        if not require_stable_boundary
        or _has_stable_stream_answer_boundary(text, match.end())
    ]
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except ValueError:
        return None


def extract_boxed_submission(
    text: str, *, require_stable_boundary: bool = False
) -> int | None:
    """Extract an explicit visible ``\\boxed{N}`` submission only."""

    if not isinstance(text, str) or not text:
        return None
    matches = [
        match
        for match in _BOXED_SUBMISSION_RE.finditer(text)
        if not require_stable_boundary
        or _has_stable_stream_answer_boundary(text, match.end())
    ]
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except ValueError:
        return None


def extract_declared_answer(
    text: str, *, require_stable_boundary: bool = False
) -> int | None:
    """Extract a direct final-answer declaration from model text.

    This intentionally does not accept generic ``Total = a + b = n`` or
    ``Sum = ...`` arithmetic. In AIME counting problems those phrases often
    appear as per-case subtotals long before the final answer, and streaming
    commit must prefer correctness over ending a few seconds earlier. Scalar
    ``Sum = N.`` is accepted only after a final-looking itemized count summary.
    """

    if not isinstance(text, str) or not text:
        return None
    latest: tuple[int, str, int] | None = None
    for pattern in (
        _DECLARED_ANSWER_RE,
        _VARIABLE_SUM_ANSWER_RE,
        _TOTAL_COUNT_ARITHMETIC_ANSWER_RE,
        _TOTAL_NUMBER_ANSWER_RE,
    ):
        for match in pattern.finditer(text):
            if require_stable_boundary and not _has_stable_stream_answer_boundary(
                text, match.end()
            ):
                continue
            candidate = match.group(1)
            if latest is None or match.start() > latest[0]:
                latest = (match.start(), candidate, match.end())
    for fallback in (
        _extract_count_summary_answer(text),
        _extract_count_total_walkdown_answer(text),
        _extract_count_arithmetic_confirmation_answer(text),
        _extract_total_desired_arithmetic_answer(text),
        _extract_divisor_count_answer(text),
        _extract_set_cardinality_answer(text),
        _extract_final_count_scalar_answer(text),
    ):
        if fallback is None:
            continue
        if require_stable_boundary and not _has_stable_stream_answer_boundary(
            text, fallback[2]
        ):
            continue
        if latest is None or fallback[0] > latest[0]:
            latest = fallback
    if latest is None:
        return None
    try:
        return int(latest[1])
    except ValueError:
        return None


def _extract_count_summary_answer(text: str) -> tuple[int, str, int] | None:
    """Return the latest scalar sum after a final-looking count summary."""

    lines = text.splitlines()
    latest: tuple[int, str, int] | None = None
    offset = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    for start_idx, line in enumerate(lines):
        if not _COUNT_SUMMARY_HEADER_RE.match(line):
            continue
        item_count = 0
        for idx in range(start_idx + 1, min(len(lines), start_idx + 32)):
            candidate_line = lines[idx]
            if not candidate_line.strip():
                continue
            total_match = _COUNT_SUMMARY_TOTAL_RE.match(candidate_line)
            if total_match is not None:
                if item_count >= 3:
                    latest = (
                        line_starts[idx],
                        total_match.group(1),
                        line_starts[idx] + total_match.end(),
                    )
                break
            if _COUNT_SUMMARY_ITEM_RE.match(candidate_line):
                item_count += 1
                continue
            if item_count:
                break

    return latest


def _extract_count_total_walkdown_answer(text: str) -> tuple[int, str, int] | None:
    """Return a count total proven by a cumulative arithmetic walkdown.

    Qwen often writes:

    ``Let's sum the counts.``
    ``Total = 32 + 16 + 8 + 4 + 2.`` or ``Total sum = ...``
    ``32 + 16 = 48.``
    ...
    ``60 + 2 = 62.``

    Generic ``Total = ...`` is unsafe because it appears in per-case scratch
    work. This fallback only accepts the walkdown when it follows a nearby
    "sum the counts" cue and the final arithmetic result equals the sum of the
    explicit total expression.
    """

    lines = text.splitlines()
    latest: tuple[int, str, int] | None = None
    offset = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    for idx, line in enumerate(lines):
        expr_match = _TOTAL_SUM_EXPRESSION_RE.match(line)
        if expr_match is None:
            continue
        context = "\n".join(lines[max(0, idx - 12) : idx + 1])
        total_sum_line = bool(re.match(r"^\s*total\s+sum\b", line, re.IGNORECASE))
        term_count = expr_match.group("expr").count("+") + 1
        if not (
            _SUM_COUNTS_CONTEXT_RE.search(context)
            or _COUNT_ARITHMETIC_CONTEXT_RE.search(context)
            or (total_sum_line and term_count >= 5)
        ):
            continue
        try:
            expected = sum(
                int(part.strip())
                for part in expr_match.group("expr").split("+")
            )
        except ValueError:
            continue
        for step_idx in range(idx + 1, min(len(lines), idx + 12)):
            step = lines[step_idx]
            if not step.strip():
                continue
            step_match = _ARITHMETIC_STEP_RE.match(step)
            if step_match is None:
                if step_idx > idx + 1:
                    break
                continue
            try:
                result = int(step_match.group("result"))
            except ValueError:
                continue
            if result == expected:
                latest = (
                    line_starts[step_idx],
                    str(result),
                    line_starts[step_idx] + step_match.end(),
                )
                break

    return latest


def _extract_count_arithmetic_confirmation_answer(
    text: str,
) -> tuple[int, str, int] | None:
    """Return a count total from a confirmed arithmetic expression.

    This catches the real app failure where the model has already written the
    final count arithmetic and calls it correct, but then keeps checking:

    ``So the calculation $5 + 15 + 20 + 15 + 6 + 1 = 62$ is correct.``

    Keep the guard narrow: the expression must be integer arithmetic, must
    evaluate to the stated result, and must sit near count/case/solution
    context. A generic ``Sum = ...`` line is still ignored.
    """

    lines = text.splitlines()
    latest: tuple[int, str, int] | None = None
    offset = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    for idx, line in enumerate(lines):
        match = _COUNT_ARITHMETIC_CONFIRM_RE.match(line)
        if match is None:
            continue
        context = "\n".join(lines[max(0, idx - 12) : idx + 1])
        if not _COUNT_ARITHMETIC_CONTEXT_RE.search(context):
            continue
        value = _safe_integer_arithmetic(match.group("expr"))
        if value is None:
            continue
        try:
            result = int(match.group("result"))
        except ValueError:
            continue
        if value == result:
            latest = (line_starts[idx], str(result), line_starts[idx] + match.end())

    return latest


def _safe_integer_arithmetic(expression: str) -> int | None:
    if re.fullmatch(r"[0-9+\-()\s]+", expression) is None:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def eval_node(node: ast.AST) -> int | None:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if left is None or right is None:
                return None
            return left + right if isinstance(node.op, ast.Add) else left - right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if isinstance(node.value, bool):
                return None
            return node.value
        return None

    return eval_node(tree)


def _extract_total_desired_arithmetic_answer(
    text: str,
) -> tuple[int, str, int] | None:
    """Return final desired-count arithmetic like ``Total good = 98 - 28 = 70``."""

    lines = text.splitlines()
    latest: tuple[int, str, int] | None = None
    offset = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    for idx, line in enumerate(lines):
        match = _TOTAL_DESIRED_ARITHMETIC_ANSWER_RE.match(line)
        if match is None:
            continue
        value = _safe_integer_arithmetic(match.group("expr"))
        if value is None:
            continue
        try:
            result = int(match.group("result"))
        except ValueError:
            continue
        if value == result:
            latest = (line_starts[idx], str(result), line_starts[idx] + match.end())

    return latest


def _extract_divisor_count_answer(text: str) -> tuple[int, str, int] | None:
    """Return the final divisor-count answer for product-of-values prompts."""

    lines = text.splitlines()
    latest: tuple[int, str, int] | None = None
    offset = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    for idx, line in enumerate(lines):
        match = _DIVISOR_COUNT_SCALAR_RE.match(line)
        if match is None:
            match = _DIVISOR_COUNT_EXPRESSION_RE.match(line)
        if match is None:
            continue
        context = "\n".join(lines[max(0, idx - 12) : idx + 1])
        if not _DIVISOR_FINAL_CONTEXT_RE.search(context):
            continue
        try:
            result = int(match.group("result"))
        except ValueError:
            continue
        latest = (line_starts[idx], str(result), line_starts[idx] + match.end())

    return latest


def _extract_set_cardinality_answer(text: str) -> tuple[int, str, int] | None:
    """Return a final set-size answer with nearby cardinality context.

    AIME counting problems often end with a set of valid values, but
    intermediate case work can also mention small sets. Keep this narrower
    than a generic "count is N": require a direct set-size line plus nearby
    wording that says the set contains the values/integers/numbers being
    counted.
    """

    latest: tuple[int, str, int] | None = None
    for match in _SET_SIZE_ANSWER_RE.finditer(text):
        context_start = max(0, match.start() - 700)
        context = text[context_start : match.start()]
        if not _SET_CARDINALITY_CONTEXT_RE.search(context):
            continue
        if latest is None or match.start() > latest[0]:
            latest = (match.start(), match.group(1), match.end())
    return latest


def _extract_final_count_scalar_answer(text: str) -> tuple[int, str, int] | None:
    """Return a final scalar count once nearby text proves it is cardinality.

    The real app can produce the correct AIME answer as a short ``Count = N.``
    after proving the valid set, then keep checking instead of writing the
    explicit marker. Keep this narrower than generic count extraction: require
    set/cardinality context and finality language, and reject local per-case
    count lines.
    """

    lines = text.splitlines()
    latest: tuple[int, str, int] | None = None
    offset = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    for idx, line in enumerate(lines):
        match = _FINAL_COUNT_SCALAR_RE.match(line)
        if match is None:
            continue
        broad_context = "\n".join(lines[max(0, idx - 16) : idx + 1])
        local_context = "\n".join(lines[max(0, idx - 3) : idx + 1])
        if not _FINAL_COUNT_CARDINALITY_CONTEXT_RE.search(broad_context):
            continue
        if not _FINAL_COUNT_FINALITY_CONTEXT_RE.search(broad_context):
            continue
        if _INTERMEDIATE_COUNT_CONTEXT_RE.search(local_context):
            continue
        latest = (
            line_starts[idx],
            match.group("result"),
            line_starts[idx] + match.end(),
        )

    return latest


def _extract_stream_commit(text: str) -> tuple[int, str] | None:
    marker = extract_candidate_answer(text, require_stable_boundary=True)
    if marker is not None:
        return marker, "candidate_answer"
    boxed = extract_boxed_submission(text, require_stable_boundary=True)
    if boxed is not None:
        return boxed, "boxed"
    return None


def _extract_verifier_stream_commit(text: str) -> int | None:
    verified = extract_verified_answer(text, require_stable_boundary=True)
    if verified is not None:
        return verified
    return extract_boxed_submission(text, require_stable_boundary=True)


def _extract_final_visible_answer(text: str) -> tuple[int, str] | None:
    boxed = extract_boxed_submission(text)
    if boxed is not None:
        return boxed, "answer_boxed"
    marker = extract_candidate_answer(text)
    if marker is not None:
        return marker, "answer_candidate_answer"
    declared = extract_declared_answer(text)
    if declared is not None:
        return declared, "answer_declared_final"
    fallback = extract_boxed(text)
    if fallback is not None:
        return fallback, "answer_validator_fallback_final"
    return None


# --- Data classes ---------------------------------------------------------


class RunState(str, Enum):
    """The runner's high-level state."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclasses.dataclass(frozen=True, slots=True)
class AIMEProblem:
    """One AIME problem, loaded from the JSONL dataset."""

    id: str
    set: str  # "AIME I" or "AIME II"
    year: int
    index: int  # 1..15 within the set
    problem: str
    answer: int
    source: str


@dataclasses.dataclass(slots=True)
class QuestionResult:
    """One question's outcome and per-question timing."""

    idx: int  # 1..30 across the run
    problem: AIMEProblem
    started_at: float | None = None
    ended_at: float | None = None
    extracted: int | None = None
    status: GradeStatus | None = None
    reasoning_token_count: int = 0
    answer_token_count: int = 0
    reasoning_text: str = ""
    answer_text: str = ""
    stream_reasoning_events: int = 0
    stream_answer_events: int = 0
    stream_reasoning_chars: int = 0
    stream_answer_chars: int = 0
    stream_progress_events: int = 0
    stream_progress_milestones: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    reasoning_finalizer_handoff: bool = False
    reasoning_finalizer_trigger_answer: int | None = None
    reasoning_finalizer_trigger_source: str | None = None
    reasoning_finalizer_trigger_completion_tokens: int | None = None
    reasoning_finalizer_grace_tokens: int = 0
    visible_submission: AIMEVisibleSubmission | None = None
    answer_verification: AIMEAnswerVerification | None = None
    cap_recovery: AIMECapRecovery | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return int(round((self.ended_at - self.started_at) * 1000.0))


# --- Errors ---------------------------------------------------------------


class ConcurrentRunError(RuntimeError):
    """Raised when start_run is called while another run is active."""

    def __init__(self, active_run_id: str) -> None:
        super().__init__(
            f"another AIME run is already active: {active_run_id}"
        )
        self.active_run_id = active_run_id


class _QuestionSkipped(RuntimeError):
    """Internal control-flow signal for a user-requested question skip."""


class _QuestionPaused(RuntimeError):
    """Internal control-flow signal for a user-requested hard pause."""


def _numeric_progress_field(progress: Mapping[str, Any], key: str) -> float | None:
    value = progress.get(key)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and number == number else None


def _observe_stream_progress(
    result: QuestionResult,
    progress: Mapping[str, Any],
) -> None:
    """Record bounded progress evidence from the OpenAI stream.

    The server owns exact token timing; this runner keeps the first visible
    progress samples so the persisted AIME row can prove whether Q3 started
    fast or inherited stale/late TPS from a prior question.
    """

    result.stream_progress_events += 1
    raw_tokens = progress.get("completion_tokens")
    if not isinstance(raw_tokens, int):
        try:
            raw_tokens = int(raw_tokens or 0)
        except (TypeError, ValueError):
            raw_tokens = 0
    completion_tokens = max(0, int(raw_tokens))
    for name, threshold in PROGRESS_TRACE_MILESTONES:
        if completion_tokens < threshold or name in result.stream_progress_milestones:
            continue
        sample: dict[str, Any] = {
            "completion_tokens": completion_tokens,
        }
        for field in (
            "decode_tok_s",
            "display_decode_tok_s",
            "request_tok_s",
            "decode_elapsed_s",
            "request_elapsed_s",
            "dashboard_progress_decision_time_s",
            "dashboard_progress_registry_update_time_s",
            "dashboard_progress_rolling_update_time_s",
            "dashboard_progress_bus_publish_time_s",
        ):
            value = _numeric_progress_field(progress, field)
            if value is not None:
                sample[field] = value
        result.stream_progress_milestones[name] = sample


def _question_progress_event(
    *,
    run_id: str,
    idx: int,
    attempt: int,
    request_id: str,
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(progress)
    payload.setdefault("request_id", request_id)
    return {
        "event": "question_progress",
        "run_id": run_id,
        "idx": idx,
        "attempt": attempt,
        "request_id": request_id,
        "progress": payload,
    }


def _int_stat(
    stats: Mapping[str, Any],
    usage: Mapping[str, Any],
    *keys: str,
) -> int | None:
    for source in (stats, usage):
        for key in keys:
            value = source.get(key)
            if isinstance(value, Mapping):
                continue
            try:
                if value is None:
                    continue
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
    return None


def _nested_int_stat(
    stats: Mapping[str, Any],
    usage: Mapping[str, Any],
    outer: str,
    inner: str,
) -> int | None:
    for source in (stats, usage):
        value = source.get(outer)
        if not isinstance(value, Mapping):
            continue
        try:
            return max(0, int(value.get(inner) or 0))
        except (TypeError, ValueError):
            continue
    return None


def _stream_text_token_estimate(text: str) -> int:
    """Small fallback for cancelled streams that never receive final usage."""

    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def _question_metrics_need_backfill(stats: Mapping[str, Any]) -> bool:
    return (
        not stats
        or "completion_tokens" not in stats
        or "request_temperature" not in stats
        or "effective_temperature" not in stats
    )


def _populate_result_token_counts(
    result: QuestionResult,
    usage: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> None:
    reasoning_tokens = _int_stat(
        stats,
        usage,
        "reasoning_tokens",
    )
    if reasoning_tokens is None:
        reasoning_tokens = _nested_int_stat(
            stats,
            usage,
            "completion_tokens_details",
            "reasoning_tokens",
        )
    answer_tokens = _int_stat(stats, usage, "answer_tokens")
    if reasoning_tokens is None and result.reasoning_text:
        reasoning_tokens = _stream_text_token_estimate(result.reasoning_text)
    if answer_tokens is None and result.answer_text:
        answer_tokens = _stream_text_token_estimate(result.answer_text)
    if answer_tokens is None:
        answer_tokens = 0
    result.reasoning_token_count = int(reasoning_tokens or 0)
    result.answer_token_count = int(answer_tokens or 0)


# --- Dataset loading ------------------------------------------------------


def load_dataset(path: Path | None = None) -> list[AIMEProblem]:
    """Load and validate the 30-problem AIME 2026 dataset."""
    dataset_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset missing: {dataset_path}")
    problems: list[AIMEProblem] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{dataset_path}:{line_no} invalid JSON: {exc}"
                ) from exc
            problems.append(
                AIMEProblem(
                    id=str(obj["id"]),
                    set=str(obj["set"]),
                    year=int(obj["year"]),
                    index=int(obj["index"]),
                    problem=str(obj["problem"]),
                    answer=int(obj["answer"]),
                    source=str(obj.get("source", "")),
                )
            )
    if len(problems) != 30:
        raise ValueError(f"expected 30 problems, got {len(problems)}")
    return problems


# --- Default chat stream factory (httpx-backed) ---------------------------


ChatStreamFactory = Callable[
    ["AIMERunner", AIMEProblem],
    Awaitable[AsyncIterator[dict[str, Any]]],
]
"""A callable returning an async iterator over parsed SSE chunk dicts.

Each yielded item is the OpenAI streaming chunk shape:
``{"choices": [{"delta": {"reasoning_content": "...", "content": "..."}}], "usage": {...}, ...}``.

The runner is robust to chunks that lack either field. The iterator
must be cancellable; the runner uses ``asyncio.CancelledError`` to abort
mid-decode on cancel.
"""

ChatCancelFactory = Callable[["AIMERunner", str], Awaitable[None]]
ChatMetricsFactory = Callable[["AIMERunner", str], Awaitable[dict[str, Any]]]
QuestionIsolationFactory = Callable[
    ["AIMERunner", "QuestionResult", str], Awaitable[dict[str, Any] | None]
]
QuestionRuntimeCleanup = Callable[[], Awaitable[dict[str, Any] | None]]
QuestionRuntimeFactory = Callable[
    ["AIMERunner", AIMEProblem], Awaitable["AIMEQuestionRuntime | None"]
]
AnswerVerifierFactory = Callable[
    ["AIMERunner", "AIMEProblem", int, str], Awaitable["AIMEAnswerVerification | None"]
]
CapRecoveryFactory = Callable[
    ["AIMERunner", "AIMEProblem", str, Mapping[str, Any]],
    Awaitable["AIMECapRecovery | None"],
]
VisibleSubmissionFactory = Callable[
    ["AIMERunner", "AIMEProblem", str, str, Mapping[str, Any]],
    Awaitable["AIMEVisibleSubmission | None"],
]


@dataclasses.dataclass(slots=True)
class AIMEVisibleSubmission:
    """Second-pass visible answer produced after a reasoning-only first pass."""

    mode: str
    request_id: str
    final_answer: int | None
    commit_source: str | None
    reasoning_text: str = ""
    answer_text: str = ""
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    duration_ms: int | None = None


@dataclasses.dataclass(slots=True)
class AIMEQuestionRuntime:
    """Optional per-question runtime override for stateless AIME requests."""

    base_url: str
    cleanup: QuestionRuntimeCleanup | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class AIMECapRecovery:
    """Bounded second-pass evidence for a capped no-answer first pass."""

    mode: str
    request_id: str
    final_answer: int | None
    commit_source: str | None
    reasoning_text: str = ""
    answer_text: str = ""
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    duration_ms: int | None = None


@dataclasses.dataclass(slots=True)
class AIMEAnswerVerification:
    """Independent no-key verification evidence for a submitted answer."""

    mode: str
    proposed_answer: int
    final_answer: int | None
    answers: list[int | None] = dataclasses.field(default_factory=list)
    request_ids: list[str] = dataclasses.field(default_factory=list)
    texts: list[str] = dataclasses.field(default_factory=list)
    stats: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    resolution: str = "none"
    duration_ms: int | None = None

    @property
    def has_verifier_answer(self) -> bool:
        return any(answer is not None for answer in self.answers)

    @property
    def agreement(self) -> bool:
        return (
            self.final_answer == self.proposed_answer
            and self.proposed_answer in self.answers
        )

    @property
    def text_tail(self) -> str:
        return "\n---\n".join(self.texts)[-500:]


def _normalize_answer_verification_mode(mode: str | None) -> str:
    if mode is None:
        return "off"
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "": "off",
        "none": "off",
        "false": "off",
        "0": "off",
        "true": "fast_majority",
        "1": "fast_majority",
        "on": "fast_majority",
        "majority": "fast_majority",
        "independent": "fast_majority",
        "self_consistency": "fast_majority",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"off", "fast_majority"}:
        raise ValueError(f"unknown AIME answer verification mode: {mode!r}")
    return normalized


def _normalize_cap_recovery_mode(mode: str | None) -> str:
    if mode is None:
        return "off"
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "": "off",
        "none": "off",
        "false": "off",
        "0": "off",
        "true": "fresh_finalizer",
        "1": "fresh_finalizer",
        "on": "fresh_finalizer",
        "fresh": "fresh_finalizer",
        "fresh_retry": "fresh_finalizer",
        "independent": "fresh_finalizer",
        "independent_retry": "fresh_finalizer",
        "independent_finalizer": "fresh_finalizer",
        "visible": "visible_finalizer",
        "finalizer": "visible_finalizer",
        "visible_retry": "visible_finalizer",
        "thinking": "thinking_retry",
        "retry": "thinking_retry",
        "gemma_thinking": "thinking_retry",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "off",
        "thinking_retry",
        "visible_finalizer",
        "fresh_finalizer",
    }:
        raise ValueError(f"unknown AIME cap recovery mode: {mode!r}")
    return normalized


def _resolve_verified_answer(
    proposed_answer: int, answers: list[int | None]
) -> tuple[int | None, str]:
    verifier_answers = [answer for answer in answers if answer is not None]
    if not verifier_answers:
        return proposed_answer, "no_verifier_answer_keep_proposed"

    if proposed_answer in verifier_answers:
        return proposed_answer, "majority_keep_proposed"

    counts: dict[int, int] = {}
    for answer in verifier_answers:
        counts[answer] = counts.get(answer, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return proposed_answer, "no_verifier_answer_keep_proposed"
    winner, votes = ranked[0]
    tied = [answer for answer, count in ranked if count == votes]
    if votes >= 2 and len(tied) == 1:
        return None, "majority_disagreed_abstain"
    return proposed_answer, "weak_verifier_disagreed_keep_proposed"


def _extract_verifier_final_answer(text: str) -> int | None:
    verified = extract_verified_answer(text)
    if verified is not None:
        return verified
    boxed = extract_boxed_submission(text)
    if boxed is not None:
        return boxed
    return None


def _completion_hit_cap(stats: Mapping[str, Any], max_tokens: int) -> bool:
    """Return true when a request exhausted its response token budget."""

    if not stats:
        return False
    completion = stats.get("completion_tokens")
    if completion is None:
        completion = stats.get("answer_tokens")
    effective = stats.get("effective_max_tokens")
    if effective is None:
        effective = stats.get("request_max_tokens")
    try:
        completion_i = int(completion)
    except (TypeError, ValueError):
        return False
    try:
        effective_i = int(effective)
    except (TypeError, ValueError):
        effective_i = int(max_tokens)
    return completion_i >= max(1, min(int(max_tokens), effective_i))


async def default_chat_stream_factory(
    runner: "AIMERunner", problem: AIMEProblem
) -> AsyncIterator[dict[str, Any]]:
    """Default factory: stream from a local OpenAI-compatible endpoint.

    Uses :mod:`httpx` to POST ``/v1/chat/completions`` with ``stream=true``
    against the runner's configured ``base_url``. Yields one parsed JSON
    object per SSE ``data:`` line (excluding the terminal ``[DONE]``).

    Sampler keys (``temperature``, ``top_p``, ``top_k``, ``max_tokens``)
    are only included in the payload when the runner has an explicit
    override. Omitting them lets the daemon apply its per-target preset
    default - which matches the launch-QA policy from
    `Restore product sampler defaults for launch QA` (25ae0fe).
    """
    import httpx

    request_id = runner.current_request_id or runner.request_id_for(
        runner.current_idx or problem.index,
        runner.current_attempt or 1,
    )
    question_idx = runner.current_idx or problem.index
    attempt = runner.current_attempt or 1
    enable_thinking = (
        bool(runner.enable_thinking)
        if runner.enable_thinking is not None
        else default_enable_thinking_for_model(runner.model_id)
    )
    lowered_model_id = str(runner.model_id or "").lower()
    if not enable_thinking and (
        "gemma4" in lowered_model_id or "gemma-4" in lowered_model_id
    ):
        logger.warning(
            "Gemma AIME request starting with thinking disabled "
            "run_id=%s idx=%s attempt=%s request_id=%s",
            runner.run_id,
            question_idx,
            attempt,
            request_id,
        )
    payload: dict[str, Any] = {
        "model": runner.model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": aime_prompt_messages(problem, enable_thinking=enable_thinking),
        "enable_thinking": enable_thinking,
        "metadata": {
            "client": "aime",
            "benchmark": f"aime_{runner.year}",
            "run_id": runner.run_id,
            "question_idx": question_idx,
            "attempt": attempt,
            "cache_mode": "bypass",
            # Stream the chain-of-thought as real `reasoning_content` so the
            # live REASONING panel populates again. Requesting
            # "visible working" (closing <think> in the prompt) routed the
            # whole CoT into the answer channel and left the REASONING panel
            # stuck on "Awaiting reasoning…". The visible boxed answer is
            # still emitted in `content` after </think> and extracted by
            # `_extract_final_visible_answer`, with the visible_submission
            # handoff as a backstop. Live reasoning rendering is already
            # O(visible) (LazyVStack + Equatable lines, commit 69bcb98), so
            # this does not reintroduce the long-reasoning perf collapse.
            "aime_visible_working": False,
            "mtplx_request_id": request_id,
            "enable_thinking": enable_thinking,
        },
    }
    if runner.temperature is not None:
        payload["temperature"] = runner.temperature
    if runner.top_p is not None:
        payload["top_p"] = runner.top_p
    if runner.top_k is not None:
        payload["top_k"] = runner.top_k
    max_tokens = runner.max_tokens
    if max_tokens is None:
        max_tokens = default_max_tokens_for_model(
            runner.model_id, enable_thinking=enable_thinking
        )
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Accept": "text/event-stream",
        "x-mtplx-client": "aime",
        "x-mtplx-cache-mode": "bypass",
        "x-mtplx-request-id": request_id,
    }
    if runner.api_key:
        headers["Authorization"] = f"Bearer {runner.api_key}"

    url = runner.request_base_url.rstrip("/") + "/v1/chat/completions"

    async def stream() -> AsyncIterator[dict[str, Any]]:
        timeout = httpx.Timeout(None)  # benchmark runs can be long
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise RuntimeError(
                        f"chat completion failed {response.status_code}: "
                        f"{body.decode('utf-8', errors='replace')[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "AIMERunner: skipping non-JSON SSE line: %r",
                            data[:120],
                        )
                        continue

    return stream()


async def default_chat_cancel_factory(runner: "AIMERunner", request_id: str) -> None:
    """Best-effort cancellation for the active AIME chat completion."""
    import httpx

    url = runner.request_base_url.rstrip("/") + f"/v1/mtplx/cancel/{request_id}"
    headers: dict[str, str] = {}
    if runner.api_key:
        headers["Authorization"] = f"Bearer {runner.api_key}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
        await client.post(url, headers=headers)


async def _noop_chat_cancel_factory(
    runner: "AIMERunner", request_id: str
) -> None:
    _ = runner
    _ = request_id


async def default_chat_metrics_factory(
    runner: "AIMERunner", request_id: str
) -> dict[str, Any]:
    """Fetch final daemon metrics for an early-committed AIME request."""
    import httpx

    url = runner.request_base_url.rstrip("/") + "/metrics"
    headers: dict[str, str] = {}
    if runner.api_key:
        headers["Authorization"] = f"Bearer {runner.api_key}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
        for _ in range(50):
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            body = response.json()
            candidates: list[dict[str, Any]] = []
            recent = body.get("recent") if isinstance(body, dict) else None
            if isinstance(recent, list):
                candidates.extend(item for item in recent if isinstance(item, dict))
            latest = body.get("latest") if isinstance(body, dict) else None
            if isinstance(latest, dict):
                candidates.append(latest)
            for item in reversed(candidates):
                if item.get("request_id") == request_id:
                    return dict(item)
            await asyncio.sleep(0.2)
    return {}


async def _noop_chat_metrics_factory(
    runner: "AIMERunner", request_id: str
) -> dict[str, Any]:
    _ = runner
    _ = request_id
    return {}


async def _noop_question_isolation_factory(
    runner: "AIMERunner", result: QuestionResult, request_id: str
) -> dict[str, Any] | None:
    _ = runner
    _ = result
    _ = request_id
    return None


async def _noop_question_runtime_factory(
    runner: "AIMERunner", problem: AIMEProblem
) -> AIMEQuestionRuntime | None:
    _ = runner
    _ = problem
    return None


async def default_visible_submission_factory(
    runner: "AIMERunner",
    problem: AIMEProblem,
    first_reasoning_text: str,
    first_answer_text: str,
    first_stats: Mapping[str, Any],
) -> AIMEVisibleSubmission | None:
    """Ask the model to visibly submit after a reasoning-only primary pass."""

    _ = first_stats

    import httpx

    primary_request_id = runner.current_request_id or runner.request_id_for(
        runner.current_idx or problem.index,
        runner.current_attempt or 1,
    )
    request_id = f"{primary_request_id}-submit1"
    idx = runner.current_idx or problem.index
    attempt = runner.current_attempt or 1
    payload: dict[str, Any] = {
        "model": runner.model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": aime_visible_submission_messages(
            problem,
            first_reasoning_text=first_reasoning_text,
            first_answer_text=first_answer_text,
        ),
        "enable_thinking": runner._primary_enable_thinking(),
        "metadata": {
            "client": "aime",
            "phase": "visible_submission",
            "benchmark": f"aime_{runner.year}",
            "run_id": runner.run_id,
            "question_idx": idx,
            "attempt": attempt,
            "cache_mode": "bypass",
            "primary_request_id": primary_request_id,
            "mtplx_request_id": request_id,
            "enable_thinking": runner._primary_enable_thinking(),
        },
    }
    if runner.temperature is not None:
        payload["temperature"] = runner.temperature
    if runner.top_p is not None:
        payload["top_p"] = runner.top_p
    if runner.top_k is not None:
        payload["top_k"] = runner.top_k
    if runner.max_tokens is not None:
        payload["max_tokens"] = runner.max_tokens
    elif runner.visible_submission_max_tokens > 0:
        payload["max_tokens"] = runner.visible_submission_max_tokens

    headers = {
        "Accept": "text/event-stream",
        "x-mtplx-client": "aime",
        "x-mtplx-cache-mode": "bypass",
        "x-mtplx-request-id": request_id,
    }
    if runner.api_key:
        headers["Authorization"] = f"Bearer {runner.api_key}"

    url = runner.request_base_url.rstrip("/") + "/v1/chat/completions"

    async def stream() -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise RuntimeError(
                        f"visible submission failed {response.status_code}: "
                        f"{body.decode('utf-8', errors='replace')[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "AIMERunner: skipping non-JSON visible submission "
                            "SSE line: %r",
                            data[:120],
                        )

    await runner._push(
        {
            "event": "visible_submission_started",
            "run_id": runner.run_id,
            "idx": idx,
            "attempt": attempt,
            "request_id": request_id,
            "primary_request_id": primary_request_id,
        }
    )

    started_at = time.time()
    old_request_id = runner._current_request_id
    runner._current_request_id = request_id
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    try:
        iterator = stream().__aiter__()
        while True:
            chunk = await runner._next_chunk_or_control(iterator, idx)
            if chunk is None:
                break
            choices = chunk.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                reasoning_text = delta.get("reasoning_content") or ""
                answer_text = delta.get("content") or ""
                if reasoning_text:
                    reasoning_text = str(reasoning_text)
                    reasoning_parts.append(reasoning_text)
                    await runner._push(
                        {
                            "event": "reasoning_delta",
                            "run_id": runner.run_id,
                            "idx": idx,
                            "attempt": attempt,
                            "request_id": request_id,
                            "text": reasoning_text,
                        }
                    )
                if answer_text:
                    answer_text = str(answer_text)
                    answer_parts.append(answer_text)
                    await runner._push(
                        {
                            "event": "answer_delta",
                            "run_id": runner.run_id,
                            "idx": idx,
                            "attempt": attempt,
                            "request_id": request_id,
                            "text": answer_text,
                        }
                    )
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("mtplx_stats"):
                stats = chunk["mtplx_stats"]
    finally:
        runner._current_request_id = old_request_id

    if not stats:
        with suppress(Exception):
            stats = await runner._chat_metrics_factory(runner, request_id)

    answer_text = "".join(answer_parts)
    final_answer = _extract_final_visible_answer(answer_text)
    if final_answer is None:
        committed_answer: int | None = None
        commit_source: str | None = None
    else:
        committed_answer, commit_source = final_answer

    submission = AIMEVisibleSubmission(
        mode="visible_submission",
        request_id=request_id,
        final_answer=committed_answer,
        commit_source=commit_source,
        reasoning_text="".join(reasoning_parts),
        answer_text=answer_text,
        usage=usage,
        stats=stats,
        duration_ms=int(round((time.time() - started_at) * 1000.0)),
    )
    await runner._push(
        {
            "event": "visible_submission_done",
            "run_id": runner.run_id,
            "idx": idx,
            "attempt": attempt,
            "request_id": request_id,
            "extracted": submission.final_answer,
            "commit_source": submission.commit_source,
            "duration_ms": submission.duration_ms,
        }
    )
    return submission


async def _noop_visible_submission_factory(
    runner: "AIMERunner",
    problem: AIMEProblem,
    first_reasoning_text: str,
    first_answer_text: str,
    first_stats: Mapping[str, Any],
) -> AIMEVisibleSubmission | None:
    _ = runner
    _ = problem
    _ = first_reasoning_text
    _ = first_answer_text
    _ = first_stats
    return None


async def default_cap_recovery_factory(
    runner: "AIMERunner",
    problem: AIMEProblem,
    first_answer_text: str,
    first_stats: Mapping[str, Any],
) -> AIMECapRecovery | None:
    """Retry a capped Gemma no-thinking row once with bounded recovery.

    This is deliberately a second request with its own request id and metrics.
    The first pass remains visible in persisted telemetry instead of being
    rewritten into a fake single-request TPS number.
    """

    _ = first_answer_text
    _ = first_stats
    if runner.cap_recovery == "off":
        return None

    import httpx

    primary_request_id = runner.current_request_id or runner.request_id_for(
        runner.current_idx or problem.index,
        runner.current_attempt or 1,
    )
    suffix = (
        "think1"
        if runner.cap_recovery == "thinking_retry"
        else "fresh1"
        if runner.cap_recovery == "fresh_finalizer"
        else "finalize1"
    )
    request_id = f"{primary_request_id}-{suffix}"
    idx = runner.current_idx or problem.index
    attempt = runner.current_attempt or 1
    enable_thinking = runner.cap_recovery == "thinking_retry"
    max_tokens = (
        GEMMA_THINKING_RECOVERY_MAX_TOKENS
        if enable_thinking
        else GEMMA_FRESH_FINALIZER_MAX_TOKENS
        if runner.cap_recovery == "fresh_finalizer"
        else GEMMA_VISIBLE_FINALIZER_MAX_TOKENS
    )
    messages = (
        aime_prompt_messages(problem, enable_thinking=True)
        if enable_thinking
        else aime_fresh_cap_recovery_messages(problem)
        if runner.cap_recovery == "fresh_finalizer"
        else aime_cap_recovery_messages(problem, first_answer_text)
    )
    payload: dict[str, Any] = {
        "model": runner.model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": messages,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "metadata": {
            "client": "aime",
            "phase": "cap_recovery",
            "recovery_mode": runner.cap_recovery,
            "benchmark": f"aime_{runner.year}",
            "run_id": runner.run_id,
            "question_idx": idx,
            "attempt": attempt,
            "cache_mode": "bypass",
            "primary_request_id": primary_request_id,
            "mtplx_request_id": request_id,
        },
    }
    if runner.temperature is not None:
        payload["temperature"] = runner.temperature
    if runner.top_p is not None:
        payload["top_p"] = runner.top_p
    if runner.top_k is not None:
        payload["top_k"] = runner.top_k

    headers = {
        "Accept": "text/event-stream",
        "x-mtplx-client": "aime",
        "x-mtplx-cache-mode": "bypass",
        "x-mtplx-request-id": request_id,
    }
    if runner.api_key:
        headers["Authorization"] = f"Bearer {runner.api_key}"

    url = runner.request_base_url.rstrip("/") + "/v1/chat/completions"

    async def stream() -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise RuntimeError(
                        f"cap recovery failed {response.status_code}: "
                        f"{body.decode('utf-8', errors='replace')[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "AIMERunner: skipping non-JSON recovery SSE line: %r",
                            data[:120],
                        )

    await runner._push(
        {
            "event": "cap_recovery_started",
            "run_id": runner.run_id,
            "idx": idx,
            "attempt": attempt,
            "request_id": request_id,
            "primary_request_id": primary_request_id,
            "mode": runner.cap_recovery,
        }
    )

    started_at = time.time()
    old_request_id = runner._current_request_id
    runner._current_request_id = request_id
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    try:
        iterator = stream().__aiter__()
        while True:
            chunk = await runner._next_chunk_or_control(iterator, idx)
            if chunk is None:
                break
            choices = chunk.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                reasoning_text = delta.get("reasoning_content") or ""
                answer_text = delta.get("content") or ""
                if reasoning_text:
                    reasoning_text = str(reasoning_text)
                    reasoning_parts.append(reasoning_text)
                    await runner._push(
                        {
                            "event": "reasoning_delta",
                            "run_id": runner.run_id,
                            "idx": idx,
                            "attempt": attempt,
                            "request_id": request_id,
                            "text": reasoning_text,
                        }
                    )
                if answer_text:
                    answer_text = str(answer_text)
                    answer_parts.append(answer_text)
                    await runner._push(
                        {
                            "event": "answer_delta",
                            "run_id": runner.run_id,
                            "idx": idx,
                            "attempt": attempt,
                            "request_id": request_id,
                            "text": answer_text,
                        }
                    )
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("mtplx_stats"):
                stats = chunk["mtplx_stats"]
    finally:
        runner._current_request_id = old_request_id

    if not stats:
        with suppress(Exception):
            stats = await runner._chat_metrics_factory(runner, request_id)

    answer_text = "".join(answer_parts)
    final_answer = _extract_final_visible_answer(answer_text)
    if final_answer is None:
        committed_answer: int | None = None
        commit_source: str | None = None
    else:
        committed_answer, commit_source = final_answer

    recovery = AIMECapRecovery(
        mode=runner.cap_recovery,
        request_id=request_id,
        final_answer=committed_answer,
        commit_source=commit_source,
        reasoning_text="".join(reasoning_parts),
        answer_text=answer_text,
        usage=usage,
        stats=stats,
        duration_ms=int(round((time.time() - started_at) * 1000.0)),
    )
    await runner._push(
        {
            "event": "cap_recovery_done",
            "run_id": runner.run_id,
            "idx": idx,
            "attempt": attempt,
            "request_id": request_id,
            "mode": recovery.mode,
            "extracted": recovery.final_answer,
            "commit_source": recovery.commit_source,
            "duration_ms": recovery.duration_ms,
        }
    )
    return recovery


async def _noop_cap_recovery_factory(
    runner: "AIMERunner",
    problem: AIMEProblem,
    first_answer_text: str,
    first_stats: Mapping[str, Any],
) -> AIMECapRecovery | None:
    _ = runner
    _ = problem
    _ = first_answer_text
    _ = first_stats
    return None


async def default_answer_verifier_factory(
    runner: "AIMERunner",
    problem: AIMEProblem,
    proposed_answer: int,
    answer_text: str,
) -> AIMEAnswerVerification | None:
    """Run one or two independent no-key answer checks.

    The checker uses the same local daemon and Gemma defaults as the first pass.
    The first verifier pass gets only the problem statement. The second pass may
    see local candidate answers, but never the official key. Disagreement is
    treated as an abstain instead of an automatic correction.
    """

    _ = answer_text
    if runner.answer_verification == "off":
        return None

    import httpx

    started_at = time.time()
    answers: list[int | None] = []
    texts: list[str] = []
    stats_list: list[dict[str, Any]] = []
    request_ids: list[str] = []
    attempt_count = max(1, min(2, int(runner.answer_verification_attempts or 2)))
    headers_base = {
        "Accept": "text/event-stream",
        "x-mtplx-client": "aime",
        "x-mtplx-cache-mode": "bypass",
    }
    if runner.api_key:
        headers_base["Authorization"] = f"Bearer {runner.api_key}"
    url = runner.request_base_url.rstrip("/") + "/v1/chat/completions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        for attempt in range(1, attempt_count + 1):
            verifier_max_tokens = (
                VERIFIER_MAX_TOKENS
                if attempt == 1
                else VERIFIER_ADJUDICATOR_MAX_TOKENS
            )
            request_id = (
                f"{runner.current_request_id or runner.request_id_for(problem.index, 1)}"
                f"-verify{attempt}"
            )
            request_ids.append(request_id)
            headers = dict(headers_base)
            headers["x-mtplx-request-id"] = request_id
            payload: dict[str, Any] = {
                "model": runner.model_id,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": verifier_max_tokens,
                "messages": aime_verifier_messages(
                    problem,
                    style="answer_only" if attempt == 1 else "adjudicator",
                    candidate_answers=[
                        int(answer)
                        for answer in [proposed_answer, *answers]
                        if answer is not None
                    ],
                ),
                "enable_thinking": False,
                "metadata": {
                    "client": "aime",
                    "phase": "answer_verification",
                    "benchmark": f"aime_{runner.year}",
                    "run_id": runner.run_id,
                    "question_idx": runner.current_idx or problem.index,
                    "attempt": runner.current_attempt,
                    "verification_attempt": attempt,
                    "cache_mode": "bypass",
                    "mtplx_request_id": request_id,
                },
            }
            if runner.temperature is not None:
                payload["temperature"] = runner.temperature
            if runner.top_p is not None:
                payload["top_p"] = runner.top_p
            if runner.top_k is not None:
                payload["top_k"] = runner.top_k
            text_parts: list[str] = []
            scan_text = ""
            answer: int | None = None
            stats: dict[str, Any] = {}
            usage: dict[str, Any] = {}
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise RuntimeError(
                        f"AIME answer verification failed {response.status_code}: "
                        f"{body.decode('utf-8', errors='replace')[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        maybe_usage = chunk.get("usage")
                        if isinstance(maybe_usage, dict):
                            usage = dict(maybe_usage)
                    if chunk.get("mtplx_stats"):
                        maybe_stats = chunk.get("mtplx_stats")
                        if isinstance(maybe_stats, dict):
                            stats = dict(maybe_stats)
                    for choice in chunk.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if not isinstance(content, str) or not content:
                            continue
                        text_parts.append(content)
                        scan_text = (scan_text + content)[-COMMIT_SCAN_TAIL_CHARS:]
                        answer = _extract_verifier_stream_commit(scan_text)
                        if answer is not None:
                            break
                    if answer is not None:
                        break
            text = "".join(text_parts)
            texts.append(text)
            if answer is None:
                answer = _extract_verifier_final_answer(text)
            answers.append(answer)
            if not stats and usage:
                stats = dict(usage)
                stats.setdefault("effective_max_tokens", verifier_max_tokens)
            if not stats:
                with suppress(Exception):
                    stats = await runner._chat_metrics_factory(runner, request_id)
            stats_list.append(stats)
            if answer is None and _completion_hit_cap(stats, verifier_max_tokens):
                break
            resolved_answer, resolution = _resolve_verified_answer(
                proposed_answer, answers
            )
            if resolution in {"majority_keep_proposed", "majority_corrected"}:
                duration_ms = int(round((time.time() - started_at) * 1000.0))
                return AIMEAnswerVerification(
                    mode=runner.answer_verification,
                    proposed_answer=proposed_answer,
                    final_answer=resolved_answer,
                    answers=answers,
                    request_ids=request_ids,
                    texts=texts,
                    stats=stats_list,
                    resolution=resolution,
                    duration_ms=duration_ms,
                )

    resolved_answer, resolution = _resolve_verified_answer(proposed_answer, answers)
    duration_ms = int(round((time.time() - started_at) * 1000.0))
    return AIMEAnswerVerification(
        mode=runner.answer_verification,
        proposed_answer=proposed_answer,
        final_answer=resolved_answer,
        answers=answers,
        request_ids=request_ids,
        texts=texts,
        stats=stats_list,
        resolution=resolution,
        duration_ms=duration_ms,
    )


async def _noop_answer_verifier_factory(
    runner: "AIMERunner",
    problem: AIMEProblem,
    proposed_answer: int,
    answer_text: str,
) -> AIMEAnswerVerification | None:
    _ = runner
    _ = problem
    _ = proposed_answer
    _ = answer_text
    return None


# --- Runner ---------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class _EventBuffer:
    """Small ring buffer of recent events for SSE reconnect replay."""

    capacity: int = TAIL_BUFFER_EVENTS
    items: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def push(self, event: dict[str, Any]) -> None:
        self.items.append(event)
        if len(self.items) > self.capacity:
            del self.items[: len(self.items) - self.capacity]

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.items)


@dataclasses.dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[dict[str, Any]]
    last_replayed: bool = False


class AIMERunner:
    """Stateful AIME 2026 benchmark runner.

    Construct via :func:`start_run`. Direct construction is supported for
    tests (pass a custom ``chat_stream_factory``).
    """

    def __init__(
        self,
        *,
        problems: list[AIMEProblem],
        model_id: str,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        run_id: str | None = None,
        persist_dir: Path | None = None,
        chat_stream_factory: ChatStreamFactory | None = None,
        chat_cancel_factory: ChatCancelFactory | None = None,
        chat_metrics_factory: ChatMetricsFactory | None = None,
        question_isolation_factory: QuestionIsolationFactory | None = None,
        question_runtime_factory: QuestionRuntimeFactory | None = None,
        answer_verifier_factory: AnswerVerifierFactory | None = None,
        visible_submission_factory: VisibleSubmissionFactory | None = None,
        cap_recovery_factory: CapRecoveryFactory | None = None,
        answer_verification: str | None = None,
        answer_verification_attempts: int | None = None,
        cap_recovery: str | None = None,
        visible_submission_max_tokens: int | None = None,
        decode_profile: str | None = None,
        mtp_enabled: bool | None = None,
        depth: int | None = None,
        year: int = 2026,
    ) -> None:
        if not problems:
            raise ValueError("problems must be non-empty")
        self.run_id: str = run_id or _new_run_id(year)
        self.year: int = year
        self.problems: list[AIMEProblem] = list(problems)
        self.model_id: str = model_id
        self.base_url: str = base_url
        self._request_base_url: str = base_url
        self.api_key: str | None = api_key
        # All sampler/cap knobs default to None so the runner inherits
        # the daemon's per-target preset (the launch-QA contract from
        # commit 25ae0fe). Callers pass explicit values only when they
        # want to override the server default for a specific run.
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.top_k: int | None = top_k
        self.max_tokens: int | None = max_tokens
        self.enable_thinking: bool | None = enable_thinking
        default_thinking = (
            bool(enable_thinking)
            if enable_thinking is not None
            else default_enable_thinking_for_model(model_id)
        )
        self.answer_verification: str = _normalize_answer_verification_mode(
            answer_verification
            if answer_verification is not None
            else default_answer_verification_for_model(
                model_id, enable_thinking=default_thinking
            )
        )
        self.answer_verification_attempts: int = max(
            1, min(2, int(answer_verification_attempts or 2))
        )
        self.cap_recovery: str = _normalize_cap_recovery_mode(
            cap_recovery
            if cap_recovery is not None
            else default_cap_recovery_for_model(
                model_id, enable_thinking=default_thinking
            )
        )
        self.visible_submission_max_tokens: int = int(
            visible_submission_max_tokens
            if visible_submission_max_tokens is not None
            else VISIBLE_SUBMISSION_MAX_TOKENS
        )
        self.decode_profile: str | None = decode_profile
        self.mtp_enabled: bool | None = mtp_enabled
        self.depth: int | None = depth

        self._chat_stream_factory: ChatStreamFactory = (
            chat_stream_factory or default_chat_stream_factory
        )
        self._chat_cancel_factory: ChatCancelFactory = (
            chat_cancel_factory
            if chat_cancel_factory is not None
            else (
                default_chat_cancel_factory
                if chat_stream_factory is None
                else _noop_chat_cancel_factory
            )
        )
        self._chat_metrics_factory: ChatMetricsFactory = (
            chat_metrics_factory
            if chat_metrics_factory is not None
            else (
                default_chat_metrics_factory
                if chat_stream_factory is None
                else _noop_chat_metrics_factory
            )
        )
        self._question_isolation_factory: QuestionIsolationFactory = (
            question_isolation_factory or _noop_question_isolation_factory
        )
        self._question_runtime_factory: QuestionRuntimeFactory = (
            question_runtime_factory or _noop_question_runtime_factory
        )
        self._answer_verifier_factory: AnswerVerifierFactory = (
            answer_verifier_factory
            if answer_verifier_factory is not None
            else (
                default_answer_verifier_factory
                if self.answer_verification != "off" and chat_stream_factory is None
                else _noop_answer_verifier_factory
            )
        )
        self._visible_submission_enabled: bool = (
            visible_submission_factory is not None or chat_stream_factory is None
        )
        self._visible_submission_factory: VisibleSubmissionFactory = (
            visible_submission_factory
            if visible_submission_factory is not None
            else (
                default_visible_submission_factory
                if chat_stream_factory is None
                else _noop_visible_submission_factory
            )
        )
        self._cap_recovery_factory: CapRecoveryFactory = (
            cap_recovery_factory
            if cap_recovery_factory is not None
            else (
                default_cap_recovery_factory
                if chat_stream_factory is None
                else _noop_cap_recovery_factory
            )
        )
        self._persist_dir: Path = persist_dir or DEFAULT_PERSIST_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._persist_path: Path = self._persist_dir / f"{self.run_id}.jsonl"

        self.state: RunState = RunState.IDLE
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.results: list[QuestionResult] = [
            QuestionResult(idx=i + 1, problem=p) for i, p in enumerate(self.problems)
        ]
        self.current_idx: int = 0  # 0 = not yet started; 1..30 once started

        self._task: asyncio.Task[None] | None = None
        self._pause_event: asyncio.Event = asyncio.Event()
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._skip_event: asyncio.Event = asyncio.Event()
        self._skip_requested_idx: int | None = None
        self._pause_announced: bool = False
        self._attempts_by_idx: dict[int, int] = {}
        self._current_attempt: int = 0
        self._current_request_id: str | None = None
        self._current_question_runtime: AIMEQuestionRuntime | None = None
        self._current_question_runtime_previous_base_url: str | None = None
        self._pending_question_runtime_cleanups: dict[str, dict[str, Any]] = {}
        self._subscribers: list[_Subscriber] = []
        self._buffer: _EventBuffer = _EventBuffer()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._finished_at_monotonic: float | None = None  # for retention GC

    # ----- public read-only properties --------------------------------------

    @property
    def total(self) -> int:
        return len(self.problems)

    @property
    def score(self) -> int:
        return sum(1 for r in self.results if r.status == "correct")

    @property
    def is_terminal(self) -> bool:
        return self.state in {RunState.DONE, RunState.CANCELLED, RunState.ERROR}

    @property
    def current_attempt(self) -> int:
        return self._current_attempt

    @property
    def current_request_id(self) -> str | None:
        return self._current_request_id

    @property
    def request_base_url(self) -> str:
        return self._request_base_url

    def request_id_for(self, idx: int, attempt: int) -> str:
        return f"chatcmpl-{self.run_id}-q{int(idx)}-a{int(attempt)}"

    def accuracy(self) -> float | None:
        completed = sum(
            1
            for r in self.results
            if r.status in {"correct", "wrong", "abstain"}
        )
        if completed == 0:
            return None
        return self.score / completed

    def elapsed_ms(self) -> int | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.time()
        return int(round((end - self.started_at) * 1000.0))

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for ``GET .../{run_id}``."""
        return {
            "run_id": self.run_id,
            "year": self.year,
            "state": self.state.value,
            "model": self.model_id,
            "total": self.total,
            "score": self.score,
            "accuracy": self.accuracy(),
            "current_idx": self.current_idx,
            "current_attempt": self._current_attempt,
            "current_request_id": self._current_request_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_ms": self.elapsed_ms(),
            "paused": self._pause_event.is_set(),
            "decode_profile": self.decode_profile,
            "mtp_enabled": self.mtp_enabled,
            "depth": self.depth,
            "answer_verification": self.answer_verification,
            "prompts": self._prompt_provenance(),
            "rescue_policy": self._rescue_policy_payload(),
            "per_question": [
                {
                    "idx": r.idx,
                    "id": r.problem.id,
                    "set": r.problem.set,
                    "expected": r.problem.answer,
                    "extracted": r.extracted,
                    "status": r.status,
                    "attempts": self._attempts_by_idx.get(r.idx, 0),
                    "duration_ms": r.duration_ms,
                    "reasoning_token_count": r.reasoning_token_count,
                    "answer_token_count": r.answer_token_count,
                    "answer_verification_resolution": (
                        r.answer_verification.resolution
                        if r.answer_verification is not None
                        else None
                    ),
                }
                for r in self.results
            ],
        }

    # ----- lifecycle --------------------------------------------------------

    async def start(self) -> asyncio.Task[None]:
        """Launch the runner's outer asyncio task. Returns the task."""
        if self._task is not None:
            return self._task
        self.started_at = time.time()
        self.state = RunState.RUNNING
        await self._push(
            {
                "event": "run_started",
                "run_id": self.run_id,
                "total": self.total,
                "model": self.model_id,
                "started_at": _iso(self.started_at),
                "answer_verification": self.answer_verification,
            }
        )
        self._task = asyncio.create_task(
            self._run_outer(), name=f"aime-runner-{self.run_id}"
        )
        return self._task

    async def pause(self) -> RunState:
        if self.is_terminal:
            return self.state
        if not self._pause_event.is_set():
            self._pause_event.set()
            self._pause_announced = False
        if self.current_idx <= 0 and self.state == RunState.RUNNING:
            await self._enter_paused()
        return self.state

    async def resume(self) -> RunState:
        if self.is_terminal:
            return self.state
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_announced = False
            await self._push({"event": "run_resumed", "run_id": self.run_id})
        if self.state == RunState.PAUSED:
            self.state = RunState.RUNNING
        return self.state

    async def cancel(self) -> RunState:
        if self.is_terminal:
            return self.state
        self._cancel_event.set()
        await self._cancel_active_request("run_cancel")
        if self._current_question_runtime is not None:
            request_id = (
                self._current_request_id
                or self.request_id_for(
                    self.current_idx or 1,
                    self._current_attempt or 1,
                )
            )
            cleanup = await self._cleanup_current_question_runtime(
                idx=self.current_idx or 0,
                attempt=self._current_attempt or 1,
                request_id=request_id,
            )
            if cleanup is not None:
                self._pending_question_runtime_cleanups[request_id] = cleanup
        # Clear pause so the loop wakes up and observes cancel.
        if self._pause_event.is_set():
            self._pause_event.clear()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        return self.state

    async def skip_current(self) -> RunState:
        """Mark the current problem as an abstain and continue the run.

        This is deliberately a user-visible control, not a hidden reasoning
        cap. It aborts only the in-flight chat stream for the active problem,
        persists that row as ``status="abstain"`` with ``error="skipped_by_user"``,
        and lets the run advance to the next question.
        """

        if self.is_terminal or self.current_idx <= 0:
            return self.state
        if self.current_idx > self.total:
            return self.state
        result = self.results[self.current_idx - 1]
        if result.status is not None:
            return self.state
        self._skip_requested_idx = self.current_idx
        self._skip_event.set()
        return self.state

    # ----- subscribers (for SSE consumers) ----------------------------------

    def subscribe(self) -> tuple[asyncio.Queue[dict[str, Any]], list[dict[str, Any]]]:
        """Subscribe a new SSE consumer. Returns (queue, replay tail).

        The replay tail is a snapshot of the last ~256 events. The consumer
        should yield these first, then drain the queue. Subsequent events
        are pushed to the queue in real time.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(_Subscriber(queue=queue))
        return queue, self._buffer.snapshot()

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers = [s for s in self._subscribers if s.queue is not queue]

    # ----- private: outer task wrapper --------------------------------------

    async def _run_outer(self) -> None:
        try:
            await self._run_loop()
            self._on_done(state=RunState.DONE)
            await self._push(self._run_done_payload())
        except asyncio.CancelledError:
            self._on_done(state=RunState.CANCELLED)
            await self._push(self._run_cancelled_payload())
            self._persist_summary()
            raise
        except Exception as exc:  # noqa: BLE001 — final-line safety net
            logger.exception("AIMERunner crashed: %s", exc)
            self._on_done(state=RunState.ERROR)
            await self._push(
                {
                    "event": "error",
                    "run_id": self.run_id,
                    "message": str(exc),
                    "recoverable": False,
                }
            )
            self._persist_summary()
        else:
            self._persist_summary()

    async def _run_loop(self) -> None:
        idx = 1
        while idx <= self.total:
            if self._cancel_event.is_set():
                raise asyncio.CancelledError
            await self._wait_if_paused()
            try:
                paused = await self._run_one_question(idx)
            finally:
                if self._current_question_runtime is not None:
                    await self._cleanup_current_question_runtime(
                        idx=idx,
                        attempt=self._current_attempt or 1,
                        request_id=(
                            self._current_request_id
                            or self.request_id_for(idx, self._current_attempt or 1)
                        ),
                    )
            if paused:
                await self._wait_if_paused()
                continue
            idx += 1

    async def _wait_if_paused(self) -> None:
        if not self._pause_event.is_set():
            return
        await self._enter_paused()
        while self._pause_event.is_set():
            if self._cancel_event.is_set():
                raise asyncio.CancelledError
            await asyncio.sleep(0.05)
        self.state = RunState.RUNNING
        self._pause_announced = False

    async def _enter_paused(self) -> None:
        self.state = RunState.PAUSED
        if self._pause_announced:
            return
        self._pause_announced = True
        await self._push(
            {
                "event": "run_paused",
                "run_id": self.run_id,
                "idx": self.current_idx,
                "attempt": self._current_attempt,
                "request_id": self._current_request_id,
            }
        )

    def _primary_enable_thinking(self) -> bool:
        if self.enable_thinking is not None:
            return bool(self.enable_thinking)
        return default_enable_thinking_for_model(self.model_id)

    def _prompt_provenance(self) -> dict[str, Any]:
        """The exact solver prompts this run used, for reproducibility.

        Embedded in snapshots and the run summary so every published score
        can be traced to the precise prompt text that produced it.
        """

        thinking = self._primary_enable_thinking()
        return {
            "enable_thinking": thinking,
            "system": SYSTEM_PROMPT if thinking else FAST_SYSTEM_PROMPT,
            "user_suffix": (
                USER_PROMPT_SUFFIX if thinking else FAST_USER_PROMPT_SUFFIX
            ),
        }

    def _rescue_policy_payload(self) -> dict[str, Any]:
        """Disclose whether any pass may override the extracted answer.

        ``answer_verification`` and ``cap_recovery`` default to "off" for
        every model except Gemma-4 in non-thinking mode (the documented
        exception); a score is rescue-free exactly when ``active`` is False.
        """

        active = (
            self.answer_verification != "off" or self.cap_recovery != "off"
        )
        return {
            "answer_verification": self.answer_verification,
            "answer_verification_attempts": int(
                self.answer_verification_attempts
            ),
            "cap_recovery": self.cap_recovery,
            "active": active,
            "gemma_non_thinking_exception": bool(
                active and not self._primary_enable_thinking()
            ),
        }

    def _should_run_cap_recovery(
        self, usage: Mapping[str, Any], stats: Mapping[str, Any]
    ) -> bool:
        if self.cap_recovery == "off":
            return False
        if self._primary_enable_thinking():
            return False
        merged = dict(usage or {})
        merged.update(dict(stats or {}))
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens_for_model(
                self.model_id, enable_thinking=self._primary_enable_thinking()
            )
        effective = (
            merged.get("effective_max_tokens")
            or merged.get("request_max_tokens")
            or max_tokens
        )
        if effective is None:
            return False
        try:
            cap = int(effective)
        except (TypeError, ValueError):
            return False
        return _completion_hit_cap(merged, cap)

    def _should_run_visible_submission(self, result: QuestionResult) -> bool:
        if not self._visible_submission_enabled:
            return False
        if not self._primary_enable_thinking():
            return False
        if result.extracted is not None:
            return False
        return bool((result.reasoning_text or "").strip())

    async def _run_one_question(self, idx: int) -> bool:
        result = self.results[idx - 1]
        problem = result.problem
        self.current_idx = idx
        attempt = self._attempts_by_idx.get(idx, 0) + 1
        self._attempts_by_idx[idx] = attempt
        self._current_attempt = attempt
        request_id = self.request_id_for(idx, attempt)
        self._current_request_id = request_id
        self._skip_requested_idx = None
        self._skip_event.clear()
        result.started_at = time.time()
        result.ended_at = None
        result.extracted = None
        result.status = None
        result.reasoning_token_count = 0
        result.answer_token_count = 0
        result.reasoning_text = ""
        result.answer_text = ""
        result.stream_reasoning_events = 0
        result.stream_answer_events = 0
        result.stream_reasoning_chars = 0
        result.stream_answer_chars = 0
        result.stream_progress_events = 0
        result.stream_progress_milestones = {}
        result.reasoning_finalizer_handoff = False
        result.reasoning_finalizer_trigger_answer = None
        result.reasoning_finalizer_trigger_source = None
        result.reasoning_finalizer_trigger_completion_tokens = None
        result.reasoning_finalizer_grace_tokens = 0
        result.visible_submission = None
        result.answer_verification = None
        result.cap_recovery = None
        result.error = None
        await self._start_question_runtime(
            problem,
            idx=idx,
            attempt=attempt,
            request_id=request_id,
        )
        await self._push(
            {
                "event": "question_started",
                "run_id": self.run_id,
                "idx": idx,
                "attempt": attempt,
                "request_id": request_id,
                "id": problem.id,
                "set": problem.set,
                "year": problem.year,
                "problem": problem.problem,
            }
        )

        reasoning_parts: list[str] = []
        answer_parts: list[str] = []
        usage: dict[str, Any] = {}
        stats: dict[str, Any] = {}
        commit_source: str | None = None
        visible_stream_commit: tuple[int, str] | None = None

        try:
            stream = await self._chat_stream_factory(self, problem)
            iterator = stream.__aiter__()
            while True:
                chunk = await self._next_chunk_or_control(iterator, idx)
                if chunk is None:
                    break
                progress = chunk.get("mtplx_progress")
                if isinstance(progress, Mapping):
                    _observe_stream_progress(result, progress)
                    await self._push(
                        _question_progress_event(
                            run_id=self.run_id,
                            idx=idx,
                            attempt=attempt,
                            request_id=request_id,
                            progress=progress,
                        )
                    )
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("mtplx_stats"):
                    stats = chunk["mtplx_stats"]
                choices = chunk.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta") or {}
                    reasoning_text = delta.get("reasoning_content") or ""
                    answer_text = delta.get("content") or ""
                    if reasoning_text:
                        reasoning_text = str(reasoning_text)
                        reasoning_parts.append(reasoning_text)
                        result.stream_reasoning_events += 1
                        result.stream_reasoning_chars += len(reasoning_text)
                        await self._push(
                            {
                                "event": "reasoning_delta",
                                "run_id": self.run_id,
                                "idx": idx,
                                "attempt": attempt,
                                "request_id": request_id,
                                "text": reasoning_text,
                            }
                        )
                    if answer_text:
                        answer_text = str(answer_text)
                        answer_parts.append(answer_text)
                        result.stream_answer_events += 1
                        result.stream_answer_chars += len(answer_text)
                        await self._push(
                            {
                                "event": "answer_delta",
                                "run_id": self.run_id,
                                "idx": idx,
                                "attempt": attempt,
                                "request_id": request_id,
                                "text": answer_text,
                            }
                        )
                        stream_commit = _extract_stream_commit(
                            "".join(answer_parts)
                        )
                        if stream_commit is not None:
                            visible_stream_commit = stream_commit
                            break
                if visible_stream_commit is not None:
                    await self._cancel_active_request("visible_answer_commit")
                    aclose = getattr(iterator, "aclose", None)
                    if aclose is not None:
                        with suppress(Exception):
                            await aclose()
                    break
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("mtplx_stats"):
                    stats = chunk["mtplx_stats"]
        except _QuestionPaused:
            result.reasoning_text = "".join(reasoning_parts)
            result.answer_text = "".join(answer_parts)
            result.extracted = None
            result.status = None
            result.error = None
            stats = await self._record_question_boundary_cleanup(
                result,
                stats,
                request_id=request_id,
            )
            result.started_at = None
            result.ended_at = None
            await self._enter_paused()
            return True
        except _QuestionSkipped:
            await self._persist_controlled_question_stop(
                result,
                reasoning_parts,
                answer_parts,
                usage,
                stats,
                request_id=request_id,
                attempt=attempt,
                status="abstain",
                error="skipped_by_user",
                commit_source="user_skip",
                emit_question_done=True,
            )
            return False
        except asyncio.CancelledError:
            # Persist partial result then propagate.
            await self._persist_controlled_question_stop(
                result,
                reasoning_parts,
                answer_parts,
                usage,
                stats,
                request_id=request_id,
                attempt=attempt,
                status=None,
                error="cancelled",
                commit_source=None,
                emit_question_done=False,
            )
            raise

        result.reasoning_text = "".join(reasoning_parts)
        result.answer_text = "".join(answer_parts)
        stats = await self._backfill_question_metrics(stats, request_id)
        _populate_result_token_counts(result, usage, stats)

        final_answer = _extract_final_visible_answer(result.answer_text)
        if final_answer is None:
            if visible_stream_commit is None:
                result.extracted = None
            else:
                result.extracted, stream_source = visible_stream_commit
                commit_source = f"visible_stream_{stream_source}"
        else:
            result.extracted, commit_source = final_answer
            if visible_stream_commit is not None:
                _, stream_source = visible_stream_commit
                commit_source = f"visible_stream_{stream_source}"

        if self._should_run_visible_submission(result):
            result.reasoning_finalizer_handoff = True
            result.reasoning_finalizer_trigger_source = (
                "reasoning_only_no_visible_answer"
            )
            result.reasoning_finalizer_trigger_completion_tokens = (
                _int_stat(
                    stats,
                    usage,
                    "completion_tokens",
                    "answer_tokens",
                )
            )
            try:
                submission = await self._visible_submission_factory(
                    self,
                    problem,
                    result.reasoning_text,
                    result.answer_text,
                    {**dict(usage or {}), **dict(stats or {})},
                )
            except _QuestionPaused:
                result.reasoning_text = "".join(reasoning_parts)
                result.answer_text = "".join(answer_parts)
                result.extracted = None
                result.status = None
                result.error = None
                stats = await self._record_question_boundary_cleanup(
                    result,
                    stats,
                    request_id=request_id,
                )
                result.started_at = None
                result.ended_at = None
                await self._enter_paused()
                return True
            except _QuestionSkipped:
                await self._persist_controlled_question_stop(
                    result,
                    reasoning_parts,
                    answer_parts,
                    usage,
                    stats,
                    request_id=request_id,
                    attempt=attempt,
                    status="abstain",
                    error="skipped_by_user",
                    commit_source="user_skip",
                    emit_question_done=True,
                )
                return False
            except asyncio.CancelledError:
                await self._persist_controlled_question_stop(
                    result,
                    reasoning_parts,
                    answer_parts,
                    usage,
                    stats,
                    request_id=request_id,
                    attempt=attempt,
                    status=None,
                    error="cancelled",
                    commit_source=commit_source,
                    emit_question_done=False,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - abstain stays visible
                logger.warning("AIME visible submission failed: %s", exc)
                result.error = f"visible_submission_failed: {exc}"
            else:
                if submission is not None:
                    result.visible_submission = submission
                    result.reasoning_text += submission.reasoning_text
                    result.answer_text += submission.answer_text
                    submission_usage = submission.usage or {}
                    submission_stats = submission.stats or {}
                    result.reasoning_token_count += int(
                        submission_usage.get("reasoning_tokens")
                        or submission_usage.get("completion_tokens_details", {}).get(
                            "reasoning_tokens"
                        )
                        or submission_stats.get("reasoning_tokens")
                        or submission_stats.get("completion_tokens_details", {}).get(
                            "reasoning_tokens"
                        )
                        or 0
                    )
                    result.answer_token_count += int(
                        submission_usage.get("answer_tokens")
                        or submission_stats.get("answer_tokens")
                        or submission_usage.get("completion_tokens")
                        or submission_stats.get("completion_tokens")
                        or 0
                    )
                    if submission.final_answer is not None:
                        result.extracted = submission.final_answer
                        commit_source = (
                            f"visible_submission_{submission.commit_source}"
                            if submission.commit_source
                            else "visible_submission_answer"
                        )
                    else:
                        commit_source = "visible_submission_no_answer"

        should_run_cap_recovery = self._should_run_cap_recovery(usage, stats)
        if result.extracted is None and should_run_cap_recovery:
            cap_recovery_scratch = result.answer_text or result.reasoning_text
            previous_cap_recovery = self.cap_recovery
            try:
                recovery = await self._cap_recovery_factory(
                    self,
                    problem,
                    cap_recovery_scratch,
                    {**dict(usage or {}), **dict(stats or {})},
                )
            except _QuestionPaused:
                result.reasoning_text = "".join(reasoning_parts)
                result.answer_text = "".join(answer_parts)
                result.extracted = None
                result.status = None
                result.error = None
                stats = await self._record_question_boundary_cleanup(
                    result,
                    stats,
                    request_id=request_id,
                )
                result.started_at = None
                result.ended_at = None
                await self._enter_paused()
                return True
            except _QuestionSkipped:
                await self._persist_controlled_question_stop(
                    result,
                    reasoning_parts,
                    answer_parts,
                    usage,
                    stats,
                    request_id=request_id,
                    attempt=attempt,
                    status="abstain",
                    error="skipped_by_user",
                    commit_source="user_skip",
                    emit_question_done=True,
                )
                return False
            except asyncio.CancelledError:
                await self._persist_controlled_question_stop(
                    result,
                    reasoning_parts,
                    answer_parts,
                    usage,
                    stats,
                    request_id=request_id,
                    attempt=attempt,
                    status=None,
                    error="cancelled",
                    commit_source=commit_source,
                    emit_question_done=False,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - abstain stays visible
                logger.warning("AIME cap recovery failed: %s", exc)
                result.error = f"cap_recovery_failed: {exc}"
            else:
                if recovery is not None:
                    result.cap_recovery = recovery
                    result.reasoning_text += recovery.reasoning_text
                    result.answer_text += recovery.answer_text
                    recovery_usage = recovery.usage or {}
                    recovery_stats = recovery.stats or {}
                    result.reasoning_token_count += int(
                        recovery_usage.get("reasoning_tokens")
                        or recovery_usage.get("completion_tokens_details", {}).get(
                            "reasoning_tokens"
                        )
                        or recovery_stats.get("reasoning_tokens")
                        or recovery_stats.get("completion_tokens_details", {}).get(
                            "reasoning_tokens"
                        )
                        or 0
                    )
                    result.answer_token_count += int(
                        recovery_usage.get("completion_tokens", 0)
                        or recovery_stats.get("completion_tokens")
                        or 0
                    )
                    if recovery.final_answer is not None:
                        result.extracted = recovery.final_answer
                        commit_source = (
                            f"cap_recovery_{recovery.commit_source}"
                            if recovery.commit_source
                            else "cap_recovery_answer"
                        )
                    else:
                        commit_source = "cap_recovery_no_answer"
            finally:
                self.cap_recovery = previous_cap_recovery

        if result.extracted is not None and self.answer_verification != "off":
            await self._push(
                {
                    "event": "answer_verification_started",
                    "run_id": self.run_id,
                    "idx": idx,
                    "attempt": attempt,
                    "request_id": request_id,
                    "mode": self.answer_verification,
                    "proposed_answer": result.extracted,
                }
            )
            try:
                verification = await self._answer_verifier_factory(
                    self, problem, result.extracted, result.answer_text
                )
            except Exception as exc:  # noqa: BLE001 - keep the scored run visible
                logger.warning("AIME answer verification failed: %s", exc)
                result.error = f"answer_verification_failed: {exc}"
                await self._push(
                    {
                        "event": "answer_verification_done",
                        "run_id": self.run_id,
                        "idx": idx,
                        "attempt": attempt,
                        "request_id": request_id,
                        "mode": self.answer_verification,
                        "proposed_answer": result.extracted,
                        "verified_answer": None,
                        "resolution": "error_keep_proposed",
                        "error": str(exc),
                    }
                )
            else:
                if verification is not None:
                    result.answer_verification = verification
                    result.extracted = verification.final_answer
                    commit_source = f"answer_verification_{verification.resolution}"
                    await self._push(
                        {
                            "event": "answer_verification_done",
                            "run_id": self.run_id,
                            "idx": idx,
                            "attempt": attempt,
                            "request_id": request_id,
                            "mode": verification.mode,
                            "proposed_answer": verification.proposed_answer,
                            "verified_answer": verification.final_answer,
                            "verifier_answers": verification.answers,
                            "verifier_request_ids": verification.request_ids,
                            "resolution": verification.resolution,
                            "duration_ms": verification.duration_ms,
                        }
                    )
        result.ended_at = time.time()
        result.status = grade(result.extracted, problem.answer)
        stats = await self._record_question_boundary_cleanup(
            result,
            stats,
            request_id=request_id,
        )

        await self._push(
            {
                "event": "question_done",
                "run_id": self.run_id,
                "idx": idx,
                "attempt": attempt,
                "request_id": request_id,
                "id": problem.id,
                "extracted": result.extracted,
                "expected": problem.answer,
                "status": result.status,
                "duration_ms": result.duration_ms,
                "reasoning_token_count": result.reasoning_token_count,
                "answer_token_count": result.answer_token_count,
                "commit_source": commit_source,
            }
        )

        self._persist_question(
            result,
            usage,
            stats,
            request_id=request_id,
            attempt=attempt,
            commit_source=commit_source,
        )
        return False

    async def _backfill_question_metrics(
        self,
        stats: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        if not _question_metrics_need_backfill(stats):
            return stats
        with suppress(Exception):
            fetched_stats = await self._chat_metrics_factory(self, request_id)
            if fetched_stats:
                return {**dict(stats), **dict(fetched_stats)}
        return stats

    async def _record_question_boundary_cleanup(
        self,
        result: QuestionResult,
        stats: dict[str, Any],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        question_cleanup = await self._isolate_question_boundary(
            result,
            request_id=request_id,
        )
        runtime_cleanup = await self._cleanup_current_question_runtime(
            idx=result.idx,
            attempt=self._current_attempt or 1,
            request_id=request_id,
        )
        if runtime_cleanup is None:
            runtime_cleanup = self._pending_question_runtime_cleanups.pop(
                request_id, None
            )
        else:
            self._pending_question_runtime_cleanups.pop(request_id, None)
        if question_cleanup is not None or runtime_cleanup is not None:
            stats = dict(stats or {})
            if question_cleanup is not None:
                stats["aime_question_boundary_cleanup"] = question_cleanup
            if runtime_cleanup is not None:
                stats["aime_question_runtime_cleanup"] = runtime_cleanup
        return stats

    async def _start_question_runtime(
        self,
        problem: AIMEProblem,
        *,
        idx: int,
        attempt: int,
        request_id: str,
    ) -> None:
        if self._current_question_runtime is not None:
            await self._cleanup_current_question_runtime(
                idx=idx,
                attempt=attempt,
                request_id=request_id,
            )
        runtime = await self._question_runtime_factory(self, problem)
        if runtime is None:
            return
        previous_base_url = self._request_base_url
        self._current_question_runtime = runtime
        self._current_question_runtime_previous_base_url = previous_base_url
        self._request_base_url = runtime.base_url or previous_base_url
        await self._push(
            {
                "event": "question_runtime_started",
                "run_id": self.run_id,
                "idx": idx,
                "attempt": attempt,
                "request_id": request_id,
                "base_url": self._request_base_url,
                "metadata": dict(runtime.metadata or {}),
            }
        )

    async def _cleanup_current_question_runtime(
        self,
        *,
        idx: int,
        attempt: int,
        request_id: str,
    ) -> dict[str, Any] | None:
        runtime = self._current_question_runtime
        if runtime is None:
            return None
        self._current_question_runtime = None
        previous_base_url = (
            self._current_question_runtime_previous_base_url or self.base_url
        )
        self._current_question_runtime_previous_base_url = None
        cleanup: dict[str, Any] | None = None
        ok = True
        if runtime.cleanup is not None:
            try:
                cleanup_result = await runtime.cleanup()
            except Exception as exc:  # noqa: BLE001 - cleanup failure is evidence
                cleanup = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                ok = False
            else:
                if cleanup_result is None:
                    cleanup = None
                else:
                    cleanup = dict(cleanup_result)
                    if cleanup.get("ok") is False:
                        ok = False
        self._request_base_url = previous_base_url
        payload = {
            "ok": ok,
            "base_url": runtime.base_url,
            "metadata": dict(runtime.metadata or {}),
            "cleanup": cleanup,
        }
        await self._push(
            {
                "event": "question_runtime_stopped",
                "run_id": self.run_id,
                "idx": idx,
                "attempt": attempt,
                "request_id": request_id,
                **payload,
            }
        )
        return payload

    async def _persist_controlled_question_stop(
        self,
        result: QuestionResult,
        reasoning_parts: Sequence[str],
        answer_parts: Sequence[str],
        usage: dict[str, Any],
        stats: dict[str, Any],
        *,
        request_id: str,
        attempt: int,
        status: GradeStatus | None,
        error: str,
        commit_source: str | None,
        emit_question_done: bool,
    ) -> None:
        result.ended_at = time.time()
        result.reasoning_text = "".join(reasoning_parts)
        result.answer_text = "".join(answer_parts)
        result.extracted = None
        result.status = status
        result.error = error
        stats = await self._backfill_question_metrics(stats, request_id)
        _populate_result_token_counts(result, usage, stats)
        stats = await self._record_question_boundary_cleanup(
            result,
            stats,
            request_id=request_id,
        )
        if emit_question_done:
            await self._push(
                {
                    "event": "question_done",
                    "run_id": self.run_id,
                    "idx": result.idx,
                    "attempt": attempt,
                    "request_id": request_id,
                    "id": result.problem.id,
                    "extracted": result.extracted,
                    "expected": result.problem.answer,
                    "status": result.status,
                    "duration_ms": result.duration_ms,
                    "reasoning_token_count": result.reasoning_token_count,
                    "answer_token_count": result.answer_token_count,
                    "commit_source": commit_source,
                }
            )
        self._persist_question(
            result,
            usage,
            stats,
            request_id=request_id,
            attempt=attempt,
            commit_source=commit_source,
        )

    async def _isolate_question_boundary(
        self,
        result: QuestionResult,
        *,
        request_id: str,
    ) -> dict[str, Any] | None:
        try:
            cleanup = await self._question_isolation_factory(
                self,
                result,
                request_id,
            )
        except Exception as exc:  # noqa: BLE001 - isolation failure is evidence
            cleanup = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if cleanup is None:
            return None
        await self._push(
            {
                "event": "question_isolation_cleanup",
                "run_id": self.run_id,
                "idx": result.idx,
                "attempt": self._current_attempt,
                "request_id": request_id,
                "cleanup": cleanup,
            }
        )
        return cleanup

    async def _next_chunk_or_control(
        self, iterator: AsyncIterator[dict[str, Any]], idx: int
    ) -> dict[str, Any] | None:
        next_task = asyncio.create_task(iterator.__anext__())
        skip_task = asyncio.create_task(self._skip_event.wait())
        pause_task = asyncio.create_task(self._pause_event.wait())
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {next_task, skip_task, pause_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done and self._cancel_event.is_set():
                await self._cancel_active_request("run_cancel")
                await self._close_iterator_after_control(next_task, iterator)
                raise asyncio.CancelledError
            if skip_task in done and self._skip_requested_idx == idx:
                await self._cancel_active_request("question_skip")
                await self._close_iterator_after_control(next_task, iterator)
                raise _QuestionSkipped
            if pause_task in done and self._pause_event.is_set():
                await self._cancel_active_request("question_pause")
                await self._close_iterator_after_control(next_task, iterator)
                raise _QuestionPaused

            skip_task.cancel()
            with suppress(asyncio.CancelledError):
                await skip_task
            pause_task.cancel()
            with suppress(asyncio.CancelledError):
                await pause_task
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task
            if next_task in done:
                try:
                    return next_task.result()
                except StopAsyncIteration:
                    return None
            try:
                return await next_task
            except StopAsyncIteration:
                return None
        finally:
            if not next_task.done():
                next_task.cancel()
            if not skip_task.done():
                skip_task.cancel()
            if not pause_task.done():
                pause_task.cancel()
            if not cancel_task.done():
                cancel_task.cancel()

    async def _close_iterator_after_control(
        self,
        next_task: asyncio.Task[dict[str, Any]],
        iterator: AsyncIterator[dict[str, Any]],
    ) -> None:
        next_task.cancel()
        with suppress(asyncio.CancelledError, StopAsyncIteration):
            await next_task
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()

    async def _cancel_active_request(self, reason: str) -> None:
        request_id = self._current_request_id
        if not request_id:
            return
        try:
            await self._chat_cancel_factory(self, request_id)
        except Exception as exc:  # noqa: BLE001 - cancellation remains best effort
            logger.warning(
                "AIMERunner: failed to cancel active chat request %s (%s): %s",
                request_id,
                reason,
                exc,
            )

    # ----- bookkeeping ------------------------------------------------------

    def _on_done(self, *, state: RunState) -> None:
        self.state = state
        self.ended_at = time.time()
        self._finished_at_monotonic = time.monotonic()
        self.current_idx = 0
        self._pause_event.clear()
        self._pause_announced = False
        self._current_attempt = 0
        self._current_request_id = None

    def _run_done_payload(self) -> dict[str, Any]:
        return {
            "event": "run_done",
            "run_id": self.run_id,
            "state": self.state.value,
            "score": self.score,
            "total": self.total,
            "accuracy": self.accuracy(),
            "duration_ms": self.elapsed_ms(),
            "model": self.model_id,
            "prompts": self._prompt_provenance(),
            "rescue_policy": self._rescue_policy_payload(),
            "per_question": [
                {
                    "idx": r.idx,
                    "id": r.problem.id,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ],
        }

    def _run_cancelled_payload(self) -> dict[str, Any]:
        completed = [r for r in self.results if r.status is not None]
        return {
            "event": "run_cancelled",
            "run_id": self.run_id,
            "state": self.state.value,
            "score": self.score,
            "completed": len(completed),
            "total": self.total,
            "duration_ms": self.elapsed_ms(),
            "model": self.model_id,
        }

    async def _push(self, event: dict[str, Any]) -> None:
        self._buffer.push(event)
        # Fan out to all subscribers. Each subscriber owns its own queue;
        # if a slow consumer falls behind we keep enqueueing (queue is
        # unbounded). The reconnect tail buffer is the durable record.
        for sub in list(self._subscribers):
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover — queues are unbounded
                logger.warning("dropping event for full subscriber queue")

    # ----- persistence ------------------------------------------------------

    def _persist_question(
        self,
        result: QuestionResult,
        usage: dict[str, Any],
        stats: dict[str, Any],
        *,
        request_id: str,
        attempt: int,
        commit_source: str | None = None,
    ) -> None:
        merged_stats = dict(stats or {})
        if not merged_stats:
            merged_stats = dict(usage or {})

        def stat(*keys: str) -> Any:
            for key in keys:
                if key in merged_stats:
                    return merged_stats.get(key)
                if key in usage:
                    return usage.get(key)
            return None

        request_session_bank_bypass = stat("request_session_bank_bypass")
        cache_miss_reason = stat("cache_miss_reason")
        if cache_miss_reason is None and request_session_bank_bypass is True:
            cache_miss_reason = "request_cache_bypass"
        session_cache_hit = stat("session_cache_hit")
        if session_cache_hit is None and request_session_bank_bypass is True:
            session_cache_hit = False
        verification = result.answer_verification
        visible_submission = result.visible_submission
        visible_submission_stats = (
            dict(visible_submission.stats or {})
            if visible_submission is not None
            else {}
        )
        visible_submission_usage = (
            dict(visible_submission.usage or {})
            if visible_submission is not None
            else {}
        )
        recovery = result.cap_recovery
        recovery_stats = dict(recovery.stats or {}) if recovery is not None else {}
        recovery_usage = dict(recovery.usage or {}) if recovery is not None else {}

        def visible_submission_stat(*keys: str) -> Any:
            for key in keys:
                if key in visible_submission_stats:
                    return visible_submission_stats.get(key)
                if key in visible_submission_usage:
                    return visible_submission_usage.get(key)
            return None

        def recovery_stat(*keys: str) -> Any:
            for key in keys:
                if key in recovery_stats:
                    return recovery_stats.get(key)
                if key in recovery_usage:
                    return recovery_usage.get(key)
            return None

        cap_recovery_enable_thinking = recovery_stat("request_enable_thinking")
        if cap_recovery_enable_thinking is None and recovery is not None:
            cap_recovery_enable_thinking = recovery.mode == "thinking_retry"
        visible_submission_enable_thinking = visible_submission_stat(
            "request_enable_thinking"
        )
        if (
            visible_submission_enable_thinking is None
            and visible_submission is not None
        ):
            visible_submission_enable_thinking = True

        row = {
            "run_id": self.run_id,
            "year": self.year,
            "model": self.model_id,
            "decode_profile": self.decode_profile,
            "mtp_enabled": self.mtp_enabled,
            "depth": self.depth,
            "request_id": request_id,
            "attempt": attempt,
            "idx": result.idx,
            "id": result.problem.id,
            "set": result.problem.set,
            "expected": result.problem.answer,
            "extracted": result.extracted,
            "status": result.status,
            "started_at": _iso(result.started_at),
            "ended_at": _iso(result.ended_at),
            "duration_ms": result.duration_ms,
            "reasoning_token_count": result.reasoning_token_count,
            "answer_token_count": result.answer_token_count,
            "total_token_count": (
                result.reasoning_token_count + result.answer_token_count
            ),
            "prompt_tokens": stat("prompt_tokens"),
            "completion_tokens": stat("completion_tokens"),
            "cached_tokens": stat("cached_tokens"),
            "new_prefill_tokens": stat("new_prefill_tokens"),
            "session_id": stat("session_id"),
            "session_cache_hit": session_cache_hit,
            "cache_source": stat("cache_source"),
            "cache_miss_reason": cache_miss_reason,
            "session_restore_mode": stat("session_restore_mode"),
            "decode_tps_p50": stat("decode_tps_p50"),
            "decode_tok_s": stat("decode_tok_s"),
            "request_tok_s": stat("request_tok_s"),
            "display_decode_tok_s": stat(
                "display_decode_tok_s",
                "server_tok_s",
                "tok_s",
            ),
            "sliding_decode_tok_s_first_32": stat("sliding_decode_tok_s_first_32"),
            "sliding_decode_tok_s_first_64": stat("sliding_decode_tok_s_first_64"),
            "sliding_decode_tok_s_first_128": stat("sliding_decode_tok_s_first_128"),
            "sliding_decode_tok_s_first_256": stat("sliding_decode_tok_s_first_256"),
            "sliding_decode_tok_s_last_32": stat("sliding_decode_tok_s_last_32"),
            "sliding_decode_tok_s_last_64": stat("sliding_decode_tok_s_last_64"),
            "sliding_decode_tok_s_last_128": stat("sliding_decode_tok_s_last_128"),
            "sliding_decode_tok_s_last_256": stat("sliding_decode_tok_s_last_256"),
            "stream_reasoning_events": result.stream_reasoning_events,
            "stream_answer_events": result.stream_answer_events,
            "stream_reasoning_chars": result.stream_reasoning_chars,
            "stream_answer_chars": result.stream_answer_chars,
            "stream_progress_events": result.stream_progress_events,
            "stream_progress_milestones": result.stream_progress_milestones,
            "reasoning_finalizer_handoff": result.reasoning_finalizer_handoff,
            "reasoning_finalizer_trigger_answer": (
                result.reasoning_finalizer_trigger_answer
            ),
            "reasoning_finalizer_trigger_source": (
                result.reasoning_finalizer_trigger_source
            ),
            "reasoning_finalizer_trigger_completion_tokens": (
                result.reasoning_finalizer_trigger_completion_tokens
            ),
            "reasoning_finalizer_grace_tokens": (
                result.reasoning_finalizer_grace_tokens
            ),
            "request_effective_mtp_depth": stat("request_effective_mtp_depth"),
            "accepted_by_depth": stat("accepted_by_depth"),
            "drafted_by_depth": stat("drafted_by_depth"),
            "mean_accept_probability_by_depth": stat(
                "mean_accept_probability_by_depth"
            ),
            "verify_calls": stat("verify_calls"),
            "verify_time_s": stat("verify_time_s"),
            "verify_target_distribution_time_s": stat(
                "verify_target_distribution_time_s"
            ),
            "target_forward_time_s": stat("target_forward_time_s"),
            "draft_time_s": stat("draft_time_s"),
            "active_memory_bytes": stat("active_memory_bytes"),
            "cache_memory_bytes": stat("cache_memory_bytes"),
            "dashboard_progress_published_events": stat(
                "dashboard_progress_published_events"
            ),
            "dashboard_progress_throttled_events": stat(
                "dashboard_progress_throttled_events"
            ),
            "dashboard_progress_last_completion_tokens": stat(
                "dashboard_progress_last_completion_tokens"
            ),
            "dashboard_progress_decision_time_s": stat(
                "dashboard_progress_decision_time_s"
            ),
            "dashboard_progress_registry_update_time_s": stat(
                "dashboard_progress_registry_update_time_s"
            ),
            "dashboard_progress_rolling_update_time_s": stat(
                "dashboard_progress_rolling_update_time_s"
            ),
            "dashboard_progress_bus_publish_time_s": stat(
                "dashboard_progress_bus_publish_time_s"
            ),
            "request_enable_thinking": stat("request_enable_thinking"),
            "request_temperature": stat("request_temperature"),
            "request_top_p": stat("request_top_p"),
            "request_top_k": stat("request_top_k"),
            "effective_temperature": stat("effective_temperature"),
            "effective_top_p": stat("effective_top_p"),
            "effective_top_k": stat("effective_top_k"),
            "mlx_cache_cleanup": stat("mlx_cache_cleanup"),
            "aime_question_boundary_cleanup": stat(
                "aime_question_boundary_cleanup"
            ),
            "aime_question_runtime_cleanup": stat(
                "aime_question_runtime_cleanup"
            ),
            "request_message_count": stat("request_message_count"),
            "request_message_roles": stat("request_message_roles"),
            "request_message_chars": stat("request_message_chars"),
            "request_client_hint": stat("request_client_hint"),
            "request_session_bank_bypass": request_session_bank_bypass,
            "request_reasoning_parser": stat("request_reasoning_parser"),
            "request_max_tokens": stat("request_max_tokens"),
            "server_max_response_tokens": stat("server_max_response_tokens"),
            "effective_max_tokens": stat("effective_max_tokens"),
            "decode_lease_tokens": stat("decode_lease_tokens"),
            "uncapped_response_requested": stat("uncapped_response_requested"),
            "server_cap_applied": stat("server_cap_applied"),
            "context_cap_applied": stat("context_cap_applied"),
            "commit_source": commit_source,
            "visible_submission_mode": (
                visible_submission.mode
                if visible_submission is not None
                else None
            ),
            "visible_submission_request_id": (
                visible_submission.request_id
                if visible_submission is not None
                else None
            ),
            "visible_submission_extracted": (
                visible_submission.final_answer
                if visible_submission is not None
                else None
            ),
            "visible_submission_commit_source": (
                visible_submission.commit_source
                if visible_submission is not None
                else None
            ),
            "visible_submission_duration_ms": (
                visible_submission.duration_ms
                if visible_submission is not None
                else None
            ),
            "visible_submission_prompt_tokens": visible_submission_stat(
                "prompt_tokens"
            ),
            "visible_submission_completion_tokens": visible_submission_stat(
                "completion_tokens"
            ),
            "visible_submission_decode_tok_s": visible_submission_stat(
                "decode_tok_s"
            ),
            "visible_submission_request_tok_s": visible_submission_stat(
                "request_tok_s"
            ),
            "visible_submission_display_decode_tok_s": visible_submission_stat(
                "display_decode_tok_s",
                "server_tok_s",
                "tok_s",
            ),
            "visible_submission_request_enable_thinking": (
                visible_submission_enable_thinking
            ),
            "visible_submission_request_reasoning_parser": visible_submission_stat(
                "request_reasoning_parser"
            ),
            "visible_submission_request_temperature": visible_submission_stat(
                "request_temperature"
            ),
            "visible_submission_request_top_p": visible_submission_stat(
                "request_top_p"
            ),
            "visible_submission_request_top_k": visible_submission_stat(
                "request_top_k"
            ),
            "visible_submission_effective_temperature": visible_submission_stat(
                "effective_temperature"
            ),
            "visible_submission_effective_top_p": visible_submission_stat(
                "effective_top_p"
            ),
            "visible_submission_effective_top_k": visible_submission_stat(
                "effective_top_k"
            ),
            "visible_submission_mlx_cache_cleanup": visible_submission_stat(
                "mlx_cache_cleanup"
            ),
            "visible_submission_request_max_tokens": visible_submission_stat(
                "request_max_tokens"
            ),
            "visible_submission_effective_max_tokens": visible_submission_stat(
                "effective_max_tokens"
            ),
            "visible_submission_decode_lease_tokens": visible_submission_stat(
                "decode_lease_tokens"
            ),
            "visible_submission_server_cap_applied": visible_submission_stat(
                "server_cap_applied"
            ),
            "visible_submission_context_cap_applied": visible_submission_stat(
                "context_cap_applied"
            ),
            "visible_submission_stats": (
                visible_submission_stats
                if visible_submission is not None
                else {}
            ),
            "visible_submission_answer_text_tail_500": (
                visible_submission.answer_text[-500:]
                if visible_submission is not None
                else ""
            ),
            "cap_recovery_mode": recovery.mode if recovery is not None else None,
            "cap_recovery_request_id": (
                recovery.request_id if recovery is not None else None
            ),
            "cap_recovery_extracted": (
                recovery.final_answer if recovery is not None else None
            ),
            "cap_recovery_commit_source": (
                recovery.commit_source if recovery is not None else None
            ),
            "cap_recovery_duration_ms": (
                recovery.duration_ms if recovery is not None else None
            ),
            "cap_recovery_prompt_tokens": recovery_stat("prompt_tokens"),
            "cap_recovery_completion_tokens": recovery_stat("completion_tokens"),
            "cap_recovery_decode_tok_s": recovery_stat("decode_tok_s"),
            "cap_recovery_request_tok_s": recovery_stat("request_tok_s"),
            "cap_recovery_display_decode_tok_s": recovery_stat(
                "display_decode_tok_s",
                "server_tok_s",
                "tok_s",
            ),
            "cap_recovery_request_enable_thinking": cap_recovery_enable_thinking,
            "cap_recovery_request_reasoning_parser": recovery_stat(
                "request_reasoning_parser"
            ),
            "cap_recovery_request_temperature": recovery_stat(
                "request_temperature"
            ),
            "cap_recovery_request_top_p": recovery_stat("request_top_p"),
            "cap_recovery_request_top_k": recovery_stat("request_top_k"),
            "cap_recovery_effective_temperature": recovery_stat(
                "effective_temperature"
            ),
            "cap_recovery_effective_top_p": recovery_stat("effective_top_p"),
            "cap_recovery_effective_top_k": recovery_stat("effective_top_k"),
            "cap_recovery_mlx_cache_cleanup": recovery_stat("mlx_cache_cleanup"),
            "cap_recovery_request_max_tokens": recovery_stat("request_max_tokens"),
            "cap_recovery_effective_max_tokens": recovery_stat(
                "effective_max_tokens"
            ),
            "cap_recovery_decode_lease_tokens": recovery_stat(
                "decode_lease_tokens"
            ),
            "cap_recovery_server_cap_applied": recovery_stat("server_cap_applied"),
            "cap_recovery_context_cap_applied": recovery_stat("context_cap_applied"),
            "cap_recovery_accepted_by_depth": recovery_stat("accepted_by_depth"),
            "cap_recovery_drafted_by_depth": recovery_stat("drafted_by_depth"),
            "cap_recovery_mean_accept_probability_by_depth": recovery_stat(
                "mean_accept_probability_by_depth"
            ),
            "cap_recovery_verify_target_distribution_time_s": recovery_stat(
                "verify_target_distribution_time_s"
            ),
            "cap_recovery_stats": (
                recovery_stats if recovery is not None else {}
            ),
            "cap_recovery_answer_text_tail_500": (
                recovery.answer_text[-500:] if recovery is not None else ""
            ),
            "answer_verification_mode": (
                verification.mode
                if verification is not None
                else self.answer_verification
            ),
            "answer_verification_proposed_answer": (
                verification.proposed_answer if verification is not None else None
            ),
            "answer_verification_final_answer": (
                verification.final_answer if verification is not None else None
            ),
            "answer_verification_answers": (
                verification.answers if verification is not None else []
            ),
            "answer_verification_request_ids": (
                verification.request_ids if verification is not None else []
            ),
            "answer_verification_resolution": (
                verification.resolution if verification is not None else None
            ),
            "answer_verification_duration_ms": (
                verification.duration_ms if verification is not None else None
            ),
            "answer_verification_agreement": (
                verification.agreement if verification is not None else None
            ),
            "answer_verification_stats": (
                verification.stats if verification is not None else []
            ),
            "answer_verification_text_tail_500": (
                verification.text_tail if verification is not None else ""
            ),
            "reasoning_text_tail_500": (result.reasoning_text or "")[-500:],
            "answer_text_tail_500": (result.answer_text or "")[-500:],
            "error": result.error,
        }
        try:
            with self._persist_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("failed to persist AIME row: %s", exc)

    def _persist_summary(self) -> None:
        summary = {
            "summary": {
                "run_id": self.run_id,
                "state": self.state.value,
                "score": self.score,
                "total": self.total,
                "accuracy": self.accuracy(),
                "duration_ms": self.elapsed_ms(),
                "model": self.model_id,
                "ended_at": _iso(self.ended_at),
                "prompts": self._prompt_provenance(),
                "rescue_policy": self._rescue_policy_payload(),
            }
        }
        try:
            with self._persist_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("failed to persist AIME summary: %s", exc)


# --- Module-level registry / public API -----------------------------------


_active_runs: dict[str, AIMERunner] = {}
_registry_lock: asyncio.Lock = asyncio.Lock()


def list_runs() -> list[AIMERunner]:
    return list(_active_runs.values())


def get_run(run_id: str) -> AIMERunner | None:
    return _active_runs.get(run_id)


def list_active_run_id() -> str | None:
    """Return the id of any currently-running or paused run, or None."""
    _gc_finished()
    for run in _active_runs.values():
        if run.state in {RunState.RUNNING, RunState.PAUSED}:
            return run.run_id
    return None


async def start_run(
    *,
    model_id: str,
    base_url: str = "http://127.0.0.1:8000",
    api_key: str | None = None,
    problems: list[AIMEProblem] | None = None,
    chat_stream_factory: ChatStreamFactory | None = None,
    chat_cancel_factory: ChatCancelFactory | None = None,
    chat_metrics_factory: ChatMetricsFactory | None = None,
    **kwargs: Any,
) -> AIMERunner:
    """Start a new AIME run if no other is active.

    Raises :class:`ConcurrentRunError` if a run is already
    ``running`` or ``paused`` (covers the API's 409 contract).
    """
    async with _registry_lock:
        _gc_finished()
        active = list_active_run_id()
        if active is not None:
            raise ConcurrentRunError(active)
        ps = problems if problems is not None else load_dataset()
        runner = AIMERunner(
            problems=ps,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            chat_stream_factory=chat_stream_factory,
            chat_cancel_factory=chat_cancel_factory,
            chat_metrics_factory=chat_metrics_factory,
            **kwargs,
        )
        _active_runs[runner.run_id] = runner
    await runner.start()
    return runner


async def stop_runs() -> None:
    """Cancel every still-running run. Used on server shutdown."""
    for run in list(_active_runs.values()):
        if not run.is_terminal:
            await run.cancel()


# --- Helpers --------------------------------------------------------------


def _new_run_id(year: int) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"aime-{year}-{stamp}-{short}"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _gc_finished() -> None:
    """Drop runs that finished more than RETENTION_S ago."""
    now = time.monotonic()
    expired: list[str] = []
    for run_id, run in _active_runs.items():
        if not run.is_terminal:
            continue
        if run._finished_at_monotonic is None:
            continue
        if now - run._finished_at_monotonic > RETENTION_S:
            expired.append(run_id)
    for run_id in expired:
        _active_runs.pop(run_id, None)
