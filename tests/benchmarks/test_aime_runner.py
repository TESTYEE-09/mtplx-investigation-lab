"""Tests for the AIME 2026 benchmark runner.

Uses a fake chat stream factory so we never touch the real MLX backend.
Each test wraps its body in asyncio.run() because pytest-asyncio isn't
in the MTPLX dev deps and adding it would expand the test footprint
unnecessarily.

Covers:
- full run completes end-to-end on a tiny mocked dataset
- per-question events have the right shape
- pause hard-stops the current question; resume retries it fresh
- cancel propagates and persists partial results
- concurrent start_run raises ConcurrentRunError (single-run guarantee)
- abstain status when the model emits no parseable answer
- snapshot reflects in-flight state
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from mtplx.benchmarks.runners import aime as aime_mod
from mtplx.benchmarks.runners.aime import (
    AIMEProblem,
    AIMERunner,
    ConcurrentRunError,
    RunState,
    start_run,
)


# ---- Test helpers --------------------------------------------------------


def _make_problems(n: int = 3) -> list[AIMEProblem]:
    return [
        AIMEProblem(
            id=f"2026-I-{i}",
            set="AIME I",
            year=2026,
            index=i,
            problem=f"Problem {i}: what is {i} * 10?",
            answer=i * 10,
            source="https://example",
        )
        for i in range(1, n + 1)
    ]


def _chunk(reasoning: str = "", content: str = "") -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if reasoning:
        delta["reasoning_content"] = reasoning
    if content:
        delta["content"] = content
    return {"choices": [{"delta": delta}]}


async def _fake_stream(chunks: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for c in chunks:
        await asyncio.sleep(0)
        yield c


def _make_factory(answer_text_by_idx: dict[int, str]):
    """Factory that returns canned text for each problem (keyed by index)."""

    async def factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        text = answer_text_by_idx.get(problem.index, f"\\boxed{{{problem.answer}}}")
        chunks = [
            _chunk(reasoning="Thinking step 1... "),
            _chunk(reasoning="step 2... "),
            _chunk(content="Final: "),
            _chunk(content=text),
        ]
        return _fake_stream(chunks)

    return factory


def _slow_factory(per_chunk_delay_s: float = 0.05):
    """Factory with per-chunk delay so pause/cancel have time to fire."""

    async def factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream() -> AsyncIterator[dict[str, Any]]:
            for _ in range(3):
                await asyncio.sleep(per_chunk_delay_s)
                yield _chunk(reasoning="thinking. ")
            await asyncio.sleep(per_chunk_delay_s)
            yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    return factory


def _blocking_first_factory(per_chunk_delay_s: float = 0.05):
    """First problem streams forever; later problems complete normally."""

    async def factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream() -> AsyncIterator[dict[str, Any]]:
            if problem.index == 1:
                while True:
                    await asyncio.sleep(per_chunk_delay_s)
                    yield _chunk(reasoning="still solving. ")
            else:
                await asyncio.sleep(0)
                yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    return factory


def _blocks_first_attempt_factory(per_chunk_delay_s: float = 0.05):
    """First attempt of first problem streams forever; retry completes."""

    async def factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream() -> AsyncIterator[dict[str, Any]]:
            if problem.index == 1 and runner.current_attempt == 1:
                while True:
                    await asyncio.sleep(per_chunk_delay_s)
                    yield _chunk(reasoning="still solving. ")
            else:
                await asyncio.sleep(0)
                yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    return factory


def _run(coro):
    """Wrap asyncio.run that also resets the module-level registry."""
    aime_mod._active_runs.clear()
    try:
        return asyncio.run(coro)
    finally:
        aime_mod._active_runs.clear()


def test_prompt_contract_closes_qwen_thinking_before_answer() -> None:
    prompt = aime_mod.SYSTEM_PROMPT + aime_mod.USER_PROMPT_SUFFIX

    assert "</think>" in prompt
    assert "\\boxed{N}" in prompt
    assert "candidate_answer=N" in prompt
    assert "scores only visible content" in prompt
    assert "Do not put candidate_answer" in prompt
    assert "Do not repeat these instructions" in prompt


def test_solver_prompts_carry_no_strategy_or_style_coaching() -> None:
    """The score must measure the model: prompts are format contract only.

    Strategy coaching (verification bans, counting recipes, commit points)
    and style demands (compactness, LaTeX, brevity) previously steered the
    solver; none of it may reappear in any solver prompt variant.
    """

    prompts = {
        "SYSTEM_PROMPT": aime_mod.SYSTEM_PROMPT,
        "USER_PROMPT_SUFFIX": aime_mod.USER_PROMPT_SUFFIX,
        "FAST_SYSTEM_PROMPT": aime_mod.FAST_SYSTEM_PROMPT,
        "FAST_USER_PROMPT_SUFFIX": aime_mod.FAST_USER_PROMPT_SUFFIX,
    }
    coaching_markers = (
        "verification pass",
        "Do not verify",
        "audit",
        "Cases:",
        "Total count",
        "commit point",
        "compact",
        "briefly",
        "brief approach note",
        "LaTeX",
        "submit immediately",
        "second pass",
        "alternate",
        "examples check",
        "source note",
    )
    for name, prompt in prompts.items():
        for marker in coaching_markers:
            assert marker.lower() not in prompt.lower(), (
                f"{name} still coaches the model: {marker!r}"
            )
    # The format contract itself stays intact in both modes.
    assert "candidate_answer=N" in aime_mod.FAST_SYSTEM_PROMPT
    assert "\\boxed{N}" in aime_mod.FAST_SYSTEM_PROMPT


def test_prompt_messages_use_active_variants() -> None:
    problems = _make_problems(1)

    thinking = aime_mod.aime_prompt_messages(problems[0], enable_thinking=True)
    assert thinking[0] == {"role": "system", "content": aime_mod.SYSTEM_PROMPT}
    assert thinking[1]["content"].endswith(aime_mod.USER_PROMPT_SUFFIX)

    fast = aime_mod.aime_prompt_messages(problems[0], enable_thinking=False)
    assert fast[0] == {"role": "system", "content": aime_mod.FAST_SYSTEM_PROMPT}
    assert fast[1]["content"].endswith(aime_mod.FAST_USER_PROMPT_SUFFIX)


def test_run_summary_embeds_prompts_and_rescue_policy(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        factory = _make_factory({1: r"\boxed{10}"})
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        run_done = next(e for e in events if e["event"] == "run_done")

        assert run_done["prompts"]["enable_thinking"] is True
        assert run_done["prompts"]["system"] == aime_mod.SYSTEM_PROMPT
        assert run_done["prompts"]["user_suffix"] == aime_mod.USER_PROMPT_SUFFIX
        assert run_done["rescue_policy"] == {
            "answer_verification": "off",
            "answer_verification_attempts": 2,
            "cap_recovery": "off",
            "active": False,
            "gemma_non_thinking_exception": False,
        }

        snapshot = runner.snapshot()
        assert snapshot["prompts"]["system"] == aime_mod.SYSTEM_PROMPT
        assert snapshot["rescue_policy"]["active"] is False

        persisted = (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        summary = json.loads(persisted[-1])["summary"]
        assert summary["prompts"]["system"] == aime_mod.SYSTEM_PROMPT
        assert summary["rescue_policy"]["active"] is False

    _run(body())


def test_gemma_non_thinking_summary_discloses_rescue_exception(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        factory = _make_factory({1: r"\boxed{10}"})
        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            enable_thinking=False,
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        snapshot = runner.snapshot()
        assert snapshot["prompts"]["enable_thinking"] is False
        assert snapshot["prompts"]["system"] == aime_mod.FAST_SYSTEM_PROMPT
        policy = snapshot["rescue_policy"]
        assert policy["answer_verification"] == "fast_majority"
        assert policy["cap_recovery"] == "fresh_finalizer"
        assert policy["active"] is True
        assert policy["gemma_non_thinking_exception"] is True

    _run(body())


def test_gemma_aime_reasoning_default_preserves_explicit_fast_contract() -> None:
    problems = _make_problems(1)
    messages = aime_mod.aime_prompt_messages(problems[0], enable_thinking=False)
    prompt = "\n".join(message["content"] for message in messages)

    assert aime_mod.default_enable_thinking_for_model(
        "gemma4-mtplx-optimized-speed"
    ) is True
    assert aime_mod.default_enable_thinking_for_model("test-model") is True
    assert (
        aime_mod.default_answer_verification_for_model(
            "gemma4-mtplx-optimized-speed",
            enable_thinking=False,
        )
        == "fast_majority"
    )
    assert (
        aime_mod.default_cap_recovery_for_model(
            "gemma4-mtplx-optimized-speed",
            enable_thinking=False,
        )
        == "fresh_finalizer"
    )
    assert (
        aime_mod.default_cap_recovery_for_model(
            "gemma4-mtplx-optimized-speed",
            enable_thinking=True,
        )
        == "off"
    )
    assert (
        aime_mod.default_cap_recovery_for_model(
            "cyankiwi-Qwen3.6-27B-AWQ-INT4-MTPLX-Speed",
            enable_thinking=True,
        )
        == "off"
    )
    assert (
        aime_mod.default_max_tokens_for_model(
            "gemma4-mtplx-optimized-speed",
            enable_thinking=False,
        )
        == aime_mod.GEMMA_FAST_AIME_MAX_TOKENS
    )
    assert (
        aime_mod.default_max_tokens_for_model(
            "gemma4-mtplx-optimized-speed",
            enable_thinking=True,
        )
        is None
    )
    assert (
        aime_mod.default_max_tokens_for_model(
            "cyankiwi-Qwen3.6-27B-AWQ-INT4-MTPLX-Speed",
            enable_thinking=True,
        )
        is None
    )
    assert "Hidden reasoning is disabled" in prompt
    assert "candidate_answer=N" in prompt
    assert "</think>" not in prompt
    recovery_messages = aime_mod.aime_fresh_cap_recovery_messages(problems[0])
    recovery_prompt = "\n".join(message["content"] for message in recovery_messages)
    assert "fresh visible AIME recovery solver" in recovery_prompt
    assert "untrusted scratch" not in recovery_prompt


def test_aime_verifier_prompt_is_independent_and_no_key() -> None:
    problem = AIMEProblem(
        id="2026-I-no-key",
        set="AIME I",
        year=2026,
        index=1,
        problem="Find the value of x.",
        answer=777,
        source="https://example",
    )
    messages = aime_mod.aime_verifier_messages(problem)
    prompt = "\n".join(message["content"] for message in messages)

    assert "official answer key" in prompt
    assert "verified_answer=N" in prompt
    assert "candidate_answer" in prompt
    assert "shortest reliable derivation" in prompt
    assert "general formula" in prompt
    assert "777" not in prompt
    assert "57" not in prompt


def test_aime_adjudicator_prompt_uses_generated_candidates_not_key() -> None:
    problem = AIMEProblem(
        id="2026-I-adjudicate",
        set="AIME I",
        year=2026,
        index=1,
        problem="Find the value of x.",
        answer=999,
        source="https://example",
    )
    messages = aime_mod.aime_verifier_messages(
        problem,
        style="adjudicator",
        candidate_answers=[57, 62],
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "official answer key" in prompt
    assert "Candidate answers from prior local solver passes: 57, 62" in prompt
    assert "verified_answer=N" in prompt
    assert "shortest reliable derivation" in prompt
    assert "candidate_answer" in prompt
    assert "999" not in prompt


def test_extract_verified_answer_ignores_first_pass_marker() -> None:
    text = "candidate_answer=57\nverified_answer=62\n\\boxed{62}"

    assert aime_mod.extract_candidate_answer(text) == 57
    assert aime_mod.extract_verified_answer(text) == 62


def test_default_chat_factory_marks_aime_requests_stateless(monkeypatch) -> None:
    async def body() -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                yield (
                    'data: {"choices":[{"delta":{"content":"'
                    r"\\boxed{10}"
                    '"}}],"usage":{"prompt_tokens":111},'
                    '"mtplx_stats":{"prompt_tokens":111,"cached_tokens":0,'
                    '"session_cache_hit":false}}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                captured["method"] = method
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problems = _make_problems(1)
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            run_id="aime-2026-test",
            base_url="http://127.0.0.1:8123",
            api_key="secret",
        )
        runner.current_idx = 1
        runner._current_attempt = 2
        runner._current_request_id = runner.request_id_for(1, 2)

        stream = await aime_mod.default_chat_stream_factory(runner, problems[0])
        chunks = [chunk async for chunk in stream]

        assert len(chunks) == 1
        assert captured["method"] == "POST"
        assert captured["url"] == "http://127.0.0.1:8123/v1/chat/completions"
        headers = captured["headers"]
        assert headers["x-mtplx-client"] == "aime"
        assert headers["x-mtplx-cache-mode"] == "bypass"
        assert headers["x-mtplx-request-id"] == "chatcmpl-aime-2026-test-q1-a2"
        assert headers["Authorization"] == "Bearer secret"
        payload = captured["json"]
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["enable_thinking"] is True
        assert payload["metadata"]["enable_thinking"] is True
        assert [message["role"] for message in payload["messages"]] == [
            "system",
            "user",
        ]
        assert len(payload["messages"]) == 2
        assert payload["metadata"]["client"] == "aime"
        assert payload["metadata"]["cache_mode"] == "bypass"
        assert payload["metadata"]["run_id"] == runner.run_id
        assert payload["metadata"]["question_idx"] == 1
        assert payload["metadata"]["attempt"] == 2
        assert (
            payload["metadata"]["mtplx_request_id"]
            == "chatcmpl-aime-2026-test-q1-a2"
        )

    _run(body())


def test_gemma_default_chat_factory_uses_reasoning_channel(monkeypatch) -> None:
    async def body() -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                yield (
                    'data: {"choices":[{"delta":{"content":"'
                    r"candidate_answer=10"
                    '"}}],"usage":{"prompt_tokens":111}}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                _ = method
                _ = url
                _ = headers
                captured["json"] = json
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problems = _make_problems(1)
        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            run_id="aime-2026-test",
            base_url="http://127.0.0.1:8123",
        )
        runner.current_idx = 1
        runner._current_attempt = 1
        runner._current_request_id = runner.request_id_for(1, 1)

        stream = await aime_mod.default_chat_stream_factory(runner, problems[0])
        _ = [chunk async for chunk in stream]

        payload = captured["json"]
        assert payload["metadata"]["enable_thinking"] is True
        assert payload["enable_thinking"] is True
        assert "max_tokens" not in payload

    _run(body())


def test_gemma_explicit_no_thinking_chat_factory_keeps_fast_cap(monkeypatch) -> None:
    async def body() -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                yield (
                    'data: {"choices":[{"delta":{"content":"'
                    r"candidate_answer=10"
                    '"}}],"usage":{"prompt_tokens":111}}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                _ = method
                _ = url
                _ = headers
                captured["json"] = json
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problems = _make_problems(1)
        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            run_id="aime-2026-test",
            base_url="http://127.0.0.1:8123",
            enable_thinking=False,
        )
        runner.current_idx = 1
        runner._current_attempt = 1
        runner._current_request_id = runner.request_id_for(1, 1)

        stream = await aime_mod.default_chat_stream_factory(runner, problems[0])
        _ = [chunk async for chunk in stream]

        payload = captured["json"]
        assert payload["enable_thinking"] is False
        assert payload["metadata"]["enable_thinking"] is False
        assert payload["max_tokens"] == aime_mod.GEMMA_FAST_AIME_MAX_TOKENS

    _run(body())


def test_extract_candidate_answer_uses_explicit_marker() -> None:
    assert aime_mod.extract_candidate_answer("candidate_answer=277") == 277
    assert aime_mod.extract_candidate_answer("candidate answer is 277") is None
    assert aime_mod.extract_candidate_answer("x = 277") is None


def test_extract_declared_answer_requires_final_answer_language() -> None:
    assert aime_mod.extract_declared_answer("The answer is 277.") == 277
    assert aime_mod.extract_declared_answer("Final answer: \\boxed{62}") == 62
    assert aime_mod.extract_declared_answer("m+n = 252 + 25 = 277") == 277
    assert aime_mod.extract_declared_answer("$p+q = 29 + 50 = 79$.") == 79
    assert aime_mod.extract_declared_answer("Total for k=5 is 6.") is None
    assert aime_mod.extract_declared_answer("Total = $32 + 16 + 8 + 4 + 2 = 62$") is None
    assert aime_mod.extract_declared_answer("Sum = $1 + 2 + 3 + 4 + 5 = 15$") is None
    assert (
        aime_mod.extract_declared_answer(
            "Total count = $32 + 16 + 8 + 4 + 2 = 62$."
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer(
            r"Total count = $\sum_{m=1}^{6}\binom{6}{m}=64$."
        )
        is None
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Total count is sum over $m=1$ to 6.",
                    "$m=1: 5$",
                    "$m=2: 15$",
                    "$m=3: 20$",
                    "$m=4: 15$",
                    "$m=5: 6$",
                    "$m=6: 1$",
                    "Sum = $5 + 15 + 20 + 15 + 6 + 1 = 62$.",
                ]
            )
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Counts:",
                    "$m=1: 5$",
                    "$m=2: 15$",
                    "$m=3: 20$",
                    "$m=4: 15$",
                    "$m=5: 6$",
                    "$m=6: 1$",
                    "So the calculation $5 + 15 + 20 + 15 + 6 + 1 = 62$ is correct.",
                    r"Wait, $\sum_{m=1}^6 \binom{6}{m} = 63$.",
                ]
            )
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Counts:",
                    "$m=1: 5$",
                    "$m=2: 15$",
                    "So sum is $(6-1) + 15 + 20 + 15 + 6 + 1 = 62$.",
                ]
            )
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Let's sum the counts.",
                    "Total = $32 + 16 + 8 + 4 + 2$.",
                    "$32 + 16 = 48$.",
                    "$48 + 8 = 56$.",
                    "$56 + 4 = 60$.",
                    "$60 + 2 = 62$.",
                    "Wait, is that all?",
                ]
            )
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Counts:",
                    "$m=1: 5$",
                    "$m=2: 15$",
                    "$m=3: 20$",
                    "$m=4: 15$",
                    "$m=5: 6$",
                    "$m=6: 1$",
                    "Total sum = $5 + 15 + 20 + 15 + 6 + 1$.",
                    "$5+15=20$.",
                    "$20+20=40$.",
                    "$40+15=55$.",
                    "$55+6=61$.",
                    "$61+1=62$.",
                    "Wait, let me reverify.",
                ]
            )
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer("Total count for k=2 = $1 + 2 + 3 = 6$.")
        is None
    )
    assert (
        aime_mod.extract_declared_answer(
            "Therefore the total number of valid palindromes is 62."
        )
        == 62
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Total bad = 28.",
                    "Total good = 98 - 28 = 70.",
                    "Let's double check the prime count.",
                ]
            )
        )
        == 70
    )
    assert aime_mod.extract_declared_answer("Total bad = 28.") is None
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "So the set of values is exactly "
                    r"$\{k-1 \mid 4 \le k \le 101, k \text{ composite}, "
                    r"k \ne p^2\}$.",
                    "The size of this set is 70.",
                    "Let's quickly check if I missed any primes.",
                ]
            )
        )
        == 70
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "So the set of valid N is exactly composites minus prime squares.",
                    "The logic covers all cases.",
                    "Composite numbers up to 100: 74.",
                    "Prime squares up to 100: 4.",
                    "Count = 70.",
                    "Let's double check the count of primes again.",
                ]
            )
        )
        == 70
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Excluded m=11 (prime). n=10. Prime. No solution. Correct.",
                    "So the logic seems solid.",
                    "The number of such integers is 70.",
                    "Let's re-verify the count of primes and squares.",
                ]
            )
        )
        == 70
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "The product of all possible positive values is P = 2026^{20}.",
                    "Number of divisors $(20+1)(20+1) = 441$.",
                    "Let's double check the question wording again.",
                ]
            )
        )
        == 441
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Product P = 2026^{20}.",
                    "Number of divisors is 441.",
                    "Is there any catch?",
                ]
            )
        )
        == 441
    )
    assert aime_mod.extract_declared_answer("Number of divisors is 12.") is None
    assert (
        aime_mod.extract_declared_answer("Total number for case k=2 is 15.")
        is None
    )
    assert aime_mod.extract_declared_answer("Count = 70.") is None
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "For case k=2, the valid integers are 1 through 5.",
                    "Count = 5.",
                ]
            )
        )
        is None
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "For case k=2, the valid integers are 1 through 5.",
                    "The number of such integers is 5.",
                ]
            )
        )
        is None
    )
    assert (
        aime_mod.extract_declared_answer(
            "For this case, the size of this set is 3."
        )
        is None
    )
    assert aime_mod.extract_declared_answer("Sum = 62") is None
    assert (
        aime_mod.extract_declared_answer(
            "So the calculation $1 + 2 + 3 = 6$ is correct."
        )
        is None
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Total sum = $1 + 2 + 3$.",
                    "$1+2=3$.",
                    "$3+3=6$.",
                ]
            )
        )
        is None
    )
    assert (
        aime_mod.extract_declared_answer(
            "\n".join(
                [
                    "Total = $32 + 16 + 8 + 4 + 2$.",
                    "$32 + 16 = 48$.",
                    "$48 + 8 = 56$.",
                    "$56 + 4 = 60$.",
                    "$60 + 2 = 62$.",
                ]
            )
        )
        is None
    )
    final_count_summary = """So the counts are:
k=1: 5
k=2: 15
k=3: 20
k=4: 15
k=5: 6
k=6: 1

Sum = 62.
"""
    assert aime_mod.extract_declared_answer(final_count_summary) == 62
    intermediate_count_summary = """For k=2, counts are:
j=1: 1
j=2: 2
j=3: 3

Sum = 6.
"""
    assert aime_mod.extract_declared_answer(intermediate_count_summary) is None
    assert aime_mod.extract_declared_answer("So m = 252 and n = 25. m+n = 252") is None
    assert aime_mod.extract_declared_answer("m+n = 252 +") is None
    assert aime_mod.extract_declared_answer("Count = 1+1+1+1+1 = 5") is None
    assert aime_mod.extract_declared_answer("Final Answer seems to be 27.") is None


def test_visible_stream_commit_requires_explicit_submission() -> None:
    assert aime_mod.extract_candidate_answer("candidate_answer=277") == 277
    assert aime_mod.extract_declared_answer("m+n = 252 + 25 = 277") == 277
    assert aime_mod._extract_stream_commit("candidate_answer=27") is None
    assert aime_mod._extract_stream_commit("candidate_answer=277\n") == (
        277,
        "candidate_answer",
    )
    assert aime_mod._extract_stream_commit(r"\boxed{27") is None
    assert aime_mod._extract_stream_commit(r"\boxed{277}") == (
        277,
        "boxed",
    )
    assert aime_mod._extract_stream_commit("m+n = 252 + 25 = 277.") is None
    assert aime_mod._extract_stream_commit("Total good = 98 - 28 = 70.") is None
    assert aime_mod._extract_stream_commit(
        "\n".join(
            [
                "So the set of valid N is exactly composites minus prime squares.",
                "The logic covers all cases.",
                "Composite numbers up to 100: 74.",
                "Prime squares up to 100: 4.",
                "Count = 70.",
            ]
        )
    ) is None
    assert aime_mod._extract_stream_commit(
        "\n".join(
            [
                "Excluded m=11 (prime). n=10. Prime. No solution. Correct.",
                "So the logic seems solid.",
                "The number of such integers is 70.",
            ]
        )
    ) is None
    assert (
        aime_mod._extract_stream_commit(
            "Product P = 2026^{20}.\nNumber of divisors is 44"
        )
        is None
    )
    assert aime_mod._extract_stream_commit(
        "Product P = 2026^{20}.\nNumber of divisors is 441."
    ) is None


def test_reasoning_total_scalar_only_is_not_a_stream_commit() -> None:
    reasoning = "\n".join(
        [
            "Counts:",
            "k=1: 5",
            "k=2: 15",
            "k=3: 20",
            "k=4: 15",
            "k=5: 6",
            "k=6: 1",
            "So everything is consistent.",
            "The total is 62.",
        ]
    )

    assert aime_mod.extract_declared_answer(reasoning) is None
    assert aime_mod._extract_stream_commit(reasoning) is None


# ---- Tests ---------------------------------------------------------------


def test_full_run_completes(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(3)
        factory = _make_factory(
            {
                1: r"\boxed{10}",
                2: r"\boxed{20}",
                3: r"\boxed{999}",  # wrong: expected 30
            }
        )
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert runner.score == 2
        statuses = [r.status for r in runner.results]
        assert statuses == ["correct", "correct", "wrong"]

        persisted = (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        assert len(persisted) == 4  # 3 question rows + 1 summary row
        last = json.loads(persisted[-1])
        assert last["summary"]["score"] == 2
        assert last["summary"]["total"] == 3

    _run(body())


def test_question_done_events_have_correct_shape(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(2)
        factory = _make_factory({1: r"\boxed{10}", 2: r"\boxed{20}"})
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        kinds = [e["event"] for e in events]
        assert kinds[0] == "run_started"
        assert "question_started" in kinds
        assert kinds.count("question_done") == 2
        assert kinds[-1] == "run_done"

        done_events = [e for e in events if e["event"] == "question_done"]
        for i, ev in enumerate(done_events, start=1):
            assert ev["idx"] == i
            assert ev["status"] == "correct"
            assert ev["extracted"] == i * 10
            assert ev["expected"] == i * 10
            assert ev["duration_ms"] is not None and ev["duration_ms"] >= 0

    _run(body())


def test_gemma_reasoning_deltas_stay_out_of_answer_stream(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(reasoning="Let x=10. ")
                yield {
                    "mtplx_progress": {
                        "completion_tokens": 32,
                        "decode_tok_s": 49.5,
                        "display_decode_tok_s": 49.0,
                        "decode_elapsed_s": 0.65,
                    },
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    f"candidate_answer={problem.answer}\n"
                                    f"\\boxed{{{problem.answer}}}"
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 111,
                        "completion_tokens": 12,
                        "total_tokens": 123,
                    },
                    "mtplx_stats": {
                        "request_id": runner.current_request_id,
                        "request_enable_thinking": True,
                        "request_reasoning_parser": "gemma4",
                        "reasoning_tokens": 3,
                        "completion_tokens": 12,
                    },
                }

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        reasoning_text = "".join(
            str(e.get("text") or "")
            for e in events
            if e["event"] == "reasoning_delta"
        )
        answer_text = "".join(
            str(e.get("text") or "")
            for e in events
            if e["event"] == "answer_delta"
        )
        combined = reasoning_text + answer_text
        assert reasoning_text == "Let x=10. "
        assert "candidate_answer=10" in answer_text
        assert "<|channel>" not in combined
        assert "<channel|>" not in combined

        progress_events = [e for e in events if e["event"] == "question_progress"]
        assert len(progress_events) == 1
        progress = progress_events[0]
        assert progress["idx"] == 1
        assert progress["attempt"] == 1
        assert progress["request_id"] == runner.request_id_for(1, 1)
        assert progress["progress"]["request_id"] == runner.request_id_for(1, 1)
        assert progress["progress"]["completion_tokens"] == 32
        assert progress["progress"]["display_decode_tok_s"] == 49.0

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["request_enable_thinking"] is True
        assert row["request_reasoning_parser"] == "gemma4"
        assert row["reasoning_token_count"] > 0
        assert row["answer_token_count"] > 0
        assert row["stream_reasoning_events"] == 1
        assert row["stream_answer_events"] >= 1
        assert row["stream_reasoning_chars"] == len("Let x=10. ")
        assert row["stream_progress_events"] == 1
        assert row["stream_progress_milestones"]["first_32"]["decode_tok_s"] == 49.5

    _run(body())


def test_question_persistence_keeps_freshness_and_speed_proof(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield {
                    "choices": [
                        {"delta": {"content": f"\\boxed{{{problem.answer}}}"}}
                    ],
                    "usage": {
                        "prompt_tokens": 123,
                        "completion_tokens": 4,
                        "total_tokens": 127,
                    },
                    "mtplx_stats": {
                        "request_id": runner.current_request_id,
                        "prompt_tokens": 123,
                        "completion_tokens": 4,
                        "cached_tokens": 0,
                        "new_prefill_tokens": 123,
                        "session_cache_hit": False,
                        "cache_miss_reason": "request_cache_bypass",
                        "session_restore_mode": "cold",
                        "decode_tok_s": 45.0,
                        "request_tok_s": 40.0,
                        "sliding_decode_tok_s_first_32": 50.0,
                        "sliding_decode_tok_s_first_64": 49.0,
                        "sliding_decode_tok_s_first_128": 48.0,
                        "sliding_decode_tok_s_first_256": 47.0,
                        "sliding_decode_tok_s_last_128": 46.0,
                        "dashboard_progress_published_events": 5,
                        "dashboard_progress_throttled_events": 9,
                        "dashboard_progress_decision_time_s": 0.001,
                        "verify_target_distribution_time_s": 12.5,
                        "target_forward_time_s": 1.25,
                        "draft_time_s": 2.5,
                        "cache_memory_bytes": 0,
                        "request_enable_thinking": False,
                        "mlx_cache_cleanup": {
                            "cleared": True,
                            "reason": "aime_stateless_question",
                        },
                        "request_effective_mtp_depth": 2,
                        "accepted_by_depth": [3, 2],
                        "drafted_by_depth": [5, 4],
                        "request_message_count": 2,
                        "request_message_roles": ["system", "user"],
                        "request_message_chars": [10, 20],
                        "request_client_hint": "aime",
                        "request_session_bank_bypass": True,
                    },
                }

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            run_id="aime-2026-proof",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads((tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0])
        assert row["request_id"] == "chatcmpl-aime-2026-proof-q1-a1"
        assert row["attempt"] == 1
        assert row["prompt_tokens"] == 123
        assert row["cached_tokens"] == 0
        assert row["new_prefill_tokens"] == 123
        assert row["session_cache_hit"] is False
        assert row["cache_miss_reason"] == "request_cache_bypass"
        assert row["decode_tok_s"] == 45.0
        assert row["request_tok_s"] == 40.0
        assert row["sliding_decode_tok_s_first_32"] == 50.0
        assert row["sliding_decode_tok_s_first_64"] == 49.0
        assert row["sliding_decode_tok_s_first_128"] == 48.0
        assert row["sliding_decode_tok_s_first_256"] == 47.0
        assert row["sliding_decode_tok_s_last_128"] == 46.0
        assert row["dashboard_progress_published_events"] == 5
        assert row["dashboard_progress_throttled_events"] == 9
        assert row["dashboard_progress_decision_time_s"] == 0.001
        assert row["verify_target_distribution_time_s"] == 12.5
        assert row["target_forward_time_s"] == 1.25
        assert row["draft_time_s"] == 2.5
        assert row["cache_memory_bytes"] == 0
        assert row["request_enable_thinking"] is False
        assert row["mlx_cache_cleanup"] == {
            "cleared": True,
            "reason": "aime_stateless_question",
        }
        assert row["request_effective_mtp_depth"] == 2
        assert row["accepted_by_depth"] == [3, 2]
        assert row["drafted_by_depth"] == [5, 4]
        assert row["request_message_count"] == 2
        assert row["request_message_roles"] == ["system", "user"]
        assert row["request_client_hint"] == "aime"
        assert row["request_session_bank_bypass"] is True

    _run(body())


def test_question_boundary_cleanup_is_recorded_after_visible_submission(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        cleanup_calls: list[tuple[int, str]] = []

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield {
                    "choices": [
                        {"delta": {"content": f"candidate_answer={problem.answer}\n"}}
                    ],
                    "mtplx_stats": {
                        "request_id": runner.current_request_id,
                        "request_enable_thinking": True,
                        "request_client_hint": "aime",
                        "completion_tokens": 8,
                    },
                }
                yield {
                    "choices": [
                        {"delta": {"content": f"\\boxed{{{problem.answer}}}"}}
                    ],
                    "usage": {"completion_tokens": 9},
                    "mtplx_stats": {
                        "request_id": runner.current_request_id,
                        "request_enable_thinking": True,
                        "request_client_hint": "aime",
                        "completion_tokens": 9,
                    },
                }

            return stream()

        async def cleanup_factory(
            runner: AIMERunner,
            result: aime_mod.QuestionResult,
            request_id: str,
        ) -> dict[str, Any]:
            _ = runner
            cleanup_calls.append((result.idx, request_id))
            return {
                "ok": True,
                "mlx_cache_cleanup": {
                    "cleared": True,
                    "reason": "aime_question_boundary",
                },
            }

        runner = AIMERunner(
            problems=_make_problems(1),
            model_id="test-model",
            run_id="aime-2026-cleanup",
            chat_stream_factory=factory,
            question_isolation_factory=cleanup_factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert cleanup_calls == [
            (1, "chatcmpl-aime-2026-cleanup-q1-a1"),
        ]
        assert row["status"] == "correct"
        assert row["extracted"] == 10
        assert row["aime_question_boundary_cleanup"] == {
            "ok": True,
            "mlx_cache_cleanup": {
                "cleared": True,
                "reason": "aime_question_boundary",
            },
        }
        cleanup_events = [
            event for event in events if event["event"] == "question_isolation_cleanup"
        ]
        assert cleanup_events
        assert cleanup_events[0]["cleanup"]["ok"] is True

    _run(body())


def test_early_candidate_commit_backfills_daemon_metrics(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content=f"candidate_answer={problem.answer}\n")

            return stream()

        async def metrics_factory(
            runner: AIMERunner, request_id: str
        ) -> dict[str, Any]:
            return {
                "request_id": request_id,
                "prompt_tokens": 222,
                "completion_tokens": 33,
                "decode_tok_s": 41.5,
                "request_tok_s": 39.0,
                "cache_memory_bytes": 0,
                "request_enable_thinking": False,
                "request_session_bank_bypass": True,
            }

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            chat_metrics_factory=metrics_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["commit_source"] == "visible_stream_candidate_answer"
        assert row["prompt_tokens"] == 222
        assert row["completion_tokens"] == 33
        assert row["decode_tok_s"] == 41.5
        assert row["cache_memory_bytes"] == 0
        assert row["request_enable_thinking"] is False

    _run(body())


def test_uncommitted_cap_backfills_daemon_metrics(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content="Still working without a final answer.")
                yield {
                    "usage": {
                        "prompt_tokens": 222,
                        "completion_tokens": 2048,
                    }
                }

            return stream()

        async def metrics_factory(
            runner: AIMERunner, request_id: str
        ) -> dict[str, Any]:
            return {
                "request_id": request_id,
                "prompt_tokens": 222,
                "completion_tokens": 2048,
                "decode_tok_s": 32.0,
                "request_max_tokens": 2048,
                "effective_max_tokens": 2048,
                "decode_lease_tokens": 2048,
                "cache_memory_bytes": 0,
                "request_enable_thinking": False,
                "request_session_bank_bypass": True,
            }

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            chat_metrics_factory=metrics_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] is None
        assert row["status"] == "abstain"
        assert row["completion_tokens"] == 2048
        assert row["request_max_tokens"] == 2048
        assert row["effective_max_tokens"] == 2048
        assert row["decode_lease_tokens"] == 2048
        assert row["decode_tok_s"] == 32.0

    _run(body())


def test_uncommitted_cap_can_recover_with_visible_finalizer(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = runner
                _ = problem
                yield _chunk(content="Still working without a final answer.")
                yield {
                    "usage": {
                        "prompt_tokens": 222,
                        "completion_tokens": 2048,
                    }
                }

            return stream()

        async def recovery(
            runner: AIMERunner,
            problem: AIMEProblem,
            first_answer_text: str,
            first_stats: Mapping[str, Any],
        ) -> aime_mod.AIMECapRecovery:
            _ = runner
            assert problem.answer == 10
            assert "without a final answer" in first_answer_text
            assert first_stats["completion_tokens"] == 2048
            return aime_mod.AIMECapRecovery(
                mode="visible_finalizer",
                request_id="chatcmpl-recovery",
                final_answer=10,
                commit_source="answer_candidate_answer",
                reasoning_text="hidden derivation",
                answer_text="candidate_answer=10\n\\boxed{10}",
                usage={"completion_tokens": 32},
                stats={
                    "prompt_tokens": 250,
                    "completion_tokens": 32,
                    "decode_tok_s": 28.0,
                    "request_tok_s": 27.0,
                    "request_enable_thinking": True,
                },
                duration_ms=1200,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            cap_recovery_factory=recovery,
            answer_verification="off",
            enable_thinking=False,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] == 10
        assert row["status"] == "correct"
        assert row["commit_source"] == "cap_recovery_answer_candidate_answer"
        assert row["answer_token_count"] < row["completion_tokens"]
        assert row["answer_token_count"] > 0
        assert row["cap_recovery_mode"] == "visible_finalizer"
        assert row["cap_recovery_request_id"] == "chatcmpl-recovery"
        assert row["cap_recovery_completion_tokens"] == 32
        assert row["cap_recovery_decode_tok_s"] == 28.0
        assert row["cap_recovery_request_enable_thinking"] is True
        assert row["cap_recovery_answer_text_tail_500"].endswith(r"\boxed{10}")

    _run(body())


def test_reasoning_only_answer_triggers_visible_submission_pass(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        cancelled: list[str] = []
        submission_inputs: list[tuple[str, str, Mapping[str, Any]]] = []

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = problem
                yield {
                    "mtplx_progress": {
                        "completion_tokens": 100,
                        "decode_tok_s": 55.0,
                    },
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    "Counts:\nk=1: 4\nk=2: 6\n"
                                    "So everything is consistent.\n"
                                    "The total is 10.\n"
                                )
                            }
                        }
                    ],
                }
                yield {
                    "mtplx_progress": {
                        "completion_tokens": 484,
                        "decode_tok_s": 44.0,
                    },
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    "Now I am still checking, but I never submitted "
                                    "visible answer content. "
                                )
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 484},
                    "mtplx_stats": {
                        "completion_tokens": 484,
                        "decode_tok_s": 44.0,
                        "request_enable_thinking": True,
                        "request_client_hint": "aime",
                        "request_session_bank_bypass": True,
                    },
                }

            return stream()

        async def cancel_factory(runner: AIMERunner, request_id: str) -> None:
            _ = runner
            cancelled.append(request_id)

        async def metrics_factory(
            runner: AIMERunner, request_id: str
        ) -> dict[str, Any]:
            return {
                "request_id": request_id,
                "prompt_tokens": 222,
                "completion_tokens": 484,
                "decode_tok_s": 44.0,
                "request_enable_thinking": True,
                "request_client_hint": "aime",
                "request_session_bank_bypass": True,
            }

        async def visible_submission(
            runner: AIMERunner,
            problem: AIMEProblem,
            first_reasoning_text: str,
            first_answer_text: str,
            first_stats: Mapping[str, Any],
        ) -> aime_mod.AIMEVisibleSubmission:
            _ = runner
            assert problem.answer == 10
            assert "The total is 10." in first_reasoning_text
            assert "still checking" in first_reasoning_text
            assert first_answer_text == ""
            assert first_stats["completion_tokens"] == 484
            submission_inputs.append(
                (first_reasoning_text, first_answer_text, first_stats)
            )
            return aime_mod.AIMEVisibleSubmission(
                mode="visible_submission",
                request_id="chatcmpl-visible-submit",
                final_answer=10,
                commit_source="answer_candidate_answer",
                reasoning_text="Submission pass checked the scratch.",
                answer_text="candidate_answer=10\n\\boxed{10}",
                usage={"completion_tokens": 18},
                stats={
                    "prompt_tokens": 333,
                    "completion_tokens": 18,
                    "decode_tok_s": 36.0,
                    "request_enable_thinking": True,
                    "request_reasoning_parser": "qwen3",
                },
                duration_ms=900,
            )

        async def recovery(
            runner: AIMERunner,
            problem: AIMEProblem,
            first_answer_text: str,
            first_stats: Mapping[str, Any],
        ) -> aime_mod.AIMECapRecovery:
            _ = runner
            _ = problem
            _ = first_answer_text
            _ = first_stats
            raise AssertionError("reasoning-on AIME must not run cap recovery")

        runner = AIMERunner(
            problems=problems,
            model_id="mtplx-qwen36-27b-optimized-speed",
            chat_stream_factory=factory,
            chat_cancel_factory=cancel_factory,
            chat_metrics_factory=metrics_factory,
            visible_submission_factory=visible_submission,
            cap_recovery_factory=recovery,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert cancelled == []
        assert len(submission_inputs) == 1
        assert row["extracted"] == 10
        assert row["status"] == "correct"
        assert row["commit_source"] == "visible_submission_answer_candidate_answer"
        assert row["reasoning_finalizer_handoff"] is True
        assert row["reasoning_finalizer_trigger_answer"] is None
        assert (
            row["reasoning_finalizer_trigger_source"]
            == "reasoning_only_no_visible_answer"
        )
        assert row["reasoning_finalizer_trigger_completion_tokens"] == 484
        assert row["visible_submission_mode"] == "visible_submission"
        assert row["visible_submission_request_id"] == "chatcmpl-visible-submit"
        assert row["visible_submission_extracted"] == 10
        assert (
            row["visible_submission_commit_source"]
            == "answer_candidate_answer"
        )
        assert row["visible_submission_completion_tokens"] == 18
        assert row["visible_submission_decode_tok_s"] == 36.0
        assert row["visible_submission_request_enable_thinking"] is True
        assert row["visible_submission_answer_text_tail_500"].endswith(r"\boxed{10}")
        assert row["cap_recovery_mode"] is None

    _run(body())


def test_visible_submission_pass_must_emit_visible_answer_to_score(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = runner
                _ = problem
                yield {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    "Hidden scratch says candidate_answer=10, "
                                    "but I never submit visible content."
                                )
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 128},
                    "mtplx_stats": {
                        "completion_tokens": 128,
                        "request_enable_thinking": True,
                    },
                }

            return stream()

        async def visible_submission(
            runner: AIMERunner,
            problem: AIMEProblem,
            first_reasoning_text: str,
            first_answer_text: str,
            first_stats: Mapping[str, Any],
        ) -> aime_mod.AIMEVisibleSubmission:
            _ = runner
            _ = problem
            assert "candidate_answer=10" in first_reasoning_text
            assert first_answer_text == ""
            assert first_stats["completion_tokens"] == 128
            return aime_mod.AIMEVisibleSubmission(
                mode="visible_submission",
                request_id="chatcmpl-visible-submit",
                final_answer=None,
                commit_source=None,
                reasoning_text="I still only put the answer in reasoning.",
                answer_text="",
                usage={"completion_tokens": 22},
                stats={
                    "completion_tokens": 22,
                    "request_enable_thinking": True,
                },
                duration_ms=500,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="mtplx-qwen36-27b-optimized-speed",
            chat_stream_factory=factory,
            visible_submission_factory=visible_submission,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] is None
        assert row["status"] == "abstain"
        assert row["commit_source"] == "visible_submission_no_answer"
        assert row["reasoning_finalizer_handoff"] is True
        assert row["reasoning_finalizer_trigger_answer"] is None
        assert row["visible_submission_extracted"] is None
        assert row["visible_submission_commit_source"] is None
        assert row["visible_submission_request_enable_thinking"] is True
        assert row["cap_recovery_mode"] is None

    _run(body())


def test_long_reasoning_stream_waits_for_visible_submission(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        cancelled: list[str] = []

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = problem
                yield {
                    "mtplx_progress": {
                        "completion_tokens": 4096,
                        "decode_tok_s": 42.0,
                    },
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    "I am still enumerating cases and have not "
                                    "submitted visible answer content. "
                                )
                            }
                        }
                    ],
                }
                yield _chunk(content="candidate_answer=10\n\\boxed{10}")
                yield {
                    "usage": {"completion_tokens": 4110},
                    "mtplx_stats": {
                        "request_id": runner.current_request_id,
                        "prompt_tokens": 222,
                        "completion_tokens": 4110,
                        "decode_tok_s": 41.0,
                        "request_enable_thinking": True,
                        "request_client_hint": "aime",
                    },
                }

            return stream()

        async def cancel_factory(runner: AIMERunner, request_id: str) -> None:
            _ = runner
            cancelled.append(request_id)

        async def metrics_factory(
            runner: AIMERunner, request_id: str
        ) -> dict[str, Any]:
            return {
                "request_id": request_id,
                "prompt_tokens": 222,
                "completion_tokens": 4110,
                "decode_tok_s": 42.0,
                "request_enable_thinking": True,
            }

        runner = AIMERunner(
            problems=problems,
            model_id="mtplx-qwen36-27b-optimized-speed",
            chat_stream_factory=factory,
            chat_cancel_factory=cancel_factory,
            chat_metrics_factory=metrics_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert cancelled == [runner.request_id_for(1, 1)]
        assert row["extracted"] == 10
        assert row["status"] == "correct"
        assert row["commit_source"] == "visible_stream_candidate_answer"
        assert row["reasoning_finalizer_handoff"] is False
        assert row["cap_recovery_mode"] is None
        assert row["answer_text_tail_500"].endswith(r"\boxed{10}")

    _run(body())


def test_custom_stream_without_visible_submission_keeps_reasoning_only_as_abstain(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        cancelled: list[str] = []

        async def default_stream_factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = problem
                yield {
                    "mtplx_progress": {
                        "completion_tokens": 100,
                        "decode_tok_s": 55.0,
                    },
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    "Cases:\na: 4\nb: 6\nThe total is 10.\n"
                                )
                            }
                        }
                    ],
                    "mtplx_stats": {
                        "completion_tokens": 100,
                        "request_enable_thinking": True,
                    },
                }
                yield {
                    "mtplx_progress": {
                        "completion_tokens": 484,
                        "decode_tok_s": 44.0,
                    },
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    "Still auditing instead of submitting. "
                                )
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 484},
                    "mtplx_stats": {
                        "request_id": runner.current_request_id,
                        "completion_tokens": 484,
                        "decode_tok_s": 44.0,
                        "request_enable_thinking": True,
                        "request_client_hint": "aime",
                    },
                }

            return stream()

        async def cancel_factory(runner: AIMERunner, request_id: str) -> None:
            _ = runner
            cancelled.append(request_id)

        runner = AIMERunner(
            problems=_make_problems(1),
            model_id="mtplx-qwen36-27b-optimized-speed",
            chat_stream_factory=default_stream_factory,
            chat_cancel_factory=cancel_factory,
            persist_dir=tmp_path,
        )
        assert runner.cap_recovery == "off"

        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert cancelled == []
        assert row["extracted"] is None
        assert row["status"] == "abstain"
        assert row["reasoning_finalizer_handoff"] is False
        assert row["cap_recovery_mode"] is None

    _run(body())


def test_default_cap_recovery_factory_uses_fresh_visible_request(
    monkeypatch,
) -> None:
    async def body() -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                yield (
                    'data: {"choices":[{"delta":{"content":"'
                    r"candidate_answer=10\n"
                    '"}}],"usage":{"completion_tokens":5},'
                    '"mtplx_stats":{"completion_tokens":5,'
                    '"request_enable_thinking":false,'
                    '"request_reasoning_parser":"gemma4",'
                    '"request_max_tokens":1280,'
                    '"effective_max_tokens":1280}}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                captured["method"] = method
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problems = _make_problems(1)
        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            run_id="aime-2026-recovery",
            base_url="http://127.0.0.1:8123",
            enable_thinking=False,
        )
        runner.current_idx = 1
        runner._current_attempt = 1
        runner._current_request_id = runner.request_id_for(1, 1)
        recovery = await aime_mod.default_cap_recovery_factory(
            runner,
            problems[0],
            "prior capped scratch that should not be copied",
            {"completion_tokens": 2048},
        )

        assert recovery is not None
        assert recovery.mode == "fresh_finalizer"
        assert recovery.request_id == (
            "chatcmpl-aime-2026-recovery-q1-a1-fresh1"
        )
        assert recovery.final_answer == 10
        assert captured["headers"]["x-mtplx-request-id"] == recovery.request_id
        payload = captured["json"]
        assert payload["enable_thinking"] is False
        assert payload["max_tokens"] == aime_mod.GEMMA_FRESH_FINALIZER_MAX_TOKENS
        assert payload["metadata"]["phase"] == "cap_recovery"
        assert payload["metadata"]["recovery_mode"] == "fresh_finalizer"
        assert recovery.stats["request_reasoning_parser"] == "gemma4"
        prompt_text = "\n".join(message["content"] for message in payload["messages"])
        assert "prior capped scratch" not in prompt_text
        assert "candidate_answer=N" in prompt_text

    _run(body())


def test_default_visible_submission_factory_uses_reasoning_request(
    monkeypatch,
) -> None:
    async def body() -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                yield (
                    'data: {"choices":[{"delta":{"reasoning_content":"'
                    r"brief hidden check"
                    '"}}]}'
                )
                yield (
                    'data: {"choices":[{"delta":{"content":"'
                    r"candidate_answer=123\n\\boxed{123}"
                    '"}}],"usage":{"completion_tokens":11},'
                    '"mtplx_stats":{"completion_tokens":11,'
                    '"request_enable_thinking":true,'
                    '"request_reasoning_parser":"qwen3"}}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                captured["method"] = method
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problem = AIMEProblem(
            id="2026-I-visible-submit",
            set="AIME I",
            year=2026,
            index=1,
            problem="Find the requested integer.",
            answer=777,
            source="https://example",
        )
        full_scratch = "scratch-start " + ("x" * 9000) + " scratch-end"
        runner = AIMERunner(
            problems=[problem],
            model_id="mtplx-qwen36-27b-optimized-speed",
            run_id="aime-2026-visible",
            base_url="http://127.0.0.1:8123",
        )
        runner.current_idx = 1
        runner._current_attempt = 1
        runner._current_request_id = runner.request_id_for(1, 1)
        submission = await aime_mod.default_visible_submission_factory(
            runner,
            problem,
            full_scratch,
            "",
            {"completion_tokens": 484},
        )

        assert submission is not None
        assert submission.request_id == (
            "chatcmpl-aime-2026-visible-q1-a1-submit1"
        )
        assert submission.final_answer == 123
        assert captured["headers"]["x-mtplx-request-id"] == submission.request_id
        payload = captured["json"]
        assert payload["enable_thinking"] is True
        assert payload["metadata"]["enable_thinking"] is True
        assert "max_tokens" not in payload
        assert payload["metadata"]["phase"] == "visible_submission"
        assert payload["metadata"]["primary_request_id"] == (
            "chatcmpl-aime-2026-visible-q1-a1"
        )
        prompt_text = "\n".join(message["content"] for message in payload["messages"])
        assert "official answer key" in prompt_text
        assert "777" not in prompt_text
        assert "scratch-start " in prompt_text
        assert " scratch-end" in prompt_text
        assert len(prompt_text) > 9000
        assert "Keep reasoning enabled" in prompt_text
        assert "Do not enumerate cases again" in prompt_text
        assert "candidate_answer=N" in prompt_text
        assert submission.stats["request_reasoning_parser"] == "qwen3"
        assert runner.current_request_id == "chatcmpl-aime-2026-visible-q1-a1"

    _run(body())


def test_fast_majority_verifier_corrects_wrong_candidate(tmp_path: Path) -> None:
    async def body() -> None:
        problems = [
            AIMEProblem(
                id="2026-I-verifier",
                set="AIME I",
                year=2026,
                index=1,
                problem="Find the requested value.",
                answer=62,
                source="https://example",
            )
        ]

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content="Fast draft.\ncandidate_answer=57\n")

            return stream()

        async def verifier(
            runner: AIMERunner,
            problem: AIMEProblem,
            proposed_answer: int,
            answer_text: str,
        ) -> aime_mod.AIMEAnswerVerification:
            _ = runner
            _ = problem
            _ = answer_text
            return aime_mod.AIMEAnswerVerification(
                mode="fast_majority",
                proposed_answer=proposed_answer,
                final_answer=62,
                answers=[62, 62],
                request_ids=["v1", "v2"],
                texts=["verified_answer=62", "verified_answer=62"],
                stats=[{"decode_tok_s": 40.0}, {"decode_tok_s": 39.0}],
                resolution="majority_corrected",
                duration_ms=12000,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            answer_verification="fast_majority",
            answer_verifier_factory=verifier,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.results[0].extracted == 62
        assert runner.results[0].status == "correct"
        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        done_events = [e for e in events if e["event"] == "question_done"]
        assert (
            done_events[-1]["commit_source"]
            == "answer_verification_majority_corrected"
        )

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] == 62
        assert row["answer_verification_proposed_answer"] == 57
        assert row["answer_verification_final_answer"] == 62
        assert row["answer_verification_answers"] == [62, 62]
        assert row["answer_verification_resolution"] == "majority_corrected"
        assert row["answer_verification_agreement"] is False
        assert row["answer_verification_stats"] == [
            {"decode_tok_s": 40.0},
            {"decode_tok_s": 39.0},
        ]

    _run(body())


def test_fast_majority_verifier_keeps_candidate_on_agreement(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content=f"candidate_answer={problem.answer}\n")

            return stream()

        async def verifier(
            runner: AIMERunner,
            problem: AIMEProblem,
            proposed_answer: int,
            answer_text: str,
        ) -> aime_mod.AIMEAnswerVerification:
            _ = runner
            _ = problem
            _ = answer_text
            return aime_mod.AIMEAnswerVerification(
                mode="fast_majority",
                proposed_answer=proposed_answer,
                final_answer=proposed_answer,
                answers=[proposed_answer],
                request_ids=["v1"],
                texts=[f"verified_answer={proposed_answer}"],
                resolution="majority_keep_proposed",
                duration_ms=5000,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            answer_verification="fast_majority",
            answer_verifier_factory=verifier,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] == 10
        assert row["status"] == "correct"
        assert row["commit_source"] == "answer_verification_majority_keep_proposed"
        assert row["answer_verification_agreement"] is True

    _run(body())


def test_fast_majority_no_verifier_answer_is_not_agreement(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content=f"candidate_answer={problem.answer}\n")

            return stream()

        async def verifier(
            runner: AIMERunner,
            problem: AIMEProblem,
            proposed_answer: int,
            answer_text: str,
        ) -> aime_mod.AIMEAnswerVerification:
            _ = runner
            _ = problem
            _ = answer_text
            return aime_mod.AIMEAnswerVerification(
                mode="fast_majority",
                proposed_answer=proposed_answer,
                final_answer=proposed_answer,
                answers=[None, None],
                request_ids=["v1", "v2"],
                texts=["no answer", "still no answer"],
                resolution="no_verifier_answer_keep_proposed",
                duration_ms=90000,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            answer_verification="fast_majority",
            answer_verifier_factory=verifier,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["answer_verification_answers"] == [None, None]
        assert row["answer_verification_resolution"] == (
            "no_verifier_answer_keep_proposed"
        )
        assert row["answer_verification_agreement"] is False

    _run(body())


def test_fast_majority_weak_verifier_disagreement_keeps_candidate_disputed(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = runner
                yield _chunk(content=f"candidate_answer={problem.answer}\n")

            return stream()

        async def verifier(
            runner: AIMERunner,
            problem: AIMEProblem,
            proposed_answer: int,
            answer_text: str,
        ) -> aime_mod.AIMEAnswerVerification:
            _ = runner
            _ = problem
            _ = answer_text
            return aime_mod.AIMEAnswerVerification(
                mode="fast_majority",
                proposed_answer=proposed_answer,
                final_answer=proposed_answer,
                answers=[4, None],
                request_ids=["v1", "v2"],
                texts=["verified_answer=4", "no answer"],
                resolution="weak_verifier_disagreed_keep_proposed",
                duration_ms=64000,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            answer_verification="fast_majority",
            answer_verifier_factory=verifier,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] == problems[0].answer
        assert row["status"] == "correct"
        assert row["commit_source"] == (
            "answer_verification_weak_verifier_disagreed_keep_proposed"
        )
        assert row["answer_verification_answers"] == [4, None]
        assert row["answer_verification_resolution"] == (
            "weak_verifier_disagreed_keep_proposed"
        )
        assert row["answer_verification_agreement"] is False

    _run(body())


def test_resolve_verified_answer_keeps_candidate_without_supported_alternative() -> None:
    assert aime_mod._resolve_verified_answer(29, [4, None]) == (
        29,
        "weak_verifier_disagreed_keep_proposed",
    )
    assert aime_mod._resolve_verified_answer(29, [4, 5]) == (
        29,
        "weak_verifier_disagreed_keep_proposed",
    )
    assert aime_mod._resolve_verified_answer(29, [4, 4]) == (
        None,
        "majority_disagreed_abstain",
    )


def test_fast_majority_verifier_disagreement_can_abstain(tmp_path: Path) -> None:
    async def body() -> None:
        problems = [
            AIMEProblem(
                id="2026-I-disputed",
                set="AIME I",
                year=2026,
                index=1,
                problem="Find the requested value.",
                answer=156,
                source="https://example",
            )
        ]

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                _ = runner
                _ = problem
                yield _chunk(content="Fast draft.\ncandidate_answer=168\n")

            return stream()

        async def verifier(
            runner: AIMERunner,
            problem: AIMEProblem,
            proposed_answer: int,
            answer_text: str,
        ) -> aime_mod.AIMEAnswerVerification:
            _ = runner
            _ = problem
            _ = answer_text
            return aime_mod.AIMEAnswerVerification(
                mode="fast_majority",
                proposed_answer=proposed_answer,
                final_answer=None,
                answers=[154, 154],
                request_ids=["v1", "v2"],
                texts=["verified_answer=154", "verified_answer=154"],
                resolution="majority_disagreed_abstain",
                duration_ms=77000,
            )

        runner = AIMERunner(
            problems=problems,
            model_id="gemma4-mtplx-optimized-speed",
            chat_stream_factory=factory,
            answer_verification="fast_majority",
            answer_verifier_factory=verifier,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert runner.results[0].extracted is None
        assert runner.results[0].status == "abstain"
        assert row["extracted"] is None
        assert row["status"] == "abstain"
        assert row["commit_source"] == "answer_verification_majority_disagreed_abstain"
        assert row["answer_verification_proposed_answer"] == 168
        assert row["answer_verification_final_answer"] is None
        assert row["answer_verification_answers"] == [154, 154]
        assert row["answer_verification_resolution"] == "majority_disagreed_abstain"
        assert row["answer_verification_agreement"] is False

    _run(body())


def test_default_verifier_stops_after_capped_no_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def body() -> None:
        calls: list[dict[str, Any]] = []

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"working"}}]}'
                yield (
                    'data: {"mtplx_stats":{"completion_tokens":1024,'
                    '"effective_max_tokens":1024}}'
                )
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                calls.append(
                    {
                        "method": method,
                        "url": url,
                        "json": json,
                        "headers": headers,
                    }
                )
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problem = _make_problems(1)[0]
        runner = AIMERunner(
            problems=[problem],
            model_id="gemma4-mtplx-optimized-speed",
            run_id="aime-2026-cap-test",
            base_url="http://127.0.0.1:8123",
            api_key="secret",
            answer_verification="fast_majority",
            persist_dir=tmp_path,
        )
        runner.current_idx = 1
        runner._current_attempt = 1
        runner._current_request_id = runner.request_id_for(1, 1)

        verification = await aime_mod.default_answer_verifier_factory(
            runner,
            problem,
            57,
            "candidate_answer=57",
        )

        assert verification is not None
        assert len(calls) == 1
        assert calls[0]["json"]["max_tokens"] == aime_mod.VERIFIER_MAX_TOKENS
        assert verification.answers == [None]
        assert verification.request_ids == ["chatcmpl-aime-2026-cap-test-q1-a1-verify1"]
        assert verification.resolution == "no_verifier_answer_keep_proposed"
        assert verification.agreement is False

    _run(body())


def test_default_verifier_abstains_on_majority_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def body() -> None:
        calls: list[dict[str, Any]] = []

        class FakeResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aread(self) -> bytes:
                return b""

            async def aiter_lines(self):
                chunk = {
                    "choices": [
                        {
                            "delta": {
                                "content": "work\nverified_answer=154\n\\boxed{154}"
                            }
                        }
                    ]
                }
                yield "data: " + json.dumps(chunk)
                yield "data: [DONE]"

        class FakeClient:
            def __init__(self, *args, **kwargs):
                _ = args
                _ = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, *, json, headers):
                calls.append(
                    {
                        "method": method,
                        "url": url,
                        "json": json,
                        "headers": headers,
                    }
                )
                return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        problem = _make_problems(1)[0]
        runner = AIMERunner(
            problems=[problem],
            model_id="gemma4-mtplx-optimized-speed",
            run_id="aime-2026-disagreement-test",
            base_url="http://127.0.0.1:8123",
            api_key="secret",
            answer_verification="fast_majority",
            persist_dir=tmp_path,
        )
        runner.current_idx = 1
        runner._current_attempt = 1
        runner._current_request_id = runner.request_id_for(1, 1)

        verification = await aime_mod.default_answer_verifier_factory(
            runner,
            problem,
            168,
            "candidate_answer=168",
        )

        assert verification is not None
        assert len(calls) == 2
        assert calls[0]["json"]["max_tokens"] == aime_mod.VERIFIER_MAX_TOKENS
        assert calls[1]["json"]["max_tokens"] == (
            aime_mod.VERIFIER_ADJUDICATOR_MAX_TOKENS
        )
        assert calls[1]["json"]["messages"][1]["content"].count("154") >= 1
        assert verification.final_answer is None
        assert verification.answers == [154, 154]
        assert verification.resolution == "majority_disagreed_abstain"
        assert verification.agreement is False

    _run(body())


def test_reasoning_candidate_answer_is_not_scored(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(reasoning="I found the value.\ncandidate_answer=10\n")
                yield _chunk(reasoning="A later thought can still change direction.")

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert runner.results[0].extracted is None
        assert runner.results[0].status == "abstain"

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        answer_events = [e for e in events if e["event"] == "answer_delta"]
        assert answer_events == []
        done_events = [e for e in events if e["event"] == "question_done"]
        assert done_events[-1]["commit_source"] is None

        rows = (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        row = json.loads(rows[0])
        assert row["commit_source"] is None
        assert row["answer_text_tail_500"] == ""
        assert "candidate_answer=10" in row["reasoning_text_tail_500"]

    _run(body())


def test_answer_candidate_answer_stops_visible_model_stream(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content="Computed final value.\ncandidate_answer=10\n")
                yield _chunk(content="Model-owned trailing work.\n")

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert runner.results[0].extracted == 10
        assert runner.results[0].status == "correct"

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        done_events = [e for e in events if e["event"] == "question_done"]
        assert done_events[-1]["commit_source"] == "visible_stream_candidate_answer"

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["commit_source"] == "visible_stream_candidate_answer"
        assert "Model-owned trailing work." not in row["answer_text_tail_500"]
        assert not row["answer_text_tail_500"].endswith(r"\boxed{10}")

    _run(body())


def test_visible_declared_answer_grades_after_stream_end(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(content="Checks are done. The answer is 10.")
                yield _chunk(content=" End of visible answer.")

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert runner.results[0].extracted == 10
        assert runner.results[0].status == "correct"

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        answer_events = [e for e in events if e["event"] == "answer_delta"]
        assert answer_events[-1]["text"] == " End of visible answer."
        done_events = [e for e in events if e["event"] == "question_done"]
        assert done_events[-1]["commit_source"] == "answer_declared_final"

    _run(body())


def test_reasoning_declared_answer_does_not_override_visible_submission(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = [
            AIMEProblem(
                id="2026-I-stream-boundary",
                set="AIME I",
                year=2026,
                index=1,
                problem="A fraction reduces to 252/25. Find m+n.",
                answer=277,
                source="https://example",
            )
        ]

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(reasoning="We need m+n. m+n = 252 + 25 = 27")
                yield _chunk(reasoning="7.")
                yield _chunk(content=r"\boxed{999}")

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert runner.results[0].extracted == 999
        assert runner.results[0].status == "wrong"
        assert "m+n = 252 + 25 = 277" in runner.results[0].reasoning_text

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        done_events = [e for e in events if e["event"] == "question_done"]
        assert done_events[-1]["commit_source"] == "visible_stream_boxed"

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] == 999
        assert row["commit_source"] == "visible_stream_boxed"

    _run(body())


def test_reasoning_cardinality_phrase_does_not_stop_stream(tmp_path: Path) -> None:
    async def body() -> None:
        problems = [
            AIMEProblem(
                id="2026-I-final-cardinality",
                set="AIME I",
                year=2026,
                index=4,
                problem="Count valid integers of the form a+b+ab.",
                answer=70,
                source="https://example",
            )
        ]

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                yield _chunk(
                    reasoning="\n".join(
                        [
                            "Excluded m=11 (prime). n=10. Prime. No solution. Correct.",
                            "So the logic seems solid.",
                            "The number of such integers is 70.",
                        ]
                    )
                )
                yield _chunk(
                    reasoning="This later recheck must not be read by the runner."
                )
                yield _chunk(content=r"\boxed{999}")

            return stream()

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert runner.results[0].extracted == 999
        assert runner.results[0].status == "wrong"
        assert "later recheck" in runner.results[0].reasoning_text

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        done_events = [e for e in events if e["event"] == "question_done"]
        assert done_events[-1]["commit_source"] == "visible_stream_boxed"

        row = json.loads(
            (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()[0]
        )
        assert row["extracted"] == 999
        assert row["commit_source"] == "visible_stream_boxed"

    _run(body())


def test_pause_hard_stops_current_question_and_resume_retries(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(3)
        cancelled: list[str] = []

        async def cancel_factory(runner: AIMERunner, request_id: str) -> None:
            _ = runner
            cancelled.append(request_id)

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=_blocks_first_attempt_factory(per_chunk_delay_s=0.02),
            chat_cancel_factory=cancel_factory,
            persist_dir=tmp_path,
        )
        queue, _ = runner.subscribe()
        await runner.start()

        # Wait until Q1 starts.
        for _ in range(200):
            if runner.current_idx >= 1:
                break
            await asyncio.sleep(0.01)

        await runner.pause()

        # Pause must stop the active stream instead of waiting for Q1 to finish.
        for _ in range(400):
            if runner.state == RunState.PAUSED:
                break
            await asyncio.sleep(0.01)
        assert runner.results[0].status is None
        assert runner.state == RunState.PAUSED
        assert runner.current_idx == 1
        assert cancelled == [runner.request_id_for(1, 1)]
        assert not (tmp_path / f"{runner.run_id}.jsonl").exists()

        await runner.resume()
        await asyncio.wait_for(runner._task, timeout=5.0)
        assert runner.state == RunState.DONE
        assert all(r.status == "correct" for r in runner.results)
        assert runner._attempts_by_idx[1] == 2

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())
        q1_started = [
            e for e in events if e["event"] == "question_started" and e["idx"] == 1
        ]
        assert [e["attempt"] for e in q1_started] == [1, 2]
        q1_done = [
            e for e in events if e["event"] == "question_done" and e["idx"] == 1
        ]
        assert [e["attempt"] for e in q1_done] == [2]
        assert "run_paused" in [e["event"] for e in events]
        assert "run_resumed" in [e["event"] for e in events]

        persisted = [
            json.loads(line)
            for line in (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        ]
        first_row = persisted[0]
        assert first_row["idx"] == 1
        assert first_row["attempt"] == 2
        assert first_row["request_id"] == runner.request_id_for(1, 2)
        assert first_row["status"] == "correct"

    _run(body())


def test_cancel_persists_partial(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(3)
        factory = _slow_factory(per_chunk_delay_s=0.05)
        cleanup_calls: list[tuple[int, str]] = []

        async def metrics_factory(
            runner: AIMERunner, request_id: str
        ) -> dict[str, Any]:
            _ = runner
            return {
                "request_id": request_id,
                "completion_tokens": 9279,
                "reasoning_tokens": 9279,
                "decode_tok_s": 42.86,
                "request_enable_thinking": True,
                "request_temperature": 1.0,
                "request_top_p": 0.8,
                "request_top_k": 60,
                "effective_temperature": 1.0,
                "effective_top_p": 0.8,
                "effective_top_k": 60,
            }

        async def cleanup_factory(
            runner: AIMERunner,
            result: aime_mod.QuestionResult,
            request_id: str,
        ) -> dict[str, Any]:
            _ = runner
            cleanup_calls.append((result.idx, request_id))
            return {
                "ok": True,
                "mlx_cache_cleanup": {
                    "cleared": True,
                    "reason": "aime_question_boundary",
                },
            }

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            temperature=1.0,
            top_p=0.8,
            top_k=60,
            enable_thinking=True,
            chat_stream_factory=factory,
            chat_metrics_factory=metrics_factory,
            question_isolation_factory=cleanup_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        for _ in range(200):
            if runner.current_idx >= 1:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.02)
        await runner.cancel()

        try:
            await asyncio.wait_for(runner._task, timeout=5.0)
        except asyncio.CancelledError:
            pass

        assert runner.state == RunState.CANCELLED
        persisted = (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        assert persisted, "expected persisted lines"
        row = json.loads(persisted[0])
        assert row["idx"] == 1
        assert row["error"] == "cancelled"
        assert cleanup_calls == [(1, runner.request_id_for(1, 1))]
        assert row["aime_question_boundary_cleanup"] == {
            "ok": True,
            "mlx_cache_cleanup": {
                "cleared": True,
                "reason": "aime_question_boundary",
            },
        }
        assert row["request_enable_thinking"] is True
        assert row["request_temperature"] == 1.0
        assert row["request_top_p"] == 0.8
        assert row["request_top_k"] == 60
        assert row["effective_temperature"] == 1.0
        assert row["effective_top_p"] == 0.8
        assert row["effective_top_k"] == 60
        assert row["reasoning_token_count"] == 9279
        summary = json.loads(persisted[-1])
        assert "summary" in summary
        assert summary["summary"]["state"] == "cancelled"

    _run(body())


def test_skip_current_marks_abstain_and_continues(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(3)
        cleanup_calls: list[tuple[int, str]] = []

        async def cleanup_factory(
            runner: AIMERunner,
            result: aime_mod.QuestionResult,
            request_id: str,
        ) -> dict[str, Any]:
            _ = runner
            cleanup_calls.append((result.idx, request_id))
            return {
                "ok": True,
                "mlx_cache_cleanup": {
                    "cleared": True,
                    "reason": "aime_question_boundary",
                },
            }

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=_blocking_first_factory(per_chunk_delay_s=0.02),
            question_isolation_factory=cleanup_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        for _ in range(200):
            if runner.current_idx == 1:
                break
            await asyncio.sleep(0.01)

        await runner.skip_current()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert [r.status for r in runner.results] == ["abstain", "correct", "correct"]
        assert runner.results[0].error == "skipped_by_user"
        assert runner.score == 2

        persisted = [
            json.loads(line)
            for line in (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        ]
        first_row = persisted[0]
        assert first_row["idx"] == 1
        assert first_row["status"] == "abstain"
        assert first_row["error"] == "skipped_by_user"
        assert first_row["commit_source"] == "user_skip"
        assert cleanup_calls == [
            (1, runner.request_id_for(1, 1)),
            (2, runner.request_id_for(2, 1)),
            (3, runner.request_id_for(3, 1)),
        ]
        assert first_row["aime_question_boundary_cleanup"] == {
            "ok": True,
            "mlx_cache_cleanup": {
                "cleared": True,
                "reason": "aime_question_boundary",
            },
        }
        assert persisted[-1]["summary"]["state"] == "done"

    _run(body())


def test_question_runtime_overrides_base_url_and_persists_cleanup(
    tmp_path: Path,
) -> None:
    async def body() -> None:
        problems = _make_problems(2)
        seen_urls: list[str] = []
        cleanup_calls: list[int] = []

        async def runtime_factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> aime_mod.AIMEQuestionRuntime:
            _ = runner

            async def cleanup() -> dict[str, Any]:
                cleanup_calls.append(problem.index)
                return {"ok": True, "pid": 1000 + problem.index}

            return aime_mod.AIMEQuestionRuntime(
                base_url=f"http://worker-{problem.index}",
                cleanup=cleanup,
                metadata={"problem_index": problem.index},
            )

        async def factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> AsyncIterator[dict[str, Any]]:
            seen_urls.append(runner.request_base_url)
            return _fake_stream([_chunk(content=f"\\boxed{{{problem.answer}}}")])

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            question_runtime_factory=runtime_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.state == RunState.DONE
        assert seen_urls == ["http://worker-1", "http://worker-2"]
        assert cleanup_calls == [1, 2]
        assert runner.request_base_url == runner.base_url

        rows = [
            json.loads(line)
            for line in (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        ]
        question_rows = [row for row in rows if "idx" in row]
        assert question_rows[0]["aime_question_runtime_cleanup"] == {
            "ok": True,
            "base_url": "http://worker-1",
            "metadata": {"problem_index": 1},
            "cleanup": {"ok": True, "pid": 1001},
        }

    _run(body())


def test_question_runtime_cleanup_runs_on_cancel(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        cleanup_calls: list[int] = []

        async def runtime_factory(
            runner: AIMERunner, problem: AIMEProblem
        ) -> aime_mod.AIMEQuestionRuntime:
            _ = runner

            async def cleanup() -> dict[str, Any]:
                cleanup_calls.append(problem.index)
                return {"ok": True, "pid": 2000 + problem.index}

            return aime_mod.AIMEQuestionRuntime(
                base_url=f"http://worker-{problem.index}",
                cleanup=cleanup,
                metadata={"problem_index": problem.index},
            )

        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=_blocking_first_factory(per_chunk_delay_s=0.02),
            question_runtime_factory=runtime_factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        for _ in range(200):
            if runner.current_idx == 1:
                break
            await asyncio.sleep(0.01)

        await runner.cancel()
        try:
            await asyncio.wait_for(runner._task, timeout=5.0)
        except asyncio.CancelledError:
            pass

        assert runner.state == RunState.CANCELLED
        assert cleanup_calls == [1]
        assert runner.request_base_url == runner.base_url

        rows = [
            json.loads(line)
            for line in (tmp_path / f"{runner.run_id}.jsonl").read_text().splitlines()
        ]
        assert rows[0]["error"] == "cancelled"
        assert rows[0]["aime_question_runtime_cleanup"] == {
            "ok": True,
            "base_url": "http://worker-1",
            "metadata": {"problem_index": 1},
            "cleanup": {"ok": True, "pid": 2001},
        }

    _run(body())


def test_concurrent_start_raises(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(2)
        factory = _slow_factory(per_chunk_delay_s=0.05)

        r1 = await start_run(
            model_id="m",
            problems=problems,
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        for _ in range(200):
            if r1.current_idx >= 1:
                break
            await asyncio.sleep(0.01)

        with pytest.raises(ConcurrentRunError) as excinfo:
            await start_run(
                model_id="m",
                problems=problems,
                chat_stream_factory=factory,
                persist_dir=tmp_path,
            )
        assert excinfo.value.active_run_id == r1.run_id

        await r1.cancel()
        try:
            await asyncio.wait_for(r1._task, timeout=5.0)
        except asyncio.CancelledError:
            pass

    _run(body())


def test_abstain_when_no_boxed(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        factory = _make_factory({1: "I am not sure how to solve this."})
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)

        assert runner.results[0].extracted is None
        assert runner.results[0].status == "abstain"
        assert runner.score == 0
        assert runner.accuracy() == 0.0

    _run(body())


def test_snapshot_reports_state(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(2)
        factory = _slow_factory(per_chunk_delay_s=0.04)
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        for _ in range(200):
            if runner.current_idx >= 1:
                break
            await asyncio.sleep(0.01)
        snap = runner.snapshot()
        assert snap["run_id"] == runner.run_id
        assert snap["total"] == 2
        assert snap["state"] in {"running", "idle"}
        assert snap["model"] == "test-model"
        assert snap["current_idx"] == 1
        assert isinstance(snap["per_question"], list) and len(snap["per_question"]) == 2

        await asyncio.wait_for(runner._task, timeout=5.0)
        snap2 = runner.snapshot()
        assert snap2["state"] == "done"
        assert snap2["score"] == 2

    _run(body())


def test_resume_after_full_finish_is_noop(tmp_path: Path) -> None:
    async def body() -> None:
        problems = _make_problems(1)
        factory = _make_factory({1: r"\boxed{10}"})
        runner = AIMERunner(
            problems=problems,
            model_id="test-model",
            chat_stream_factory=factory,
            persist_dir=tmp_path,
        )
        await runner.start()
        await asyncio.wait_for(runner._task, timeout=5.0)
        assert runner.state == RunState.DONE
        state = await runner.resume()
        assert state == RunState.DONE

    _run(body())
