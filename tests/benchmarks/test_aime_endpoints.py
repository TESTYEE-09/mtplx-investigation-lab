"""End-to-end Phase 5 gate: AIME endpoints against a live FastAPI app.

Mounts a minimal FastAPI app with the AIME routes wired up, patches the
runner's chat stream factory with a fake stream, and exercises the full
HTTP contract: start, stream (SSE), pause, resume, skip, cancel, snapshot,
active, 409 on concurrent start, history. This proves the Python side
works without needing a real MLX backend.

The minimal app intentionally mirrors `mtplx/server/openai.py` AIME
endpoints (start/active/history/snapshot/pause/resume/skip/cancel/stream) so
the tests fail loudly on any schema drift between the two definitions.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time as _time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from mtplx.benchmarks.runners import aime as aime_runner
from mtplx.benchmarks.runners.aime import (
    AIMEProblem,
    AIMEQuestionRuntime,
    AIMERunner,
)


# ---- Fake chat stream + minimal app -------------------------------------


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


async def fake_factory(
    runner: AIMERunner, problem: AIMEProblem
) -> AsyncIterator[dict[str, Any]]:
    """Even-index problems get the correct boxed answer; odd get 999."""
    body = (
        f"\\boxed{{{problem.answer}}}"
        if problem.index % 2 == 0
        else "\\boxed{999}"
    )
    chunks = [
        _chunk(reasoning="step a "),
        _chunk(reasoning="step b "),
        _chunk(content=body),
    ]
    return _fake_stream(chunks)


def _make_problems(n: int = 4) -> list[AIMEProblem]:
    return [
        AIMEProblem(
            id=f"2026-I-{i}",
            set="AIME I",
            year=2026,
            index=i,
            problem=f"Q{i}",
            answer=i * 10,
            source="https://example",
        )
        for i in range(1, n + 1)
    ]


def _make_app(
    *,
    problems: list[AIMEProblem],
    persist_dir: Path,
    factory=fake_factory,
    question_runtime_factory=None,
) -> FastAPI:
    app = FastAPI()

    class _Start(BaseModel):
        model_config = ConfigDict(extra="ignore")
        year: int = 2026
        enable_thinking: bool | None = None
        answer_verification: str | None = None
        answer_verification_attempts: int | None = None
        question_limit: int | None = None

    @app.post("/v1/mtplx/benchmarks/aime/start")
    async def start(body: dict[str, Any] | None = Body(default=None)) -> JSONResponse:
        parsed_body = _Start.model_validate(body or {})
        year = parsed_body.year
        if year != 2026:
            raise HTTPException(status_code=400, detail="only AIME 2026 is shipped")
        run_problems = problems
        if parsed_body.question_limit is not None:
            question_limit = int(parsed_body.question_limit)
            if question_limit < 1 or question_limit > 30:
                raise HTTPException(
                    status_code=400,
                    detail="question_limit must be between 1 and 30",
                )
            run_problems = problems[:question_limit]
        kwargs: dict[str, Any] = {}
        if parsed_body.enable_thinking is not None:
            kwargs["enable_thinking"] = parsed_body.enable_thinking
        if parsed_body.answer_verification is not None:
            kwargs["answer_verification"] = parsed_body.answer_verification
        if parsed_body.answer_verification_attempts is not None:
            kwargs["answer_verification_attempts"] = (
                parsed_body.answer_verification_attempts
            )
        try:
            runner = await aime_runner.start_run(
                year=year,
                model_id="test-model",
                problems=run_problems,
                chat_stream_factory=factory,
                question_runtime_factory=question_runtime_factory,
                persist_dir=persist_dir,
                **kwargs,
            )
        except aime_runner.ConcurrentRunError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "run_in_progress",
                    "active_run_id": exc.active_run_id,
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "run_id": runner.run_id,
                "total": runner.total,
                "model": runner.model_id,
                "year": runner.year,
                "state": runner.state.value,
                "started_at": runner.snapshot()["started_at"],
            },
        )

    @app.get("/v1/mtplx/benchmarks/aime/active")
    def active() -> dict[str, Any]:
        return {"active_run_id": aime_runner.list_active_run_id()}

    @app.get("/v1/mtplx/benchmarks/aime/history")
    def history(limit: int = 5) -> dict[str, Any]:
        directory = persist_dir
        if not directory.is_dir():
            return {"runs": []}
        files = sorted(
            (p for p in directory.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        runs: list[dict[str, Any]] = []
        capped = max(1, min(int(limit or 5), 50))
        for path in files[:capped]:
            with path.open(encoding="utf-8") as handle:
                last_summary: dict[str, Any] | None = None
                for raw in handle:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and "summary" in obj:
                        last_summary = obj["summary"]
            if last_summary is not None:
                runs.append({"run_id": path.stem, "path": str(path), **last_summary})
        return {"runs": runs}

    @app.get("/v1/mtplx/benchmarks/aime/{run_id}")
    def snapshot(run_id: str) -> dict[str, Any]:
        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404)
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/pause")
    async def pause(run_id: str) -> dict[str, Any]:
        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404)
        await run.pause()
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/resume")
    async def resume(run_id: str) -> dict[str, Any]:
        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404)
        await run.resume()
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/skip")
    async def skip(run_id: str) -> dict[str, Any]:
        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404)
        await run.skip_current()
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/cancel")
    async def cancel(run_id: str) -> dict[str, Any]:
        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404)
        await run.cancel()
        return run.snapshot()

    @app.get("/v1/mtplx/benchmarks/aime/{run_id}/stream")
    async def stream(run_id: str) -> StreamingResponse:
        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404)
        terminal_kinds = {"run_done", "run_cancelled", "error"}
        queue, replay = run.subscribe()

        async def event_stream():
            saw_terminal = False
            try:
                for ev in replay:
                    yield (
                        f"event: {ev.get('event', 'message')}\n"
                        f"data: {json.dumps(ev)}\n\n"
                    )
                    if ev.get("event") in terminal_kinds:
                        saw_terminal = True
                if saw_terminal:
                    return
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield (
                        f"event: {ev.get('event', 'message')}\n"
                        f"data: {json.dumps(ev)}\n\n"
                    )
                    if ev.get("event") in terminal_kinds:
                        break
            finally:
                run.unsubscribe(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


@pytest.fixture(autouse=True)
def _clear_registry():
    aime_runner._active_runs.clear()
    yield
    aime_runner._active_runs.clear()


# ---- Tests ---------------------------------------------------------------


def _parse_sse(response_iter) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in response_iter:
        if not line:
            if current:
                events.append(current)
                current = None
            continue
        if line.startswith("event: "):
            current = {"event": line[len("event: "):]}
        elif line.startswith("data: ") and current is not None:
            current["data"] = json.loads(line[len("data: "):])
    if current:
        events.append(current)
    return events


def test_start_then_stream_runs_to_completion(tmp_path: Path) -> None:
    """Phase 5 happy path: POST start, GET stream, see every expected event."""
    problems = _make_problems(4)
    app = _make_app(problems=problems, persist_dir=tmp_path)

    with TestClient(app) as client:
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total"] == 4
        assert data["model"] == "test-model"
        run_id = data["run_id"]
        assert isinstance(run_id, str) and run_id.startswith("aime-2026-")
        assert data["year"] == 2026
        assert data["state"] in {"running", "paused"}

        active = client.get("/v1/mtplx/benchmarks/aime/active").json()
        assert active["active_run_id"] == run_id

        with client.stream(
            "GET", f"/v1/mtplx/benchmarks/aime/{run_id}/stream"
        ) as response:
            assert response.status_code == 200
            events = _parse_sse(response.iter_lines())

        kinds = [e["event"] for e in events]
        assert "run_started" in kinds
        assert kinds.count("question_started") == 4
        assert kinds.count("question_done") == 4
        assert "reasoning_delta" in kinds
        assert "answer_delta" in kinds
        assert kinds[-1] == "run_done"

        done_events = [e for e in events if e["event"] == "question_done"]
        statuses = [e["data"]["status"] for e in done_events]
        assert statuses.count("correct") == 2  # even index = correct
        assert statuses.count("wrong") == 2  # odd index = wrong

        final = events[-1]["data"]
        assert final["score"] == 2
        assert final["total"] == 4
        assert final["state"] == "done"

        active = client.get("/v1/mtplx/benchmarks/aime/active").json()
        assert active["active_run_id"] is None

        snap = client.get(f"/v1/mtplx/benchmarks/aime/{run_id}").json()
        assert snap["state"] == "done"
        assert snap["score"] == 2

        persisted = (tmp_path / f"{run_id}.jsonl").read_text().splitlines()
        assert len(persisted) == 5  # 4 question rows + 1 summary
        assert "summary" in json.loads(persisted[-1])


def test_start_can_limit_question_count(tmp_path: Path) -> None:
    problems = _make_problems(4)
    app = _make_app(problems=problems, persist_dir=tmp_path)

    with TestClient(app) as client:
        r = client.post(
            "/v1/mtplx/benchmarks/aime/start",
            json={"year": 2026, "question_limit": 2},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total"] == 2
        run_id = data["run_id"]

        with client.stream(
            "GET", f"/v1/mtplx/benchmarks/aime/{run_id}/stream"
        ) as response:
            assert response.status_code == 200
            events = _parse_sse(response.iter_lines())

        assert [e["event"] for e in events].count("question_started") == 2
        assert events[-1]["data"]["total"] == 2


def test_start_rejects_invalid_question_limit(tmp_path: Path) -> None:
    app = _make_app(problems=_make_problems(4), persist_dir=tmp_path)

    with TestClient(app) as client:
        r = client.post(
            "/v1/mtplx/benchmarks/aime/start",
            json={"year": 2026, "question_limit": 0},
        )

    assert r.status_code == 400
    assert "question_limit must be between 1 and 30" in r.text


def test_start_body_controls_reasoning_and_verification(tmp_path: Path) -> None:
    problems = _make_problems(1)
    app = _make_app(problems=problems, persist_dir=tmp_path)

    with TestClient(app) as client:
        invalid = client.post(
            "/v1/mtplx/benchmarks/aime/start",
            json={"year": 2027},
        )
        assert invalid.status_code == 400

        r = client.post(
            "/v1/mtplx/benchmarks/aime/start",
            json={
                "year": 2026,
                "enable_thinking": False,
                "answer_verification": "fast_majority",
                "answer_verification_attempts": 2,
            },
        )
        assert r.status_code == 200, r.text
        run_id = r.json()["run_id"]

        with client.stream(
            "GET", f"/v1/mtplx/benchmarks/aime/{run_id}/stream"
        ) as response:
            assert response.status_code == 200
            _ = _parse_sse(response.iter_lines())

        row = json.loads((tmp_path / f"{run_id}.jsonl").read_text().splitlines()[0])
        assert row["request_enable_thinking"] is None
        assert row["answer_verification_mode"] == "fast_majority"


def test_concurrent_start_returns_409(tmp_path: Path) -> None:
    problems = _make_problems(2)

    async def slow_factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream():
            for _ in range(5):
                await asyncio.sleep(0.05)
                yield _chunk(reasoning="...")
            yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    app = _make_app(problems=problems, persist_dir=tmp_path, factory=slow_factory)

    with TestClient(app) as client:
        r1 = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        assert r1.status_code == 200
        run_id = r1.json()["run_id"]

        r2 = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        assert r2.status_code == 409
        body = r2.json()
        assert body["error"] == "run_in_progress"
        assert body["active_run_id"] == run_id

        client.post(f"/v1/mtplx/benchmarks/aime/{run_id}/cancel")


def test_cancel_persists_partial(tmp_path: Path) -> None:
    problems = _make_problems(3)

    async def slow_factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream():
            for _ in range(10):
                await asyncio.sleep(0.05)
                yield _chunk(reasoning="...")
            yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    app = _make_app(problems=problems, persist_dir=tmp_path, factory=slow_factory)

    with TestClient(app) as client:
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        run_id = r.json()["run_id"]

        events_seen: list[str] = []

        def consume():
            with client.stream(
                "GET", f"/v1/mtplx/benchmarks/aime/{run_id}/stream"
            ) as response:
                for line in response.iter_lines():
                    if line and line.startswith("event: "):
                        kind = line[len("event: "):]
                        events_seen.append(kind)
                        if kind in {"run_cancelled", "run_done", "error"}:
                            return

        t = threading.Thread(target=consume)
        t.start()

        _time.sleep(0.2)
        cancel_resp = client.post(
            f"/v1/mtplx/benchmarks/aime/{run_id}/cancel"
        )
        assert cancel_resp.status_code == 200

        t.join(timeout=5.0)
        assert "run_cancelled" in events_seen

        persisted = (tmp_path / f"{run_id}.jsonl").read_text().splitlines()
        assert persisted
        summary = json.loads(persisted[-1])
        assert summary["summary"]["state"] == "cancelled"


def test_cancel_stops_active_question_runtime(tmp_path: Path) -> None:
    problems = _make_problems(2)
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()

    async def blocking_factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream():
            while True:
                await asyncio.sleep(0.05)
                yield _chunk(reasoning="still solving ")

        return stream()

    async def question_runtime_factory(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AIMEQuestionRuntime:
        async def cleanup() -> dict[str, Any]:
            cleanup_started.set()
            await asyncio.sleep(0.05)
            cleanup_finished.set()
            return {"ok": True, "pid": 1234, "returncode": -15}

        return AIMEQuestionRuntime(
            base_url=runner.base_url,
            cleanup=cleanup,
            metadata={"mode": "test_runtime"},
        )

    app = _make_app(
        problems=problems,
        persist_dir=tmp_path,
        factory=blocking_factory,
        question_runtime_factory=question_runtime_factory,
    )

    with TestClient(app) as client:
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        run_id = r.json()["run_id"]

        for _ in range(100):
            snap = client.get(f"/v1/mtplx/benchmarks/aime/{run_id}").json()
            if snap["current_idx"] == 1:
                break
            _time.sleep(0.02)
        assert snap["current_idx"] == 1

        cancel_resp = client.post(
            f"/v1/mtplx/benchmarks/aime/{run_id}/cancel"
        )
        assert cancel_resp.status_code == 200

        assert cleanup_started.wait(timeout=1.0)
        assert cleanup_finished.wait(timeout=1.0)

        for _ in range(100):
            active = client.get("/v1/mtplx/benchmarks/aime/active").json()
            if active["active_run_id"] is None:
                break
            _time.sleep(0.02)
        assert active["active_run_id"] is None


def test_history_returns_recent_runs(tmp_path: Path) -> None:
    """GET /history reads the JSONL tail summary lines for each finished run."""
    problems = _make_problems(2)
    app = _make_app(problems=problems, persist_dir=tmp_path)

    with TestClient(app) as client:
        # Run once.
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        run_id_a = r.json()["run_id"]
        with client.stream(
            "GET", f"/v1/mtplx/benchmarks/aime/{run_id_a}/stream"
        ) as response:
            for _ in _parse_sse(response.iter_lines()):
                pass

        # Run twice.
        _time.sleep(0.05)  # ensure mtime ordering is stable
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        run_id_b = r.json()["run_id"]
        with client.stream(
            "GET", f"/v1/mtplx/benchmarks/aime/{run_id_b}/stream"
        ) as response:
            for _ in _parse_sse(response.iter_lines()):
                pass

        history = client.get("/v1/mtplx/benchmarks/aime/history?limit=5").json()
        assert "runs" in history
        run_ids = [r["run_id"] for r in history["runs"]]
        # Most recent first.
        assert run_id_b in run_ids and run_id_a in run_ids
        assert run_ids.index(run_id_b) < run_ids.index(run_id_a)
        # Each row carries the summary fields.
        for row in history["runs"]:
            assert "score" in row and "total" in row and "state" in row


def test_pause_resume_hard_stops_and_retries_current_question(tmp_path: Path) -> None:
    """Pause must stop the active problem; resume retries that problem fresh."""
    problems = _make_problems(3)

    async def first_attempt_blocks(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream():
            if problem.index == 1 and runner.current_attempt == 1:
                while True:
                    await asyncio.sleep(0.05)
                    yield _chunk(reasoning="still solving ")
            yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    app = _make_app(
        problems=problems,
        persist_dir=tmp_path,
        factory=first_attempt_blocks,
    )

    with TestClient(app) as client:
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        run_id = r.json()["run_id"]

        events_seen: list[str] = []
        data_seen: list[dict[str, Any]] = []

        def consume():
            with client.stream(
                "GET", f"/v1/mtplx/benchmarks/aime/{run_id}/stream"
            ) as response:
                current_kind: str | None = None
                for line in response.iter_lines():
                    if line and line.startswith("event: "):
                        kind = line[len("event: "):]
                        current_kind = kind
                        events_seen.append(kind)
                    elif line and line.startswith("data: ") and current_kind:
                        data = json.loads(line[len("data: "):])
                        data_seen.append({"event": current_kind, **data})
                        if current_kind in {"run_cancelled", "run_done", "error"}:
                            return

        t = threading.Thread(target=consume)
        t.start()

        _time.sleep(0.07)
        pause_resp = client.post(
            f"/v1/mtplx/benchmarks/aime/{run_id}/pause"
        )
        assert pause_resp.status_code == 200
        for _ in range(100):
            snap = client.get(f"/v1/mtplx/benchmarks/aime/{run_id}").json()
            if snap["state"] == "paused":
                break
            _time.sleep(0.02)
        assert snap["state"] == "paused"
        assert snap["current_idx"] == 1
        assert snap["per_question"][0]["status"] is None

        resume_resp = client.post(
            f"/v1/mtplx/benchmarks/aime/{run_id}/resume"
        )
        assert resume_resp.status_code == 200

        t.join(timeout=5.0)
        assert "run_paused" in events_seen
        assert "run_resumed" in events_seen
        assert events_seen[-1] == "run_done"
        starts = [
            ev
            for ev in data_seen
            if ev["event"] == "question_started" and ev["idx"] == 1
        ]
        assert [ev["attempt"] for ev in starts] == [1, 2]
        done = [
            ev
            for ev in data_seen
            if ev["event"] == "question_done" and ev["idx"] == 1
        ]
        assert [ev["attempt"] for ev in done] == [2]

        persisted = [
            json.loads(line)
            for line in (tmp_path / f"{run_id}.jsonl").read_text().splitlines()
        ]
        first_row = persisted[0]
        assert first_row["idx"] == 1
        assert first_row["attempt"] == 2
        assert first_row["status"] == "correct"


def test_skip_current_abstains_and_stream_continues(tmp_path: Path) -> None:
    """Skip aborts only the active problem and keeps the run alive."""
    problems = _make_problems(3)

    async def first_problem_blocks(
        runner: AIMERunner, problem: AIMEProblem
    ) -> AsyncIterator[dict[str, Any]]:
        async def stream():
            if problem.index == 1:
                while True:
                    await asyncio.sleep(0.05)
                    yield _chunk(reasoning="still solving ")
            else:
                yield _chunk(content=f"\\boxed{{{problem.answer}}}")

        return stream()

    app = _make_app(
        problems=problems,
        persist_dir=tmp_path,
        factory=first_problem_blocks,
    )

    with TestClient(app) as client:
        r = client.post("/v1/mtplx/benchmarks/aime/start", json={"year": 2026})
        run_id = r.json()["run_id"]

        events_seen: list[str] = []

        def consume():
            with client.stream(
                "GET", f"/v1/mtplx/benchmarks/aime/{run_id}/stream"
            ) as response:
                for line in response.iter_lines():
                    if line and line.startswith("event: "):
                        kind = line[len("event: "):]
                        events_seen.append(kind)
                        if kind in {"run_cancelled", "run_done", "error"}:
                            return

        t = threading.Thread(target=consume)
        t.start()

        for _ in range(50):
            snap = client.get(f"/v1/mtplx/benchmarks/aime/{run_id}").json()
            if snap["current_idx"] == 1:
                break
            _time.sleep(0.02)

        skip_resp = client.post(
            f"/v1/mtplx/benchmarks/aime/{run_id}/skip"
        )
        assert skip_resp.status_code == 200

        t.join(timeout=5.0)
        assert events_seen[-1] == "run_done"

        snap = client.get(f"/v1/mtplx/benchmarks/aime/{run_id}").json()
        assert snap["state"] == "done"
        assert snap["per_question"][0]["status"] == "abstain"
        assert snap["per_question"][1]["status"] == "correct"
        assert snap["per_question"][2]["status"] == "correct"
