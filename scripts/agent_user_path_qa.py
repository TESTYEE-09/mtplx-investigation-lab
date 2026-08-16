#!/usr/bin/env python3
"""User-path QA for OpenCode and Claude-Code-style MTPLX traffic.

This is intentionally higher level than a unit or smoke test. It creates fresh
coding projects, streams through the OpenAI or Anthropic-compatible endpoints,
can drive the real OpenCode CLI, and records TTFT/TPS/cache/tool-loop evidence
as JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _now() -> float:
    return time.time()


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"ts": _now(), "event": event, "payload": payload or {}},
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _server_health(base_url: str) -> dict[str, Any]:
    return _http_json("GET", base_url.rstrip("/") + "/health", timeout_s=20.0)


def _server_snapshot(base_url: str) -> dict[str, Any]:
    return _http_json(
        "GET", base_url.rstrip("/") + "/v1/mtplx/snapshot", timeout_s=20.0
    )


def _fan_summary() -> dict[str, Any]:
    try:
        from mtplx.thermal import fan_summary

        return fan_summary()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "fans": []}


def _fans_are_actual_max(summary: dict[str, Any], *, fraction: float = 0.85) -> bool:
    fans = summary.get("fans") or []
    if not summary.get("ok") or not fans:
        return False
    for fan in fans:
        actual = fan.get("actual_rpm")
        capacity = fan.get("max_capacity_rpm")
        target = fan.get("target_rpm")
        mode = str(fan.get("mode") or "").lower()
        if actual is None:
            return False
        if capacity:
            if int(actual) < int(float(capacity) * fraction):
                return False
            continue
        if target:
            if int(actual) < int(float(target) * fraction):
                return False
            continue
        if mode not in {"manual", "max"}:
            return False
    return True


def _start_fan_monitor(
    writer: JsonlWriter, *, require_max_fans: bool, interval_s: float
) -> tuple[threading.Event, queue.Queue[dict[str, Any]], threading.Thread]:
    stop = threading.Event()
    failures: queue.Queue[dict[str, Any]] = queue.Queue()

    def poll() -> None:
        while not stop.is_set():
            summary = _fan_summary()
            actual_max = _fans_are_actual_max(summary)
            writer.write("fan_state", {"actual_max": actual_max, "summary": summary})
            if require_max_fans and not actual_max:
                failures.put(summary)
            stop.wait(interval_s)

    thread = threading.Thread(target=poll, name="mtplx-user-path-fan-monitor", daemon=True)
    thread.start()
    return stop, failures, thread


def _project_dir(root: Path, index: int) -> Path:
    return root / f"fresh_project_{index:02d}"


def _write_project(root: Path, index: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    (root / "README.md").write_text(
        "\n".join(
            [
                f"# MTPLX QA Project {index}",
                "",
                "This is a fresh OpenCode-style project for user-path QA.",
                "The agent should edit files, avoid loops, and produce runnable Python.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "src" / "symbol_graph.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def normalize_symbol(name: str) -> str:",
                "    return name.strip().replace('-', '_')",
                "",
                "def build_edges(lines: list[str]) -> list[tuple[str, str]]:",
                "    edges: list[tuple[str, str]] = []",
                "    for line in lines:",
                "        if '->' in line:",
                "            left, right = line.split('->', 1)",
                "            edges.append((normalize_symbol(left), normalize_symbol(right)))",
                "    return edges",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_symbol_graph.py").write_text(
        "\n".join(
            [
                "from src.symbol_graph import build_edges, normalize_symbol",
                "",
                "def test_normalize_symbol():",
                "    assert normalize_symbol(' hello-world ') == 'hello_world'",
                "",
                "def test_build_edges():",
                "    assert build_edges(['A -> B']) == [('A', 'B')]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _workspace_context(lines: int, *, index: int) -> str:
    rows = []
    for i in range(lines):
        rows.append(
            "workspace-file-{i:05d}: src/module_{bucket}/feature_{feature}.py "
            "defines SymbolNode, EdgePlan, RetryPolicy, and async repair task "
            "for OpenCode session {index}; keep exact naming stable.".format(
                i=i, bucket=i % 97, feature=i % 41, index=index
            )
        )
    return "\n".join(rows)


def _prompt(kind: str, index: int, *, project: Path | None, context_lines: int) -> str:
    project_text = f"\nProject path: {project}\n" if project is not None else "\n"
    if kind == "huge":
        return (
            "You are a coding agent working with a very large repository context."
            " Use the context to produce a concise implementation plan and a small"
            " Python patch. Do not repeat yourself. Return code and a test plan.\n"
            f"{project_text}\n"
            f"{_workspace_context(context_lines, index=index)}\n\n"
            f"Task {index}: implement a robust symbol graph migration helper."
        )
    if kind == "project":
        return (
            "You are in a fresh coding project. Inspect the existing intent from"
            " the file names below and produce a concrete patch plan plus code."
            " Avoid raw tool XML and avoid repeating the same paragraph.\n"
            f"{project_text}\n"
            "Files: README.md, src/symbol_graph.py, tests/test_symbol_graph.py.\n"
            f"Task {index}: extend the parser to support comments, duplicate"
            " edges, cycle detection, and clear tests."
        )
    if kind == "quick_project":
        return (
            "You are in a fresh coding project. Make one small concrete edit and"
            " then stop. Use the edit tool once to add a concise module docstring"
            " to src/symbol_graph.py that describes symbol graph parsing. Do not"
            " inspect every file and do not run tests. Avoid raw tool XML and do"
            " not repeat yourself.\n"
            f"{project_text}\n"
            "Files: README.md, src/symbol_graph.py, tests/test_symbol_graph.py.\n"
            f"Task {index}: add the docstring and finish."
        )
    if kind == "tool":
        return (
            "You are an OpenCode-style agent. Use the write_file tool exactly once"
            " to create qa_result.py containing a small parser and tests. Do not"
            " leak XML tags in assistant text."
            f"{project_text}"
        )
    repeated = _workspace_context(420, index=index)
    return (
        "You are inside a coding agent. Read the workspace notes below, then write"
        " one complete Python module with typed functions, error handling, and"
        " tests. Return code only.\n\n"
        f"{repeated}\n\n"
        f"Task {index}: build a dependency graph normalizer with a self-test block."
    )


def _tools_payload() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write UTF-8 text to a project file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read UTF-8 text from a project file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def _anthropic_tools_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": "write_file",
            "description": "Write UTF-8 text to a project file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "read_file",
            "description": "Read UTF-8 text from a project file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ]


def _extract_openai_delta(chunk: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    tools: list[dict[str, Any]] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        for key in ("content", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str):
                text_parts.append(value)
        for tool in delta.get("tool_calls") or []:
            if isinstance(tool, dict):
                tools.append(tool)
    return "".join(text_parts), tools


def _parse_sse_payloads(response: Any):
    event = None
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        yield event, "\n".join(data_lines)


def _detect_loop(text: str) -> dict[str, Any]:
    raw_tool_leak = any(tag in text for tag in ("<tool_call", "</tool_call>", "<function="))
    normalized = re.sub(r"\s+", " ", text).strip()
    repeated_window = False
    if len(normalized) >= 800:
        seen: set[str] = set()
        for start in range(0, len(normalized) - 240, 80):
            window = normalized[start : start + 240]
            if window in seen:
                repeated_window = True
                break
            seen.add(window)
    return {
        "raw_tool_leak": raw_tool_leak,
        "repeated_window": repeated_window,
        "chars": len(text),
    }


def _run_openai_request(
    *,
    base_url: str,
    model: str,
    index: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    depth: int | None,
    session_prefix: str,
    use_tools: bool,
    writer: JsonlWriter,
) -> dict[str, Any]:
    request_id = f"openai-{index}"
    session_id = f"{session_prefix}-openai-{index}"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": bool(enable_thinking),
        "metadata": {
            "client": "opencode_user_path_qa",
            "session_id": session_id,
        },
    }
    if depth is not None:
        body["depth"] = int(depth)
    if use_tools:
        body["tools"] = _tools_payload()
        body["tool_choice"] = "auto"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-MTPLX-Client": "opencode-user-path-qa",
            "X-MTPLX-Session-ID": session_id,
        },
    )
    started = time.perf_counter()
    writer.write("request_start", {"request_id": request_id, "session_id": session_id})
    first_token_s: float | None = None
    text_parts: list[str] = []
    tool_deltas: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for _event, data in _parse_sse_payloads(response):
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                text, tools = _extract_openai_delta(payload)
                if text or tools:
                    if first_token_s is None:
                        first_token_s = time.perf_counter()
                        writer.write(
                            "request_first_token",
                            {
                                "request_id": request_id,
                                "ttft_s": first_token_s - started,
                            },
                        )
                    text_parts.append(text)
                    tool_deltas.extend(tools)
                if isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
                if isinstance(payload.get("mtplx_stats"), dict):
                    stats = payload["mtplx_stats"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    text = "".join(text_parts)
    result = {
        "request_id": request_id,
        "session_id": session_id,
        "ok": error is None,
        "error": error,
        "wall_s": finished - started,
        "ttft_client_s": None if first_token_s is None else first_token_s - started,
        "usage": usage,
        "mtplx_stats": stats,
        "tool_delta_count": len(tool_deltas),
        "loop": _detect_loop(text),
        "text_tail": text[-1000:],
    }
    writer.write("request_finish", result)
    return result


def _run_anthropic_request(
    *,
    base_url: str,
    model: str,
    index: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    session_prefix: str,
    use_tools: bool,
    writer: JsonlWriter,
) -> dict[str, Any]:
    request_id = f"anthropic-{index}"
    session_id = f"{session_prefix}-anthropic-{index}"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "metadata": {"session_id": session_id, "client": "claude_code_user_path_qa"},
    }
    if use_tools:
        body["tools"] = _anthropic_tools_payload()
        body["tool_choice"] = {"type": "auto"}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-MTPLX-Client": "claude-code-user-path-qa",
            "X-MTPLX-Session-ID": session_id,
        },
    )
    started = time.perf_counter()
    writer.write("request_start", {"request_id": request_id, "session_id": session_id})
    first_token_s: float | None = None
    text_parts: list[str] = []
    tool_blocks = 0
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    stop_reason: str | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for event, data in _parse_sse_payloads(response):
                if not data:
                    continue
                payload = json.loads(data)
                if event == "content_block_start":
                    block = payload.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        tool_blocks += 1
                        if first_token_s is None:
                            first_token_s = time.perf_counter()
                            writer.write(
                                "request_first_token",
                                {
                                    "request_id": request_id,
                                    "ttft_s": first_token_s - started,
                                },
                            )
                if event == "content_block_delta":
                    delta = payload.get("delta") or {}
                    text = delta.get("text") or delta.get("thinking") or ""
                    if text:
                        if first_token_s is None:
                            first_token_s = time.perf_counter()
                            writer.write(
                                "request_first_token",
                                {
                                    "request_id": request_id,
                                    "ttft_s": first_token_s - started,
                                },
                            )
                        text_parts.append(str(text))
                if event == "message_delta":
                    usage = payload.get("usage") or usage
                    stats = payload.get("mtplx_stats") or stats
                    delta = payload.get("delta") or {}
                    stop_reason = delta.get("stop_reason") or stop_reason
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    text = "".join(text_parts)
    result = {
        "request_id": request_id,
        "session_id": session_id,
        "ok": error is None,
        "error": error,
        "wall_s": finished - started,
        "ttft_client_s": None if first_token_s is None else first_token_s - started,
        "usage": usage,
        "mtplx_stats": stats,
        "tool_block_count": tool_blocks,
        "stop_reason": stop_reason,
        "loop": _detect_loop(text),
        "text_tail": text[-1000:],
    }
    writer.write("request_finish", result)
    return result


def _write_opencode_config(
    config_home: Path,
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    enable_thinking: bool,
    depth: int | None,
) -> Path:
    config_dir = config_home / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "opencode.json"
    payload = {
        "provider": {
            "mtplx": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MTPLX (local QA)",
                "options": {
                    "baseURL": base_url.rstrip("/") + "/v1",
                    "apiKey": "mtplx-local",
                    "timeout": False,
                    "chunkTimeout": 900000,
                },
                "models": {
                    model: {
                        "name": "MTPLX local QA",
                        "reasoning": True,
                        "interleaved": {"field": "reasoning_content"},
                        "tool_call": True,
                        "temperature": True,
                        "limit": {"context": 262144, "output": 262144},
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "options": {
                            "enable_thinking": bool(enable_thinking),
                            "topP": 0.95,
                            **({} if depth is None else {"depth": int(depth)}),
                        },
                    }
                },
            }
        },
        "model": f"mtplx/{model}",
        "small_model": f"mtplx/{model}",
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def _run_opencode_request(
    *,
    model: str,
    index: int,
    prompt: str,
    project: Path,
    config_home: Path,
    dangerous: bool,
    timeout_s: float,
    writer: JsonlWriter,
) -> dict[str, Any]:
    request_id = f"opencode-{index}"
    command = [
        "opencode",
        "run",
        "--model",
        f"mtplx/{model}",
        "--format",
        "json",
        "--dir",
        str(project),
    ]
    if dangerous:
        command.append("--dangerously-skip-permissions")
    command.append(prompt)
    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(config_home)
    env["OPENCODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    started = time.perf_counter()
    writer.write(
        "request_start",
        {"request_id": request_id, "command": command, "project": str(project)},
    )
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=max(1.0, float(timeout_s)))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
    finished = time.perf_counter()
    steps: list[dict[str, Any]] = []
    errors: list[Any] = []
    first_step_ms: int | None = None
    last_step_ms: int | None = None
    output_tokens = 0
    tool_events = 0
    for raw_line in stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        ts = event.get("timestamp")
        if event.get("type") == "error":
            errors.append(event.get("error") or event)
        if event.get("type") == "step_start" and isinstance(ts, int):
            first_step_ms = ts if first_step_ms is None else min(first_step_ms, ts)
        if event.get("type") == "step_finish":
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            output_tokens += int(tokens.get("output") or 0)
            if isinstance(ts, int):
                last_step_ms = ts if last_step_ms is None else max(last_step_ms, ts)
            steps.append(
                {
                    "reason": part.get("reason"),
                    "input_tokens": int(tokens.get("input") or 0),
                    "output_tokens": int(tokens.get("output") or 0),
                    "timestamp_ms": ts,
                }
            )
        if "tool" in str(event.get("type") or "").lower():
            tool_events += 1
    project_files = sorted(
        str(path.relative_to(project))
        for path in project.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    result = {
        "request_id": request_id,
        "ok": proc.returncode == 0 and not errors and not timed_out,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "errors": errors[-5:],
        "wall_s": finished - started,
        "completion_tokens": output_tokens,
        "request_tok_s": output_tokens / (finished - started)
        if output_tokens and finished > started
        else None,
        "first_step_start_ms": first_step_ms,
        "last_step_finish_ms": last_step_ms,
        "steps": steps,
        "tool_events": tool_events,
        "project_files": project_files,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "loop": _detect_loop(stdout + "\n" + stderr),
    }
    writer.write("request_finish", result)
    return result


def _count_tokens(base_url: str, model: str, prompt: str) -> dict[str, Any]:
    try:
        return _http_json(
            "POST",
            base_url.rstrip("/") + "/v1/messages/count_tokens",
            payload={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
            },
            timeout_s=120.0,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _summarize(results: list[dict[str, Any]], *, wall_s: float) -> dict[str, Any]:
    total_completion = 0
    per_request: list[dict[str, Any]] = []
    for result in results:
        stats = result.get("mtplx_stats") or {}
        usage = result.get("usage") or {}
        completion = int(
            stats.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completion_tokens")
            or result.get("completion_tokens")
            or 0
        )
        total_completion += completion
        request_wall = float(result.get("wall_s") or 0.0)
        per_request.append(
            {
                "request_id": result.get("request_id"),
                "ok": result.get("ok"),
                "wall_s": request_wall,
                "completion_tokens": completion,
                "ttft_s": stats.get("ttft_s") or result.get("ttft_client_s"),
                "decode_tok_s": stats.get("decode_tok_s"),
                "request_tok_s": stats.get("request_tok_s")
                or result.get("request_tok_s")
                or (completion / request_wall if completion and request_wall else None),
                "prompt_tokens": stats.get("prompt_tokens") or usage.get("input_tokens"),
                "cached_tokens": stats.get("cached_tokens"),
                "new_prefill_tokens": stats.get("new_prefill_tokens"),
                "cache_source": stats.get("cache_source"),
                "session_restore_mode": stats.get("session_restore_mode"),
                "scheduler_lane": stats.get("scheduler_lane"),
                "accepted_by_depth": stats.get("accepted_by_depth"),
                "verify_calls": stats.get("verify_calls"),
                "verify_ms_per_call": stats.get("verify_ms_per_call"),
                "openai_bridge_policy_version": stats.get(
                    "openai_bridge_policy_version"
                ),
                "tool_contract_policy_version": stats.get(
                    "tool_contract_policy_version"
                ),
                "tool_contract_active": stats.get("tool_contract_active"),
                "active_batch_size": stats.get("active_batch_size")
                or stats.get("ar_batch_max_observed"),
                "mtp_disabled_reason": stats.get("mtp_disabled_reason"),
                "tool_events": result.get("tool_delta_count")
                or result.get("tool_block_count")
                or result.get("tool_events"),
                "loop": result.get("loop"),
            }
        )
    decode_values = [
        float(item["decode_tok_s"])
        for item in per_request
        if item.get("decode_tok_s") is not None
    ]
    ttft_values = [
        float(item["ttft_s"]) for item in per_request if item.get("ttft_s") is not None
    ]
    return {
        "wall_s": wall_s,
        "total_completion_tokens": total_completion,
        "aggregate_completion_tok_s": total_completion / wall_s if wall_s else 0.0,
        "decode_tok_s_min": min(decode_values) if decode_values else None,
        "decode_tok_s_max": max(decode_values) if decode_values else None,
        "decode_fairness_ratio": (
            min(decode_values) / max(decode_values)
            if decode_values and max(decode_values) > 0
            else None
        ),
        "ttft_s_max": max(ttft_values) if ttft_values else None,
        "loop_failures": [
            item["request_id"]
            for item in per_request
            if (item.get("loop") or {}).get("raw_tool_leak")
            or (item.get("loop") or {}).get("repeated_window")
        ],
        "per_request": per_request,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18083")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode", choices=["openai", "anthropic", "opencode"], default="openai"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--prompt-kind",
        choices=["long", "project", "quick_project", "tool", "huge"],
        default="project",
    )
    parser.add_argument("--context-lines", type=int, default=5200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--depth", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--session-prefix", default=None)
    parser.add_argument("--tools", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--count-tokens", action="store_true")
    parser.add_argument("--opencode-dangerously-skip-permissions", action="store_true")
    parser.add_argument("--opencode-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--require-max-fans", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fan-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--fail-on-loop", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    writer = JsonlWriter(Path(args.output_jsonl).expanduser().resolve())
    session_prefix = args.session_prefix or f"user-path-{int(time.time())}"
    writer.write("run_start", {**vars(args), "session_prefix": session_prefix})
    projects: dict[int, Path] = {}
    for index in range(1, int(args.concurrency) + 1):
        project = _project_dir(project_root, index)
        _write_project(project, index)
        projects[index] = project
    if args.mode == "opencode":
        config_path = _write_opencode_config(
            project_root / "opencode_xdg",
            base_url=args.base_url,
            model=args.model,
            max_tokens=int(args.max_tokens),
            enable_thinking=bool(args.enable_thinking),
            depth=args.depth,
        )
        writer.write("opencode_config", {"config_path": str(config_path)})

    before_fans = _fan_summary()
    writer.write(
        "fan_state_before",
        {"actual_max": _fans_are_actual_max(before_fans), "summary": before_fans},
    )
    if args.require_max_fans and not _fans_are_actual_max(before_fans):
        writer.write("run_failed", {"reason": "fans_not_actual_max_before_run"})
        return 2
    try:
        writer.write("health_before", _server_health(args.base_url))
        writer.write("snapshot_before", _server_snapshot(args.base_url))
    except Exception as exc:
        writer.write(
            "run_failed",
            {"reason": "server_probe_failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        return 2

    first_prompt = _prompt(
        args.prompt_kind,
        1,
        project=projects[1],
        context_lines=int(args.context_lines),
    )
    if args.count_tokens:
        writer.write("prompt_token_count", _count_tokens(args.base_url, args.model, first_prompt))

    stop_fans, fan_failures, fan_thread = _start_fan_monitor(
        writer,
        require_max_fans=bool(args.require_max_fans),
        interval_s=max(0.5, float(args.fan_poll_interval_s)),
    )
    started = time.perf_counter()
    results_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    threads: list[threading.Thread] = []
    use_tools = bool(args.tools if args.tools is not None else args.prompt_kind == "tool")
    try:
        for index in range(1, int(args.concurrency) + 1):
            prompt = _prompt(
                args.prompt_kind,
                index,
                project=projects[index],
                context_lines=int(args.context_lines),
            )
            if args.mode == "openai":
                def target(i: int = index, p: str = prompt) -> None:
                    results_queue.put(
                        _run_openai_request(
                            base_url=args.base_url,
                            model=args.model,
                            index=i,
                            prompt=p,
                            max_tokens=int(args.max_tokens),
                            temperature=float(args.temperature),
                            enable_thinking=bool(args.enable_thinking),
                            depth=args.depth,
                            session_prefix=session_prefix,
                            use_tools=use_tools,
                            writer=writer,
                        )
                    )
            elif args.mode == "anthropic":
                def target(i: int = index, p: str = prompt) -> None:
                    results_queue.put(
                        _run_anthropic_request(
                            base_url=args.base_url,
                            model=args.model,
                            index=i,
                            prompt=p,
                            max_tokens=int(args.max_tokens),
                            temperature=float(args.temperature),
                            session_prefix=session_prefix,
                            use_tools=use_tools,
                            writer=writer,
                        )
                    )
            else:
                def target(i: int = index, p: str = prompt) -> None:
                    results_queue.put(
                        _run_opencode_request(
                            model=args.model,
                            index=i,
                            prompt=p,
                            project=projects[i],
                            config_home=project_root / "opencode_xdg",
                            dangerous=bool(args.opencode_dangerously_skip_permissions),
                            timeout_s=float(args.opencode_timeout_s),
                            writer=writer,
                        )
                    )
            thread = threading.Thread(target=target, name=f"mtplx-user-path-{index}")
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    finally:
        stop_fans.set()
        fan_thread.join(timeout=5)
    wall_s = time.perf_counter() - started

    results: list[dict[str, Any]] = []
    while not results_queue.empty():
        results.append(results_queue.get())
    try:
        writer.write("health_after", _server_health(args.base_url))
        writer.write("snapshot_after", _server_snapshot(args.base_url))
    except Exception as exc:
        writer.write("server_probe_after_failed", {"error": f"{type(exc).__name__}: {exc}"})
    after_fans = _fan_summary()
    writer.write(
        "fan_state_after",
        {"actual_max": _fans_are_actual_max(after_fans), "summary": after_fans},
    )
    summary = _summarize(results, wall_s=wall_s)
    summary["fan_failures"] = fan_failures.qsize()
    writer.write("run_summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if fan_failures.qsize() and args.require_max_fans:
        return 3
    if any(not result.get("ok") for result in results):
        return 4
    if args.fail_on_loop and summary["loop_failures"]:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
