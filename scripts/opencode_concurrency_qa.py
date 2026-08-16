#!/usr/bin/env python3
"""Real OpenCode-style concurrency QA for MTPLX.

This harness is intentionally user-path shaped: long enough prompts, streaming
responses, per-request metrics, server snapshots, and fan-state proof. A
64-token run is useful plumbing smoke only; performance runs should use 256,
512, or 1024 max output tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
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
        record = {
            "ts": _now(),
            "event": event,
            "payload": payload or {},
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _server_snapshot(base_url: str, *, api_key: str | None = None) -> dict[str, Any]:
    return _http_json(
        "GET",
        base_url.rstrip("/") + "/v1/mtplx/snapshot",
        api_key=api_key,
    )


def _optional_server_snapshot(base_url: str, *, api_key: str | None = None) -> dict[str, Any]:
    try:
        return _server_snapshot(base_url, api_key=api_key)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"available": False, "reason": "not_mtplx_server"}
        raise


def _server_health(base_url: str, *, api_key: str | None = None) -> dict[str, Any]:
    return _http_json("GET", base_url.rstrip("/") + "/health", api_key=api_key)


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


def _prompt(kind: str, index: int) -> str:
    if kind == "long":
        repeated = "\n".join(
            f"# workspace file {i}: implement robust parser and tests for symbol graph {i}"
            for i in range(420)
        )
        return (
            "You are inside a coding agent. Read the workspace notes below, then "
            "write one complete Python module with typed functions, error handling, "
            "and tests. Return code only.\n\n"
            f"{repeated}\n\n"
            f"Task {index}: build a dependency graph normalizer with at least 90 "
            "lines of code and a compact self-test block."
        )
    if kind == "mixed" and index == 1:
        return _prompt("long", index)
    if kind == "mixed":
        return (
            f"Task {index}: return code only. Write a complete Python helper module "
            "for applying text edits to files. Include dataclasses, validation, "
            "three functions, and a small pytest-style test section."
        )
    return (
        f"Task {index}: return code only. Write a complete Python module named "
        f"opencode_concurrency_task_{index}. It must include dataclasses, input "
        "validation, a parser, a planner, a formatter, and at least six tests. "
        "Make it substantial enough to use the full output budget."
    )


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    text_parts: list[str] = []
    for choice in choices:
        delta = choice.get("delta") or {}
        for key in ("content", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str):
                text_parts.append(value)
    return "".join(text_parts)


def _run_http_request(
    *,
    base_url: str,
    model: str,
    index: int,
    prompt_kind: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    writer: JsonlWriter,
) -> dict[str, Any]:
    request_id = f"http-{index}"
    started = time.perf_counter()
    writer.write(
        "request_start",
        {"request_id": request_id, "index": index, "max_tokens": max_tokens},
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(prompt_kind, index)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": True,
        "metadata": {"client": "opencode_qa", "qa_index": index},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-MTPLX-Client": "opencode-qa",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    first_token_s: float | None = None
    chunks = 0
    chars = 0
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                chunk = json.loads(payload)
                delta_text = _extract_delta_text(chunk)
                if delta_text:
                    chunks += 1
                    chars += len(delta_text)
                    if first_token_s is None:
                        first_token_s = time.perf_counter()
                        writer.write(
                            "request_first_token",
                            {
                                "request_id": request_id,
                                "ttft_s": first_token_s - started,
                            },
                        )
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                if isinstance(chunk.get("mtplx_stats"), dict):
                    stats = chunk["mtplx_stats"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    result = {
        "request_id": request_id,
        "index": index,
        "ok": error is None,
        "error": error,
        "wall_s": finished - started,
        "ttft_client_s": None if first_token_s is None else first_token_s - started,
        "chunks": chunks,
        "streamed_chars": chars,
        "usage": usage,
        "mtplx_stats": stats,
    }
    writer.write("request_finish", result)
    return result


def _run_opencode_request(
    *,
    model: str,
    index: int,
    prompt_kind: str,
    max_tokens: int,
    writer: JsonlWriter,
) -> dict[str, Any]:
    request_id = f"opencode-{index}"
    prompt = _prompt(prompt_kind, index)
    env = dict(os.environ)
    env["MTPLX_QA_MAX_TOKENS"] = str(max_tokens)
    command = [
        "opencode",
        "run",
        "--model",
        model,
        "--format",
        "json",
        "--dir",
        "/tmp/mtplx-opencode-concurrency-qa",
        prompt,
    ]
    started = time.perf_counter()
    writer.write("request_start", {"request_id": request_id, "command": command})
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout, stderr = proc.communicate(timeout=1800)
    finished = time.perf_counter()
    parsed = _parse_opencode_json_events(stdout)
    result = {
        "request_id": request_id,
        "index": index,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "wall_s": finished - started,
        "completion_tokens": parsed["completion_tokens"],
        "opencode_steps": parsed["steps"],
        "opencode_first_step_start_ms": parsed["first_step_start_ms"],
        "opencode_last_step_finish_ms": parsed["last_step_finish_ms"],
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    writer.write("request_finish", result)
    return result


def _parse_opencode_json_events(stdout: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    first_start: int | None = None
    last_finish: int | None = None
    completion_tokens = 0
    for raw_line in stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        timestamp = event.get("timestamp")
        if event.get("type") == "step_start" and isinstance(timestamp, int):
            first_start = timestamp if first_start is None else min(first_start, timestamp)
        if event.get("type") != "step_finish":
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
        output = int(tokens.get("output") or 0)
        completion_tokens += output
        if isinstance(timestamp, int):
            last_finish = timestamp if last_finish is None else max(last_finish, timestamp)
        steps.append(
            {
                "reason": part.get("reason"),
                "input_tokens": int(tokens.get("input") or 0),
                "output_tokens": output,
                "reasoning_tokens": int(tokens.get("reasoning") or 0),
                "timestamp_ms": timestamp,
            }
        )
    return {
        "completion_tokens": completion_tokens,
        "steps": steps,
        "first_step_start_ms": first_start,
        "last_step_finish_ms": last_finish,
    }


def _start_fan_monitor(
    *,
    writer: JsonlWriter,
    require_max_fans: bool,
    interval_s: float,
) -> tuple[threading.Event, queue.Queue[dict[str, Any]], threading.Thread]:
    stop = threading.Event()
    failures: queue.Queue[dict[str, Any]] = queue.Queue()

    def poll() -> None:
        while not stop.is_set():
            summary = _fan_summary()
            ok = _fans_are_actual_max(summary)
            writer.write("fan_state", {"actual_max": ok, "summary": summary})
            if require_max_fans and not ok:
                failures.put(summary)
            stop.wait(interval_s)

    thread = threading.Thread(target=poll, name="mtplx-qa-fan-monitor", daemon=True)
    thread.start()
    return stop, failures, thread


def _summarize(results: list[dict[str, Any]], *, wall_s: float) -> dict[str, Any]:
    per_request: list[dict[str, Any]] = []
    total_completion = 0
    for result in results:
        stats = result.get("mtplx_stats") or {}
        usage = result.get("usage") or {}
        completion = int(
            stats.get("completion_tokens")
            or usage.get("completion_tokens")
            or result.get("completion_tokens")
            or 0
        )
        client_wall_s = float(result.get("wall_s") or 0.0)
        client_request_tok_s = (
            completion / client_wall_s if completion > 0 and client_wall_s > 0 else None
        )
        total_completion += completion
        per_request.append(
            {
                "request_id": result.get("request_id"),
                "ok": result.get("ok"),
                "completion_tokens": completion,
                "ttft_s": stats.get("ttft_s") or result.get("ttft_client_s"),
                "decode_tok_s": stats.get("decode_tok_s"),
                "request_tok_s": stats.get("request_tok_s") or client_request_tok_s,
                "client_wall_s": client_wall_s,
                "queue_wait_s": stats.get("queue_wait_s")
                or stats.get("lock_wait_time_s"),
                "prompt_tokens": stats.get("prompt_tokens")
                or usage.get("prompt_tokens"),
                "cached_tokens": stats.get("cached_tokens"),
                "new_prefill_tokens": stats.get("new_prefill_tokens"),
                "prefill_tok_s": stats.get("prompt_target_prefill_tok_s")
                or stats.get("prompt_tps"),
                "session_cache_hit": stats.get("session_cache_hit"),
                "session_restore_mode": stats.get("session_restore_mode"),
                "ar_batch_shared_prefix_tokens": stats.get(
                    "ar_batch_shared_prefix_tokens"
                ),
                "ar_batch_shared_prefix_prefill_s": stats.get(
                    "ar_batch_shared_prefix_prefill_s"
                ),
                "scheduler_lane": stats.get("scheduler_lane"),
                "scheduler_policy": stats.get("scheduler_policy"),
                "active_batch_size": stats.get("active_batch_size")
                or stats.get("ar_batch_max_observed"),
                "mtp_disabled_reason": stats.get("mtp_disabled_reason"),
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
        "aggregate_completion_tok_s": total_completion / wall_s if wall_s > 0 else 0.0,
        "per_request": per_request,
        "decode_tok_s_min": min(decode_values) if decode_values else None,
        "decode_tok_s_max": max(decode_values) if decode_values else None,
        "decode_fairness_ratio": (
            min(decode_values) / max(decode_values)
            if decode_values and max(decode_values) > 0
            else None
        ),
        "ttft_s_max": max(ttft_values) if ttft_values else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18083")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["http", "opencode"], default="http")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-kind", choices=["short", "long", "mixed"], default="short")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fan-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--require-max-fans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Bearer token for app-owned daemons that require an API key. Defaults to OPENAI_API_KEY.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    writer = JsonlWriter(Path(args.output_jsonl).expanduser())
    writer.write("run_start", vars(args))
    before_fans = _fan_summary()
    writer.write("fan_state_before", {"actual_max": _fans_are_actual_max(before_fans), "summary": before_fans})
    if args.require_max_fans and not _fans_are_actual_max(before_fans):
        writer.write("run_failed", {"reason": "fans_not_actual_max_before_run"})
        return 2
    try:
        writer.write("health_before", _server_health(args.base_url, api_key=args.api_key))
        writer.write(
            "snapshot_before",
            _optional_server_snapshot(args.base_url, api_key=args.api_key),
        )
    except Exception as exc:
        writer.write("run_failed", {"reason": "server_probe_failed", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    stop_fans, fan_failures, fan_thread = _start_fan_monitor(
        writer=writer,
        require_max_fans=bool(args.require_max_fans),
        interval_s=max(0.5, float(args.fan_poll_interval_s)),
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        worker_results: queue.Queue[dict[str, Any]] = queue.Queue()
        threads: list[threading.Thread] = []
        for index in range(1, int(args.concurrency) + 1):
            if args.mode == "http":
                def target(i: int = index) -> None:
                    worker_results.put(
                        _run_http_request(
                            base_url=args.base_url,
                            model=args.model,
                            index=i,
                            prompt_kind=args.prompt_kind,
                            max_tokens=int(args.max_tokens),
                            temperature=float(args.temperature),
                            api_key=args.api_key,
                            writer=writer,
                        )
                    )
            else:
                def target(i: int = index) -> None:
                    worker_results.put(
                        _run_opencode_request(
                            model=args.model,
                            index=i,
                            prompt_kind=args.prompt_kind,
                            max_tokens=int(args.max_tokens),
                            writer=writer,
                        )
                    )
            thread = threading.Thread(target=target, name=f"mtplx-qa-request-{index}")
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        while not worker_results.empty():
            results.append(worker_results.get())
    finally:
        stop_fans.set()
        fan_thread.join(timeout=5)
    wall_s = time.perf_counter() - started

    after_health: dict[str, Any] = {}
    after_snapshot: dict[str, Any] = {}
    try:
        after_health = _server_health(args.base_url, api_key=args.api_key)
        after_snapshot = _optional_server_snapshot(args.base_url, api_key=args.api_key)
        writer.write("health_after", after_health)
        writer.write("snapshot_after", after_snapshot)
    except Exception as exc:
        writer.write("server_probe_after_failed", {"error": f"{type(exc).__name__}: {exc}"})
    after_fans = _fan_summary()
    writer.write("fan_state_after", {"actual_max": _fans_are_actual_max(after_fans), "summary": after_fans})
    summary = _summarize(results, wall_s=wall_s)
    scheduler = after_health.get("scheduler") or after_snapshot.get("scheduler") or {}
    summary["scheduler"] = scheduler
    summary["fan_failures"] = fan_failures.qsize()
    writer.write("run_summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if fan_failures.qsize() and args.require_max_fans:
        return 3
    if any(not result.get("ok") for result in results):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
