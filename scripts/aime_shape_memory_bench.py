#!/usr/bin/env python3
"""AIME-shaped memory regression harness for MTPLX OpenAI serving.

The failure this targets is small benchmark prompts with a huge response budget
(`max_tokens=65536`) creating oversized paged-KV allocations. The harness uses
the live OpenAI-compatible server so it measures the actual product path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None


AIME_PROMPTS = [
    "Find the sum of all positive integers n such that n^2 + 19n + 89 is a perfect square.",
    "Let a and b be positive integers with ab=432. What is the minimum possible value of a+b?",
    "A circle has radius 5. A chord is 6 units long. Find the distance from the center to the chord.",
    "How many ordered pairs of integers (x,y) satisfy x^2 + y^2 = 25?",
    "The sequence a_n is defined by a_1=2 and a_{n+1}=3a_n+1. Find a_5.",
    "A fair six-sided die is rolled three times. What is the probability that the sum is 10?",
    "Find the remainder when 7^2026 is divided by 13.",
    "The roots of x^2 - kx + 36 = 0 differ by 5. Find k if k is positive.",
    "A rectangle has integer side lengths and perimeter 50. What is the largest possible area?",
    "How many subsets of {1,2,3,4,5,6,7,8} have an even sum?",
]


CODING_PROMPTS = [
    "Write a compact Python function that validates and normalizes OpenAI chat messages for a local model server. Include edge cases.",
    "Design a small CLI command that prints before/after benchmark summaries for decode TPS, TTFT, and memory. Show the Python code.",
    "Given a slow FastAPI endpoint, explain how you would instrument request latency, first-token latency, and memory growth without changing behavior.",
]


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_s: float = 1200.0,
) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {body}") from exc
    return json.loads(raw) if raw else {}


def _try_json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_s: float = 1200.0,
) -> dict[str, Any] | None:
    try:
        return _json_request(
            url,
            method=method,
            payload=payload,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _system_memory() -> dict[str, Any]:
    if psutil is None:
        return {}
    vm = psutil.virtual_memory()
    data = {
        "total_bytes": int(getattr(vm, "total", 0) or 0),
        "available_bytes": int(getattr(vm, "available", 0) or 0),
        "used_bytes": int(getattr(vm, "used", 0) or 0),
        "percent": float(getattr(vm, "percent", 0.0) or 0.0),
    }
    wired = getattr(vm, "wired", None)
    if wired is not None:
        data["wired_bytes"] = int(wired)
    return data


def _process_memory(pid: int | None) -> dict[str, Any]:
    if pid is None or psutil is None:
        return {}
    try:
        proc = psutil.Process(pid)
        info = proc.memory_info()
    except Exception:
        return {}
    data = {"pid": int(pid), "rss_bytes": int(getattr(info, "rss", 0) or 0)}
    vms = getattr(info, "vms", None)
    if vms is not None:
        data["vms_bytes"] = int(vms)
    return data


class _MemorySampler:
    def __init__(
        self,
        *,
        pid: int | None,
        poll_s: float,
        abort_rss_gb: float | None,
        abort_system_used_gb: float | None,
    ) -> None:
        self.pid = pid
        self.poll_s = max(float(poll_s), 0.05)
        self.abort_rss_bytes = (
            int(abort_rss_gb * 1_000_000_000) if abort_rss_gb else None
        )
        self.abort_system_used_bytes = (
            int(abort_system_used_gb * 1_000_000_000)
            if abort_system_used_gb
            else None
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.aborted = False
        self.abort_reason: str | None = None
        self.process_peak: dict[str, Any] = {}
        self.system_peak: dict[str, Any] = {}

    def __enter__(self) -> "_MemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.poll_s * 2, 0.2))

    def _record_process(self, sample: dict[str, Any]) -> None:
        if not sample:
            return
        current = int(sample.get("rss_bytes") or 0)
        previous = int(self.process_peak.get("rss_bytes") or 0)
        if current >= previous:
            self.process_peak = sample
        if self.abort_rss_bytes is not None and current >= self.abort_rss_bytes:
            self._abort_server(f"process_rss_bytes>={self.abort_rss_bytes}")

    def _record_system(self, sample: dict[str, Any]) -> None:
        if not sample:
            return
        current = int(sample.get("used_bytes") or 0)
        previous = int(self.system_peak.get("used_bytes") or 0)
        if current >= previous:
            self.system_peak = sample
        if (
            self.abort_system_used_bytes is not None
            and current >= self.abort_system_used_bytes
        ):
            self._abort_server(f"system_used_bytes>={self.abort_system_used_bytes}")

    def _abort_server(self, reason: str) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.abort_reason = reason
        if self.pid is None:
            return
        try:
            os.kill(int(self.pid), signal.SIGTERM)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples += 1
            self._record_process(_process_memory(self.pid))
            self._record_system(_system_memory())
            self._stop.wait(self.poll_s)


def _server_pid_from_arg(args: argparse.Namespace) -> int | None:
    if args.server_pid:
        return int(args.server_pid)
    if args.server_pid_file:
        try:
            return int(Path(args.server_pid_file).read_text().strip())
        except Exception:
            return None
    return None


def _percent_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return ((after - before) / before) * 100.0


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _mean(values: list[Any]) -> float | None:
    numbers = _finite(values)
    return statistics.fmean(numbers) if numbers else None


def _max(values: list[Any]) -> float | None:
    numbers = _finite(values)
    return max(numbers) if numbers else None


def _last(values: list[Any]) -> Any:
    return values[-1] if values else None


def _slope_per_request(values: list[Any]) -> float | None:
    numbers = _finite(values)
    if len(numbers) < 2:
        return None
    first = numbers[0]
    last = numbers[-1]
    return (last - first) / float(len(numbers) - 1)


def _summarize_rows(rows: list[dict[str, Any]], *, suite: str) -> dict[str, Any]:
    stats = [row.get("mtplx_stats") or {} for row in rows]
    process = [
        row.get("process_memory_peak") or row.get("process_memory") or {}
        for row in rows
    ]
    system = [
        row.get("system_memory_peak") or row.get("system_memory") or {}
        for row in rows
    ]
    dynamic = [item.get("dynamic_paged_kv") or {} for item in stats]
    return {
        "suite": suite,
        "count": len(rows),
        "decode_tok_s_mean": _mean([item.get("decode_tok_s") for item in stats]),
        "request_tok_s_mean": _mean([item.get("request_tok_s") for item in stats]),
        "ttft_s_mean": _mean([item.get("ttft_s") for item in stats]),
        "prompt_eval_time_s_mean": _mean(
            [item.get("prompt_eval_time_s") for item in stats]
        ),
        "prefill_tok_s_mean": _mean([item.get("prefill_tok_s") for item in stats]),
        "completion_tokens_total": int(
            sum(int(item.get("completion_tokens") or 0) for item in stats)
        ),
        "prompt_tokens_mean": _mean([item.get("prompt_tokens") for item in stats]),
        "peak_memory_bytes_max": _max([item.get("peak_memory_bytes") for item in stats]),
        "active_memory_bytes_last": _last(
            [item.get("active_memory_bytes") for item in stats]
        ),
        "active_memory_bytes_slope_per_request": _slope_per_request(
            [item.get("active_memory_bytes") for item in stats]
        ),
        "cache_memory_bytes_last": _last(
            [item.get("cache_memory_bytes") for item in stats]
        ),
        "cache_memory_bytes_slope_per_request": _slope_per_request(
            [item.get("cache_memory_bytes") for item in stats]
        ),
        "process_rss_bytes_max": _max([item.get("rss_bytes") for item in process]),
        "process_rss_bytes_slope_per_request": _slope_per_request(
            [item.get("rss_bytes") for item in process]
        ),
        "system_used_bytes_max": _max([item.get("used_bytes") for item in system]),
        "system_used_bytes_slope_per_request": _slope_per_request(
            [item.get("used_bytes") for item in system]
        ),
        "system_wired_bytes_max": _max([item.get("wired_bytes") for item in system]),
        "system_wired_bytes_slope_per_request": _slope_per_request(
            [item.get("wired_bytes") for item in system]
        ),
        "paged_kv_capacity_tokens_max": _max(
            [item.get("paged_kv_capacity_tokens") for item in stats]
        ),
        "paged_kv_num_blocks_max": _max(
            [item.get("paged_kv_num_blocks") for item in stats]
        ),
        "paged_kv_grow_events_total": int(
            sum(int(item.get("paged_kv_grow_events") or 0) for item in stats)
        ),
        "sessionbank_snapshot_bytes_max": _max(
            [item.get("sessionbank_snapshot_bytes") for item in stats]
        ),
        "session_keep_live_ref_values": sorted(
            {
                str(item.get("request_session_keep_live_ref"))
                for item in stats
                if "request_session_keep_live_ref" in item
            }
        ),
        "dynamic_requested_new_tokens_max": _max(
            [item.get("requested_new_tokens") for item in dynamic]
        ),
        "dynamic_reserved_new_tokens_max": _max(
            [item.get("reserved_new_tokens") for item in dynamic]
        ),
        "dynamic_reservation_capped_count": int(
            sum(1 for item in dynamic if item.get("reservation_capped"))
        ),
    }


def _prompts_for_suite(suite: str) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    if suite in {"aime10", "all"}:
        prompts.extend((f"aime_shape_{idx:02d}", prompt) for idx, prompt in enumerate(AIME_PROMPTS, 1))
    if suite in {"coding3", "all"}:
        prompts.extend((f"coding_control_{idx:02d}", prompt) for idx, prompt in enumerate(CODING_PROMPTS, 1))
    return prompts


def _expanded_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    base = _prompts_for_suite(args.suite)
    repeat = max(1, int(args.repeat))
    prompts = [
        (f"{row_id}_r{repeat_index:02d}", prompt)
        for repeat_index in range(1, repeat + 1)
        for row_id, prompt in base
    ]
    if args.limit is not None:
        prompts = prompts[: max(0, int(args.limit))]
    return prompts


def _payload(args: argparse.Namespace, prompt: str, row_id: str) -> dict[str, Any]:
    if row_id.startswith("coding_control"):
        content = (
            "Complete the coding task concisely. Prefer concrete code or steps over "
            "background explanation.\n\n"
            f"Task: {prompt}"
        )
    elif args.prompt_mode == "answer-only":
        content = (
            "Answer with only the final answer. Do not explain or show reasoning.\n\n"
            f"Problem: {prompt}\nFinal answer:"
        )
    else:
        content = (
            "Solve the problem. Give the final answer clearly after the reasoning.\n\n"
            f"Problem: {prompt}"
        )
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "enable_thinking": bool(args.enable_thinking),
        "stream": False,
        "metadata": {
            "mtplx_benchmark": "aime_shape_memory",
            "row_id": row_id,
            "phase": args.phase,
            "sampler_source": _sampler_source(args),
        },
    }
    _apply_sampler_overrides(payload, args)
    if args.stop:
        payload["stop"] = args.stop
    return payload


def _sampler_source(args: argparse.Namespace) -> str:
    return (
        "explicit_payload"
        if any(
            getattr(args, field, None) is not None
            for field in ("temperature", "top_p", "top_k")
        )
        else "server_defaults"
    )


def _apply_sampler_overrides(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.temperature is not None:
        payload["temperature"] = float(args.temperature)
    if args.top_p is not None:
        payload["top_p"] = float(args.top_p)
    if args.top_k is not None:
        payload["top_k"] = int(args.top_k)


def _request_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_tokens": int(args.max_tokens),
        "sampler_source": _sampler_source(args),
        "temperature": (
            None if args.temperature is None else float(args.temperature)
        ),
        "top_p": None if args.top_p is None else float(args.top_p),
        "top_k": None if args.top_k is None else int(args.top_k),
        "seed": int(args.seed),
        "enable_thinking": bool(args.enable_thinking),
        "prompt_mode": args.prompt_mode,
        "stop": args.stop,
    }


def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{args.phase}-{args.suite}-rows.jsonl"
    summary_path = output_dir / f"{args.phase}-{args.suite}-summary.json"
    pid = _server_pid_from_arg(args)
    health_before = _try_json_request(f"{base_url}/health", api_key=args.api_key)
    if args.clear_cache_first:
        _try_json_request(
            f"{base_url}/admin/cache/clear",
            method="POST",
            payload={},
            api_key=args.api_key,
        )
    rows = []
    with rows_path.open("w", encoding="utf-8") as handle:
        for index, (row_id, prompt) in enumerate(_expanded_prompts(args), 1):
            started = time.perf_counter()
            payload = _payload(args, prompt, row_id)
            monitor = _MemorySampler(
                pid=pid,
                poll_s=float(args.memory_poll_s),
                abort_rss_gb=args.abort_rss_gb,
                abort_system_used_gb=args.abort_system_used_gb,
            )
            try:
                with monitor:
                    response = _json_request(
                        f"{base_url}/v1/chat/completions",
                        method="POST",
                        payload=payload,
                        api_key=args.api_key,
                        timeout_s=float(args.timeout_s),
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                row = {
                    "index": index,
                    "id": row_id,
                    "phase": args.phase,
                    "suite": args.suite,
                    "elapsed_wall_s": elapsed,
                    "prompt_preview": prompt[:160],
                    "error": str(exc),
                    "memory_sampler": {
                        "samples": monitor.samples,
                        "aborted": monitor.aborted,
                        "abort_reason": monitor.abort_reason,
                    },
                    "process_memory_peak": monitor.process_peak,
                    "system_memory_peak": monitor.system_peak,
                    "process_memory": _process_memory(pid),
                    "system_memory": _system_memory(),
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
                if monitor.aborted:
                    print(
                        json.dumps(
                            {
                                "id": row_id,
                                "aborted": True,
                                "abort_reason": monitor.abort_reason,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    break
                raise
            elapsed = time.perf_counter() - started
            stats = response.get("mtplx_stats") or {}
            metrics = _try_json_request(f"{base_url}/metrics", api_key=args.api_key)
            sessions = _try_json_request(f"{base_url}/admin/sessions", api_key=args.api_key)
            row = {
                "index": index,
                "id": row_id,
                "phase": args.phase,
                "suite": args.suite,
                "elapsed_wall_s": elapsed,
                "prompt_preview": prompt[:160],
                "request": _request_snapshot(args),
                "response_finish_reason": (
                    (response.get("choices") or [{}])[0].get("finish_reason")
                    if isinstance(response.get("choices"), list)
                    else None
                ),
                "usage": response.get("usage"),
                "mtplx_stats": stats,
                "metrics_latest": (metrics or {}).get("latest") if metrics else None,
                "sessions": sessions,
                "memory_sampler": {
                    "samples": monitor.samples,
                    "aborted": monitor.aborted,
                    "abort_reason": monitor.abort_reason,
                },
                "process_memory_peak": monitor.process_peak,
                "system_memory_peak": monitor.system_peak,
                "process_memory": _process_memory(pid),
                "system_memory": _system_memory(),
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "id": row_id,
                        "decode_tok_s": stats.get("decode_tok_s"),
                        "ttft_s": stats.get("ttft_s"),
                        "peak_memory_gb": (
                            (stats.get("peak_memory_bytes") or 0) / 1e9
                            if stats.get("peak_memory_bytes") is not None
                            else None
                        ),
                        "reserved_new_tokens": (
                            (stats.get("dynamic_paged_kv") or {}).get(
                                "reserved_new_tokens"
                            )
                        ),
                        "session_keep_live_ref": stats.get(
                            "request_session_keep_live_ref"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "phase": args.phase,
        "suite": args.suite,
        "created_at_unix": time.time(),
        "git": {"commit": _git_commit(), "branch": _git_branch()},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "server": {
            "base_url": base_url,
            "pid": pid,
            "health_before": health_before,
        },
        "request": {
            **_request_snapshot(args),
            "repeat": int(args.repeat),
            "limit": args.limit,
        },
        "pillars": _summarize_rows(rows, suite=args.suite),
        "artifacts": {"rows_jsonl": str(rows_path), "summary_json": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_summary(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gate_ok(metric: str, before: dict[str, Any], after: dict[str, Any], *, max_drop_pct: float) -> dict[str, Any]:
    before_value = before.get(metric)
    after_value = after.get(metric)
    delta = _percent_delta(
        float(before_value) if before_value is not None else None,
        float(after_value) if after_value is not None else None,
    )
    ok = delta is None or delta >= -abs(max_drop_pct)
    return {
        "metric": metric,
        "before": before_value,
        "after": after_value,
        "delta_pct": delta,
        "ok": ok,
        "max_allowed_drop_pct": max_drop_pct,
    }


def compare(args: argparse.Namespace) -> int:
    before = _load_summary(args.before)["pillars"]
    after = _load_summary(args.after)["pillars"]
    gates = [
        _gate_ok("decode_tok_s_mean", before, after, max_drop_pct=args.max_regression_pct),
        _gate_ok("ttft_s_mean", before, after, max_drop_pct=args.max_regression_pct),
        _gate_ok("prefill_tok_s_mean", before, after, max_drop_pct=args.max_regression_pct),
    ]
    memory_before = before.get("peak_memory_bytes_max")
    memory_after = after.get("peak_memory_bytes_max")
    memory_delta = _percent_delta(
        float(memory_before) if memory_before is not None else None,
        float(memory_after) if memory_after is not None else None,
    )
    memory_ok = memory_delta is not None and memory_delta <= -abs(args.min_memory_improvement_pct)
    comparison = {
        "before": args.before,
        "after": args.after,
        "gates": gates,
        "memory": {
            "metric": "peak_memory_bytes_max",
            "before": memory_before,
            "after": memory_after,
            "delta_pct": memory_delta,
            "ok": memory_ok,
            "min_required_improvement_pct": args.min_memory_improvement_pct,
        },
    }
    comparison["passed"] = bool(memory_ok and all(gate["ok"] for gate in gates))
    if args.output:
        Path(args.output).write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if comparison["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run AIME-shape or coding-control requests")
    run_p.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_p.add_argument("--api-key", default=os.environ.get("MTPLX_API_KEY"))
    run_p.add_argument("--model", default="mtplx-qwen36-27b-optimized-speed")
    run_p.add_argument("--phase", choices=("before", "after", "candidate"), required=True)
    run_p.add_argument("--suite", choices=("aime10", "coding3", "all"), default="aime10")
    run_p.add_argument("--repeat", type=int, default=1)
    run_p.add_argument("--limit", type=int)
    run_p.add_argument("--output-dir", required=True)
    run_p.add_argument("--server-pid", type=int)
    run_p.add_argument("--server-pid-file")
    run_p.add_argument("--max-tokens", type=int, default=65536)
    run_p.add_argument(
        "--temperature",
        type=float,
        help="Optional explicit sampler override. Omitted by default so serve profile defaults own QA.",
    )
    run_p.add_argument(
        "--top-p",
        type=float,
        help="Optional explicit sampler override. Omitted by default so serve profile defaults own QA.",
    )
    run_p.add_argument(
        "--top-k",
        type=int,
        help="Optional explicit sampler override. Omitted by default so serve profile defaults own QA.",
    )
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--enable-thinking", action="store_true", default=True)
    run_p.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    run_p.add_argument(
        "--prompt-mode",
        choices=("aime", "answer-only"),
        default="aime",
        help="Use answer-only for quick memory-policy runs that keep max_tokens high.",
    )
    run_p.add_argument(
        "--stop",
        action="append",
        help="OpenAI stop sequence. Repeat for multiple sequences.",
    )
    run_p.add_argument("--clear-cache-first", action="store_true")
    run_p.add_argument("--timeout-s", type=float, default=1200.0)
    run_p.add_argument("--memory-poll-s", type=float, default=0.5)
    run_p.add_argument(
        "--abort-rss-gb",
        type=float,
        help="Terminate the benchmark server if its RSS reaches this many GB.",
    )
    run_p.add_argument(
        "--abort-system-used-gb",
        type=float,
        help="Terminate the benchmark server if system used memory reaches this many GB.",
    )
    run_p.set_defaults(func=run)

    compare_p = sub.add_parser("compare", help="Compare before/after summaries")
    compare_p.add_argument("--before", required=True)
    compare_p.add_argument("--after", required=True)
    compare_p.add_argument("--output")
    compare_p.add_argument("--max-regression-pct", type=float, default=2.0)
    compare_p.add_argument("--min-memory-improvement-pct", type=float, default=10.0)
    compare_p.set_defaults(func=compare)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
