#!/usr/bin/env python3
"""Answer-checking AIME-style gate for public ``mtplx serve``.

This is a compatibility gate, not a model-quality leaderboard. It verifies that
the OpenAI-compatible server can run a 20-prompt math-shaped workload, return
extractable answers, expose MTPLX stats, and survive a 4-process client burst.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


AIME_LITE_PROMPTS: list[tuple[str, str, str]] = [
    (
        "aime_shape_01",
        "Find the sum of all positive integers n such that n^2 + 19n + 89 is a perfect square.",
        "0",
    ),
    (
        "aime_shape_02",
        "Let a and b be positive integers with ab=432. What is the minimum possible value of a+b?",
        "42",
    ),
    (
        "aime_shape_03",
        "A circle has radius 5. A chord is 6 units long. Find the distance from the center to the chord.",
        "4",
    ),
    (
        "aime_shape_04",
        "How many ordered pairs of integers (x,y) satisfy x^2 + y^2 = 25?",
        "12",
    ),
    (
        "aime_shape_05",
        "The sequence a_n is defined by a_1=2 and a_{n+1}=3a_n+1. Find a_5.",
        "202",
    ),
    (
        "aime_shape_06",
        "A fair six-sided die is rolled three times. What is the probability that the sum is 10?",
        "1/8",
    ),
    (
        "aime_shape_07",
        "Find the remainder when 7^2026 is divided by 13.",
        "4",
    ),
    (
        "aime_shape_08",
        "The roots of x^2 - kx + 36 = 0 differ by 5. Find k if k is positive.",
        "13",
    ),
    (
        "aime_shape_09",
        "A rectangle has integer side lengths and perimeter 50. What is the largest possible area?",
        "156",
    ),
    (
        "aime_shape_10",
        "How many subsets of {1,2,3,4,5,6,7,8} have an even sum?",
        "128",
    ),
]


def _expanded_prompts(repeat: int, limit: int | None) -> list[tuple[str, str, str]]:
    rows = [
        (f"{row_id}_r{idx:02d}", prompt, expected)
        for idx in range(1, max(1, repeat) + 1)
        for row_id, prompt, expected in AIME_LITE_PROMPTS
    ]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def _client_request_id(row_id: str) -> str:
    safe_row_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", row_id).strip("-")[:48]
    suffix = time.time_ns() % 1_000_000_000
    return f"aime-{safe_row_id or 'row'}-{os.getpid()}-{suffix}"


def _response_id_for_client_request_id(client_request_id: str) -> str:
    return (
        client_request_id
        if client_request_id.startswith("chatcmpl-")
        else f"chatcmpl-{client_request_id}"
    )


def _submit_answer_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer after finishing the reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The exact final answer, with no explanation.",
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def _payload(
    args: argparse.Namespace,
    row_id: str,
    prompt: str,
    *,
    client_request_id: str,
) -> dict[str, Any]:
    if args.answer_contract == "tool":
        system = (
            "You are running an MTPLX launch-readiness answer contract. "
            "Solve this short AIME-style sanity-check problem decisively. "
            "Reason only until the final answer is determined; do not keep "
            "exploring after the answer is known. If a requested set is empty, "
            "the sum or count is 0. If you prove that no integer or case can "
            "satisfy the condition, submit 0 immediately instead of rechecking "
            "the same algebra. When done, immediately call `submit_answer` with "
            "only the exact final answer. Do not put the final answer in normal "
            "assistant text."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Problem: {prompt}"},
        ]
    elif args.prompt_mode == "answer-only":
        content = (
            "Answer with only the final answer. Do not explain.\n\n"
            f"Problem: {prompt}\nFinal answer:"
        )
        messages = [{"role": "user", "content": content}]
    else:
        content = (
            "Solve briefly. Show only the essential calculation, then end with "
            f"exactly `Final answer: <answer>`.\n\nProblem: {prompt}"
        )
        messages = [{"role": "user", "content": content}]
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "seed": int(args.seed),
        "enable_thinking": bool(args.enable_thinking),
        "stream": bool(args.stream),
        "metadata": {
            "mtplx_benchmark": "aime_serve_gate",
            "mtplx_request_id": client_request_id,
            "row_id": row_id,
            "client_mode": args.mode,
            "answer_contract": args.answer_contract,
            "sampler_source": _sampler_source(args),
        },
    }
    _apply_sampler_overrides(payload, args)
    if args.answer_contract == "tool":
        payload["tools"] = [_submit_answer_tool_schema()]
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "submit_answer"},
        }
    if args.max_tokens is not None:
        payload["max_tokens"] = int(args.max_tokens)
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


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None,
    client_request_id: str,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-MTPLX-Request-ID": client_request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _append_stream_tool_call(
    tool_calls: dict[int, dict[str, Any]],
    item: dict[str, Any],
) -> None:
    try:
        index = int(item.get("index") or 0)
    except (TypeError, ValueError):
        index = 0
    current = tool_calls.setdefault(
        index,
        {
            "id": item.get("id") or f"call_stream_{index}",
            "type": item.get("type") or "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if item.get("id"):
        current["id"] = item["id"]
    if item.get("type"):
        current["type"] = item["type"]
    function = item.get("function")
    if not isinstance(function, dict):
        return
    current_function = current.setdefault("function", {"name": "", "arguments": ""})
    name = function.get("name")
    if name:
        current_function["name"] = str(name)
    if "arguments" in function:
        current_function["arguments"] = str(current_function.get("arguments") or "") + str(
            function.get("arguments") or ""
        )


def _post_stream_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None,
    client_request_id: str,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-MTPLX-Request-ID": client_request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    response_id: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    stats: dict[str, Any] = {}
    saw_done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                response_id = str(event.get("id") or response_id or "")
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                if isinstance(event.get("mtplx_stats"), dict):
                    stats = event["mtplx_stats"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice.get("finish_reason")
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    content_parts.append(content)
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                for item in delta.get("tool_calls") or []:
                    if isinstance(item, dict):
                        _append_stream_tool_call(tool_calls, item)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
    }
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "id": response_id,
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": usage,
        "mtplx_stats": stats,
        "_stream_complete": bool(saw_done and finish_reason),
    }


def _cancel_request(
    base_url: str,
    request_id: str,
    *,
    api_key: str | None,
) -> dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/mtplx/cancel/{request_id}",
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout_s: float, api_key: str | None) -> dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_answer_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    text = text.replace("\\boxed", " boxed ")
    text = text.replace("-", "-")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_correct(text: str, expected: str) -> bool:
    normalized = _normalize_answer_text(text)
    if expected == "1/8":
        return bool(
            re.search(r"(^|[^0-9])1\s*/\s*8([^0-9]|$)", normalized)
            or re.search(r"(^|[^0-9])0\.125([^0-9]|$)", normalized)
        )
    numbers = re.findall(r"-?\d+", normalized)
    return bool(numbers and numbers[-1] == expected)


def _submit_answer_from_message(message: dict[str, Any]) -> str | None:
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != "submit_answer":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            parsed = arguments
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return None
        else:
            return None
        answer = parsed.get("answer")
        return str(answer).strip() if answer is not None else None
    return None


def _run_one(task: tuple[dict[str, Any], str, str, str]) -> dict[str, Any]:
    config, row_id, prompt, expected = task
    base_url = str(config["base_url"]).rstrip("/")
    args = argparse.Namespace(**config["request_args"])
    client_request_id = _client_request_id(row_id)
    response_id = _response_id_for_client_request_id(client_request_id)
    payload = _payload(
        args,
        row_id,
        prompt,
        client_request_id=client_request_id,
    )
    started = time.perf_counter()
    try:
        post = _post_stream_json if args.stream else _post_json
        response = post(
            f"{base_url}/v1/chat/completions",
            payload,
            timeout_s=float(config["timeout_s"]),
            api_key=config.get("api_key"),
            client_request_id=client_request_id,
        )
        elapsed = time.perf_counter() - started
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        tool_answer = _submit_answer_from_message(message)
        answer_text = tool_answer if tool_answer is not None else content
        answer_source = "tool" if tool_answer is not None else "content"
        stream_complete = bool(response.get("_stream_complete", True))
        row_error = (
            None
            if stream_complete
            else "stream_ended_without_finish_reason"
        )
        stats = response.get("mtplx_stats") or {}
        return {
            "id": row_id,
            "request_id": response.get("id") or response_id,
            "pid": os.getpid(),
            "expected": expected,
            "payload_has_max_tokens": "max_tokens" in payload,
            "payload_has_max_completion_tokens": "max_completion_tokens" in payload,
            "payload_has_temperature": "temperature" in payload,
            "payload_has_top_p": "top_p" in payload,
            "payload_has_top_k": "top_k" in payload,
            "payload_sampler_source": payload["metadata"]["sampler_source"],
            "payload_stream": bool(payload.get("stream")),
            "content": content,
            "reasoning_tail": str(message.get("reasoning_content") or "")[-1200:],
            "tool_answer": tool_answer,
            "answer_source": answer_source,
            "tool_call_count": len(message.get("tool_calls") or []),
            "stream_complete": stream_complete,
            "normalized_tail": _normalize_answer_text(answer_text)[-320:],
            "correct": row_error is None and _is_correct(answer_text, expected),
            "finish_reason": choice.get("finish_reason"),
            "elapsed_wall_s": elapsed,
            "usage": response.get("usage"),
            "mtplx_stats": stats,
            "error": row_error,
            "cancel_after_error": None,
        }
    except Exception as exc:
        cancel_after_error = None
        try:
            cancel_after_error = _cancel_request(
                base_url,
                response_id,
                api_key=config.get("api_key"),
            )
        except Exception as cancel_exc:
            cancel_after_error = {"ok": False, "error": repr(cancel_exc)}
        return {
            "id": row_id,
            "request_id": response_id,
            "pid": os.getpid(),
            "expected": expected,
            "payload_has_max_tokens": "max_tokens" in payload,
            "payload_has_max_completion_tokens": "max_completion_tokens" in payload,
            "payload_has_temperature": "temperature" in payload,
            "payload_has_top_p": "top_p" in payload,
            "payload_has_top_k": "top_k" in payload,
            "payload_sampler_source": payload["metadata"]["sampler_source"],
            "payload_stream": bool(payload.get("stream")),
            "content": "",
            "reasoning_tail": "",
            "tool_answer": None,
            "answer_source": None,
            "tool_call_count": 0,
            "stream_complete": False,
            "normalized_tail": "",
            "correct": False,
            "finish_reason": None,
            "elapsed_wall_s": time.perf_counter() - started,
            "usage": None,
            "mtplx_stats": {},
            "error": repr(exc),
            "cancel_after_error": cancel_after_error,
        }


def _finite(values: list[Any]) -> list[float]:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numbers.append(float(value))
    return numbers


def _mean(values: list[Any]) -> float | None:
    numbers = _finite(values)
    return statistics.fmean(numbers) if numbers else None


def _min(values: list[Any]) -> float | None:
    numbers = _finite(values)
    return min(numbers) if numbers else None


def _max(values: list[Any]) -> float | None:
    numbers = _finite(values)
    return max(numbers) if numbers else None


def _summary(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    wall_s: float,
    health_before: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = [row.get("mtplx_stats") or {} for row in rows]
    dynamic = [item.get("dynamic_paged_kv") or {} for item in stats]
    return {
        "surface": "public mtplx serve",
        "base_url": args.base_url.rstrip("/"),
        "mode": args.mode,
        "workers": int(args.workers) if args.mode == "concurrent" else 1,
        "count": len(rows),
        "correct": sum(1 for row in rows if row.get("correct")),
        "errors": sum(1 for row in rows if row.get("error")),
        "wall_s": wall_s,
        "prompt_mode": args.prompt_mode,
        "answer_contract": args.answer_contract,
        "stream": bool(args.stream),
        "max_tokens": None if args.max_tokens is None else int(args.max_tokens),
        "sampler_source": _sampler_source(args),
        "temperature": (
            None if args.temperature is None else float(args.temperature)
        ),
        "top_p": None if args.top_p is None else float(args.top_p),
        "top_k": None if args.top_k is None else int(args.top_k),
        "enable_thinking": bool(args.enable_thinking),
        "finish_reasons": sorted({str(row.get("finish_reason")) for row in rows}),
        "tool_answer_rows": sum(1 for row in rows if row.get("answer_source") == "tool"),
        "tool_call_finish_rows": sum(
            1 for row in rows if row.get("finish_reason") == "tool_calls"
        ),
        "stream_incomplete_rows": sum(
            1 for row in rows if row.get("payload_stream") and not row.get("stream_complete")
        ),
        "answer_contract_ok": (
            True
            if args.answer_contract != "tool"
            else bool(rows)
            and all(row.get("answer_source") == "tool" for row in rows)
            and all(row.get("finish_reason") == "tool_calls" for row in rows)
        ),
        "payload_max_tokens_present_count": sum(
            1 for row in rows if row.get("payload_has_max_tokens")
        ),
        "payload_max_completion_tokens_present_count": sum(
            1 for row in rows if row.get("payload_has_max_completion_tokens")
        ),
        "payload_temperature_present_count": sum(
            1 for row in rows if row.get("payload_has_temperature")
        ),
        "payload_top_p_present_count": sum(
            1 for row in rows if row.get("payload_has_top_p")
        ),
        "payload_top_k_present_count": sum(
            1 for row in rows if row.get("payload_has_top_k")
        ),
        "uncapped_response_rows": sum(
            1 for item in stats if item.get("uncapped_response_requested") is True
        ),
        "decode_tok_s_mean": _mean([item.get("decode_tok_s") for item in stats]),
        "decode_tok_s_min": _min([item.get("decode_tok_s") for item in stats]),
        "request_tok_s_mean": _mean([item.get("request_tok_s") for item in stats]),
        "ttft_s_mean": _mean([item.get("ttft_s") for item in stats]),
        "ttft_s_max": _max([item.get("ttft_s") for item in stats]),
        "server_cap_applied_count": sum(
            1 for item in stats if item.get("server_cap_applied")
        ),
        "reasoning_token_rows": sum(
            1 for item in stats if int(item.get("reasoning_tokens") or 0) > 0
        ),
        "scheduler_lanes": sorted(
            {str(item.get("scheduler_lane")) for item in stats}
        ),
        "ar_batch_bypass_reasons": sorted(
            {str(item.get("ar_batch_bypass_reason")) for item in stats}
        ),
        "reserved_new_tokens_max": _max(
            [item.get("reserved_new_tokens") for item in dynamic]
        ),
        "health_before": health_before,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{args.phase}-{args.mode}-rows.jsonl"
    summary_path = output_dir / f"{args.phase}-{args.mode}-summary.json"
    health_before = None
    try:
        health_before = _get_json(
            f"{args.base_url.rstrip('/')}/health",
            timeout_s=float(args.timeout_s),
            api_key=args.api_key,
        )
    except Exception:
        health_before = None
    request_args = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "prompt_mode": args.prompt_mode,
        "answer_contract": args.answer_contract,
        "stream": args.stream,
        "mode": args.mode,
    }
    config = {
        "base_url": args.base_url.rstrip("/"),
        "api_key": args.api_key,
        "timeout_s": args.timeout_s,
        "request_args": request_args,
    }
    tasks = [(config, *row) for row in _expanded_prompts(args.repeat, args.limit)]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as handle:
        if args.mode == "concurrent":
            context_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
            context = mp.get_context(context_name)
            with futures.ProcessPoolExecutor(
                max_workers=int(args.workers),
                mp_context=context,
            ) as pool:
                pending = [pool.submit(_run_one, task) for task in tasks]
                for future in futures.as_completed(pending):
                    row = future.result()
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    _print_row(row)
        else:
            for task in tasks:
                row = _run_one(task)
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                _print_row(row)
    summary = _summary(
        args=args,
        rows=rows,
        wall_s=time.perf_counter() - started,
        health_before=health_before,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "artifacts": {
        "rows_jsonl": str(rows_path),
        "summary_json": str(summary_path),
    }}, indent=2, sort_keys=True))
    passed = (
        summary["correct"] == summary["count"]
        and not summary["errors"]
        and bool(summary["answer_contract_ok"])
    )
    return 0 if passed else 1


def _print_row(row: dict[str, Any]) -> None:
    stats = row.get("mtplx_stats") or {}
    print(
        json.dumps(
            {
                "id": row.get("id"),
                "pid": row.get("pid"),
                "correct": row.get("correct"),
                "finish": row.get("finish_reason"),
                "answer_source": row.get("answer_source"),
                "tool_answer": row.get("tool_answer"),
                "stream": row.get("payload_stream"),
                "tail": str(row.get("normalized_tail") or "")[-120:],
                "reasoning_tail": str(row.get("reasoning_tail") or "")[-120:],
                "decode_tok_s": stats.get("decode_tok_s"),
                "request_tok_s": stats.get("request_tok_s"),
                "ttft_s": stats.get("ttft_s"),
                "lane": stats.get("scheduler_lane"),
                "bypass": stats.get("ar_batch_bypass_reason"),
                "cap": stats.get("server_cap_applied"),
                "error": row.get("error"),
                "cancel_after_error": row.get("cancel_after_error"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18183")
    parser.add_argument("--api-key", default=os.environ.get("MTPLX_API_KEY"))
    parser.add_argument("--model", default="mtplx-qwen36-27b-optimized-speed")
    parser.add_argument("--phase", default="candidate")
    parser.add_argument("--mode", choices=("sequential", "concurrent"), default="sequential")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-mode", choices=("brief", "answer-only"), default="brief")
    parser.add_argument(
        "--answer-contract",
        choices=("text", "tool"),
        default="text",
        help="Use normal final-answer text or a required submit_answer tool call.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--no-max-tokens",
        action="store_true",
        help="Omit max_tokens/max_completion_tokens entirely so the server owns the full response lease.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional explicit sampler override. Omitted by default so serve profile defaults own QA.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Optional explicit sampler override. Omitted by default so serve profile defaults own QA.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional explicit sampler override. Omitted by default so serve profile defaults own QA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    args = parser.parse_args()
    if args.no_max_tokens:
        args.max_tokens = None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
