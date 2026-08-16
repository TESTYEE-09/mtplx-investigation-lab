#!/usr/bin/env python3
"""OpenCode-style same-session cache QA.

This exercises the user-visible failure mode directly:

1. Send a long coding-agent request.
2. Preserve the assistant response in the transcript.
3. Send a short follow-up in the same session.
4. Record whether SessionBank restored a large prefix and only prefetched the
   suffix.

It intentionally uses the OpenAI-compatible streaming endpoint because that is
the contract OpenCode Desktop ultimately hits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _now() -> float:
    return time.time()


def _http_json(method: str, url: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


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


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {"ts": _now(), "event": event, "payload": payload or {}}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _workspace_context(lines: int) -> str:
    rows = []
    for index in range(lines):
        rows.append(
            "repo-file-{index:05d}: src/game/system_{bucket}/module_{feature}.ts "
            "contains camera, WASD movement, bow aiming, terrain props, destructible "
            "environment state, and TypeScript strict errors. Keep identifiers stable.".format(
                index=index,
                bucket=index % 113,
                feature=index % 47,
            )
        )
    return "\n".join(rows)


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        for key in ("reasoning_content", "content"):
            value = delta.get(key)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _stream_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    session_id: str,
    writer: JsonlWriter,
    label: str,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": True,
        "metadata": {"client": "session_cache_followup_qa", "session_id": session_id},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-MTPLX-Client": "session-cache-followup-qa",
            "X-MTPLX-Session-ID": session_id,
        },
    )
    started = time.perf_counter()
    first_token_s: float | None = None
    chunks = 0
    text_parts: list[str] = []
    stats: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    writer.write("request_start", {"label": label, "session_id": session_id})
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            payload = json.loads(data)
            delta = _extract_delta_text(payload)
            if delta:
                chunks += 1
                text_parts.append(delta)
                if first_token_s is None:
                    first_token_s = time.perf_counter()
                    writer.write(
                        "request_first_token",
                        {"label": label, "ttft_s": first_token_s - started},
                    )
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            if isinstance(payload.get("mtplx_stats"), dict):
                stats = payload["mtplx_stats"]
    wall_s = time.perf_counter() - started
    text = "".join(text_parts)
    result = {
        "label": label,
        "ok": True,
        "wall_s": wall_s,
        "ttft_client_s": None if first_token_s is None else first_token_s - started,
        "chunks": chunks,
        "chars": len(text),
        "usage": usage,
        "mtplx_stats": stats,
        "text_tail": text[-1200:],
    }
    writer.write("request_finish", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18123")
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-id", default=f"followup-cache-{int(time.time())}")
    parser.add_argument("--context-lines", type=int, default=5200)
    parser.add_argument("--first-max-tokens", type=int, default=512)
    parser.add_argument("--followup-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--require-max-fans", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    writer = JsonlWriter(Path(args.output_jsonl).expanduser().resolve())
    writer.write("run_start", vars(args))
    fans_before = _fan_summary()
    writer.write(
        "fan_state_before",
        {"actual_max": _fans_are_actual_max(fans_before), "summary": fans_before},
    )
    if args.require_max_fans and not _fans_are_actual_max(fans_before):
        writer.write("run_failed", {"reason": "fans_not_actual_max_before_run"})
        return 2
    writer.write("health_before", _http_json("GET", args.base_url.rstrip("/") + "/health"))
    writer.write(
        "snapshot_before",
        _http_json("GET", args.base_url.rstrip("/") + "/v1/mtplx/snapshot"),
    )

    long_user = (
        "You are OpenCode inside a large TypeScript game project. Read the repo "
        "context and produce a concrete repair plan plus code-level guidance. "
        "Do not repeat yourself and do not stop after a preamble.\n\n"
        f"{_workspace_context(int(args.context_lines))}\n\n"
        "User report: camera direction, WASD movement, bow aiming, destructible "
        "props, terrain scale, animation, and particle effects are broken. Fix it."
    )
    first = _stream_chat(
        base_url=args.base_url,
        model=args.model,
        messages=[{"role": "user", "content": long_user}],
        max_tokens=int(args.first_max_tokens),
        temperature=float(args.temperature),
        session_id=str(args.session_id),
        writer=writer,
        label="seed_long_turn",
    )
    followup_messages = [
        {"role": "user", "content": long_user},
        {"role": "assistant", "content": str(first.get("text_tail") or "")},
        {"role": "user", "content": "Ok fix it."},
    ]
    followup = _stream_chat(
        base_url=args.base_url,
        model=args.model,
        messages=followup_messages,
        max_tokens=int(args.followup_max_tokens),
        temperature=float(args.temperature),
        session_id=str(args.session_id),
        writer=writer,
        label="followup_ok_fix_it",
    )
    writer.write(
        "snapshot_after",
        _http_json("GET", args.base_url.rstrip("/") + "/v1/mtplx/snapshot"),
    )
    fans_after = _fan_summary()
    writer.write(
        "fan_state_after",
        {"actual_max": _fans_are_actual_max(fans_after), "summary": fans_after},
    )

    first_stats = first.get("mtplx_stats") or {}
    followup_stats = followup.get("mtplx_stats") or {}
    summary = {
        "session_id": args.session_id,
        "seed": {
            "prompt_tokens": first_stats.get("prompt_tokens"),
            "completion_tokens": first_stats.get("completion_tokens"),
            "ttft_s": first_stats.get("ttft_s") or first.get("ttft_client_s"),
            "decode_tok_s": first_stats.get("decode_tok_s"),
            "cache_source": first_stats.get("cache_source"),
            "new_prefill_tokens": first_stats.get("new_prefill_tokens"),
            "cached_tokens": first_stats.get("cached_tokens"),
            "accepted_by_depth": first_stats.get("accepted_by_depth"),
            "verify_calls": first_stats.get("verify_calls"),
            "verify_ms_per_call": first_stats.get("verify_ms_per_call"),
            "openai_bridge_policy_version": first_stats.get("openai_bridge_policy_version"),
            "tool_contract_policy_version": first_stats.get("tool_contract_policy_version"),
        },
        "followup": {
            "prompt_tokens": followup_stats.get("prompt_tokens"),
            "completion_tokens": followup_stats.get("completion_tokens"),
            "ttft_s": followup_stats.get("ttft_s") or followup.get("ttft_client_s"),
            "decode_tok_s": followup_stats.get("decode_tok_s"),
            "cache_source": followup_stats.get("cache_source"),
            "session_cache_hit": followup_stats.get("session_cache_hit"),
            "session_restore_mode": followup_stats.get("session_restore_mode"),
            "cached_tokens": followup_stats.get("cached_tokens"),
            "new_prefill_tokens": followup_stats.get("new_prefill_tokens"),
            "cache_miss_reason": followup_stats.get("cache_miss_reason"),
            "ssd_cache_hit": followup_stats.get("ssd_cache_hit"),
            "ssd_cached_tokens": followup_stats.get("ssd_cached_tokens"),
            "ssd_restore_s": followup_stats.get("ssd_restore_s"),
            "accepted_by_depth": followup_stats.get("accepted_by_depth"),
            "verify_calls": followup_stats.get("verify_calls"),
            "verify_ms_per_call": followup_stats.get("verify_ms_per_call"),
            "openai_bridge_policy_version": followup_stats.get("openai_bridge_policy_version"),
            "tool_contract_policy_version": followup_stats.get("tool_contract_policy_version"),
        },
    }
    writer.write("run_summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
