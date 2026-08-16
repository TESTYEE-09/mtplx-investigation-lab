import argparse
import importlib.util
import json
from pathlib import Path


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aime_serve_gate.py"
    spec = importlib.util.spec_from_file_location("aime_serve_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "model": "mtplx-test",
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "seed": 42,
        "enable_thinking": True,
        "prompt_mode": "brief",
        "answer_contract": "tool",
        "stream": False,
        "mode": "sequential",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_tool_answer_contract_payload_omits_token_cap():
    gate = _load_gate_module()

    payload = gate._payload(
        _args(),
        "row",
        "What is 40+2?",
        client_request_id="aime-row",
    )

    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert payload["metadata"]["sampler_source"] == "server_defaults"
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_answer"},
    }
    assert payload["tools"][0]["function"]["name"] == "submit_answer"
    assert payload["metadata"]["answer_contract"] == "tool"
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "immediately call `submit_answer`" in payload["messages"][0]["content"]
    assert payload["messages"][1]["role"] == "user"


def test_payload_includes_only_explicit_sampler_overrides():
    gate = _load_gate_module()

    payload = gate._payload(
        _args(temperature=0.6, top_p=0.95, top_k=20),
        "row",
        "What is 40+2?",
        client_request_id="aime-row",
    )

    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["metadata"]["sampler_source"] == "explicit_payload"


def test_run_one_scores_submit_answer_tool_call(monkeypatch):
    gate = _load_gate_module()

    def fake_post_json(_url, payload, **_kwargs):
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "submit_answer",
                                    "arguments": json.dumps({"answer": "42"}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "mtplx_stats": {"uncapped_response_requested": True},
        }

    monkeypatch.setattr(gate, "_post_json", fake_post_json)

    row = gate._run_one(
        (
            {
                "base_url": "http://127.0.0.1:18183",
                "api_key": None,
                "timeout_s": 5,
                "request_args": vars(_args()),
            },
            "aime_shape_02",
            "Let a and b be positive integers with ab=432.",
            "42",
        )
    )

    assert row["correct"] is True
    assert row["payload_sampler_source"] == "server_defaults"
    assert row["payload_has_temperature"] is False
    assert row["payload_has_top_p"] is False
    assert row["payload_has_top_k"] is False
    assert row["answer_source"] == "tool"
    assert row["tool_answer"] == "42"
    assert row["finish_reason"] == "tool_calls"
