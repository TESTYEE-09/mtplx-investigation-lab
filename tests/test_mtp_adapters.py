import json
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

import scripts.collect_mtp_hidden_calib as collect_mtp_hidden_calib
import scripts.train_mtp_adapter_c4 as train_mtp_adapter_c4
from scripts.train_mtp_adapter_c4 import (
    _evaluate_rollout_d2_adapter,
    _live_cycle_gate_reasons,
    _load_live_cycle_row_weights,
    _render_report,
    _rollout_d2_logits_for_rows,
)
from mtplx.mtp_adapters import (
    LoRALinear,
    install_mtp_lora_adapters,
    install_saved_mtp_lora_adapter,
    iter_mtp_lora_modules,
    merge_installed_mtp_lora_adapters,
    save_mtp_lora_adapter,
    save_combined_mtp_lora_adapter,
    save_filtered_mtp_lora_adapter,
    set_mtp_adapter_depth,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.layers = [_Layer()]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mtp = _MTP()


class _FakeRolloutRuntime:
    def __init__(self):
        self.depth_tokens: list[tuple[int, int]] = []

    def make_mtp_cache(self):
        return {}

    def draft_mtp(
        self,
        _hidden,
        tokens,
        *,
        mtp_cache,
        return_hidden,
        mtp_hidden_variant,
        mtp_depth,
    ):
        del mtp_cache, mtp_hidden_variant
        token = int(np.asarray(tokens).reshape(-1)[0])
        self.depth_tokens.append((int(mtp_depth), token))
        logits_np = np.zeros((1, 1, 128), dtype=np.float32)
        if mtp_depth == 1:
            logits_np[:, :, 99] = 10.0
        logits = mx.array(logits_np)
        hidden = mx.ones((1, 1, 4), dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def test_collect_hidden_calibration_passes_mtp_quant_contract(monkeypatch, tmp_path):
    captured = {}

    def fake_load(_model_path, *, contract, **_kwargs):
        captured["contract"] = contract
        return SimpleNamespace(
            tokenizer=object(),
            contract=contract,
            mtp_adapter_metadata=None,
            mtp_adapter_merge_report=None,
        )

    monkeypatch.setattr(collect_mtp_hidden_calib, "load", fake_load)
    monkeypatch.setattr(collect_mtp_hidden_calib, "load_prompt_suite", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="collector produced no rows"):
        collect_mtp_hidden_calib.collect_hidden_calibration(
            tmp_path / "model",
            tmp_path / "prompts.jsonl",
            tmp_path / "out",
            mtp_quant_bits=4,
            mtp_quant_group_size=32,
            mtp_quant_mode="affine",
        )

    assert captured["contract"].mtp_quant_bits == 4
    assert captured["contract"].mtp_quant_group_size == 32
    assert captured["contract"].mtp_quant_mode == "affine"


def test_train_mtp_adapter_passes_mtp_quant_contract(monkeypatch, tmp_path):
    captured = {}
    calib_npz = tmp_path / "hidden_calib.npz"
    np.savez_compressed(
        calib_npz,
        recursive_input_hidden=np.zeros((2, 4), dtype=np.float32),
        recursive_input_tokens=np.array([1, 2], dtype=np.int64),
        target_next_hidden=np.zeros((2, 4), dtype=np.float32),
        depths=np.array([1, 1], dtype=np.int64),
        prompt_ids=np.array(["a", "b"]),
        window_indices=np.array([0, 0], dtype=np.int64),
        target_tokens=np.array([3, 4], dtype=np.int64),
        recursive_matches=np.array([True, False]),
        target_ar_p_recursive_draft=np.array([0.8, 0.1], dtype=np.float32),
        recursive_prefix_active=np.array([True, True]),
        recursive_top_indices=np.zeros((2, 16), dtype=np.int64),
        target_ar_top_indices=np.zeros((2, 16), dtype=np.int64),
        target_ar_top_probs=np.ones((2, 16), dtype=np.float32) / 16.0,
    )

    def fake_load(_model_path, *, contract, **_kwargs):
        captured["contract"] = contract
        return SimpleNamespace(model=object(), contract=contract)

    def stop_after_load(*_args, **_kwargs):
        raise RuntimeError("stop after load")

    monkeypatch.setattr(train_mtp_adapter_c4, "load", fake_load)
    monkeypatch.setattr(
        train_mtp_adapter_c4,
        "deterministic_train_mask",
        lambda *_args, **_kwargs: np.array([True, False]),
    )
    monkeypatch.setattr(train_mtp_adapter_c4, "ensure_nonempty_split", lambda mask, *_args: mask)
    monkeypatch.setattr(train_mtp_adapter_c4, "_model_identity_hash", lambda _path: "hash")
    monkeypatch.setattr(train_mtp_adapter_c4, "install_mtp_lora_adapters", stop_after_load)

    with pytest.raises(RuntimeError, match="stop after load"):
        train_mtp_adapter_c4.train_mtp_adapter_c4(
            calib_npz,
            model_path=tmp_path / "model",
            depths=[1],
            targets=["fc"],
            steps=1,
            mtp_quant_bits=4,
            mtp_quant_group_size=32,
            mtp_quant_mode="affine",
        )

    assert captured["contract"].mtp_quant_bits == 4
    assert captured["contract"].mtp_quant_group_size == 32
    assert captured["contract"].mtp_quant_mode == "affine"


def test_install_lora_adapter_preserves_zero_delta_output():
    model = _Model()
    x = mx.ones((1, 1, 4), dtype=mx.float32)
    before = model.mtp.fc(x)

    installed = install_mtp_lora_adapters(model, rank=2, targets=["fc"])
    after = model.mtp.fc(x)

    assert installed == ["fc"]
    assert isinstance(model.mtp.fc, LoRALinear)
    np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=1e-6, atol=1e-6)


def test_quantized_linear_lora_uses_dense_input_width():
    linear = nn.Linear(64, 24, bias=False)
    qlinear = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    adapter = LoRALinear(qlinear, rank=2)
    x = mx.ones((1, 3, 64), dtype=mx.float32)
    before = qlinear(x)
    after = adapter(x)

    assert adapter.lora_a.shape == (2, 64)
    assert adapter.lora_b.shape == (24, 2)
    np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=1e-6, atol=1e-6)

    adapter.lora_b = mx.ones(adapter.lora_b.shape, dtype=mx.float32) * 0.25
    changed = adapter(x)
    assert changed.shape == before.shape
    assert not np.allclose(np.asarray(changed), np.asarray(before))


def test_save_and_load_lora_adapter_round_trips_sidecar(tmp_path):
    model = _Model()
    install_mtp_lora_adapters(model, rank=2, alpha=4.0, targets=["fc", "layers.0.self_attn.q_proj"])
    for _target, module in iter_mtp_lora_modules(model):
        module.lora_b = mx.ones(module.lora_b.shape, dtype=mx.float32) * 0.25
    path = save_mtp_lora_adapter(tmp_path / "adapter.npz", model, metadata={"model_hash": "test"})

    loaded = _Model()
    metadata = install_saved_mtp_lora_adapter(loaded, path)

    assert metadata["kind"] == "c4_mtp_lora_adapter"
    assert metadata["model_hash"] == "test"
    assert isinstance(loaded.mtp.fc, LoRALinear)
    assert isinstance(loaded.mtp.layers[0].self_attn.q_proj, LoRALinear)

    for (target_a, module_a), (target_b, module_b) in zip(
        iter_mtp_lora_modules(model),
        iter_mtp_lora_modules(loaded),
    ):
        assert target_a == target_b
        np.testing.assert_allclose(np.asarray(module_b.lora_a), np.asarray(module_a.lora_a))
        np.testing.assert_allclose(np.asarray(module_b.lora_b), np.asarray(module_a.lora_b))


def test_combine_lora_adapters_keeps_non_overlapping_targets(tmp_path):
    first = _Model()
    second = _Model()
    install_mtp_lora_adapters(first, rank=1, alpha=2.0, targets=["fc"])
    install_mtp_lora_adapters(second, rank=2, alpha=4.0, targets=["layers.0.self_attn.q_proj"])
    first.mtp.fc.lora_b = mx.ones(first.mtp.fc.lora_b.shape, dtype=mx.float32) * 0.5
    second.mtp.layers[0].self_attn.q_proj.lora_b = (
        mx.ones(second.mtp.layers[0].self_attn.q_proj.lora_b.shape, dtype=mx.float32) * 0.25
    )
    first_path = save_mtp_lora_adapter(tmp_path / "first.npz", first, metadata={"model_hash": "same"})
    second_path = save_mtp_lora_adapter(tmp_path / "second.npz", second, metadata={"model_hash": "same"})

    combined_path = save_combined_mtp_lora_adapter(
        tmp_path / "combined.npz",
        [first_path, second_path],
        metadata={"run_id": "combined-test"},
    )
    loaded = _Model()
    metadata = install_saved_mtp_lora_adapter(loaded, combined_path)

    assert metadata["run_id"] == "combined-test"
    assert metadata["model_hash"] == "same"
    assert [entry["target"] for entry in metadata["targets"]] == [
        "fc",
        "layers.0.self_attn.q_proj",
    ]
    assert len(metadata["combined_from"]) == 2
    assert isinstance(loaded.mtp.fc, LoRALinear)
    assert isinstance(loaded.mtp.layers[0].self_attn.q_proj, LoRALinear)


def test_combine_lora_adapters_rejects_duplicate_targets(tmp_path):
    first = _Model()
    second = _Model()
    install_mtp_lora_adapters(first, rank=1, targets=["fc"])
    install_mtp_lora_adapters(second, rank=1, targets=["fc"])
    first_path = save_mtp_lora_adapter(tmp_path / "first.npz", first)
    second_path = save_mtp_lora_adapter(tmp_path / "second.npz", second)

    try:
        save_combined_mtp_lora_adapter(tmp_path / "combined.npz", [first_path, second_path])
    except ValueError as exc:
        assert "duplicate MTP adapter target: fc" in str(exc)
    else:
        raise AssertionError("duplicate adapter targets must be rejected")


def test_filter_lora_adapter_keeps_selected_targets(tmp_path):
    model = _Model()
    install_mtp_lora_adapters(model, rank=2, targets=["fc", "layers.0.self_attn.q_proj"])
    source_path = save_mtp_lora_adapter(tmp_path / "source.npz", model, metadata={"model_hash": "same"})

    filtered_path = save_filtered_mtp_lora_adapter(
        tmp_path / "filtered.npz",
        source_path,
        targets=["layers.0.self_attn.q_proj"],
        metadata={"run_id": "filtered-test"},
    )
    loaded = _Model()
    metadata = install_saved_mtp_lora_adapter(loaded, filtered_path)

    assert metadata["run_id"] == "filtered-test"
    assert metadata["model_hash"] == "same"
    assert [entry["target"] for entry in metadata["targets"]] == ["layers.0.self_attn.q_proj"]
    assert metadata["filtered_from"]["targets"] == ["layers.0.self_attn.q_proj"]
    assert not isinstance(loaded.mtp.fc, LoRALinear)
    assert isinstance(loaded.mtp.layers[0].self_attn.q_proj, LoRALinear)


def test_filter_lora_adapter_rejects_missing_target(tmp_path):
    model = _Model()
    install_mtp_lora_adapters(model, rank=1, targets=["fc"])
    source_path = save_mtp_lora_adapter(tmp_path / "source.npz", model)

    try:
        save_filtered_mtp_lora_adapter(tmp_path / "filtered.npz", source_path, targets=["missing"])
    except ValueError as exc:
        assert "adapter targets not found: missing" in str(exc)
    else:
        raise AssertionError("missing adapter targets must be rejected")


def test_depth_gated_lora_can_leave_depth_one_unchanged():
    model = _Model()
    x = mx.ones((1, 1, 4), dtype=mx.float32)
    base = model.mtp.fc(x)
    install_mtp_lora_adapters(model, rank=2, targets=["fc"], depth_scales=[0.0, 1.0])
    model.mtp.fc.lora_b = mx.ones(model.mtp.fc.lora_b.shape, dtype=mx.float32)

    set_mtp_adapter_depth(model, 1)
    d1 = model.mtp.fc(x)
    set_mtp_adapter_depth(model, 2)
    d2 = model.mtp.fc(x)

    np.testing.assert_allclose(np.asarray(d1), np.asarray(base), rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(d2), np.asarray(base))


def test_merge_installed_lora_adapter_bakes_dense_output():
    model = _Model()
    model.mtp.fc.weight = mx.zeros((4, 4), dtype=mx.float32)
    x = mx.ones((1, 1, 4), dtype=mx.float32)
    install_mtp_lora_adapters(model, rank=1, alpha=1.0, targets=["fc"])
    model.mtp.fc.lora_a = mx.ones(model.mtp.fc.lora_a.shape, dtype=mx.float32)
    model.mtp.fc.lora_b = mx.ones(model.mtp.fc.lora_b.shape, dtype=mx.float32) * 0.25

    with_adapter = model.mtp.fc(x)
    report = merge_installed_mtp_lora_adapters(model)
    merged = model.mtp.fc(x)

    assert report["merged"] == 1
    assert report["targets"][0]["target"] == "fc"
    assert not isinstance(model.mtp.fc, LoRALinear)
    assert iter_mtp_lora_modules(model) == []
    np.testing.assert_allclose(np.asarray(merged), np.asarray(with_adapter), rtol=1e-6, atol=1e-6)


def test_live_cycle_weights_align_by_generated_token_offset(tmp_path):
    live_path = tmp_path / "live.json"
    live_path.write_text(
        json.dumps(
            {
                "depths": [
                    {
                        "rows": [
                            {
                                "prompt_id": "p1",
                                "tokens": [101, 201, 301, 302, 401],
                                "events": [
                                    {
                                        "primary": 101,
                                        "accepted_depths": 0,
                                        "rejected_at_depth": 1,
                                        "timing_s": {"repair_forward": 0.02},
                                        "drafts": [
                                            {"depth": 1, "accepted": False},
                                            {"depth": 2, "accepted": None},
                                        ],
                                    },
                                    {
                                        "primary": 301,
                                        "accepted_depths": 2,
                                        "bonus_token": 401,
                                        "drafts": [
                                            {"depth": 1, "accepted": True},
                                            {"depth": 2, "accepted": True},
                                        ],
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        )
    )
    weights, summary = _load_live_cycle_row_weights(
        [live_path],
        prompt_ids=np.asarray(["p1", "p1", "p1", "p1"]),
        window_indices=np.asarray([0, 0, 2, 2]),
        depths=np.asarray([1, 2, 1, 2]),
        loss_weight=1.0,
        accepted_boost=0.5,
        rejected_boost=2.0,
        unverified_boost=0.0,
        repair_s_boost=10.0,
        max_weight=6.0,
    )

    np.testing.assert_allclose(weights, np.asarray([3.2, 1.0, 1.5, 1.5], dtype=np.float32))
    assert summary["matched_rows"] == 3
    assert summary["weighted_rows"] == 3
    assert summary["accepted_by_depth"] == {"1": 1, "2": 1}
    assert summary["rejected_by_depth"] == {"1": 1}


def test_rollout_d2_evaluator_requires_persistent_replay():
    result = _evaluate_rollout_d2_adapter(
        None,
        np.asarray([0, 1]),
        input_hidden=np.zeros((2, 4), dtype=np.float32),
        input_tokens=np.zeros((2,), dtype=np.int64),
        target_tokens=np.zeros((2,), dtype=np.int64),
        target_top_indices=np.zeros((2, 2), dtype=np.int64),
        target_top_probs=np.ones((2, 2), dtype=np.float32),
        mtp_hidden_variant="pre_norm",
        prompt_ids=np.asarray(["p", "p"]),
        window_indices=np.asarray([0, 0]),
        depths=np.asarray([1, 2]),
        row_lookup=None,
        batch_size=1,
        sampler_temperature=0.6,
    )

    assert result == {
        "rows": 0,
        "requested_rows": 0,
        "dropped_rows": 0,
        "reason": "requires_persistent_replay",
    }


def test_rollout_d2_uses_recorded_sampled_d1_token():
    rt = _FakeRolloutRuntime()

    logits, rows = _rollout_d2_logits_for_rows(
        rt,
        np.asarray([1]),
        input_hidden=np.zeros((2, 4), dtype=np.float32),
        input_tokens=np.asarray([11, 42], dtype=np.int64),
        mtp_hidden_variant="pre_norm",
        prompt_ids=np.asarray(["p", "p"]),
        window_indices=np.asarray([0, 0], dtype=np.int64),
        depths=np.asarray([1, 2], dtype=np.int64),
        row_lookup={("p", 0, 1): 0, ("p", 0, 2): 1},
    )

    assert rows.tolist() == [1]
    assert logits is not None
    assert rt.depth_tokens == [(1, 11), (2, 42)]


def test_c4_report_includes_rollout_d2_gate_reasons():
    metrics = {
        "adapter_path": "outputs/adapters/example.npz",
        "model_path": "/models/step",
        "model_hash": "abc",
        "cache_policy": "persistent",
        "hidden_variant": "pre_norm",
        "train_depths": [2],
        "rank": 4,
        "alpha": 4.0,
        "targets_installed": ["layers.1.shared_head_head"],
        "depth_gate": "train-depths",
        "train_rows": 10,
        "heldout_rows": 4,
        "run_id": "example",
        "runtime_integration_gate": False,
        "runtime_integration_gate_reasons": {
            "one_step_heldout_improved": True,
            "rollout_d2_heldout_improved": False,
            "live_cycle_baseline_improved": None,
            "live_cycle_pretrain_preserved": None,
        },
        "evaluations": {
            "baseline": {
                "heldout": {
                    "top1": 0.2,
                    "top4": 0.4,
                    "top8": 0.5,
                    "mean_greedy_acceptance_lb": 0.2,
                    "greedy_target_topk_hit_rate": 0.4,
                    "by_depth": {
                        "2": {
                            "rows": 4,
                            "top1": 0.2,
                            "top4": 0.4,
                            "top8": 0.5,
                            "mean_greedy_acceptance_lb": 0.2,
                            "greedy_target_topk_hit_rate": 0.4,
                        }
                    },
                }
            },
            "c4_adapter": {
                "heldout": {
                    "top1": 0.6,
                    "top4": 0.7,
                    "top8": 0.8,
                    "mean_greedy_acceptance_lb": 0.6,
                    "greedy_target_topk_hit_rate": 0.7,
                    "mean_target_top_q_mass": 0.8,
                    "mean_target_top_overlap_mass": 0.5,
                    "mean_target_top_excess_q_mass": 0.3,
                }
            },
            "c4_adapter_rollout_d2": {
                "heldout": {
                    "rows": 4,
                    "top1": 0.1,
                    "top4": 0.3,
                    "top8": 0.4,
                    "mean_greedy_acceptance_lb": 0.1,
                    "greedy_target_topk_hit_rate": 0.3,
                }
            },
        },
    }

    report = _render_report(metrics)

    assert "## Adapter-Conditioned D2 Rollout" in report
    assert "| C4 rollout D2 | 4 | 0.1000 | 0.3000 | 0.4000 | 0.1000 | 0.3000 |" in report
    assert "- one-step heldout improved: True" in report
    assert "- rollout D2 heldout improved: False" in report
    assert "- live-cycle baseline improved: None" in report
    assert "- live-cycle pretrain preserved: None" in report


def test_live_cycle_gate_rejects_pretrain_regression():
    reasons = _live_cycle_gate_reasons(
        {
            "baseline": {"mean_greedy_acceptance_lb": 0.17},
            "pretrain_adapter": {"mean_greedy_acceptance_lb": 0.59},
            "c4_adapter": {"mean_greedy_acceptance_lb": 0.48},
        }
    )

    assert reasons == {
        "live_cycle_baseline_improved": True,
        "live_cycle_pretrain_preserved": False,
    }


def test_c4_report_includes_live_cycle_pretrain_row():
    metrics = {
        "adapter_path": "outputs/adapters/example.npz",
        "model_path": "/models/step",
        "model_hash": "abc",
        "cache_policy": "persistent",
        "hidden_variant": "pre_norm",
        "train_depths": [2],
        "rank": 4,
        "alpha": 4.0,
        "targets_installed": ["layers.1.shared_head_head"],
        "depth_gate": "all",
        "train_rows": 10,
        "heldout_rows": 4,
        "run_id": "example",
        "runtime_integration_gate": False,
        "runtime_integration_gate_reasons": {
            "one_step_heldout_improved": True,
            "rollout_d2_heldout_improved": True,
            "live_cycle_baseline_improved": True,
            "live_cycle_pretrain_preserved": False,
        },
        "live_cycle": {
            "enabled": True,
            "matched_rows": 12,
            "weighted_rows": 9,
            "mean_weight": 1.5,
            "max_observed_weight": 3.0,
        },
        "evaluations": {
            "baseline": {
                "heldout": {
                    "top1": 0.2,
                    "top4": 0.4,
                    "top8": 0.5,
                    "mean_greedy_acceptance_lb": 0.2,
                    "greedy_target_topk_hit_rate": 0.4,
                    "by_depth": {},
                }
            },
            "c4_adapter": {
                "heldout": {
                    "top1": 0.6,
                    "top4": 0.7,
                    "top8": 0.8,
                    "mean_greedy_acceptance_lb": 0.6,
                    "greedy_target_topk_hit_rate": 0.7,
                    "mean_target_top_q_mass": 0.8,
                    "mean_target_top_overlap_mass": 0.5,
                    "mean_target_top_excess_q_mass": 0.3,
                }
            },
            "live_cycle": {
                "baseline": {
                    "top1": 0.1,
                    "top4": 0.2,
                    "top8": 0.3,
                    "mean_greedy_acceptance_lb": 0.17,
                    "greedy_target_topk_hit_rate": 0.3,
                },
                "pretrain_adapter": {
                    "top1": 0.6,
                    "top4": 0.8,
                    "top8": 0.83,
                    "mean_greedy_acceptance_lb": 0.59,
                    "greedy_target_topk_hit_rate": 0.86,
                },
                "c4_adapter": {
                    "top1": 0.5,
                    "top4": 0.72,
                    "top8": 0.79,
                    "mean_greedy_acceptance_lb": 0.48,
                    "greedy_target_topk_hit_rate": 0.80,
                },
            },
        },
    }

    report = _render_report(metrics)

    assert "| pretrain adapter | 0.6000 | 0.8000 | 0.8300 | 0.5900 | 0.8600 |" in report
    assert "- live-cycle pretrain preserved: False" in report
