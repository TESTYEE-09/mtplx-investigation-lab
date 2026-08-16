from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase0h_paged_verifier_exactness.py"
SPEC = importlib.util.spec_from_file_location("phase0h_paged_verifier_exactness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_distribution_metrics_identical_logits_pass_shape() -> None:
    logits = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    metrics = mod._distribution_metrics(
        logits,
        logits.copy(),
        mod.SamplerConfig(temperature=0.6, top_p=0.95, top_k=3),
        top_k_compare=3,
        sample_seed=1,
        sample_draws=32,
    )

    assert metrics["logits"]["max_abs_diff"] == 0.0
    assert metrics["logits"]["argmax_match"] is True
    assert metrics["topk"]["overlap_ratio"] == 1.0
    assert metrics["distribution"]["support_equal"] is True
    assert metrics["distribution"]["total_variation"] == 0.0
    assert metrics["controlled_rng_sample"]["agreement"] == 1.0


def test_distribution_metrics_detects_support_divergence() -> None:
    stock = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    paged = np.array([3.0, 0.0, 1.0, 2.0], dtype=np.float32)
    metrics = mod._distribution_metrics(
        stock,
        paged,
        mod.SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
        top_k_compare=2,
        sample_seed=1,
        sample_draws=32,
    )

    assert metrics["topk"]["overlap_ratio"] < 1.0
    assert metrics["distribution"]["support_equal"] is False
    assert metrics["distribution"]["total_variation"] > 0.0
