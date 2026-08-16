from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mtplx.mtp_activation_stats import (
    ActivationStatsLinear,
    activation_stats_to_npz_payload,
    discover_mtp_linear_targets,
    install_mtp_activation_recorders,
)


class TinyMTP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=False)


class TinyWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False)

    def __call__(self, x):
        return self.base(x)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mtp = TinyMTP()


def test_mtp_activation_recorders_wrap_targets_and_collect_input_stats():
    model = TinyModel()

    assert discover_mtp_linear_targets(model) == ["fc"]
    recorders = install_mtp_activation_recorders(model, targets=["fc"])

    assert len(recorders) == 1
    assert isinstance(model.mtp.fc, ActivationStatsLinear)
    _ = model.mtp.fc(mx.array([[[1.0, -2.0, 4.0], [3.0, 0.0, -1.0]]]))

    stats = recorders[0].stats()
    assert stats.calls == 1
    assert stats.rows == 2
    assert np.allclose(stats.mean_abs, [2.0, 1.0, 2.5])
    assert np.allclose(stats.max_abs, [3.0, 2.0, 4.0])
    summary = stats.summary(top_n=2, group_size=2)
    assert [item["index"] for item in summary["top_channels"]] == [2, 0]

    payload = activation_stats_to_npz_payload([stats])
    assert payload["target_names"].tolist() == ["fc"]
    assert np.allclose(payload["target_0.mean_abs"], [2.0, 1.0, 2.5])


def test_mtp_activation_recorders_use_stable_name_for_wrapped_linears():
    model = TinyModel()
    model.mtp.fc = TinyWrapper()

    assert discover_mtp_linear_targets(model) == ["fc"]
    recorders = install_mtp_activation_recorders(model, targets=["fc"])

    assert isinstance(model.mtp.fc.base, ActivationStatsLinear)
    _ = model.mtp.fc(mx.array([[[1.0, 2.0, 3.0]]]))
    assert recorders[0].target == "fc"
    assert recorders[0].stats().rows == 1
