"""Activation-stat collection for MTP-side quantization calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _mtp_root(model: Any) -> Any:
    text_model = _text_model(model)
    root = getattr(text_model, "mtp", None)
    if root is None:
        raise RuntimeError("model has no injected MTP module")
    return root


def _get_child(obj: Any, part: str) -> Any:
    if isinstance(obj, (list, tuple)):
        return obj[int(part)]
    return getattr(obj, part)


def _set_child(obj: Any, part: str, value: Any) -> None:
    if isinstance(obj, list):
        obj[int(part)] = value
    else:
        setattr(obj, part, value)


def _resolve_parent(root: Any, path: str) -> tuple[Any, str]:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("empty MTP activation target path")
    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    return parent, parts[-1]


def _get_target(root: Any, path: str) -> Any:
    parent, leaf = _resolve_parent(root, path)
    return _get_child(parent, leaf)


def _set_target(root: Any, path: str, value: Any) -> None:
    parent, leaf = _resolve_parent(root, path)
    _set_child(parent, leaf, value)


def _is_linear(module: Any) -> bool:
    return isinstance(module, (nn.Linear, nn.QuantizedLinear))


def _linear_base(module: Any) -> Any | None:
    base = getattr(module, "base", None)
    return base if _is_linear(base) else None


def _activation_stats_base(module: Any) -> Any | None:
    base = getattr(module, "base", None)
    return base if isinstance(base, ActivationStatsLinear) else None


def _safe_top_indices(values: np.ndarray, n: int) -> list[int]:
    if values.size == 0:
        return []
    count = min(max(int(n), 0), int(values.size))
    if count <= 0:
        return []
    return [int(index) for index in np.argsort(-values)[:count]]


def _share_of_top(values: np.ndarray, fraction: float) -> float | None:
    if values.size == 0:
        return None
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return None
    count = max(1, int(np.ceil(float(fraction) * int(values.size))))
    top = np.sort(values)[-count:]
    return float(top.sum() / total)


def _group_summary(values: np.ndarray, group_size: int) -> dict[str, Any]:
    if values.size == 0:
        return {"group_size": int(group_size), "groups": 0}
    group_size = max(1, int(group_size))
    pad = (-int(values.size)) % group_size
    padded = np.pad(values, (0, pad), constant_values=0.0) if pad else values
    groups = padded.reshape(-1, group_size)
    group_mean = groups.mean(axis=1)
    group_max = groups.max(axis=1)
    return {
        "group_size": group_size,
        "groups": int(groups.shape[0]),
        "mean_abs_group_mean_p50": float(np.percentile(group_mean, 50)),
        "mean_abs_group_mean_p95": float(np.percentile(group_mean, 95)),
        "mean_abs_group_max_p95": float(np.percentile(group_max, 95)),
        "mean_abs_group_max_global": float(group_max.max()),
    }


@dataclass
class ActivationStats:
    target: str
    calls: int
    rows: int
    sum_abs: np.ndarray
    sum_sq: np.ndarray
    max_abs: np.ndarray

    @property
    def mean_abs(self) -> np.ndarray:
        if self.rows <= 0:
            return np.zeros_like(self.sum_abs)
        return self.sum_abs / float(self.rows)

    @property
    def rms(self) -> np.ndarray:
        if self.rows <= 0:
            return np.zeros_like(self.sum_sq)
        return np.sqrt(np.maximum(self.sum_sq / float(self.rows), 0.0))

    def summary(self, *, top_n: int = 16, group_size: int = 64) -> dict[str, Any]:
        mean_abs = self.mean_abs
        rms = self.rms
        top = _safe_top_indices(mean_abs, top_n)
        return {
            "target": self.target,
            "calls": int(self.calls),
            "rows": int(self.rows),
            "input_dims": int(mean_abs.size),
            "mean_abs_avg": float(mean_abs.mean()) if mean_abs.size else 0.0,
            "mean_abs_p95": float(np.percentile(mean_abs, 95)) if mean_abs.size else 0.0,
            "mean_abs_max": float(mean_abs.max()) if mean_abs.size else 0.0,
            "rms_avg": float(rms.mean()) if rms.size else 0.0,
            "rms_p95": float(np.percentile(rms, 95)) if rms.size else 0.0,
            "max_abs_global": float(self.max_abs.max()) if self.max_abs.size else 0.0,
            "top_1pct_mean_abs_share": _share_of_top(mean_abs, 0.01),
            "top_5pct_mean_abs_share": _share_of_top(mean_abs, 0.05),
            "group_summary": _group_summary(mean_abs, group_size),
            "top_channels": [
                {
                    "index": int(index),
                    "mean_abs": float(mean_abs[index]),
                    "rms": float(rms[index]),
                    "max_abs": float(self.max_abs[index]),
                }
                for index in top
            ],
        }


class ActivationStatsLinear(nn.Module):
    """Wrap a Linear/QuantizedLinear and accumulate input-channel stats."""

    def __init__(self, base: nn.Module, *, target: str) -> None:
        super().__init__()
        if not _is_linear(base):
            raise TypeError(f"activation stats target is not linear: {type(base)!r}")
        self.base = base
        self.target = str(target)
        self.calls = 0
        self.rows = 0
        self._sum_abs: np.ndarray | None = None
        self._sum_sq: np.ndarray | None = None
        self._max_abs: np.ndarray | None = None

    def __call__(self, x: mx.array) -> mx.array:
        self.observe(x)
        return self.base(x)

    def observe(self, x: mx.array) -> None:
        rows = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        abs_rows = mx.abs(rows)
        sum_abs = mx.sum(abs_rows, axis=0)
        sum_sq = mx.sum(rows * rows, axis=0)
        max_abs = mx.max(abs_rows, axis=0)
        mx.eval(sum_abs, sum_sq, max_abs)
        sum_abs_np = np.asarray(sum_abs, dtype=np.float64)
        sum_sq_np = np.asarray(sum_sq, dtype=np.float64)
        max_abs_np = np.asarray(max_abs, dtype=np.float64)
        if self._sum_abs is None:
            self._sum_abs = np.zeros_like(sum_abs_np)
            self._sum_sq = np.zeros_like(sum_sq_np)
            self._max_abs = np.zeros_like(max_abs_np)
        self._sum_abs += sum_abs_np
        self._sum_sq += sum_sq_np
        self._max_abs = np.maximum(self._max_abs, max_abs_np)
        self.calls += 1
        self.rows += int(rows.shape[0])

    def stats(self) -> ActivationStats:
        if self._sum_abs is None or self._sum_sq is None or self._max_abs is None:
            weight = getattr(self.base, "weight", None)
            if weight is None or len(weight.shape) != 2:
                input_dims = 0
            elif isinstance(self.base, nn.QuantizedLinear):
                bits = int(getattr(self.base, "bits", 0) or 0)
                input_dims = 0 if bits <= 0 else (int(weight.shape[1]) * 32) // bits
            else:
                input_dims = int(weight.shape[1])
            zeros = np.zeros(input_dims, dtype=np.float64)
            return ActivationStats(self.target, self.calls, self.rows, zeros, zeros, zeros)
        return ActivationStats(
            self.target,
            self.calls,
            self.rows,
            self._sum_abs.copy(),
            self._sum_sq.copy(),
            self._max_abs.copy(),
        )


def discover_mtp_linear_targets(model: Any) -> list[str]:
    root = _mtp_root(model)
    targets: list[str] = []
    for path, module in root.named_modules():
        if not path or isinstance(module, ActivationStatsLinear):
            continue
        if path.endswith(".base"):
            continue
        if (
            _is_linear(module)
            or _linear_base(module) is not None
            or _activation_stats_base(module) is not None
        ):
            targets.append(str(path))
    return sorted(targets)


def install_mtp_activation_recorders(
    model: Any,
    *,
    targets: Iterable[str] | None = None,
    strict: bool = True,
) -> list[ActivationStatsLinear]:
    root = _mtp_root(model)
    selected = list(targets) if targets is not None else discover_mtp_linear_targets(model)
    if not selected:
        raise ValueError("no MTP activation targets selected")
    recorders: list[ActivationStatsLinear] = []
    for target in selected:
        try:
            module = _get_target(root, target)
        except (AttributeError, IndexError, ValueError) as exc:
            if strict:
                raise ValueError(f"MTP activation target not found: {target}") from exc
            continue
        if isinstance(module, ActivationStatsLinear):
            recorders.append(module)
            continue
        installed_base = _activation_stats_base(module)
        if installed_base is not None:
            recorders.append(installed_base)
            continue
        base = _linear_base(module)
        if base is not None:
            recorder = ActivationStatsLinear(base, target=target)
            module.base = recorder
            recorders.append(recorder)
            continue
        if not _is_linear(module):
            if strict:
                raise TypeError(f"MTP activation target is not linear: {target}")
            continue
        recorder = ActivationStatsLinear(module, target=target)
        _set_target(root, target, recorder)
        recorders.append(recorder)
    return recorders


def activation_stats_to_npz_payload(
    stats: Iterable[ActivationStats],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    target_names: list[str] = []
    for index, item in enumerate(stats):
        target_names.append(item.target)
        prefix = f"target_{index}"
        payload[f"{prefix}.mean_abs"] = item.mean_abs.astype(np.float32)
        payload[f"{prefix}.rms"] = item.rms.astype(np.float32)
        payload[f"{prefix}.max_abs"] = item.max_abs.astype(np.float32)
    payload["target_names"] = np.asarray(target_names)
    return payload
