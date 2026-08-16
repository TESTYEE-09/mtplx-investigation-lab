"""Canonical MTPLX fan-mode values shared by CLI and server code."""

from __future__ import annotations

from typing import Any


FAN_MODE_DEFAULT = "default"
FAN_MODE_SMART = "smart"
FAN_MODE_MAX = "max"
FAN_MODES = (FAN_MODE_DEFAULT, FAN_MODE_SMART, FAN_MODE_MAX)
FAN_MODE_CHOICES = FAN_MODES


def normalize_fan_mode(value: Any, *, default: str = FAN_MODE_DEFAULT) -> str:
    raw = str(value if value is not None else default).strip().lower()
    if raw in {"", "default", "auto", "apple", "apple-default", "system"}:
        return FAN_MODE_DEFAULT
    if raw in {"smart", "request", "request-scoped"}:
        return FAN_MODE_SMART
    if raw in {"max", "maximum", "performance", "sustained-max"}:
        return FAN_MODE_MAX
    raise ValueError("fan mode must be default, smart, or max")


def fan_mode_from_args(args: Any) -> str:
    mode = normalize_fan_mode(getattr(args, "fan_mode", FAN_MODE_DEFAULT))
    if bool(getattr(args, "max", False)):
        if mode not in {FAN_MODE_DEFAULT, FAN_MODE_MAX}:
            raise ValueError("--max cannot be combined with --fan-mode")
        return FAN_MODE_MAX
    return mode
