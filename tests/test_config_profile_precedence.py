from __future__ import annotations

import argparse

from mtplx.config import apply_user_config
from mtplx.profiles import DEFAULT_PROFILE_NAME


def _args(*, command: str, flags: set[str], profile: str = DEFAULT_PROFILE_NAME):
    return argparse.Namespace(
        command=command,
        model=None,
        cache_dir=None,
        profile=profile,
        max="max" in flags,
        _cli_flags=flags,
    )


def test_quickstart_max_keeps_sustained_over_stale_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="quickstart", flags={"max"})

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_serve_max_keeps_sustained_over_stale_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="serve", flags={"max"})

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_start_max_keeps_sustained_over_stale_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="start", flags={"max"})

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_explicit_profile_still_beats_config_when_using_max(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="quickstart", flags={"profile", "max"}, profile="sustained")

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_config_profile_still_applies_without_max_flag(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="quickstart", flags=set())

    apply_user_config(args, config_path=config)

    assert args.profile == "performance-cold"
