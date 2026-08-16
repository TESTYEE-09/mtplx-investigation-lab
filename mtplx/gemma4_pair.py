"""Gemma 4 assistant-pair bundle helpers.

The public Gemma artifacts are not single MLX-LM folders.  They are bundle
roots with a target verifier under ``target/`` and an external assistant
drafter under ``assistant/``.  These helpers keep that layout explicit so the
normal Qwen native-MTP path does not learn any Gemma-specific assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mtplx.backends.descriptors import GEMMA4_ASSISTANT_DESCRIPTOR


GEMMA4_PAIR_REPO_IDS = {
    "Youssofal/Gemma4-MTPLX-Optimized-Speed",
    "Youssofal/Gemma4-MTPLX-Optimized-Quality",
}
GEMMA4_PAIR_FILE = "mtplx_pair.json"
GEMMA4_BACKEND = GEMMA4_ASSISTANT_DESCRIPTOR.backend_id
GEMMA4_ARCH_ID = GEMMA4_ASSISTANT_DESCRIPTOR.architecture_id
GEMMA4_DEFAULT_SAMPLER = GEMMA4_ASSISTANT_DESCRIPTOR.sampler_defaults.to_dict()
GEMMA4_RUNTIME_STATUS = GEMMA4_ASSISTANT_DESCRIPTOR.status


def is_gemma4_pair_repo_id(value: str | None) -> bool:
    normalized = str(value or "").strip().strip("/")
    return normalized in GEMMA4_PAIR_REPO_IDS


def load_gemma4_pair_metadata(bundle_root: str | Path) -> dict[str, Any] | None:
    path = Path(bundle_root).expanduser() / GEMMA4_PAIR_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_gemma4_pair_paths(bundle_root: str | Path) -> dict[str, Any] | None:
    root = Path(bundle_root).expanduser()
    metadata = load_gemma4_pair_metadata(root)
    if metadata is None:
        return None
    layout = metadata.get("layout") if isinstance(metadata.get("layout"), dict) else {}
    target_name = str(layout.get("target") or "target")
    assistant_name = str(layout.get("assistant") or "assistant")
    target = root / target_name
    assistant = root / assistant_name
    if not (target / "config.json").is_file() or not (assistant / "config.json").is_file():
        return None
    return {
        "bundle_root": str(root),
        "target_model": str(target),
        "assistant_model": str(assistant),
        "metadata": metadata,
    }


def _sampler_from_mapping(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    try:
        temperature = float(data["temperature"])
        top_p = float(data["top_p"])
        top_k = int(data["top_k"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def _sampler_from_generation_config(model_path: str | Path) -> dict[str, Any] | None:
    config_path = Path(model_path).expanduser() / "generation_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _sampler_from_mapping(config)


def _load_local_config(model_path: str | Path) -> dict[str, Any] | None:
    path = Path(model_path).expanduser() / "config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _looks_like_real_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict) or not config:
        return False
    return any(key in config for key in ("model_type", "architectures", "text_config"))


def _unsupported_variant_message(
    *,
    target_model: str | Path,
    assistant_model: str | Path,
) -> str | None:
    target_config = _load_local_config(target_model)
    assistant_config = _load_local_config(assistant_model)
    if not (_looks_like_real_config(target_config) or _looks_like_real_config(assistant_config)):
        return None
    try:
        from mtplx.backends.gemma4_assistant import (
            Gemma4AssistantUnsupported,
            validate_gemma4_31b_pair_configs,
        )

        validate_gemma4_31b_pair_configs(
            target_config or {},
            assistant_config or {},
        )
    except Gemma4AssistantUnsupported as exc:
        return str(exc)
    except Exception:
        return None
    return None


def gemma4_pair_sampler_defaults(
    *,
    target_model: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return Gemma's target sampler defaults without borrowing Qwen settings."""

    return (
        _sampler_from_generation_config(target_model)
        or _sampler_from_mapping(
            metadata.get("benchmark") if isinstance(metadata, dict) else None
        )
        or dict(GEMMA4_DEFAULT_SAMPLER)
    )


def gemma4_pair_inspection(
    *,
    model_ref: str,
    bundle_root: str | Path,
    target_model: str | Path,
    assistant_model: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    target_meta = metadata.get("target") if isinstance(metadata.get("target"), dict) else {}
    assistant_meta = (
        metadata.get("assistant") if isinstance(metadata.get("assistant"), dict) else {}
    )
    benchmark = (
        metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
    )
    sampler = gemma4_pair_sampler_defaults(
        target_model=target_model,
        metadata=metadata,
    )
    unsupported_variant = _unsupported_variant_message(
        target_model=target_model,
        assistant_model=assistant_model,
    )
    can_run = unsupported_variant is None
    return {
        "source": model_ref,
        "model_dir": str(bundle_root),
        "runtime_model": str(target_model),
        "assistant_model": str(assistant_model),
        "architecture": "Gemma4AssistantPair",
        "model_type": "gemma4_pair",
        "mtp_arch": GEMMA4_ARCH_ID,
        "mtp_supported": True,
        "recommended_backend": GEMMA4_BACKEND,
        "recommended_profile": "sustained",
        "recommended_sampler": sampler,
        "runtime_compatibility": "assistant-pair-native",
        "backend_status": GEMMA4_RUNTIME_STATUS,
        "gemma4_pair": {
            "bundle_root": str(bundle_root),
            "target_model": str(target_model),
            "assistant_model": str(assistant_model),
            "variant": metadata.get("variant"),
            "target_quantization": target_meta.get("quantization"),
            "assistant_quantization": assistant_meta.get("quantization"),
            "sampler": sampler,
            "benchmark": benchmark,
        },
        "compatibility": {
            "tier": "family-compatible-unverified",
            "can_run": can_run,
            "supported": can_run,
            "recognized": True,
            "exit_code": 0,
            "message": (
                "Gemma 4 assistant-pair bundle is runnable through the Gemma backend, "
                "but full public QA is still pending for this artifact."
            )
            if can_run
            else unsupported_variant,
            "arch_id": GEMMA4_ARCH_ID,
            "recommended_backend": GEMMA4_BACKEND,
            "recommended_profile": "sustained",
            "mtp_supported": "yes",
            "runtime_compatibility": "assistant-pair-native"
            if can_run
            else "unsupported_model_variant",
            "support_level": GEMMA4_RUNTIME_STATUS if can_run else "unsupported_model_variant",
            "support_notes": (
                "External assistant drafter; target and assistant live in bundle "
                "subdirectories and are loaded together. Promotion to verified requires "
                "the 160-token gate, full user-surface QA, and Qwen regression proof."
            )
            if can_run
            else "Unsupported Gemma variant for this integration phase.",
            "unverified_model": True,
            "unsupported_model_variant": None if can_run else unsupported_variant,
        },
    }
