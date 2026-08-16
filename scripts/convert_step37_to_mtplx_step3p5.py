#!/usr/bin/env python3
"""Build an MTPLX-loadable Step-3.7-Flash text model from an mlx_vlm 4-bit export.

The public MLX Step-3.7-Flash quants (e.g. ``mlx-community/Step-3.7-Flash-4bit``)
are mlx_vlm artifacts: ``model_type: step3p7`` with every tensor under a
``language_model.`` prefix, and the 3 NextN/MTP layers dropped. mlx-lm (which
MTPLX uses) has a text-only ``step3p5`` model. This converter produces a flat
``step3p5`` directory that mlx-lm can load, then attaches the BF16 MTP overlay
(extracted NextN layers) as ``mtp.safetensors`` plus an ``mtplx_runtime.json``
contract so ``mtplx.runtime.load`` injects native MTP via ``step3p5_mtp_patch``.

Transforms (pure, unit-tested via --self-test):
  - strip the ``language_model.`` prefix from every weight + quantization key
  - flatten ``text_config`` to a top-level ``step3p5`` config (keep the length-48
    per-layer arrays so MTP layers 45-47 build correctly)

Streaming (needs the weights present + ~trunk-sized free disk):
  - rewrite each trunk shard with stripped keys + a fresh index.json
  - merge the overlay shards into ``mtp.safetensors`` (NextN layers kept verbatim)

Disk note: the trunk is ~100 GB; the output needs comparable free space. The
transforms are validated without the weights via ``--self-test``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

_LM_PREFIX = "language_model."


def strip_lm_prefix(key: str) -> str:
    """``language_model.model.x`` -> ``model.x``; ``language_model.lm_head`` -> ``lm_head``."""
    if key.startswith(_LM_PREFIX):
        return key[len(_LM_PREFIX):]
    return key


def fix_quant_config(quant: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in quant.items():
        out[strip_lm_prefix(key) if isinstance(key, str) else key] = value
    return out


def flatten_step_config(src: dict[str, Any]) -> dict[str, Any]:
    """Produce a flat mlx-lm ``step3p5`` config from a wrapped ``step3p7`` config."""
    tcfg = dict(src.get("text_config") or src)
    cfg = dict(tcfg)
    cfg["model_type"] = "step3p5"
    cfg.setdefault("architectures", ["Step3p5ForCausalLM"])
    # Carry quantization (with stripped keys) from whichever level holds it.
    quant = src.get("quantization") or src.get("quantization_config") or tcfg.get("quantization")
    if isinstance(quant, dict) and quant:
        cfg["quantization"] = fix_quant_config(quant)
        cfg["quantization_config"] = fix_quant_config(quant)
    # Drop any VLM-only keys that leaked into the text config.
    for vlm_key in ("vision_config", "image_token_id", "auto_map"):
        cfg.pop(vlm_key, None)
    return cfg


def build_runtime_contract(num_mtp_layers: int, mtplx_version: str = "0.1.0rc1") -> dict[str, Any]:
    return {
        "mtplx_version": mtplx_version,
        "arch_id": "step3p5-mtp",
        "mtp_depth_max": int(num_mtp_layers),
        "recommended_profile": "performance-cold",
        "exactness_baseline": {
            "gate": "pending-first-load-smoke",
            "max_abs_diff": None,
        },
        "verified_on": {
            "hardware": "pending",
            "timestamp": "pending",
        },
        "mtplx_mtp_hidden_variant": "pre_norm",
        "mtplx_mtp_quantization": {"prequantized": False, "policy": None, "bits": None},
    }


def _self_test() -> int:
    fails: list[str] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    ck("strip lm_head", strip_lm_prefix("language_model.lm_head.weight") == "lm_head.weight")
    ck("strip model layer", strip_lm_prefix("language_model.model.layers.0.mlp.down_proj.weight")
       == "model.layers.0.mlp.down_proj.weight")
    ck("non-lm key untouched", strip_lm_prefix("model.layers.45.enorm.weight") == "model.layers.45.enorm.weight")

    src = {
        "model_type": "step3p7",
        "architectures": ["Step3p7ForConditionalGeneration"],
        "vision_config": {"foo": 1},
        "auto_map": {"AutoModel": "x"},
        "text_config": {
            "model_type": "step3p5",
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 3,
            "hidden_size": 4096,
            "vocab_size": 128896,
            "layer_types": ["full_attention"] * 48,
            "partial_rotary_factors": [1.0] * 48,
            "rope_theta": [10000.0] * 48,
        },
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.3.mlp.gate.gate": {"group_size": 64, "bits": 8},
        },
    }
    flat = flatten_step_config(src)
    ck("flat model_type step3p5", flat["model_type"] == "step3p5")
    ck("flat drops vision_config", "vision_config" not in flat and "auto_map" not in flat)
    ck("flat keeps len-48 arrays", len(flat["layer_types"]) == 48 and len(flat["partial_rotary_factors"]) == 48)
    ck("flat quant keys stripped",
       "model.layers.3.mlp.gate.gate" in flat["quantization"]
       and not any(k.startswith("language_model.") for k in flat["quantization"]))
    contract = build_runtime_contract(3)
    ck("contract arch/depth/variant",
       contract["arch_id"] == "step3p5-mtp" and contract["mtp_depth_max"] == 3
       and contract["mtplx_mtp_hidden_variant"] == "pre_norm")

    print()
    if fails:
        print(f"SELF-TEST FAILED: {fails}")
        return 1
    print("CONVERTER SELF-TEST PASSED")
    return 0


def _load_index(model_dir: Path) -> dict[str, str]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return dict(json.loads(idx.read_text()).get("weight_map", {}))
    return {}


def convert(trunk_dir: Path, overlay_dir: Path, out_dir: Path) -> dict[str, Any]:
    import mlx.core as mx

    out_dir.mkdir(parents=True, exist_ok=False)
    src_cfg = json.loads((trunk_dir / "config.json").read_text())
    flat = flatten_step_config(src_cfg)
    num_mtp = int(flat.get("num_nextn_predict_layers") or 0)
    num_hidden = int(flat.get("num_hidden_layers") or 0)

    # --- 1. stream trunk shards with stripped keys ---
    weight_map: dict[str, str] = {}
    shards = sorted(trunk_dir.glob("model*.safetensors"))
    for i, shard in enumerate(shards):
        tensors = {strip_lm_prefix(k): v for k, v in mx.load(str(shard)).items()}
        # drop dropped MTP placeholders / vision if any slipped through
        tensors = {k: v for k, v in tensors.items() if not k.startswith("vision")}
        out_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out_dir / out_name), tensors)
        for k in tensors:
            weight_map[k] = out_name
        del tensors

    # --- 2. merge overlay shards into mtp.safetensors (NextN layers verbatim) ---
    overlay: dict[str, Any] = {}
    start = num_hidden
    wanted = tuple(
        tag
        for off in range(num_mtp)
        for tag in (f"model.layers.{start + off}.", f"language_model.model.layers.{start + off}.")
    )
    o_index = _load_index(overlay_dir)
    o_shards = (
        sorted({overlay_dir / rel for k, rel in o_index.items() if str(k).startswith(wanted)})
        if o_index
        else sorted(overlay_dir.glob("model*.safetensors"))
    )
    for shard in o_shards:
        for k, v in mx.load(str(shard)).items():
            ks = strip_lm_prefix(k)
            if ks.startswith(wanted) or ks.startswith("model.embed_tokens"):
                overlay[ks] = v
    if not any(".enorm." in k for k in overlay):
        raise RuntimeError(f"overlay has no NextN enorm tensors under layers {start}..{start+num_mtp-1}")
    mx.save_safetensors(str(out_dir / "mtp.safetensors"), overlay)

    # --- 3. config + index + contract + tokenizer ---
    flat["mlx_lm_extra_tensors"] = {"mtp_file": "mtp.safetensors", "mtp_tensor_count": len(overlay)}
    (out_dir / "config.json").write_text(json.dumps(flat, indent=2, sort_keys=True) + "\n")
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2) + "\n"
    )
    (out_dir / "mtplx_runtime.json").write_text(
        json.dumps(build_runtime_contract(num_mtp), indent=2) + "\n"
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        src = trunk_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    return {"out_dir": str(out_dir), "trunk_tensors": len(weight_map), "mtp_tensors": len(overlay)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="Validate the pure transforms (no weights needed)")
    p.add_argument("--trunk", help="mlx_vlm Step-3.7-Flash 4-bit dir (language_model.* keys)")
    p.add_argument("--overlay", help="BF16 MTP overlay dir (NextN layers 45-47 + embed)")
    p.add_argument("--out", help="Output MTPLX step3p5 model dir to create")
    args = p.parse_args()
    if args.self_test:
        return _self_test()
    if not (args.trunk and args.overlay and args.out):
        p.error("--trunk, --overlay, and --out are required unless --self-test")
    result = convert(Path(args.trunk).expanduser(), Path(args.overlay).expanduser(), Path(args.out).expanduser())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
