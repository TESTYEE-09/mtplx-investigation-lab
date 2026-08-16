#!/usr/bin/env python3
"""Deterministic validation of the Step-3.7-Flash MTP integration.

Runs WITHOUT the ~100 GB trunk. It exercises every code path that must be
correct for acceptance to work, using:
  - the real downloaded Step MTP config (config detection + arg parsing),
  - a tiny synthetic Step config (build the MTP module, remap real-named
    checkpoint keys onto it, assert exact load coverage, assert the +1.0
    zero-centered norm shift),
  - an on-disk safetensors fixture (registry recognition via inspect_model).

Exits non-zero on any failure so it can gate CI / the build.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mtplx.step3p5_mtp_patch import (  # noqa: E402
    _apply_zero_centered_norm_shift,
    _make_step_mtp_module,
    _rewrite_step_mtp_weights,
    _validate_mtp_load_coverage,
    is_step3p5_mtp_config,
)

FAILS: list[str] = []
def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# --- 1. Backend facade imports + health -------------------------------------
try:
    from mtplx.backends.step3p5_mtp import Step3p5MTPBackend

    health = Step3p5MTPBackend().health()
    check(
        "backend.health",
        health.get("contract_required") is True
        and health.get("hidden_variant") == "pre_norm"
        and "step3p5" in health.get("supported_model_types", []),
        json.dumps({k: health[k] for k in ("arch_id", "hidden_variant", "verify_strategy")}),
    )
except Exception as exc:  # pragma: no cover
    check("backend.health", False, repr(exc))


# --- 2. Config detection against the real Step config -----------------------
real_cfg_path = REPO / "models/Step-3.7-Flash-MTP-draft-src/config.json"
if real_cfg_path.exists():
    real_cfg = json.loads(real_cfg_path.read_text())
    check(
        "is_step3p5_mtp_config(real Step-3.7-Flash config)",
        is_step3p5_mtp_config(real_cfg),
        f"model_type={real_cfg.get('model_type')} "
        f"num_nextn={real_cfg.get('text_config', {}).get('num_nextn_predict_layers')}",
    )
else:
    # Fall back to a representative config matching the verified HF metadata.
    synthetic_top = {
        "model_type": "step3p7",
        "architectures": ["Step3p7ForConditionalGeneration"],
        "text_config": {"model_type": "step3p5", "num_nextn_predict_layers": 3, "num_hidden_layers": 45},
    }
    check("is_step3p5_mtp_config(representative config)", is_step3p5_mtp_config(synthetic_top))

check("is_step3p5_mtp_config rejects non-Step", not is_step3p5_mtp_config({"model_type": "qwen3_5", "num_nextn_predict_layers": 1}))
check("is_step3p5_mtp_config rejects no-MTP Step", not is_step3p5_mtp_config({"model_type": "step3p5", "num_nextn_predict_layers": 0}))


# --- 3. Build the MTP module + exact remap/coverage (tiny synthetic config) ---
from mlx_lm.models.step3p5 import ModelArgs  # noqa: E402

H = 128
START = 4
NMTP = 3
TOTAL = START + NMTP  # per-layer arrays must cover the MTP indices
tiny = {
    "model_type": "step3p5",
    "hidden_size": H,
    "num_hidden_layers": START,
    "vocab_size": 512,
    "num_attention_heads": 4,
    "num_attention_groups": 2,
    "head_dim": 32,
    "intermediate_size": 256,
    "rms_norm_eps": 1e-5,
    "rope_theta": [10000.0] * TOTAL,
    "max_position_embeddings": 4096,
    "sliding_window": 64,
    "layer_types": ["full_attention"] * TOTAL,
    "partial_rotary_factors": [1.0] * TOTAL,
    "use_head_wise_attn_gate": True,
    "moe_num_experts": 8,
    "moe_top_k": 2,
    "moe_intermediate_size": 64,
    "share_expert_dim": 64,
    "num_nextn_predict_layers": NMTP,
}
args = ModelArgs.from_dict(tiny)

mtp = _make_step_mtp_module(args, NMTP, START)

from mlx.utils import tree_flatten  # noqa: E402

module_params = tree_flatten(mtp.parameters(), destination={})

# Reverse-map each module param to a real-named checkpoint key, so we test the
# forward remap is the exact inverse, with realistic naming + an outer prefix.
def to_checkpoint_key(module_key: str, local: int) -> str:
    spec = START + local
    base = f"model.layers.{spec}."
    rest = module_key.split(".", 2)[2]  # strip "layers.{local}."
    if rest.startswith("shared_head_norm."):
        return base + "transformer.shared_head.norm." + rest[len("shared_head_norm."):]
    if rest.startswith("shared_head_head."):
        return base + "transformer.shared_head.output." + rest[len("shared_head_head."):]
    if rest.startswith("mtp_block."):
        return base + rest[len("mtp_block."):]
    return base + rest  # enorm./hnorm./eh_proj.

raw: dict[str, mx.array] = {}
for mkey, val in module_params.items():
    local = int(mkey.split(".")[1])
    ckey = to_checkpoint_key(mkey, local)
    # Exercise the language_model. outer-prefix stripping on half the keys.
    if local == 1:
        ckey = "language_model." + ckey
    raw[ckey] = mx.zeros(val.shape)  # zeros => centered norms => exercises +1 shift
# add an embed_tokens key that must be dropped (shared with trunk)
raw["model.embed_tokens.weight"] = mx.zeros((args.vocab_size, H))

mapped = _rewrite_step_mtp_weights(raw, start_layer=START, num_mtp_layers=NMTP)
check("remap drops shared embed_tokens", not any("embed_tokens" in k for k in mapped))
check(
    "remap renames shared_head.output -> shared_head_head",
    any(k.endswith("shared_head_head.weight") for k in mapped)
    and not any("shared_head.output" in k for k in mapped),
)

# +1.0 zero-centered shift makes centered (zero) norm weights ~1.0.
pre_mean = float(next(v for k, v in mapped.items() if k.endswith("enorm.weight")).mean().item())
mapped = _apply_zero_centered_norm_shift(mapped)
post_mean = float(next(v for k, v in mapped.items() if k.endswith("enorm.weight")).mean().item())
check("zero-centered norm +1.0 shift applied", pre_mean < 0.5 <= post_mean, f"{pre_mean} -> {post_mean}")
# non-norm weights (eh_proj) must NOT be shifted.
ehp = float(next(v for k, v in mapped.items() if k.endswith("eh_proj.weight")).mean().item())
check("eh_proj not shifted", abs(ehp) < 1e-6, f"mean={ehp}")

# Exact module/weight coverage + strict load.
try:
    _validate_mtp_load_coverage(mtp, mapped)
    mtp.load_weights(list(mapped.items()), strict=True)
    mx.eval(mtp.parameters())
    check("MTP module strict load coverage (no missing/extra/mismatch)", True, f"{len(mapped)} tensors")
except Exception as exc:
    check("MTP module strict load coverage (no missing/extra/mismatch)", False, repr(exc))


# --- 4. Registry recognition via on-disk fixture (inspect_model) -------------
try:
    from mtplx.artifacts import inspect_model

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "config.json").write_text(json.dumps({
            "model_type": "step3p7",
            "architectures": ["Step3p7ForConditionalGeneration"],
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 3,
            "hidden_size": 4096,
            "vocab_size": 128896,
        }))
        mx.save_safetensors(str(d / "model.safetensors"), {
            "model.layers.45.enorm.weight": mx.zeros((8,)),
            "model.layers.46.enorm.weight": mx.zeros((8,)),
            "model.layers.47.enorm.weight": mx.zeros((8,)),
            "model.layers.0.input_layernorm.weight": mx.zeros((8,)),
        })
        insp = inspect_model(str(d)).to_dict()
        comp = insp.get("compatibility", {})
        check(
            "registry recognizes step3p7 as runnable appended-layer MTP",
            comp.get("arch_id") == "step3p5-mtp"
            and comp.get("can_run") is True
            and comp.get("recommended_backend") == "step3p5_mtp",
            f"tier={comp.get('tier')} arch={comp.get('arch_id')} "
            f"can_run={comp.get('can_run')} backend={comp.get('recommended_backend')}",
        )
except Exception as exc:
    check("registry recognizes step3p7 as runnable appended-layer MTP", False, repr(exc))


print()
if FAILS:
    print(f"VALIDATION FAILED: {len(FAILS)} check(s): {FAILS}")
    sys.exit(1)
print("ALL STEP MTP INJECTOR VALIDATION CHECKS PASSED")
