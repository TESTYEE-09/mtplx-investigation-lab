"""StepFun Step-3.5 / Step-3.7-Flash native MTP backend facade.

Like the DeepSeek and GLM facades, the speculative sampler is shared through
``mtplx.generation``; this facade keeps the architecture registry honest: Step
now has a concrete runtime loader (``mtplx.step3p5_mtp_patch``), but it remains
verified-contract gated until real checkpoints pass the same exactness and
acceptance gates as Qwen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from mtplx.profiles import DEFAULT_PROFILE_NAME


class Step3p5MTPBackend(MTPBackend):
    arch_id = "step3p5-mtp"

    def load(self, model_path: Path) -> ModelState:
        from mtplx.mtp_patch import MTPContract
        from mtplx.runtime import load

        runtime = load(model_path, mtp=True, contract=MTPContract())
        return ModelState(
            model_path=Path(model_path),
            runtime=runtime,
            metadata={"arch_id": self.arch_id, "contract_gated": True},
        )

    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError("Step3p5MTPBackend.verify is wired through generation.py")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("Step3p5MTPBackend.propose is wired through generation.py")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.step3p5_mtp_patch + mtplx.generation",
            "support_level": "experimental-native-contract-gated",
            "contract_required": True,
            "supported_model_types": ["step3p5", "step3p7"],
            "mtp_style": "appended-nextn-layers",
            "hidden_variant": "pre_norm",
            "verify_strategy": "batched",
            "notes": (
                "Step ships 3 distinct appended NextN layers (dense MLP, GQA). "
                "MTP consumes the trunk pre-final-norm hidden; norms are "
                "zero-centered (Gemma-style)."
            ),
            "references": [
                "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/step3p5_mtp.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/step3p5.py",
            ],
        }
