"""Compatibility keys for AR batches and future MTP verify cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ARBatchKey:
    model_id: str
    tokenizer_template_hash: str | None
    quant_policy: str | None
    cache_layout: str
    sampler_fingerprint: str
    stop_fingerprint: str

    @classmethod
    def from_request(
        cls,
        request: Any,
        *,
        model_id: str,
        tokenizer_template_hash: str | None,
        quant_policy: str | None = None,
        cache_layout: str = "dynamic_paged_kv",
    ) -> "ARBatchKey":
        return cls(
            model_id=str(model_id),
            tokenizer_template_hash=tokenizer_template_hash,
            quant_policy=quant_policy,
            cache_layout=cache_layout,
            sampler_fingerprint=_fingerprint(getattr(request, "sampler", None)),
            stop_fingerprint=_fingerprint(sorted(getattr(request, "stop_token_ids", []) or [])),
        )

    def as_batch_key(self) -> str:
        return "|".join(
            (
                "ar",
                self.model_id,
                str(self.tokenizer_template_hash or ""),
                str(self.quant_policy or ""),
                self.cache_layout,
                self.sampler_fingerprint,
                self.stop_fingerprint,
            )
        )


@dataclass(frozen=True)
class MTPBatchKey:
    model_id: str
    quant_policy: str | None
    speculative_depth: int
    verify_width: int
    mtp_hidden_variant: str
    mtp_history_policy: str
    cache_kind: str
    verify_core: str

    def as_batch_key(self) -> str:
        return "|".join(
            (
                "mtp",
                self.model_id,
                str(self.quant_policy or ""),
                str(self.speculative_depth),
                str(self.verify_width),
                self.mtp_hidden_variant,
                self.mtp_history_policy,
                self.cache_kind,
                self.verify_core,
            )
        )


def _fingerprint(value: Any) -> str:
    if value is None:
        return "none"
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        value = vars(value)
    return repr(value)
