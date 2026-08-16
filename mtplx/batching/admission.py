"""Admission and memory-pressure policy for concurrent batching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .state import PresetConfig, RequestPriority, RequestState, preset_config


class MemoryPressure(StrEnum):
    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"


class AdmissionStatus(StrEnum):
    ADMIT = "admit"
    WAIT = "wait"
    REJECT = "reject"


@dataclass(frozen=True)
class AdmissionDecision:
    status: AdmissionStatus
    reason: str
    effective_max_active_requests: int
    effective_decode_batch_max: int
    effective_prefill_chunk_tokens: int
    memory_pressure: MemoryPressure

    @property
    def admitted(self) -> bool:
        return self.status == AdmissionStatus.ADMIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "effective_max_active_requests": self.effective_max_active_requests,
            "effective_decode_batch_max": self.effective_decode_batch_max,
            "effective_prefill_chunk_tokens": self.effective_prefill_chunk_tokens,
            "memory_pressure": self.memory_pressure.value,
        }


@dataclass
class AdmissionPolicy:
    """Conservative request-admission policy for coding-agent UX."""

    preset: PresetConfig
    max_active_requests: int | None = None
    decode_batch_max: int | None = None
    prefill_chunk_tokens: int | None = None

    @classmethod
    def from_preset(cls, name: str) -> "AdmissionPolicy":
        return cls(preset=preset_config(name))

    def classify_memory(
        self,
        *,
        active_memory_bytes: int | None,
        total_memory_bytes: int | None,
    ) -> MemoryPressure:
        if not active_memory_bytes or not total_memory_bytes or total_memory_bytes <= 0:
            return MemoryPressure.NORMAL
        fraction = float(active_memory_bytes) / float(total_memory_bytes)
        if fraction >= self.preset.hard_memory_fraction:
            return MemoryPressure.HARD
        if fraction >= self.preset.soft_memory_fraction:
            return MemoryPressure.SOFT
        return MemoryPressure.NORMAL

    def effective_limits(self, pressure: MemoryPressure) -> tuple[int, int, int]:
        active = int(self.max_active_requests or self.preset.max_active_requests)
        batch = int(self.decode_batch_max or self.preset.decode_batch_max)
        chunk = int(self.prefill_chunk_tokens or self.preset.prefill_chunk_tokens)
        if pressure == MemoryPressure.HARD:
            return 1, 1, max(128, min(chunk, 512))
        if pressure == MemoryPressure.SOFT:
            return max(1, min(active, 2)), max(1, min(batch, 2)), max(256, chunk // 2)
        return max(1, active), max(1, batch), max(128, chunk)

    def decide(
        self,
        request: RequestState,
        *,
        active_count: int,
        active_memory_bytes: int | None = None,
        total_memory_bytes: int | None = None,
    ) -> AdmissionDecision:
        pressure = self.classify_memory(
            active_memory_bytes=active_memory_bytes,
            total_memory_bytes=total_memory_bytes,
        )
        max_active, max_batch, chunk = self.effective_limits(pressure)
        if pressure == MemoryPressure.HARD and request.priority < RequestPriority.FOREGROUND:
            return AdmissionDecision(
                status=AdmissionStatus.WAIT,
                reason="hard_memory_pressure",
                effective_max_active_requests=max_active,
                effective_decode_batch_max=max_batch,
                effective_prefill_chunk_tokens=chunk,
                memory_pressure=pressure,
            )
        if active_count >= max_active:
            return AdmissionDecision(
                status=AdmissionStatus.WAIT,
                reason="active_request_limit",
                effective_max_active_requests=max_active,
                effective_decode_batch_max=max_batch,
                effective_prefill_chunk_tokens=chunk,
                memory_pressure=pressure,
            )
        return AdmissionDecision(
            status=AdmissionStatus.ADMIT,
            reason="admitted",
            effective_max_active_requests=max_active,
            effective_decode_batch_max=max_batch,
            effective_prefill_chunk_tokens=chunk,
            memory_pressure=pressure,
        )
