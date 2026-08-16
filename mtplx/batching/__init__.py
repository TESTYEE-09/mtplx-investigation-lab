"""Concurrent batching primitives for MTPLX serving.

The package is intentionally MLX-agnostic. The live model still belongs to
the single owner thread; these types describe admission, request state, bucket
compatibility, and cooperative scheduling without importing FastAPI or MLX.
"""

from .admission import (
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionStatus,
    MemoryPressure,
)
from .buckets import ARBatchKey, MTPBatchKey
from .scheduler import (
    BatchSchedulerConfig,
    BatchSchedulerStats,
    MTPContinuousScheduler,
    SchedulerHooks,
    StepResult,
)
from .state import (
    DecodeState,
    RequestPhase,
    RequestPriority,
    RequestState,
    SchedulerMode,
    SchedulerPreset,
    preset_config,
)

__all__ = [
    "ARBatchKey",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionStatus",
    "BatchSchedulerConfig",
    "BatchSchedulerStats",
    "DecodeState",
    "MTPBatchKey",
    "MTPContinuousScheduler",
    "MemoryPressure",
    "RequestPhase",
    "RequestPriority",
    "RequestState",
    "SchedulerHooks",
    "SchedulerMode",
    "SchedulerPreset",
    "StepResult",
    "preset_config",
]
