"""Generic MoE expert-layout adapters.

Some Hugging Face checkpoints store MoE experts as numbered modules:

    layers.N.mlp.experts.E.gate_proj.weight

MLX's switch-MoE layers load the same experts stacked under:

    layers.N.mlp.switch_mlp.gate_proj.weight

This module owns that translation once so Forge, runtime MTP injection, and
future model families do not grow model-by-model key patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


StackFn = Callable[[list[Any]], Any]

EXPERT_COUNT_CONFIG_KEYS = (
    "num_experts",
    "n_routed_experts",
    "n_experts",
    "moe_num_experts",
    "num_local_experts",
    "num_moe_experts",
)


@dataclass(frozen=True)
class NumberedExpertKey:
    source_key: str
    expert_index: int
    module_prefix: str
    leaf: str

    @property
    def output_key(self) -> str:
        return f"{self.module_prefix}.{self.leaf}"


def parse_numbered_expert_key(key: str) -> NumberedExpertKey | None:
    """Return parsed numbered-expert metadata, or None for ordinary keys."""
    text = str(key)
    marker = ".experts."
    if marker not in text:
        return None
    before, after = text.split(marker, 1)
    parts = after.split(".")
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    expert_index = int(parts[0])
    projection = parts[1]
    leaf = ".".join(parts[2:])
    if not projection or not leaf:
        return None
    return NumberedExpertKey(
        source_key=text,
        expert_index=expert_index,
        module_prefix=f"{before}.switch_mlp.{projection}",
        leaf=leaf,
    )


def num_experts_from_config(config: dict[str, Any]) -> int:
    tcfg = config.get("text_config", config) if isinstance(config, dict) else {}
    for key in EXPERT_COUNT_CONFIG_KEYS:
        try:
            value = tcfg.get(key) if isinstance(tcfg, dict) else None
            if value is None:
                value = config.get(key)
            if value:
                return int(value)
        except Exception:
            continue
    return 0


def _default_stack(values: list[Any]) -> Any:
    import mlx.core as mx

    return mx.stack(values, axis=0)


def stack_numbered_experts(
    weights: dict[str, Any],
    *,
    num_experts: int | None = None,
    stack_fn: StackFn | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Stack complete numbered expert leaves into switch-MoE leaves.

    Ordinary keys pass through unchanged. In strict mode, any partial numbered
    expert group raises instead of leaking keys the MLX model cannot load.
    """
    stack = stack_fn or _default_stack
    passthrough: dict[str, Any] = {}
    grouped: dict[tuple[str, str], dict[int, tuple[str, Any]]] = {}
    for key, value in weights.items():
        parsed = parse_numbered_expert_key(key)
        if parsed is None:
            passthrough[key] = value
            continue
        grouped.setdefault((parsed.module_prefix, parsed.leaf), {})[parsed.expert_index] = (
            parsed.source_key,
            value,
        )

    if not grouped:
        return dict(weights)

    result = dict(passthrough)
    incomplete: list[str] = []
    for (module_prefix, leaf), experts in sorted(grouped.items()):
        expected = int(num_experts or (max(experts) + 1))
        expected_indexes = set(range(expected))
        present_indexes = set(experts)
        if present_indexes == expected_indexes:
            result[f"{module_prefix}.{leaf}"] = stack(
                [experts[index][1] for index in range(expected)]
            )
            continue
        missing = sorted(expected_indexes - present_indexes)
        label = f"{module_prefix}.{leaf}"
        if missing:
            label += f" missing experts {missing[:8]}"
            if len(missing) > 8:
                label += f" (+{len(missing) - 8} more)"
        incomplete.append(label)
        if not strict:
            for _expert_index, (source_key, value) in sorted(experts.items()):
                result[source_key] = value

    if incomplete and strict:
        raise ValueError(
            "incomplete numbered MoE expert groups: " + "; ".join(incomplete[:8])
        )
    return result


class NumberedExpertAccumulator:
    """Streaming accumulator for shard-by-shard expert stacking."""

    def __init__(self, *, num_experts: int | None = None, stack_fn: StackFn | None = None) -> None:
        self.num_experts = num_experts
        self.stack = stack_fn or _default_stack
        self._groups: dict[tuple[str, str], dict[int, tuple[str, Any]]] = {}

    def add(self, key: str, value: Any) -> bool:
        parsed = parse_numbered_expert_key(key)
        if parsed is None:
            return False
        self._groups.setdefault((parsed.module_prefix, parsed.leaf), {})[
            parsed.expert_index
        ] = (parsed.source_key, value)
        return True

    def flush_complete(self) -> dict[str, Any]:
        if self.num_experts is None:
            return {}
        complete: dict[str, Any] = {}
        for group_key, experts in list(self._groups.items()):
            expected = int(self.num_experts)
            if set(experts) != set(range(expected)):
                continue
            module_prefix, leaf = group_key
            complete[f"{module_prefix}.{leaf}"] = self.stack(
                [experts[index][1] for index in range(expected)]
            )
            del self._groups[group_key]
        return complete

    def flush_remaining(self, *, strict: bool = False) -> dict[str, Any]:
        if not self._groups:
            return {}
        pending: dict[str, Any] = {}
        for experts in self._groups.values():
            for _expert_index, (source_key, value) in experts.items():
                pending[source_key] = value
        self._groups.clear()
        return stack_numbered_experts(
            pending,
            num_experts=self.num_experts,
            stack_fn=self.stack,
            strict=strict,
        )
