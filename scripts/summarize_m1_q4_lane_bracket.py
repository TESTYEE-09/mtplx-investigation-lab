#!/usr/bin/env python3
"""Pool an A/B/B/A lane bracket into one comparison table per depth."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    ("tok_s", "tok/s", 3, True),
    ("verify_ms_per_call", "verify total ms", 3, False),
    ("verify_forward_ms_per_call", "verify forward ms", 3, False),
    ("verify_eval_ms_per_call", "eval interval ms", 3, False),
    ("draft_ms_per_call", "draft/call ms", 3, False),
    ("active_memory_bytes", "active RAM GiB", 3, False),
    ("peak_memory_bytes", "peak RAM GiB", 3, False),
)
# "shipped" is the default-configuration arm used by the stock-vs-shipped
# comparisons; the other three name individual lanes in the 4-arm bracket.
ARM_ORDER = ("stock", "mlp", "gdn", "both", "shipped")


def _load(directory: Path) -> dict[str, list[dict[str, Any]]]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".stdout.json"):
            continue
        arm = path.stem.split("-", 1)[1]
        arms.setdefault(arm, []).append(json.loads(path.read_text()))
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arms = _load(args.directory)

    # Guard against the failure that invalidated an earlier round of this
    # experiment: if two arms report the same lane signature, they are the
    # same configuration and any difference between them is noise, not signal.
    lanes = {
        arm: sorted({payload.get("m1_q4_lane_signature", "?") for payload in payloads})
        for arm, payloads in arms.items()
    }
    stats = {
        arm: [payload.get("m1_q4_lane_stats", {}) for payload in payloads]
        for arm, payloads in arms.items()
    }
    print("=== lane configuration per arm ===")
    for arm in ARM_ORDER:
        if arm not in lanes:
            continue
        print(f"{arm:<8} signature={lanes[arm]}")
        print(f"{'':<8} engagement={stats[arm]}")
    distinct = {arm: tuple(sig) for arm, sig in lanes.items()}
    collisions = [
        (a, b)
        for i, a in enumerate(sorted(distinct))
        for b in sorted(distinct)[i + 1 :]
        if distinct[a] == distinct[b]
    ]
    if collisions:
        print(f"!! arms share a lane signature (invalid comparison): {collisions}")
    if any(len(sig) > 1 for sig in lanes.values()):
        print("!! an arm's repeats disagree on lane signature (invalid comparison)")

    depths = sorted(
        {int(result["depth"]) for payloads in arms.values() for payload in payloads
         for result in payload["results"]}
    )

    summary: dict[str, Any] = {"depths": {}}
    for depth in depths:
        pooled: dict[str, dict[str, float]] = {}
        tokens: dict[str, list[list[int]]] = {}
        acceptance: dict[str, Any] = {}
        for arm, payloads in arms.items():
            samples: list[dict[str, Any]] = []
            for payload in payloads:
                for result in payload["results"]:
                    if int(result["depth"]) != depth:
                        continue
                    samples.extend(result["samples"])
                    acceptance[arm] = result["acceptance_by_position"]
            if not samples:
                continue
            pooled[arm] = {
                key: statistics.median(float(row[key]) for row in samples)
                for key, _, _, _ in METRICS
            }
            pooled[arm]["n_samples"] = float(len(samples))
            tokens[arm] = [row["tokens"] for row in samples]

        identical = {}
        reference = tokens.get("stock", [[]])[0]
        for arm, runs in tokens.items():
            identical[arm] = all(run == reference for run in runs)
        summary["depths"][str(depth)] = {
            "lane_signatures": lanes,
            "lane_stats": stats,
            "medians": pooled,
            "acceptance_by_position": acceptance,
            "tokens_identical_to_stock": identical,
        }

        present = [arm for arm in ARM_ORDER if arm in pooled]
        print(f"\n=== depth {depth} ===")
        header = f"{'metric':<20}" + "".join(f"{arm:>14}" for arm in present)
        print(header)
        print("-" * len(header))
        for key, label, _, higher_better in METRICS:
            row = f"{label:<20}"
            for arm in present:
                value = pooled[arm][key]
                if "RAM" in label:
                    value /= 1024 ** 3
                row += f"{value:>14.3f}"
            print(row)
        row = f"{'delta vs stock %':<20}"
        for arm in present:
            delta = 100.0 * (pooled[arm]["tok_s"] / pooled["stock"]["tok_s"] - 1.0)
            row += f"{delta:>+14.1f}"
        print(row)
        row = f"{'fwd delta %':<20}"
        for arm in present:
            delta = 100.0 * (
                pooled[arm]["verify_forward_ms_per_call"]
                / pooled["stock"]["verify_forward_ms_per_call"]
                - 1.0
            )
            row += f"{delta:>+14.1f}"
        print(row)
        row = f"{'samples':<20}"
        for arm in present:
            row += f"{int(pooled[arm]['n_samples']):>14d}"
        print(row)
        print("tokens identical to stock:", identical)
        print("acceptance:", {a: acceptance.get(a) for a in present})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
