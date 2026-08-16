#!/usr/bin/env python3
"""Combine non-overlapping MTP LoRA adapter sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.mtp_adapters import load_mtp_lora_adapter, save_combined_mtp_lora_adapter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("adapters", nargs="+", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--metadata-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = json.loads(args.metadata_json) if args.metadata_json else {}
    if args.run_id:
        metadata["run_id"] = args.run_id
    output = save_combined_mtp_lora_adapter(
        args.output,
        args.adapters,
        metadata=metadata,
    )
    state = load_mtp_lora_adapter(output)
    print(
        json.dumps(
            {
                "adapter_path": str(output),
                "targets": [entry["target"] for entry in state.metadata["targets"]],
                "combined_from": state.metadata.get("combined_from", []),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
