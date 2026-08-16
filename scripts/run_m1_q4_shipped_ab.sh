#!/bin/bash
# Matched stock-vs-shipped A/B/B/A over a longer generation and a realistic
# coding prompt.
#
# The 64-token bracket answers "is the fast path faster"; this answers "does it
# stay faster once generation is long enough that launch/setup cost is amortised
# away". Same cross-process discipline: one process per arm, forward then
# reversed.
set -euo pipefail

# Defaults to the installed model, read-only. The repo-local lab copies were
# deleted in the 2026-08-16 cleanup; nothing here writes to the model tree.
MODEL="${MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16}"
OUTDIR="${OUTDIR:-outputs/m1-q4-verify/shipped-ab}"
PY="${PY:-.venv/bin/python}"
DEPTHS="${DEPTHS:-1 2}"
REPEATS="${REPEATS:-2}"
MAXTOK="${MAXTOK:-384}"
LABEL="${LABEL:-long}"
PROMPT="${PROMPT:-}"

DIR="$OUTDIR/$LABEL"
mkdir -p "$DIR"

# "shipped" is the shipped default configuration -- no per-lane overrides, so
# this measures exactly what a normal `mtplx serve` gets. "stock" is the
# documented opt-out.
run_arm() {
  local tag="$1" disable="$2" slot="$3"
  echo "=== $LABEL/$slot: arm=$tag disable=$disable ==="
  # `set -u` treats an empty array expansion as unbound on bash 3.2 (macOS),
  # so the array always carries at least one element.
  local -a extra=(--seed 20260816)
  if [ -n "$PROMPT" ]; then extra+=(--prompt "$PROMPT"); fi
  env MTPLX_DISABLE_M1_Q4_VERIFY="$disable" \
      MTPLX_M1_Q4_STATS=1 \
      "$PY" scripts/benchmark_qwen38_gdn_verify.py \
        --model "$MODEL" \
        --depths $DEPTHS \
        --repeats "$REPEATS" \
        --max-tokens "$MAXTOK" \
        "${extra[@]}" \
        --output "$DIR/$slot-$tag.json" \
    > "$DIR/$slot-$tag.log" 2>&1
  echo "    -> $DIR/$slot-$tag.json"
}

run_arm stock    1 a1
run_arm shipped  0 b1
run_arm shipped  0 b2
run_arm stock    1 a2

echo "$LABEL A/B/B/A complete"
