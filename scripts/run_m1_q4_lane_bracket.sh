#!/bin/bash
# Cross-process A/B/C/D/D/C/B/A bracket for the M1 q4 verify lanes.
#
# Every arm gets its own process. The compiled verify step is traced once and
# cached process-wide (graphbank._SHARED_VERIFY_STEPS); the cache key now
# carries the lane signature so an in-process flip can no longer alias graphs,
# but a fresh process is still the honest way to compare arms -- it also gives
# each arm its own allocator state and warm-up.
#
# Arms are run forward then reversed so host frequency/thermal drift loads on
# every arm equally instead of on the last one.
set -euo pipefail

# Defaults to the installed model, read-only. The repo-local lab copies were
# deleted in the 2026-08-16 cleanup; nothing here writes to the model tree.
MODEL="${MODEL:-$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16}"
OUTDIR="${OUTDIR:-outputs/m1-q4-verify/bracket}"
PY="${PY:-.venv/bin/python}"
DEPTHS="${DEPTHS:-1 2}"
REPEATS="${REPEATS:-3}"
MAXTOK="${MAXTOK:-64}"

mkdir -p "$OUTDIR"

run_arm() {
  local tag="$1" mlp="$2" gdn="$3" slot="$4"
  echo "=== $slot: arm=$tag mlp=$mlp gdn=$gdn ==="
  env MTPLX_M1_Q4_MLP="$mlp" \
      MTPLX_M1_Q4_GDN_PROJ="$gdn" \
      MTPLX_M1_Q4_STATS=1 \
      "$PY" scripts/benchmark_qwen38_gdn_verify.py \
        --model "$MODEL" \
        --depths $DEPTHS \
        --repeats "$REPEATS" \
        --max-tokens "$MAXTOK" \
        --output "$OUTDIR/$slot-$tag.json" \
    > "$OUTDIR/$slot-$tag.log" 2>&1
  echo "    -> $OUTDIR/$slot-$tag.json"
}

run_arm stock 0 0 a1
run_arm mlp   1 0 b1
run_arm gdn   0 1 c1
run_arm both  1 1 d1
run_arm both  1 1 d2
run_arm gdn   0 1 c2
run_arm mlp   1 0 b2
run_arm stock 0 0 a2

echo "bracket complete"
