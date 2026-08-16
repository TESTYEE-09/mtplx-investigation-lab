# Invalid performance evidence in this directory

Preserved for provenance, **not** to be cited as a result.

## `lanes-rotated-d1d2.json` — INVALID

Produced by an in-process rotated A/B that flipped the lane environment between
generations inside one process. At the time, `graphbank._SHARED_VERIFY_STEPS`
keyed the compiled verify callable without the kernel lane configuration, so
after the first trace every "arm" replayed whichever graph was traced first.
The flat, arm-indistinguishable numbers in this file are that artifact, not a
measurement of the lanes.

The generating script (`scripts/benchmark_m1_q4_lanes.py`) has been deleted, and
the cache key now carries `m1_q4_verify.lane_signature()` — see
`docs/perf/m1-q4-verify-lanes.md` and the regression test
`tests/test_graphbank_compiled_verify.py::test_lane_change_never_replays_a_graph_traced_under_another_lane_set`.

## `lanes-exactness.json` — WEAK

Same in-process lane toggling. The token equality it reports is not evidence the
lanes differed. Kernel-level exactness in `kernel-multi-paired.json` is
unaffected and remains valid; cross-process token equality is re-established in
`outputs/m1-q4-verify/`.

## `bracket/` — VALID but INCOMPLETE

A cross-process bracket that was interrupted after 5 of 8 arms (`a1`, `b1`,
`c1`, `d1`, `d2`; `c2`, `b2`, `a2` never ran). Directionally consistent with the
completed bracket in `outputs/m1-q4-verify/bracket/`, but unbalanced — the
missing reverse-order arms are exactly the ones that cancel thermal drift.
Superseded; kept as prior evidence.
