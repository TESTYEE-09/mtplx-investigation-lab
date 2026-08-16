# Exact fused q4 GDN input projections, M=2/M=3, Apple M1 Pro

Date: 2026-08-16
Device: Apple M1 Pro, `applegpu_g13s`, MLX 0.32
Model: `models/qwen38-27b-optimized-speed` (local lab copy; `~/.mtplx/models/`
was never read or written)
Runtime path: Turbo profile, `capture_commit`, linear GDN from convolution
tape, committed MTP history, exact speculative rejection sampling

This extends the accepted MLP gate/up dual-launch experiment to the GDN q4
row-independent input projections, as the prior report proposed.

## What was built

`verify_kernels.exact_qmv_m23_multi` — an *N*-way fused affine-q4 QMV launch.
The output columns of several projections are laid out in one virtual
concatenated column space, so each simdgroup still owns a BN=4 tile of a single
weight matrix and performs exactly MLX 0.32 `qmv_fast`'s per-column dot product
and SIMD reduction. Only the scheduling changes: one Metal launch and one
graph op replace *n*. Every projection still materializes its own separate
BF16 output array, so no downstream numerical boundary moves.

Applied to the GDN family this replaces four launches
(`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) with one, in each of the
48 GDN layers.

## Exactness — MEASURED, bit-identical

Against stock `mx.quantized_matmul` on the model's own loaded weights.
Artifact: `kernel-multi-paired.json`.

| Fused combination | M=2 | M=3 |
|---|---|---|
| `qkv` alone `(5120,10240)` | bit-identical | bit-identical |
| `z` alone `(5120,6144)` | bit-identical | bit-identical |
| `b`, `a` alone `(5120,48)` | bit-identical | bit-identical |
| `qkv+z` | bit-identical | bit-identical |
| `b+a` | bit-identical | bit-identical |
| `qkv+z+b+a` (4 launches -> 1) | bit-identical | bit-identical |

`max_abs = 0.0` on every output stream at every `nsg` tested (1, 2, 4, 8).
Synthetic pre-checks additionally covered a threadgroup straddling a
projection boundary (`N=(68,28)`), which the real shapes do not exercise.

Note the small-N result: MLX serves `N=48` from the same `qmv_fast` arithmetic,
so `in_proj_b`/`in_proj_a` fold into the same exact launch. They did not need a
separate numerical treatment.

## Isolated kernel timing — MEASURED

Adjacent paired A/B, 25 epochs x 5 calls, medians. Artifact:
`kernel-multi-paired.json`.

| Combination | M=2 stock | M=2 fused | M=2 speedup | M=3 stock | M=3 fused | M=3 speedup |
|---|---:|---:|---:|---:|---:|---:|
| `qkv+z` (best nsg) | 0.887 ms | 0.838 ms | 1.06x | 1.248 ms | 1.207 ms | 1.03x |
| whole family vs 4 stock launches | 0.881 ms | 0.840 ms | 1.05x | 1.226 ms | 1.118 ms | 1.10x |

This is far below the MLP lane's isolated 1.22x/1.29x, and it correctly
predicted the production result below. The GDN matrices are small enough that
sharing the `x` read and the `xt` prep buys much less than it does on the
`(5120,17408)` MLP pair.

## Production A/B/B/A — MEASURED

**Methodology correction, load-bearing.** `graphbank._shared_or_new_verify_step`
caches the compiled verify step in a *process-wide* `_SHARED_VERIFY_STEPS` dict.
The Python projection hooks therefore execute only while a graph is being
traced; afterwards the graph is replayed. Any in-process A/B that toggles a
lane after the first trace measures **nothing** — all arms replay the graph
traced by the first arm. My first attempt did exactly this
(`lanes-rotated-d1d2.json`) and showed a flat, meaningless result; it is kept
only as the record of that error. Its token-equality matrix is likewise void.
Arms must be separate processes with the lane fixed before load.

Engagement was then verified directly rather than inferred, via lane counters
split across the trace and replay phases (`cpu-profile-d1-both.json`): at trace
time the fused MLP launch served **56** times (all 56 q4 MLP layers) and the
fused GDN launch **48** times (all 48 GDN layers); `compiled_calls 23/23`,
`fallback_calls 0`.

Bracket: 8 processes, `a1 b1 c1 d1 d2 c2 b2 a2` (stock/mlp/gdn/both, forward
then reverse). Same prompt, seed 20260816, temperature 1.0, top-p 0.95, top-k
20, 16-token warmup, 64 generated tokens, 3 repetitions per process, 6 pooled
samples per arm, medians. Artifacts: `bracket/`, `bracket-summary.json`.

### D1 (verify M=2)

| Metric | stock | MLP lane | GDN lane | both |
|---|---:|---:|---:|---:|
| tok/s | 7.971 | 8.796 | 8.135 | 8.768 |
| verify total | 227.124 ms | 205.121 ms | 222.923 ms | 205.359 ms |
| verify forward | 202.849 ms | 179.946 ms | 199.140 ms | 180.253 ms |
| eval interval | 24.297 ms | 24.922 ms | 23.681 ms | 25.146 ms |
| draft/call | 16.293 ms | 15.607 ms | 15.673 ms | 16.001 ms |
| acceptance | 0.84375 | 0.84375 | 0.84375 | 0.84375 |
| active RAM | 18.974 GiB | 18.974 GiB | 18.974 GiB | 18.974 GiB |
| peak RAM | 19.497 GiB | 19.497 GiB | 19.497 GiB | 19.497 GiB |
| **tok/s delta** | — | **+10.4%** | **+2.1%** | **+10.0%** |

### D2 (verify M=3)

| Metric | stock | MLP lane | GDN lane | both |
|---|---:|---:|---:|---:|
| tok/s | 8.016 | 8.760 | 8.106 | 8.786 |
| verify total | 317.554 ms | 288.672 ms | 314.461 ms | 287.447 ms |
| verify forward | 284.507 ms | 254.508 ms | 281.871 ms | 253.220 ms |
| eval interval | 33.047 ms | 34.164 ms | 32.657 ms | 34.439 ms |
| draft/call | 29.955 ms | 29.417 ms | 29.545 ms | 29.774 ms |
| acceptance | [0.9545, 0.8571] | [0.9545, 0.8571] | [0.9545, 0.8571] | [0.9545, 0.8571] |
| active RAM | 18.974 GiB | 18.974 GiB | 18.974 GiB | 18.974 GiB |
| peak RAM | 19.537 GiB | 19.537 GiB | 19.537 GiB | 19.537 GiB |
| **tok/s delta** | — | **+9.3%** | **+1.1%** | **+9.6%** |

Token equality: every one of the 48 generations in the bracket produced a token
stream identical to stock. Acceptance by position is unchanged in every arm.
RAM is unchanged to the byte at D1 and D2.

### Interpretation

- The MLP lane reproduced independently: **+10.4% / +9.3%**, against the prior
  report's +9.9% / +8.3%. The bracket therefore validates itself.
- The GDN lane is exact and free, but worth only **+2.1% / +1.1%** alone.
- **The two do not stack.** `both` equals `mlp` within noise (D1 +10.0 vs
  +10.4, D2 +9.6 vs +9.3). The GDN fusion contributes approximately zero on
  top of the MLP fusion.
- Launch count is therefore *not* the governing variable. The GDN lane removes
  144 dispatches per verify forward and buys ~1-2%; the MLP lane removes 56 and
  buys ~10%. Production gain tracks the *isolated kernel speedup* (1.05-1.10x
  vs 1.22-1.29x), i.e. it comes from the work each launch does, not from how
  many launches there are.

## Where the verify forward actually goes — MEASURED

`verify_forward` is instrumented as pure graph construction (no `mx.eval`), and
at D1 it is ~200 ms against a ~24 ms final eval wait, which invited the reading
that decode is Python-graph-bound. cProfile of a full generation
(`cpu-profile-d1-both.json`) refutes that: `graphbank.forward_ar_capture` is
**3.688 s of tottime over 23 calls (160 ms/call) with no Python children**,
against roughly 5 ms of Python across the entire generation. The time is Metal
encode plus GPU wait inside the compiled replay. Reducing GPU work and making
each launch cheaper is the lever; shaving Python is not.

One secondary Python cost is visible and unexplained:
`fast_sampling._device_serial_support_arrays`, 0.435 s over 71 calls
(~6 ms/call, ~6% of generation wall time). Not investigated.

## Gating and reversibility

Lane env var `MTPLX_M1_Q4_GDN_PROJ` (renamed by the concurrent consolidation
into `mtplx/m1_q4_verify.py`; my original was
`MTPLX_EXPERIMENTAL_M1_Q4_GDN_PROJ`), plus `MTPLX_M1_Q4_GDN_NSG` (default 2).
Requires `applegpu_g13s`. At load, every candidate launch is compared against
stock MLX on the model's own weights; only bit-identical `(kind, M, K, N, ...)`
keys may serve, and anything unchecked, failing, or raising stays on stock.
Given the ~1-2% production result, defaulting this lane **off** is the correct
call.

## Not done / open

- The `a+b` micro-fusion was measured, not assumed: it is exact and it is
  inside the family launch, but on this evidence it is not worth a separate
  lane.
- **Next lever, designed but not yet run:** column-tile width. `qmv_fast` fixes
  BN=4 columns per simdgroup. BN is a pure scheduling parameter — each output
  column keeps its own accumulator, its own lane->K mapping and its own
  reduction tree — so widening it to 8 or 16 cannot change a BF16 output bit,
  unlike split-K, which reassociates the K reduction itself and is why those
  prototypes failed. The dual kernel's win is consistent with this being the
  real mechanism: it already amortizes the `x` read and `xt` prep over 8
  columns instead of 4. `scripts/bench_m1_q4_bn_sweep.py` sweeps BN x NSG for
  the MLP and GDN shapes with exactness checked at every point. **Not yet
  executed** — see below.
- Isolated stock gate+up at M=2 moves 89 MB in 2.122 ms = ~42 GB/s, roughly a
  quarter of this machine's achievable bandwidth. If that gap is latency, not
  bandwidth, BN widening is the exactness-preserving way to attack it. This is
  an UNCONFIRMED HYPOTHESIS.

## Concurrency hazard

At 19:19-19:20 on 2026-08-16 another agent rewrote `mtplx/nax_verify.py`,
`mtplx/gdn_capture.py`, `mtplx/generation.py` and `mtplx/runtime.py` in this
same working tree, consolidating both lanes into a new `mtplx/m1_q4_verify.py`.
That refactor consumes this experiment's `exact_qmv_m23_multi` kernel and adds
a `lane_signature()` into the graphbank compile-cache key, which is the correct
fix for the `_SHARED_VERIFY_STEPS` aliasing described above. Nothing was lost.
Work stopped at that point rather than re-applying edits into a tree being
rewritten concurrently: two agents sharing one repo *and one GPU* will also
corrupt each other's timings.
