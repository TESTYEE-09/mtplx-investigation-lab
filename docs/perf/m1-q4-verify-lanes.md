# Exact q4 M=2/M=3 verify lanes (Apple M1 Pro)

Status: **shipped**, both lanes on by default.
Hardware: Apple M1 Pro, 32 GB, `applegpu_g13s`, MLX 0.32.
Model: `Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` (affine q4, group 32; **fp16**
residual, **fp16** scales/biases — see "Scale dtype and MLX's promotion"). The
repo-local `models/qwen38-27b-optimized-speed` copy is bf16 throughout; the two
are different computations and both are validated.

## The problem

Speculative verification replays `M = speculative_depth + 1` rows through the
decoder. MLX 0.32 only has a row-parallel quantized verify kernel from `M >= 4`
(`vk_k`); at `M = 2` and `M = 3` every quantized projection falls back to
`qmv_fast` — one Metal launch each, each re-reading the same hidden state.

The dominant row-scaled cost of a verify forward on this model is quantized
matmul, not the GDN recurrence (measured: the recurrence is ~15–16 ms total
across all 48 GDN layers and essentially flat in `M`):

| Component              | Approx marginal share |
| ---------------------- | --------------------: |
| Decoder MLP q4 matmuls |                  ~66% |
| GDN input projections  |                  ~16% |
| Attention              |                   ~7% |
| GDN output projection  |                   ~5% |
| LM head                |                   ~4% |
| Other                  |                   ~2% |
| GDN recurrence         |                   ~0% |

Straight split-K prototypes were faster (~1.25–1.61x isolated) but changed the
reduction order and therefore the output bits. They were rejected: the shipping
path preserves stock arithmetic exactly.

## What ships

Two shape-specialized kernels that keep `qmv_fast`'s per-column dot product and
SIMD reduction order **byte for byte**, and change only which simdgroup computes
which column and how many launches carry the work.

### `mlp` lane (default ON on `applegpu_g13s`)

`gate_proj` and `up_proj` share one Metal launch and one read of `x`
(`exact_qmv_m23_dual`), stock `swiglu` follows, and `down_proj` runs through the
exact single-projection kernel (`exact_qmv_m23`). 56 of the 64 decoder layers
use the q4 MLP (42 GDN + 14 attention); the other eight are q8 and stay stock.

### `gdn_proj` lane (default ON on `applegpu_g13s`)

`in_proj_qkv` + `in_proj_z` + `in_proj_b` + `in_proj_a` share one launch
(`exact_qmv_m23_multi`) instead of four, across all 48 GDN layers. The four
output column spaces are laid out in one virtual concatenated space, so each
simdgroup still owns a `BN=4` tile of a single weight matrix.

Together the two lanes remove roughly 200 Metal dispatches per verify forward.
The GDN lane is close to neutral *on its own* (+0.6% / +0.3%) but adds
materially on top of the MLP lane (+9.6% / +9.7% combined vs +4.9% / +8.6% for
MLP alone), so the two ship together rather than separately.

## Safety model

Three independent gates, all of which must pass before a fused launch serves:

1. **Device.** `mx.device_info()["architecture"] == "applegpu_g13s"`. Every
   other GPU keeps stock MLX; nothing about this module changes behaviour there.
2. **Shape / dtype / format.** Affine q4, group 32, `M in (2, 3)`, the exact
   Qwen3.8 matrix shapes, and a non-prefill attention phase. The quant format
   is checked separately from the shape because this model carries q4 *and* q8
   projections at the same `(K, N)` — a shape-only key would hand an 8-bit
   weight to a nibble-unpacking kernel.
3. **Loaded weights.** `m1_q4_verify.validate(model)` runs every candidate
   launch against stock MLX on the model's own weights at load, before any
   verify graph is traced, and records a per-key pass/fail. Only keys that
   compared *bit*-identical are allowed to serve. Anything unchecked, failing,
   or raising stays on stock, and a failure is logged at WARNING.

### Scale dtype and MLX's promotion (the FP16 siblings)

The activation dtype and the scale/bias dtype are **independent**, and getting
this wrong is silent. `Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` — the model the
`mtplx serve` flow actually downloads — stores its q4 scales and biases in
**fp16** while the residual stream is **bf16**. MLX promotes a quantized matmul
across both, so on that model:

* every q4 projection *returns fp32*, and
* the next projection in the block is therefore called with **fp32 rows**
  (`down_proj` never sees bf16 there).

Two consequences are baked into the implementation:

* The kernels carry a separate `S` template parameter for scales/biases. Typing
  the scale pointer with the activation dtype does not merely round — it fails
  to compile (`cannot initialize 'const device bfloat *' with 'const device
  float16_t *'`), and every check falls back to stock.
* `verify_kernels._promote` casts the rows to `mx.result_type(x.dtype,
  scale_dtype)` before the launch, exactly as MLX does internally. The kernel
  mirrors `qmv_fast`'s arithmetic *in the activation type*, including its bf16
  partial sums; reading the bf16 rows against an fp32 reference is wrong by
  **~1%**, far outside any ULP argument.

Because the same projection is a different computation depending on the rows it
is handed, every `_VALIDATED` key carries the activation dtype, and `validate()`
checks each shape under both the residual dtype and the promoted one. A dtype
that was never validated simply never dispatches.

This was caught only by running against the installed model: a lab copy with
bf16 scales passed every check while the shipped model failed all ten. Fixture
coverage now pins the fp16-scale/bf16-row combination directly
(`test_fp16_scales_against_bf16_rows_stay_bit_identical`).

Validation runs at load rather than inside the projection call because the first
verify invocation may be an `mx.compile` trace, where an eager comparison cannot
run.

## Compiled-graph cache correctness

`graphbank._SHARED_VERIFY_STEPS` caches one compiled verify callable per logical
key, process-wide. That key used to omit the kernel lane configuration, so a
harness that changed lanes after the first trace silently replayed the graph
built for the *previous* configuration — an A/B that compared an arm against
itself. Any in-process lane-rotation result from before this fix is invalid,
including `outputs/m1-q4-gdn-proj/lanes-rotated-d1d2.json`.

The key now carries `m1_q4_verify.lane_signature()`, which covers the lane flags,
the GDN simdgroup count, *and* the set of shapes that cleared validation — the
complete description of what the traced graph emits. Caching is otherwise
unchanged. Regression test:
`tests/test_graphbank_compiled_verify.py::test_lane_change_never_replays_a_graph_traced_under_another_lane_set`
asserts a lane change hands back a different callable in both directions, and
that an unchanged lane still reuses the cached one.

The signature renders the validated keys to text before sorting. Those keys
carry `mx.Dtype` members, which are equality-comparable but *not* orderable, so
sorting the raw tuples raises `TypeError` as soon as two keys differ only in
their activation dtype — the exact state a model with fp16 scales leaves behind.
Since the signature is computed on every trace, that would have been a hard
failure on the shipped model rather than a silent one
(`test_lane_signature_survives_mixed_dtype_validation_keys`).

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MTPLX_DISABLE_M1_Q4_VERIFY` | unset | `1` disables both lanes; wins over everything below. |
| `MTPLX_M1_Q4_MLP` | auto (ON) | `0` off, `1` on. |
| `MTPLX_M1_Q4_GDN_PROJ` | auto (ON) | `0` off, `1` on. |
| `MTPLX_M1_Q4_GDN_NSG` | `2` | Simdgroups per threadgroup for the fused GDN launch. |
| `MTPLX_M1_Q4_STATS` | unset | `1` enables lane engagement counters (research only). |

Lanes are process-level: they are resolved once at load and never flip under a
live compiled graph.

## Measuring

Cross-process only. Each arm gets its own process:

```bash
OUTDIR=outputs/m1-q4-verify/bracket bash scripts/run_m1_q4_lane_bracket.sh
```

```bash
.venv/bin/python scripts/summarize_m1_q4_lane_bracket.py --directory outputs/m1-q4-verify/bracket
```

The bracket runs `stock, mlp, gdn, both, both, gdn, mlp, stock` so host
frequency and thermal drift load on every arm equally. The summarizer refuses to
present a comparison where two arms report the same lane signature, and prints
the engagement counters so "the tokens matched" is never mistaken for "the fused
launch actually served".

Because "the lane engaged" and "the tokens matched" are different claims, the
counters distinguish trace-phase engagement from replay. `M1_LANE_STATS` ticks
during Python graph construction; the compiled graph then replays the fused
launches without re-entering Python.

## Results

Balanced 8-process bracket, 2026-08-16. Apple M1 Pro 32 GB, Turbo profile,
capture-commit, `linear-gdn-from-conv-tape`, committed MTP history, seed
20260816, temperature 1.0 / top-p 0.95 / top-k 20, 24-token prompt, 16-token
warmup, 64 generated tokens, 3 repeats per process, 6 pooled samples per arm,
medians. Run order `stock, mlp, gdn, both, both, gdn, mlp, stock`.

### D1 (M=2)

| Metric | Stock | MLP | GDN | Combined |
| --- | ---: | ---: | ---: | ---: |
| tok/s | 8.276 | 8.680 | 8.326 | **9.068** |
| vs stock | — | +4.9% | +0.6% | **+9.6%** |
| verify total (ms) | 219.237 | 206.435 | 217.497 | 198.540 |
| verify forward (ms) | 195.992 | 181.271 | 194.359 | 174.466 |
| eval interval (ms) | 23.267 | 25.072 | 23.138 | 24.084 |
| draft/call (ms) | 15.360 | 16.340 | 15.573 | 15.368 |
| active RAM (GiB) | 18.974 | 18.974 | 18.974 | 18.974 |
| peak RAM (GiB) | 19.497 | 19.497 | 19.497 | 19.497 |

### D2 (M=3)

| Metric | Stock | MLP | GDN | Combined |
| --- | ---: | ---: | ---: | ---: |
| tok/s | 8.242 | 8.951 | 8.265 | **9.045** |
| vs stock | — | +8.6% | +0.3% | **+9.7%** |
| verify total (ms) | 309.208 | 282.659 | 308.165 | 279.669 |
| verify forward (ms) | 277.195 | 249.210 | 276.114 | 246.463 |
| eval interval (ms) | 32.013 | 33.513 | 31.877 | 33.205 |
| draft/call (ms) | 28.865 | 28.529 | 28.721 | 28.479 |
| active RAM (GiB) | 18.974 | 18.974 | 18.974 | 18.974 |
| peak RAM (GiB) | 19.537 | 19.537 | 19.537 | 19.537 |

Acceptance by MTP position was identical in every arm: D1 `[0.84375]`,
D2 `[0.954545, 0.857143]`. Every generated token stream was identical to stock,
within and across arms. All four arms reported distinct lane signatures, and
the engagement counters confirm the fused launches were baked into the traced
graph (56 MLP and 48 GDN fused launches per verify trace).

The saving lands in `verify_forward` (-11.0% / -11.1% combined), which is
compiled Metal encode/execute/wait work — profiling puts `forward_ar_capture`
at ~160 ms/call in C++ with essentially no Python children and total Python
overhead across a generation around 5 ms. This is not a Python
graph-construction win.

Raw artifacts: `outputs/m1-q4-verify/bracket/`, summary
`outputs/m1-q4-verify/bracket-summary.json`, narrative
`outputs/m1-q4-verify/REPORT.md` (which also carries the longer-generation and
coding-workload runs).
