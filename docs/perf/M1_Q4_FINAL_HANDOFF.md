# M1 q4 verify lanes — handoff

Everything a fresh session needs to resume this project. Design detail lives in
`docs/perf/m1-q4-verify-lanes.md`; this file is the map.

Branch: **`m1-gdn-speed`** (nothing pushed anywhere).
Hardware this targets: Apple M1 Pro, 32 GB, `applegpu_g13s`, MLX 0.32.
Model: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`, installed read-only at
`~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`.

## What ships

Two shape-specialized Metal kernels that keep `qmv_fast`'s per-column dot
product and SIMD reduction order **byte for byte**, changing only which
simdgroup computes which column and how many launches carry the work. Both
default ON, `applegpu_g13s` only.

* **`mlp` lane** — fused `gate_proj`+`up_proj` in one launch and one read of
  `x`, stock `swiglu`, exact `down_proj`. 56 of 64 decoder layers (the other 8
  MLPs are q8 and stay stock).
* **`gdn_proj` lane** — fused `in_proj_qkv`+`z`+`b`+`a` in one launch instead of
  four, across all 48 GDN layers.

Together they remove ~200 Metal dispatches per verify forward. Active only at
`M = 2` and `M = 3` (D1/D2). AR (M=1) and D3 (M=4) decline by construction —
MLX's own `vk_k` path handles M>=4.

Deliberately **not** done: split-K (faster but changes output bits — rejected
under the exactness requirement), SwiGLU fusion, top-k LM head wiring, session
bank policy changes.

## Changed files

| File | Role |
| --- | --- |
| `mtplx/verify_kernels.py` | The three Metal kernels, `_promote`, eligibility predicates |
| `mtplx/m1_q4_verify.py` | Gating, load-time validation, dispatch, patches, `lane_signature()` |
| `mtplx/graphbank.py` | Lane signature added to the compiled-verify cache key |
| `mtplx/runtime.py` | Load hook: `reset_config()` → `validate(model)` → install patches |
| `mtplx/gdn_capture.py` | Routes GDN input projections through the fused lane |
| `tests/test_m1_q4_verify.py` | 21 tests |

Additive diff, no deletions to existing behaviour.

## Exactness

**30/30 bit-identical** against stock MLX on the installed model's own weights
(`passed: 30, failed: 0, ready: true, gdn_layers: 48`).

Coverage: `single` both directions (K=5120→17408 and K=17408→5120), `dual`, full
`mlp` output, fused `gdn` qkv/z/b/a — each at M=2 and M=3, each under
**bfloat16, float16 and float32** activations. Bit-identical, never tolerance-based.

Three gates, all must pass before a fused launch serves: device architecture;
shape/dtype/quant-format (quant format checked *separately from shape* — this
model carries q4 and q8 at the same `(K, N)`); and load-time validation against
stock MLX on the real weights, before any verify graph is traced.

### Three silent bugs worth remembering

1. **fp16 scales with an activation-typed pointer** — kernel refused to compile,
   everything silently fell back to stock. Fixed with an independent `S`
   template parameter for scales/biases.
2. **Wrong arithmetic precision** — the kernel mirrors `qmv_fast`'s partial sums
   *in the activation type*, so bf16 rows against an fp32-promoted reference
   were off by **~1%**. Proven with an fp64 NumPy ground truth. Fixed by
   `_promote()` casting rows to `mx.result_type(x.dtype, scale_dtype)` exactly
   as MLX does internally.
3. **Validation derived its dtypes from an assumed bf16 residual** — the
   installed model's residual is **fp16**. Validation checked bf16 and fp32, the
   model ran fp16, and **every dispatch declined while reporting 20 passing
   checks** (`mlp_fused: 0, mlp_stock: 2620`). Fixed by validating all three
   supported dtypes unconditionally.

> **"Validation passed" is not "the fast path ran."** Bug 3 produced a perfect
> validation report and zero fused launches. Always read the engagement
> counters. This is the single most important lesson in this project.

## Engagement

Proven from `mtplx tune` candidate subprocesses, identical across two
independent runs:

| Candidate | MLP fused | GDN fused | Expected |
| --- | ---: | ---: | --- |
| AR (M=1) | 0 | 0 | declines |
| D1 (M=2) | 10,304 | 8,832 | engages |
| D2 (M=3) | 7,000 | 6,000 | engages |
| D3 (M=4) | 0 | 0 | declines to `vk_k` |

Enable with `MTPLX_M1_Q4_STATS=1`; add `MTPLX_M1_Q4_STATS_FILE=<path>` to get a
JSONL dump per process at exit (needed when a harness spawns subprocesses).

## Performance evidence

**Best valid evidence** (controlled cross-process bracket, repeats + medians,
A/B/B/A ordering) — measured on the repo-local **bf16** copy, which no longer
exists:

| Metric | Stock D1 | Shipped D1 | Δ | Stock D2 | Shipped D2 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 8.492 | 9.503 | +11.9% | 8.281 | 9.243 | +11.6% |
| verify forward (ms) | 184.75 | 159.87 | -13.5% | 266.15 | 233.04 | -12.4% |

RAM identical between arms. Tokens identical to stock in every sample.
The saving lands in `verify_forward` — compiled Metal encode/execute/wait.
An earlier claim that this was mainly a Python graph-construction win was
**wrong**: `forward_ar_capture` is ~160 ms/call in C++ with ~5 ms total Python
overhead across a generation.

**Installed fp16 model: no validated speedup number exists yet.** The lanes are
proven correct and proven to engage on it, but the matched stock-vs-shipped
measurement was never completed. Do not quote a speedup for this model.

### Two invalid-methodology warnings

1. **In-process lane toggling is invalid.** `graphbank._SHARED_VERIFY_STEPS`
   cached compiled verify graphs process-wide without the lane configuration in
   the key, so toggling lanes after the first trace replayed the graph traced
   under the *previous* configuration — an A/B comparing an arm against itself.
   Fixed (the key now carries `lane_signature()`), but any pre-fix in-process
   result is invalid: see `outputs/m1-q4-gdn-proj/INVALID-EVIDENCE.md`.
   **Always benchmark cross-process, one process per arm.**
2. **`mtplx tune` cannot resolve a ~10% effect.** It takes one sample per mode
   (`--limit 1`). A four-arm bracket showed within-arm spreads of 42–62% on
   identical binaries — stock D2 moved 8.389 → 13.575 between two runs. AR, where
   the lanes provably ran zero fused launches, "showed" −13%. Evidence retained
   in `outputs/tune-compare/` as a noise-floor record. Raise `--limit` if you
   want a usable answer from the tuner.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MTPLX_DISABLE_M1_Q4_VERIFY` | unset | `1` disables both lanes; wins over everything below |
| `MTPLX_M1_Q4_MLP` | auto (ON) | `0` off, `1` on |
| `MTPLX_M1_Q4_GDN_PROJ` | auto (ON) | `0` off, `1` on |
| `MTPLX_M1_Q4_GDN_NSG` | `2` | Simdgroups per threadgroup for the fused GDN launch |
| `MTPLX_M1_Q4_STATS` | unset | Engagement counters (research only) |
| `MTPLX_M1_Q4_STATS_FILE` | unset | Append per-process counters as JSONL at exit |

## Commands

Tests:

```bash
.venv/bin/python -m pytest tests/test_m1_q4_verify.py -q
```

Pre-existing unrelated failures (confirmed on clean HEAD): 5 parity tests in
`tests/test_graphbank_compiled_verify.py`, and
`tests/test_laguna_model.py::test_laguna_s_2_1_ar_route_skips_qwen_performance_hooks`
(requires 85.3 GiB unified memory; this machine has 32).

Controlled bracket, then summarize:

```bash
OUTDIR=outputs/m1-q4-verify/bracket bash scripts/run_m1_q4_lane_bracket.sh
```

```bash
.venv/bin/python scripts/summarize_m1_q4_lane_bracket.py --directory outputs/m1-q4-verify/bracket
```

Stock-vs-shipped A/B/B/A:

```bash
LABEL=run1 MAXTOK=384 REPEATS=2 DEPTHS="1 2" bash scripts/run_m1_q4_shipped_ab.sh
```

Both scripts now default `MODEL` to the installed model, read-only. The
repo-local lab copies were deleted; there is no need to re-download them.

Use the optimized build as the real CLI (reversible, env-only):

```bash
MTPLX_BREW_VENV=~/Documents/mtplx-investigation-lab/repo/.venv mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16
```

## Rollback

Three independent levels, all non-destructive:

* Disable the lanes: `export MTPLX_DISABLE_M1_Q4_VERIFY=1`
* Return to stock Homebrew MTPLX: `unset MTPLX_BREW_VENV`. The launcher
  `/opt/homebrew/bin/mtplx` falls back to `/opt/homebrew/var/mtplx/venv-2.7.1`,
  which was never modified.
* Discard the code: `git checkout main`.

## Next ideas

* **MEASURED** — GDN recurrence is *not* a bottleneck (~15–16 ms total across 48
  layers, flat in M). Do not optimize it.
* **MEASURED** — Cost shares of a verify forward: MLP q4 matmuls ~66%, GDN input
  projections ~16%, attention ~7%, GDN out_proj ~5%, LM head ~4%.
* **MEASURED** — Split-K is 1.25–1.61× in isolation but changes output bits.
* **ESTIMATED** — SwiGLU fusion would remove a full-width intermediate
  round-trip on 56 layers.
* **THEORETICAL** — Same treatment for the 8 q8 MLPs and the q8 GDN out_proj;
  needs an 8-bit unpacking variant.
* **UNCONFIRMED HYPOTHESIS** — The session bank may evict usefully-warm
  prefixes. Never investigated.

**First job for the next session:** complete the matched stock-vs-shipped
measurement on the installed fp16 model, cross-process, with repeats and
medians. That is the one missing number.

## Retained artifacts

| Path | What it proves |
| --- | --- |
| `outputs/KEEP/INDEX.md` | Index of retained evidence |
| `outputs/m1-q4-verify/REPORT.md` | Full shipping report, all workloads |
| `outputs/m1-q4-verify/bracket-summary.json` | Controlled 8-arm bracket |
| `outputs/m1-q4-verify/{long,coding,d3}-summary.json` | Long / coding / D3 A/B |
| `outputs/m1-q4-gdn-proj/INVALID-EVIDENCE.md` | Why in-process A/B is invalid |
| `outputs/m1-q4-gdn-proj/lanes-exactness.json` | Kernel exactness |
| `outputs/m1-q4-m23/` | MLP and GDN kernel microbenchmarks |
| `outputs/tune-compare/` | `mtplx tune` noise floor |
| `docs/perf/m1-q4-verify-lanes.md` | Design and safety model |
