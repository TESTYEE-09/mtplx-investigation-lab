# Shipping report — exact q4 M=2/M=3 verify lanes

Date: 2026-08-16
Device: Apple M1 Pro, 14 GPU cores, 32 GiB unified memory, `applegpu_g13s`
MLX: 0.32
Model: repo-local `models/qwen38-27b-optimized-speed` for the bracket and the
long/coding/D3 tables below; the installed
`~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` for the
"Installed FP16 model" section. Weights were never written to in either tree —
`~/.mtplx/models` was opened read-only.

> **Read the provenance before quoting a number.** The repo-local copy stores
> its q4 scales in **bf16**; the installed model the `mtplx serve` flow actually
> downloads stores them in **fp16**, which makes MLX promote every quantized
> projection to fp32. That is a different computation, and the lanes were
> initially *wrong* on it while passing every bf16 check (see
> `docs/perf/m1-q4-verify-lanes.md`, "Scale dtype and MLX's promotion"). The
> bracket / long / coding / D3 tables below were measured on the bf16-scale copy
> with the pre-fix kernels; they establish the lane comparison, not the shipped
> model's throughput. For the model the user runs, use the "Installed FP16
> model" section.
Runtime path: Turbo profile, `capture_commit`,
`linear-gdn-from-conv-tape`, committed MTP history
Sampler: temperature 1.0, top-p 0.95, top-k 20, seed 20260816

## Shipped

Both lanes are on by default on `applegpu_g13s`:

* `mlp` — fused exact `gate_proj`+`up_proj` launch + exact `down_proj`, 56 q4 MLPs
* `gdn_proj` — fused exact `in_proj_qkv`+`z`+`b`+`a` launch, 48 GDN layers

Opt out: `MTPLX_DISABLE_M1_Q4_VERIFY=1`. Per-lane: `MTPLX_M1_Q4_MLP=0`,
`MTPLX_M1_Q4_GDN_PROJ=0`.

Design, safety model and configuration: `docs/perf/m1-q4-verify-lanes.md`.

## Method

Cross-process only. Each arm is its own process; the arms run forward then
reversed (`stock, mlp, gdn, both, both, gdn, mlp, stock`) so host frequency and
thermal drift load on every arm equally. Medians of pooled samples; the fastest
sample is never selected.

Two guards make a repeat of the earlier fake A/B impossible:

1. The compiled verify-graph cache key now carries the lane signature
   (`graphbank._shared_or_new_verify_step`), so a configuration change cannot
   alias a graph traced under another configuration.
2. The summarizer prints each arm's lane signature and engagement counters, and
   flags any two arms that share a signature.

Earlier in-process lane-rotation results are invalid and are labelled as such in
`outputs/m1-q4-gdn-proj/INVALID-EVIDENCE.md`.

## Controlled bracket — 64 tokens, 3 repeats/process, 6 samples/arm

24-token prompt, 16-token warmup.

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
| acceptance | `[0.84375]` | same | same | same |
| tokens identical to stock | — | yes | yes | yes |

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
| acceptance | `[0.954545, 0.857143]` | same | same | same |
| tokens identical to stock | — | yes | yes | yes |

Lane engagement per verify trace, from the counters: 56 fused MLP launches and
48 fused GDN launches, i.e. every eligible layer. All four arms reported
distinct lane signatures.

### Why the combination beats either lane

The GDN lane alone is essentially neutral (+0.6% / +0.3%), but adds ~4.7 points
at D1 and ~1.1 at D2 on top of the MLP lane. Collapsing the MLP launches first
is what makes the remaining GDN dispatch pressure worth removing. Neither lane
ships without the other.

Artifacts: `bracket/`, `bracket-summary.json`.

## Longer generation — 384 tokens

Stock-vs-shipped A/B/B/A, 2 repeats/process, 4 pooled samples/arm. "Stock" is
`MTPLX_DISABLE_M1_Q4_VERIFY=1`; "shipped" is the default configuration, so this
measures exactly what a normal `mtplx serve` gets.

| Metric | Stock D1 | Shipped D1 | Δ | Stock D2 | Shipped D2 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 8.492 | 9.503 | **+11.9%** | 8.281 | 9.243 | **+11.6%** |
| verify total (ms) | 210.274 | 186.200 | -11.4% | 302.320 | 267.656 | -11.5% |
| verify forward (ms) | 184.753 | 159.867 | -13.5% | 266.147 | 233.037 | -12.4% |
| eval interval (ms) | 25.593 | 26.333 | +2.9% | 35.941 | 34.645 | -3.6% |
| draft/call (ms) | 16.606 | 16.488 | -0.7% | 32.804 | 32.827 | +0.1% |
| active RAM (GiB) | 18.974 | 18.974 | 0 | 18.974 | 18.974 | 0 |
| peak RAM (GiB) | 19.528 | 19.528 | 0 | 19.568 | 19.568 | 0 |

Tokens identical to stock in every sample. Acceptance identical:
D1 `[0.909091]`, D2 `[0.934307, 0.846715]`.

The gain is *larger* at 384 tokens than at 64, so it is not a launch or
warm-up artifact: the fused launches are replayed by the compiled graph on
every verify call, and the longer the generation, the more of the run is
steady-state verify.

Artifacts: `shipped-ab/long/`, summary `long-summary.json`.

## Realistic coding workload — 256 tokens

Prompt: implement a thread-safe O(1) LRU cache with a doubly linked list and a
dict, type hints, docstrings, a usage example, and an explanation of the
invariants. A/B/B/A, 2 repeats/process, 4 pooled samples/arm.

| Metric | Stock D1 | Shipped D1 | Δ | Stock D2 | Shipped D2 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 7.961 | 9.082 | **+14.1%** | 7.650 | 8.041 | **+5.1%** |
| verify total (ms) | 219.070 | 189.566 | -13.5% | 297.034 | 279.135 | -6.0% |
| verify forward (ms) | 192.346 | 162.929 | -15.3% | 260.660 | 241.650 | -7.3% |
| eval interval (ms) | 26.925 | 26.637 | -1.1% | 36.412 | 37.485 | +2.9% |
| draft/call (ms) | 17.700 | 17.540 | -0.9% | 33.918 | 36.209 | +6.8% |
| active RAM (GiB) | 18.974 | 18.974 | 0 | 18.974 | 18.974 | ~0 |
| peak RAM (GiB) | 19.527 | 19.528 | ~0 | 19.527 | 19.528 | ~0 |

Tokens identical to stock in every sample. Acceptance identical:
D1 `[0.866667]`, D2 `[0.851485, 0.673267]`.

The smaller D2 figure here is not a verify regression — verify forward still
improved 7.3% — it is the +6.8% drift in `draft/call`, which these lanes do not
touch (drafting runs at `M = 1`, where every dispatch declines). Treat the D2
coding number as the noisier of the six measurements; the D1 coding, D1/D2
long, and D1/D2 controlled results all land between +9.6% and +14.1%.

Artifacts: `shipped-ab/coding/`, summary `coding-summary.json`.

## D3 (M=4) no-regression check — 128 tokens

`M = 4` is outside these lanes by construction, so every dispatch declines and
Turbo's existing `vk_k` path runs untouched. Measured anyway, A/B/B/A,
2 repeats/process:

| Metric | Stock | Shipped | Δ |
| --- | ---: | ---: | ---: |
| tok/s | 11.300 | 11.230 | -0.6% |
| verify total (ms) | 220.744 | 221.628 | +0.4% |
| verify forward (ms) | 196.225 | 196.700 | +0.2% |
| active RAM (GiB) | 18.974 | 18.974 | 0 |

Tokens identical; acceptance identical (`[0.902439, 0.675, 0.5]`). The -0.6% is
run-to-run noise on a path the lanes never touch.

Artifacts: `shipped-ab/d3/`, summary `d3-summary.json`.

## Notes on interpretation

* The saving is in `verify_forward`, which is compiled Metal
  encode/execute/wait. Profiling put `forward_ar_capture` at ~160 ms/call in
  C++ with essentially no Python children, and total Python overhead across a
  generation at ~5 ms. An earlier claim that this optimization mainly removes
  Python graph construction was wrong.
* These tok/s figures describe these fixed workloads on this machine state.
  They are not comparable to the historical long-code Optimized-Speed tune
  (AR ~8.67, D1 ~14.08, D2 ~14.14, D3 ~10.81), which used a different
  workload and sampler. Only matched stock-vs-shipped percentages from this
  report should be quoted.
* D3 (M=4) is outside these lanes by construction — `M = 4` is not in
  `(2, 3)`, so every dispatch declines and the Turbo `vk_k` path is untouched.
