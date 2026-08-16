# Optimized-Speed M1 Pro q4 M=2/M=3 verification investigation

Date: 2026-08-16  
Device: Apple M1 Pro, `applegpu_g13s`  
Model: `models/qwen38-27b-optimized-speed`  
Runtime path: Turbo profile, capture-commit, linear GDN from convolution tape,
committed MTP history  
Experimental gate: `MTPLX_EXPERIMENTAL_M1_Q4_M23=1`

## Exact matrix inventory

The 64 decoder layers contain 48 GDN layers and 16 full-attention layers.

Main q4 MLPs occur in 56 layers (42 GDN and 14 attention):

| Projection | Logical `(K,N)` | Packed weight | Quantization |
|---|---:|---:|---|
| gate | `(5120,17408)` | `(17408,640)` | affine q4, group 32 |
| up | `(5120,17408)` | `(17408,640)` | affine q4, group 32 |
| down | `(17408,5120)` | `(5120,2176)` | affine q4, group 32 |

The last eight MLPs (six GDN and two attention) retain the same logical shapes
but use affine q8, group 64: packed gate/up `(17408,1280)` and down
`(5120,4352)`. They are deliberately outside this experiment.

All 48 GDN layers use:

| Projection | Logical `(K,N)` | Packed weight | Quantization |
|---|---:|---:|---|
| `in_proj_qkv` | `(5120,10240)` | `(10240,640)` | affine q4, group 32 |
| `in_proj_z` | `(5120,6144)` | `(6144,640)` | affine q4, group 32 |
| `in_proj_b` | `(5120,48)` | `(48,640)` | affine q4, group 32 |
| `in_proj_a` | `(5120,48)` | `(48,640)` | affine q4, group 32 |
| `out_proj` | `(12288,5120)` | `(5120,1536)` | affine q8, group 64 |

This experiment changes only the two q4 MLP shapes. GDN projections and the
recurrent implementation are untouched.

## Dispatch diagnosis

For M=2 and M=3, the existing MTPLX q4 patch did not engage because its verify
route begins at M=4. MLX 0.32 therefore dispatched affine `qmv_fast`: one SIMD
group computes four output columns, two SIMD groups per threadgroup, with grid
`M x ceil(N/8)`. G13 does not take MLX's newer wide-QMV route.

At M=4, Turbo dispatches MTPLX `vk_k`: a compile-time-K, q4 affine split-K
kernel with four row accumulators and two K partitions for these large-N
shapes. It creates substantially more K-parallel work and better GPU occupancy
than QMV. That is why M=4 is materially faster. Its reduction order differs
from stock MLX, however, so simply shrinking it to M=2/M=3 failed the exact
output requirement.

The rejected exact-M split-K prototypes were 1.25-1.61x faster in isolated
tests, but changed many BF16 outputs (maximum differences: gate/up about
0.02-0.033 and down 0.125-0.1875). They are not wired into the runtime.

## Implemented experiment

The accepted path mirrors MLX `qmv_fast`'s dot products and SIMD reduction
order at fixed M/K/N. Its main MLP kernel computes gate and up in one Metal
launch, shares input-row reads, and still emits separate BF16 gate and up
arrays. Stock SwiGLU then runs unchanged, followed by an exact fixed-shape down
QMV. It therefore removes a launch without changing quantization, BF16
materialization, activation behavior, or model semantics.

The path is restricted to:

- `applegpu_g13s`
- affine q4, group 32, BF16 input
- M=2 or M=3
- gate/up `(5120,17408)` and down `(17408,5120)`
- non-prefill calls

At model load it eagerly checks both individual shapes, both fused gate/up
outputs, and the complete MLP output for M=2 and M=3 against stock MLX using
the actual loaded weights. All eight checks must pass. Missing or failed keys
fall back to stock before verify compilation.

## Isolated kernel result

Adjacent paired timing, 30 A/B epochs with five calls per measurement:

| Rows | Two stock gate/up QMV | Exact fused gate/up | Speedup | Equality |
|---:|---:|---:|---:|---|
| M=2 | 2.122 ms | 1.741 ms | 1.22x | bit-identical |
| M=3 | 3.525 ms | 2.732 ms | 1.29x | bit-identical |

Artifact: `kernel-dual-paired.json`.

## Production-style A/B/B/A result

Identical prompt, seed 20260816, temperature 1.0, top-p 0.95, top-k 20,
16-token warmup, 64 generated tokens, three repetitions per process. Values
below pool the six samples in each arm from the stabilized A/B/B/A bracket.

| Metric | D1 stock | D1 experiment | Delta | D2 stock | D2 experiment | Delta |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 8.195 | 9.009 | +9.9% | 8.113 | 8.787 | +8.3% |
| verify total | 221.010 ms | 200.280 ms | -9.4% | 314.021 ms | 287.744 ms | -8.4% |
| forward/dispatch | 197.554 ms | 176.152 ms | -10.8% | 281.391 ms | 253.747 ms | -9.8% |
| `mx.eval` interval | 23.519 ms | 24.137 ms | +2.6% | 32.631 ms | 34.004 ms | +4.2% |
| draft/call | 15.742 ms | 15.128 ms | -3.9% | 29.536 ms | 29.359 ms | -0.6% |
| active RAM | 18.974 GiB | 18.974 GiB | 0 | 18.974 GiB | 18.974 GiB | 0 |
| peak RAM | 19.497 GiB | 19.497 GiB | +16 KiB | 19.537 GiB | 19.537 GiB | 0 |

The saved time appears in graph construction/dispatch, not the final eval
wait: each verify forward eliminates one gate/up launch in each of 56 q4 MLP
layers. D1 acceptance remained 0.84375. D2 acceptance remained
`[0.954545, 0.857143]`. Every repeated token stream and every stock-versus-
experiment token stream was identical.

Artifacts: `prod-stable-a1-stock.json`, `prod-stable-b1-dual.json`,
`prod-stable-b2-dual.json`, and `prod-stable-a2-stock.json`.

These tok/s values describe this fixed 64-token production-style workload and
current machine state; they are not substituted for a separately controlled
production-server headline run. Earlier and later process epochs showed large
host-frequency/background variation, which is why only the stabilized bracket
is reported.

## Next optimization

The next exact experiment should extend the same dual-projection morphology to
the GDN q4 row-independent input projections where compatible, starting with a
paired `qkv + z` launch and retaining their stock BF16 boundaries. It should
not revisit the recurrence. A more ambitious MLP fusion across SwiGLU is not
the next choice because reproducing MLX activation rounding exactly is a
harder correctness boundary.

