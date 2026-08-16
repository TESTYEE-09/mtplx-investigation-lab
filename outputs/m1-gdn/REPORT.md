# Optimized-Speed M1 Pro GDN verification localization

Model: repo-local `qwen38-27b-optimized-speed` (unchanged weights)  
Machine: 14-GPU-core M1 Pro, 32 GiB unified memory  
Verify path: `capture_commit` / `linear-gdn-from-conv-tape` / committed MTP history

## Result

The GatedDeltaNet recurrence is not the per-row bottleneck. Across all 48 GDN
layers it costs about 15--16 ms per verify forward and stays effectively flat
from M=2 through M=4. No experimental GDN implementation was installed.

The eager M=2 -> M=3 incremental 87.585 ms is attributed as follows:

| Operation | Increment | Share |
|---|---:|---:|
| Decoder MLPs | 57.826 ms | 66.0% |
| GDN input projections | 14.198 ms | 16.2% |
| Attention work | 6.426 ms | 7.3% |
| GDN output projections | 4.342 ms | 5.0% |
| LM head | 3.843 ms | 4.4% |
| GDN convolution | 0.739 ms | 0.8% |
| GDN recurrence | -0.328 ms | -0.4% |

M=3 -> M=4 has the same morphology before Turbo's existing M4 specialization:
MLPs 57.593 ms, GDN input projections 14.598 ms, attention 5.757 ms,
GDN output projections 4.191 ms, LM head 3.882 ms, recurrence 0.866 ms.

The supplied Optimized-Speed verify-eval values are almost a constant factor
faster than the eager local measurements: 153.1/192.4=0.796,
221.3/279.9=0.790, and 289.0/367.1=0.787. That proportionality rules out an
extra row-dependent GDN operation in the local path. The current machine was
not suitable for an absolute timing reconciliation: max-fan state was not
available/verified, WindowServer and Firefox were using the GPU, and initial
server samples overlapped extended background warmup and asynchronous
postcommit work. Those contaminated server samples are retained as evidence
but are not headline medians.

## Production realization boundary

The product profile sets `MTPLX_LAZY_VERIFY_LOGITS=1`, so `verify_eval`
realizes hidden state only. The corrected low-level Turbo probe measured:

| Depth | M | Hidden-only eval | Forced all-output eval | GDN recurrence (48 layers) |
|---|---:|---:|---:|---:|
| D1 | 2 | 184.600 ms | 203.759 ms | 16.356 ms |
| D2 | 3 | 268.374 ms | 281.063 ms | 15.150 ms |
| D3 | 4 | 174.089 ms | 182.685 ms | 15.257 ms |

D3 is not comparable to the eager supplied D3 value under Turbo: the existing
`vk_k` M4 quantized-matmul specialization engages and changes the entire
projection/MLP cost.

Compiled verify also moves realization work from the named `verify_eval`
timer into `verify_forward` through its pre/post boundaries. Therefore
`verify_eval` alone is not comparable between eager and compiled runs.

## Clean in-process Turbo generation benchmark

Identical prompt, seed 20260816, target and draft sampler 1.0/0.95/20,
16-token warmup, 64 generated tokens, three repetitions:

| Depth | tok/s median | verify total | verify eval | verify forward | draft/call | active | peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 7.997 | 227.159 ms | 24.503 ms | 202.657 ms | 16.182 ms | 18.974 GiB | 19.497 GiB |
| D2 | 7.904 | 321.847 ms | 33.743 ms | 288.104 ms | 30.607 ms | 18.974 GiB | 19.537 GiB |
| D3 | 11.355 | 238.439 ms | 22.551 ms | 215.920 ms | 43.680 ms | 18.974 GiB | 19.542 GiB |

Tokens were identical across the three repetitions at each depth. These are
not release-quality thermal numbers because the machine had no verified fan
control and other GUI GPU load was present.

## Implementation inspection

The tape recurrence implementation already:

- issues one recurrence Metal launch per GDN layer, not per speculative row;
- loads recurrent state once into registers;
- executes the mathematically sequential T loop inside that kernel;
- writes only final recurrent state once;
- produces rollback tape in the same recurrence launch.

Projection, convolution/gate, norm, output projection, attention, MLP, and LM
head remain separate operations. The dominant row scaling is the quantized
MLP/projection family, not state round-trips or recurrence launches.

## LM head

The Optimized-Speed target LM head is q8, while the separate draft-only head
installed by the server is q4. The prototype kernels return top-k logits and
indices only. Exact sampler semantics require the full-vocabulary logsumexp
before top-p truncation, followed by the exact proposal probability used by
rejection sampling. Top-20 values alone cannot supply that normalization or
prove whether the top-p boundary extends beyond those 20 entries. The kernel
was not wired in and no LM-head experimental flag was added.

## RAM

The clean generation path is roughly 18.97 GiB active / 19.50--19.54 GiB peak.
The exact server accumulated 1.639 GiB in nine in-RAM session-bank entries
from independent synthetic requests, and asynchronous retokenized postcommit
work materially interfered with subsequent request timings. This is a real
2--4 GiB-class runtime opportunity over longer multi-session workloads, but it
is session/prefix-cache policy rather than GDN verification memory. No cache
policy was changed without a dedicated warm-prefix latency and throughput A/B.

## Next optimization

Optimize the row-scaled q4 quantized decoder MLPs first, then GDN input/output
projections. A fixed-M fused/persistent recurrence is not justified by these
measurements. Separately, benchmark anonymous/stateless session-bank retention
and postcommit scheduling behind its own experiment, with warm-prefix hit rate
and request throughput as hard regressions gates.
