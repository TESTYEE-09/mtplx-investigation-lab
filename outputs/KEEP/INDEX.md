# Retained evidence index

Compact evidence kept after the 2026-08-16 cleanup. Start at
`docs/perf/M1_Q4_FINAL_HANDOFF.md`; this file says which artifact proves what.

Total retained evidence: ~3 MB. The 34 GB of repo-local model copies were
deleted; the installed model at
`~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` is the
source for all future testing and was never modified.

## Correctness

| Artifact | Proves |
| --- | --- |
| `../m1-q4-gdn-proj/lanes-exactness.json` | Fused kernels bit-identical to stock MLX |
| `../m1-q4-gdn-proj/kernel-multi-paired.json` | Fused GDN multi-projection exactness |
| `../m1-q4-m23/` | MLP and GDN kernel microbenchmarks and exactness checks |

The authoritative exactness result — 30/30 bit-identical on the *installed*
model across bf16/fp16/fp32 at M=2 and M=3 — is reproduced at load time by
`m1_q4_verify.validate(model)`. Re-run it any time; it needs no stored artifact.

## Performance

| Artifact | Proves |
| --- | --- |
| `../m1-q4-verify/REPORT.md` | Full shipping report, all workloads, with provenance warnings |
| `../m1-q4-verify/bracket-summary.json` | Controlled 8-arm cross-process bracket (best valid evidence) |
| `../m1-q4-verify/long-summary.json` | 384-token stock-vs-shipped A/B/B/A |
| `../m1-q4-verify/coding-summary.json` | Realistic coding workload A/B/B/A |
| `../m1-q4-verify/d3-summary.json` | D3 (M=4) no-regression check |
| `../m1-gdn/` | Component profiling: where verify time actually goes |

All of the above were measured on the repo-local **bf16** copy (now deleted).
They are a valid lane-vs-lane comparison on that model. The installed model has
**fp16** scales *and* an fp16 residual — a different computation — and its
matched stock-vs-shipped number was never completed.

## Invalid methodology — kept deliberately

| Artifact | Proves |
| --- | --- |
| `../m1-q4-gdn-proj/INVALID-EVIDENCE.md` | Why in-process lane toggling produced a fake A/B |
| `../m1-q4-gdn-proj/lanes-rotated-d1d2.json` | The invalid result itself, retained as a marker |
| `../tune-compare/` | `mtplx tune` noise floor: 42–62% spread on identical binaries |

Both are kept so nobody repeats them. The graph-cache aliasing bug is fixed;
the tuner's single-sample noise is inherent and needs `--limit` raised.

## Engagement

Lane engagement counters from the tune bracket live in
`../tune-compare/{o1-opt,o2-opt}/lane-stats.jsonl`: 10,304 fused MLP and 8,832
fused GDN launches at D1, 7,000 / 6,000 at D2, zero at AR and D3 — identical
across two independent runs.
