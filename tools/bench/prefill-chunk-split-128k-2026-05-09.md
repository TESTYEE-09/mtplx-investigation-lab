# Prefill chunk-size split (dense=4096 / repage=2048) - 128k validation

> Superseded 2026-05-11: the dense/repage split is no longer the sustained
> product default. Follow-up OpenCode/Pi-shaped cold prefill QA showed 2048
> wins the user-facing metrics at 32k/64k, so sustained mode now uses 2048 for
> both dense and repage defaults and keeps the v0.2-style dense path through
> 128k.

**Status: COMPLETE.** Bench executed 2026-05-09 against the local M5 Max.

## Hypothesis (from PR #33 maintainer review)

PR #33 originally bumped a single chunk-size knob from 2048 -> 4096. Maintainer
review on 2026-05-09 flagged that 4096 in the **repage path** (contexts > 64k)
regresses 128k-context behavior, while it remains a clear win on the **dense path**
(contexts <= 64k). The reshape splits the knob:

- `MTPLX_PREFILL_CHUNK_SIZE_DENSE`  default `4096` (contexts <= 64k)
- `MTPLX_PREFILL_CHUNK_SIZE_REPAGE` default `2048` (contexts >  64k)
- `MTPLX_PREFILL_CHUNK_SIZE` retained as a legacy single-knob fallback.
- `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT` locked at `65536` so any prompt
  above 64k tokens always takes the repage path.

## Results

Apple M5 Max, 128 GB unified memory. Single 128k context row, sustained profile,
fan max, generation_mode=mtp, seed=0, max_tokens=256, prompt-style=coding-agent,
chat format, thinking disabled. Each row is one bench invocation.

| Config | effective chunk | TTFT (s) | Prompt TPS | Decode TPS | Peak memory |
|--------|-----------------|----------|------------|------------|-------------|
| A: `_REPAGE=2048` (this PR, post-reshape) | 2048 | 339.8 | 385.8 | 27.1 | **37.5 GB** |
| B: `_REPAGE=4096` (pre-reshape, forced via `--prefill-chunk-size 4096`) | 4096 | 344.9 | 380.0 | 29.7 | **51.8 GB** |

### Headline delta

- **Memory: +14.3 GB peak (+38%)** with 4096 in repage path at 128k context.
  This is the real regression. On a 128 GB M5 Max it fits; on a 96 GB or
  smaller machine the 4096 repage path risks OOM / heavy swap.
- **TTFT essentially identical** (339.8 vs 344.9 s, ~1.5% diff, within
  run-to-run variance).
- **Prompt TPS within 1.5%** (385.8 vs 380.0).
- **Decode TPS slightly favors 4096** (29.7 vs 27.1, ~9%) but at +38% memory
  cost - not worth it.

### Conclusion

The split is the right call. `_REPAGE=2048` keeps memory in budget for 128k
contexts on machines below 128 GB while staying within noise of `_REPAGE=4096`
on TTFT and prompt rate. The dense path keeps the +29% decode / -35% TTFT
win at <=64k from `tools/bench/findings-chunk-4096-2026-05-08.md`.

## Reproduction

Local model dir: `/Users/dan/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed`

```bash
# A. _REPAGE=2048 (new behavior - profile defaults)
uv run python -m mtplx.cli bench prefill-ladder \
    --model "$MODEL_DIR" \
    --profile sustained --max \
    --prompt-style coding-agent \
    --prompt-format chat \
    --disable-thinking \
    --max-tokens 256 \
    --contexts 131072 \
    --output benchmarks/results/prefill-chunk-split-new-128k-2026-05-09.json

# B. _REPAGE=4096 (forced via --prefill-chunk-size override).
# Note: the profile env hardcodes _REPAGE=2048 since the reshape, so
# shell env vars don't override. Use the CLI flag to bypass profile env
# at the legacy single-knob level (sets MTPLX_PREFILL_CHUNK_SIZE=4096
# absolute, which short-circuits the auto-split).
uv run python -m mtplx.cli bench prefill-ladder \
    --model "$MODEL_DIR" \
    --profile sustained --max \
    --prompt-style coding-agent \
    --prompt-format chat \
    --disable-thinking \
    --max-tokens 256 \
    --contexts 131072 \
    --prefill-chunk-size 4096 \
    --output benchmarks/results/prefill-chunk-split-old-128k-2026-05-09.json
```

Result JSONs are at `benchmarks/results/prefill-chunk-split-{new,old}-128k-2026-05-09.json`
and committed alongside this markdown for reference.

## Caveats

- Single iteration per config. The TTFT/throughput numbers are within run-to-run
  variance (no statistical significance on speed); only the memory delta is
  large enough to be a clear signal at n=1. Run-to-run variance is ~5% in
  prior bench runs at this context size.
- Memory figures are MLX active-bytes peak as reported by the bench, not
  full process RSS.
- The 4096 measurement was obtained by forcing the legacy single-knob env
  (`--prefill-chunk-size 4096`) since the post-reshape profile actively pins
  `_REPAGE` to 2048; this is functionally equivalent to the pre-reshape
  behavior at 128k.
