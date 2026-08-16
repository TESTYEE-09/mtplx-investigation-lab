# Prefill chunk-size 4096 vs 2048 (sustained profile, 2026-05-08)

> Superseded 2026-05-11: this cached 40-pass growth benchmark is useful
> historical evidence, but it is not the product default. A later
> OpenCode/Pi-shaped cold prefill ladder showed 2048 is better for TTFT,
> prompt TPS, decode TPS, and memory at 32k/64k, so sustained mode reverted to
> 2048 for both dense and repage defaults.

## Headline

Bumping `MTPLX_PREFILL_CHUNK_SIZE_DENSE` and `MTPLX_PREFILL_CHUNK_SIZE_REPAGE`
from 2048 -> 4096 yields a clean win on the 40-pass context-growth benchmark.

## Comparison

|                  | chunk 2048 (prior) | chunk 4096 (this run) |
|------------------|--------------------|-----------------------|
| Successful       | 39/40              | **40/40**             |
| Cache hits       | 39/40              | 39/40                 |
| Decode mean t/s  | 39.3               | **50.6 (+29%)**       |
| Decode max t/s   | ~55                | 61.1                  |
| TTFT mean        | 2.29s              | **1.48s (-35%)**      |
| TTFT P40 (38K)   | 1.87s              | 1.84s (flat)          |
| TTFT P30 (29K)   | 6.62s              | **1.66s (-75%)**      |
| Peak memory MB   | 38,754             | 37,865 (lower)        |

## Notable

- **TTFT cliff at >29K is gone.** Baseline jumped from 1.7s -> 6.6s when
  context crossed ~29K tokens. With chunk 4096 it stays around 1.7-1.9s
  through 38K. The bigger chunk amortizes Python loop overhead during
  prefill, which dominated TTFT at high context.
- **Decode at >25K stays in the 45-55 t/s band**, vs baseline 30-38 t/s
  range at the same contexts.
- No reliability regression - 40/40 vs 39/40 baseline.
- Memory peak slightly lower (37.9 GB vs 38.8 GB) - bigger chunks mean
  fewer cleanup/repage cycles at long ctx.

## Historical action

Initially promoted to default in `mtplx/profiles.py` SUSTAINED_PREFILL_ENV.
This was later reverted because the user-facing cold long-context path
regressed.

## Per-pass detail (chunk 4096)

- ctx range: 34 -> 38192
- TTFT (s): min=0.24 max=1.87 mean=1.48
- Decode (t/s): min=32.5 max=61.1 mean=50.6
- Peak mem (MB): max=37865 mean=30818

| # | obj | ctx | ttft | dec | mem | hit |
|---|---|---|---|---|---|---|
| 1 | Airplane | 34 | 0.24 | 32.5 | 18414 | False |
| 2 | AIFollower | 1010 | 1.01 | 45.5 | 18707 | True |
| 3 | Airport | 1995 | 1.02 | 37.0 | 19018 | True |
| 4 | Balloons | 2981 | 1.03 | 48.8 | 19411 | True |
| 5 | Birds | 3958 | 1.12 | 52.0 | 19819 | True |
| 6 | Bridges | 4937 | 1.17 | 52.6 | 20404 | True |
| 7 | Building | 5915 | 1.20 | 53.7 | 20934 | True |
| 8 | Car | 6891 | 1.22 | 56.7 | 21642 | True |
| 9 | Castle | 7869 | 1.24 | 61.1 | 22283 | True |
| 10 | Checkpoints | 8847 | 1.26 | 49.4 | 23047 | True |
| 11 | City | 9828 | 1.32 | 45.9 | 23961 | True |
| 12 | Clouds | 10810 | 1.31 | 53.5 | 24798 | True |
| 13 | Coins | 11787 | 1.36 | 53.8 | 25779 | True |
| 14 | ControlTower | 12765 | 1.42 | 60.0 | 26823 | True |
| 15 | Drone | 13746 | 1.46 | 57.5 | 27978 | True |
| 16 | ExplosionEffect | 14725 | 1.49 | 55.6 | 29093 | True |
| 17 | Fence | 15704 | 1.52 | 57.1 | 30376 | True |
| 18 | Fireworks | 16682 | 1.54 | 51.2 | 31631 | True |
| 19 | Helicopter | 17660 | 1.56 | 54.7 | 33004 | True |
| 20 | Highway | 18638 | 1.58 | 55.6 | 34458 | True |
| 21 | Hills | 19615 | 1.59 | 48.4 | 35789 | True |
| 22 | HotAirBalloon | 20594 | 1.59 | 54.0 | 36426 | True |
| 23 | House | 21574 | 1.59 | 55.6 | 36426 | True |
| 24 | Lighthouse | 22552 | 1.59 | 53.7 | 36612 | True |
| 25 | Mountains | 23530 | 1.60 | 53.2 | 36890 | True |
| 26 | Pine | 24506 | 1.62 | 53.3 | 36890 | True |
| 27 | PowerLines | 25485 | 1.64 | 54.9 | 36890 | True |
| 28 | Rain | 26462 | 1.64 | 48.5 | 36890 | True |
| 29 | Refinery | 27440 | 1.64 | 51.4 | 36991 | True |
| 30 | River | 28417 | 1.74 | 49.2 | 36991 | True |
| 31 | Road | 29394 | 1.75 | 49.1 | 37095 | True |
| 32 | RoadCones | 30373 | 1.73 | 48.6 | 37095 | True |
| 33 | Rocks | 31350 | 1.76 | 47.1 | 37095 | True |
| 34 | Runway | 32327 | 1.73 | 47.9 | 37370 | True |
| 35 | Sky | 33304 | 1.78 | 45.3 | 37370 | True |
| 36 | Stadium | 34281 | 1.79 | 47.3 | 37370 | True |
| 37 | Streetlights | 35259 | 1.83 | 47.5 | 37370 | True |
| 38 | Tank | 36236 | 1.80 | 43.2 | 37865 | True |
| 39 | Trains | 37215 | 1.87 | 47.2 | 37865 | True |
| 40 | Trees | 38192 | 1.81 | 45.7 | 37865 | True |
