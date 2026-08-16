# Prepared session-bank/postcommit accumulation A/B

Status: protocol only. No session-bank or postcommit behavior was changed.

## Question

Determine whether the observed approximately 1.64 GiB growth is:

1. retained SessionBank entries,
2. an async postcommit snapshot/future kept alive after bank supersession, or
3. MLX allocator cache rather than active tensors.

## Primary retention A/B

Use the exact production server command, model, profile, prompt sequence, and
session ID in both arms. Restart the server before each arm.

- A: shipped configuration: async postcommit and the default per-session
  retention cap (`MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES=3`).
- B: async postcommit unchanged, but set
  `MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES=1`.

This is a runtime retention-only isolation. It does not disable the bank,
postcommit, restores, or model features. If the extra 1.64 GiB tracks the
second/third retained entry in A and disappears in B after postcommit settles,
the bank owns the accumulation. Throughput and next-turn restore time must also
be recorded because cap=1 may reduce divergent-prefix coverage.

## Postcommit-lifetime follow-up

Only after the retention A/B, repeat cap=1 with:

- A: `--session-postcommit-mode async`
- B: `--session-postcommit-mode inline`

Inline is a scheduling/lifetime diagnostic, not a postcommit-off switch. The
current server exposes only `async` and `inline`; claiming that inline disables
postcommit would be incorrect. A true no-postcommit arm would require a new
test-only gate and is intentionally not added yet.

## Sampling points per turn

Collect all counters at:

1. server idle before the request,
2. response completion before async postcommit settles,
3. pending postcommit completion,
4. immediately before the next request,
5. after the final turn and a 10-second idle window.

At every point record:

- MLX active, peak, and allocator-cache bytes,
- memory attribution: model weights, `session_bank_bytes`, and generation
  working bytes,
- `session_bank.entries`, `total_nbytes`, and every prefix's `nbytes`,
  `has_live_ref`, `has_mtp_history_live_ref`, `live_ref_only`, and `lazy_kv`,
- session `pending_postcommit`, outcome, prefix length, and boundary count,
- process RSS and compressed-memory pressure,
- TTFT, decode tok/s, verify time, draft time, and restored/cached tokens.

The server already exposes the required bank ledger through
`SessionBank.to_dict()` and live MLX/memory-attribution counters; no production
instrumentation change is needed for the first A/B.

## Interpretation and acceptance gates

- Bank ledger and active memory both fall by about 1.64 GiB in cap=1: retained
  snapshots are causal.
- Ledger falls but active memory does not, while allocator-cache bytes rise:
  buffers were released but remain in MLX's reusable cache.
- A transient extra allocation exists only while async postcommit is pending
  and disappears in inline: the future/job lifetime is causal.
- Extra active memory remains after pending work completes and is absent only
  after a clean server restart: inspect references held by postcommit results,
  live leases, or queued cold-tier work before changing policy.

Do not accept a RAM arm if median D1/D2 throughput drops by more than 1%, token
streams/acceptance change, or next-turn TTFT regresses materially. Do not use
`mx.clear_cache()` as the optimization; it is diagnostic only.

