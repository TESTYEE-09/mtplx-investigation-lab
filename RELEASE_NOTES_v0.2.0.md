# MTPLX v0.2.0

v0.2.0 is the fast-prefill and agent-client release.

## Highlights

- Sustained is now the default long-context public path for `mtplx start`,
  `quickstart`, `serve`, and benchmark commands unless Burst is explicitly
  selected.
- Long-context Sustained prompt prefill is substantially faster while keeping
  bounded memory and zero large-query split fallback in release QA.
- Pi is now a first-class onboarding target: `mtplx start pi` configures Pi,
  starts the local OpenAI-compatible server, and keeps `/reasoning`, `/mtp`,
  `/stats`, and `/help` controls available in the MTPLX terminal.
- OpenAI streaming with `tools` present now streams normal text incrementally,
  and actual tool-call responses emit structured `delta.tool_calls` chunks.
- Async SessionBank postcommit now runs for tool-call responses after the
  foreground request goes idle, preserving Daniel Farina's PR #17 contribution.

## Targeted Issue Fixes

- Fixes #9: streamed tool-call arguments are chunked instead of sent as one
  complete JSON blob.
- Fixes #13: requests that include a `tools` array but produce normal content
  continue to stream `delta.content` incrementally.
- Fixes #15: Pi setup and streaming are now covered by first-class CLI
  onboarding and compatibility QA.

## Release QA Snapshot

Fresh M5 Max release QA was run from the v0.2.0 integration worktree with
`--profile sustained --max`, 128 generated tokens, and
`MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK=1`.

| Context | Prompt TPS | Decode TPS | Peak Memory | Fallback Calls |
|---:|---:|---:|---:|---:|
| 32k | 620.6 tok/s | 39.1 tok/s | 22.1 GB | 0 |
| 64k | 504.3 tok/s | 31.2 tok/s | 27.2 GB | 0 |
| 128k | 372.1 tok/s | 25.3 tok/s | 37.5 GB | 0 |

The JSON artifact is
`benchmarks/results/v0.2.0-release-m5max-32k-64k-128k.json`.

Direct OpenAI-compatible streaming QA passed for tools-present normal content
and forced tool-call streaming. Pi 0.74.0 was also verified against a live local
MTPLX server with incremental `message_update` / `text_delta` events.

## Release Honesty

- This release does not claim Gemma runtime support.
- This release does not claim continuous batching.
- This release does not claim direct M5 Neural Accelerator use; eligibility is
  reported separately from proof.
- Long no-fan decode decay remains a future runtime track.
