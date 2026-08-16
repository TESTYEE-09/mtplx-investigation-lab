# MTPLX v0.2.1

v0.2.1 is the current v0.2 release line: it includes the full v0.2.0 fast
prefill and agent-client update, plus an urgent safety fix for the public
server quickstart path.

## Highlights

- Sustained long-context prefill is now the default public path for `mtplx
  start`, `quickstart`, `serve`, and benchmark commands.
- Long-context prompt processing is dramatically faster than v0.1.6 while
  staying on the bounded-memory Sustained route used for the 32K/64K/128K M5
  Max release QA.
- `mtplx start pi` connects MTPLX to Pi: it writes Pi's model config, starts the
  local OpenAI-compatible MTPLX server, and opens Pi automatically.
- Pi mode keeps the original MTPLX terminal useful with live server controls:
  `/reasoning`, `/mtp`, `/stats`, and `/help`.
- OpenAI-compatible streaming now works correctly when clients send `tools`:
  normal text still streams incrementally, and real tool calls stream as
  structured `delta.tool_calls` instead of raw model markup.
- SessionBank has safer async postcommit behavior for tool-call responses and
  configurable capacity environment variables.

## v0.2.1 Hotfix

- `mtplx quickstart --max`, `mtplx serve --max`, and `mtplx start --max` now keep
  the v0.2 Sustained Max default even when an older `~/.mtplx/config.toml`
  contains `profile = "performance-cold"`.
- Non-Sustained MTP prefill above 16K prompt tokens now fails with a clear
  configuration error instead of taking the full hidden/logits path that can
  allocate hundreds of GB at 64K+ context.

## Immediate User Guidance

For long-context server benchmarks, use:

```bash
mtplx config set profile sustained
mtplx quickstart --profile sustained --max
```

`--profile performance-cold --max` remains available as the short-context Burst
lane, but it is not the long-context v0.2 Sustained prefill path.
