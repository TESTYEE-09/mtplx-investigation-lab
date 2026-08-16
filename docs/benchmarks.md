# Benchmarks

Every benchmark claim should record:

- hardware and RAM
- macOS version
- model and quantization
- sampler settings
- prompt suite
- token count
- profile
- fan mode
- date and commit

Separate cold headline runs from sustained no-fan runs and fan-controlled diagnostics.

```bash
mtplx bench run --suite cold-long-code-192 --max-tokens 192 --strict-cold
mtplx bench run --suite flappy --max-tokens 10000 --no-fanmax
```
