#!/usr/bin/env python3
"""Memory-safe load smoke for the converted Step-3.7-Flash MTPLX model.

Confirms the real model loads through mtplx.runtime, the Step MTP injector
attaches, a target forward runs, and one MTP draft is produced -- plus peak
memory (the Phase 0b fail-fast gate). NOT a benchmark; acceptance/speed come next.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mtplx.runtime import load  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/Step-3.7-Flash-MTPLX-step3p5"

t0 = time.time()
rt = load(MODEL, mtp=True)
print(
    f"LOADED mtp_enabled={rt.mtp_enabled} load_s={time.time() - t0:.1f} "
    f"peak_GB={mx.get_peak_memory() / 1e9:.1f}",
    flush=True,
)

prompt = "Write a Python function that returns the nth Fibonacci number."
ids = mx.array([rt.tokenizer.encode(prompt)])
cache = rt.model.make_cache()
logits, hidden = rt.forward_ar(ids, cache=cache, return_hidden=True, hidden_variant="pre_norm")
mx.eval(logits, hidden)
nxt = mx.argmax(logits[:, -1, :], axis=-1)
print(
    f"TARGET_FORWARD_OK next_id={int(nxt[0])} tok={rt.tokenizer.decode([int(nxt[0])])!r} "
    f"hidden_shape={tuple(hidden.shape)}",
    flush=True,
)

mtp_cache = rt.model.make_mtp_cache()
dlogits, dhidden = rt.model.mtp_forward(
    hidden[:, -1:, :],
    nxt.reshape(1, 1),
    mtp_cache=mtp_cache,
    return_hidden=True,
    mtp_depth=1,
)
mx.eval(dlogits, dhidden)
dtok = int(mx.argmax(dlogits[:, -1, :], axis=-1)[0])
print(f"MTP_DRAFT_OK draft_id={dtok} tok={rt.tokenizer.decode([dtok])!r}", flush=True)
print(f"PEAK_GB={mx.get_peak_memory() / 1e9:.1f}", flush=True)
print("STEP_SMOKE_OK", flush=True)
