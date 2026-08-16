#!/usr/bin/env bash
set -euo pipefail

mtplx bench run \
  --suite flappy \
  --max-tokens 10000 \
  --no-fanmax
