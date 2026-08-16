#!/usr/bin/env bash
set -euo pipefail

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx","messages":[{"role":"user","content":"hello"}],"stream":true}'
