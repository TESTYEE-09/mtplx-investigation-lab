#!/usr/bin/env bash
set -euo pipefail

curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mtplx",
    "max_tokens": 128,
    "system": "Be concise.",
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "Write one sentence about native MTP."}]
      }
    ]
  }'
