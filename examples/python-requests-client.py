from __future__ import annotations

import requests


response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "mtplx",
        "messages": [
            {
                "role": "user",
                "content": "Return a compact JSON object with one greeting key.",
            }
        ],
        "max_tokens": 128,
    },
    timeout=120,
)
response.raise_for_status()

payload = response.json()
print(payload["choices"][0]["message"]["content"])
