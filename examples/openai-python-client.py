from __future__ import annotations

from openai import OpenAI


client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

stream = client.chat.completions.create(
    model="mtplx",
    messages=[{"role": "user", "content": "Write a tiny TOML parser example."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
