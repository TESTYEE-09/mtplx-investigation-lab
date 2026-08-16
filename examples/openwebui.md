# Open WebUI

Start MTPLX:

```bash
mtplx serve --host 127.0.0.1 --port 8000 --no-stats-footer
```

In Open WebUI, set the OpenAI-compatible base URL to:

```text
http://127.0.0.1:8000/v1
```

Use a local API key only if you started the server with `--api-key`.

Dockerized Open WebUI must talk back to MTPLX through the host gateway:

```bash
mtplx openwebui docker-command
```

For an isolated smoke test with the Open WebUI Python package:

```bash
DATA_DIR=/tmp/mtplx-openwebui \
HF_HOME=/tmp/mtplx-openwebui-hf \
WEBUI_SECRET_KEY=replace-with-a-local-test-key \
WEBUI_AUTH=True \
ENABLE_SIGNUP=True \
ENABLE_OLLAMA_API=False \
ENABLE_OPENAI_API=True \
OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1 \
OPENAI_API_KEYS=local-test \
ENABLE_TITLE_GENERATION=False \
ENABLE_TAGS_GENERATION=False \
ENABLE_FOLLOW_UP_GENERATION=False \
ENABLE_AUTOCOMPLETE_GENERATION=False \
DEFAULT_MODELS=mtplx-qwen36-27b-optimized-speed \
uvx --python 3.11 --from open-webui open-webui serve --host 127.0.0.1 --port 8080
```

`--no-stats-footer` keeps benchmark text out of assistant messages. MTPLX still exposes runtime details through `/metrics`.
The smoke disables Open WebUI's Ollama probe and background title, tag, follow-up, and autocomplete generations so the only model load is the visible chat turn.

The smoke above is pinned to Python 3.11 because one tested Open WebUI package path failed under Python 3.13 on `audioop` / `pyaudioop` import compatibility before the server started.
