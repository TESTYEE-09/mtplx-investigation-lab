"""Suite-wide isolation from the developer machine's live MTPLX state.

Without these guards the suite is machine-dependent: a daemon left running
by the macOS app would make ``mtplx start`` flows offer attach prompts, the
real ``~/Library/Application Support/MTPLX/settings.json`` would inject
"same as the app" options, and the real ``~/.mtplx/models`` cache would
change picker numbering. Tests that exercise those features explicitly
override these variables with their own fixtures.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermetic_mtplx_state(monkeypatch, tmp_path_factory):
    isolated = tmp_path_factory.mktemp("hermetic-mtplx")
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "off")
    monkeypatch.setenv(
        "MTPLX_APP_SETTINGS_PATH", str(isolated / "app-settings.json")
    )
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(isolated / "models"))
