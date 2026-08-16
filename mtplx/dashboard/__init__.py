"""MTPLX live dashboard package.

This package is intentionally tiny and *must not* import MLX, the MTPLX
runtime, or any other heavy module at import time. The CLI uses it as a
lightweight browser opener (``mtplx dashboard``) that probes ``/health`` and
calls ``webbrowser.open``. The actual SPA is shipped as static files in
``mtplx/dashboard/_static/`` and mounted by the running server.

The import-discipline guard below makes the contract enforceable: anyone
who breaks it (by adding ``import mlx`` here or in a top-level helper)
will see the test in ``tests/test_dashboard_endpoints.py`` fail.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_FORBIDDEN_AT_IMPORT_TIME = (
    "mlx",
    "mlx.core",
    "mlx_lm",
    "mtplx.runtime",
    "mtplx.generation",
)


def _assert_no_heavy_imports() -> None:
    """Fail loud if a contributor accidentally pulls MLX into this package."""

    bad = sorted(name for name in _FORBIDDEN_AT_IMPORT_TIME if name in _sys.modules)
    if bad:
        raise RuntimeError(
            "mtplx.dashboard must not import "
            + ", ".join(bad)
            + " at import time; keep this package thin so `mtplx dashboard`"
            " stays sub-second."
        )


DASHBOARD_STATIC_DIR: _Path = _Path(__file__).resolve().parent / "_static"


def has_static_bundle() -> bool:
    """Return True iff a built SPA bundle exists on disk."""

    return DASHBOARD_STATIC_DIR.joinpath("index.html").is_file()


__all__ = ["DASHBOARD_STATIC_DIR", "has_static_bundle"]
