"""Smoke tests for ``mtplx.ui.download_progress``.

The rich progress bar is hard to assert against frame-by-frame, but we can
pin the public API contract:

* ``start``, ``update``, ``complete`` are idempotent and never raise.
* ``update`` before ``start`` and after ``stop`` are no-ops.
* When rich is unavailable the class degrades to plain-stdlib status prints.
* The ``hf_loader`` event-shape adapter routes start/progress/complete to
  the right bar lifecycle and never lets a UX glitch break the download.
"""

from __future__ import annotations

import builtins
import sys

from mtplx.ui.download_progress import (
    RichDownloadProgress,
    from_progress_event_callback,
    _format_bytes,
)


def test_format_bytes_handles_known_scales():
    assert _format_bytes(0) == "0 B"
    assert _format_bytes(999) == "999 B"
    assert _format_bytes(1_000) == "1.0 KB"
    assert _format_bytes(1_500_000) == "1.5 MB"
    assert _format_bytes(20_400_000_000) == "20.4 GB"


def test_format_bytes_returns_placeholder_for_none():
    assert _format_bytes(None) == "? B"


def test_progress_lifecycle_runs_clean(capsys):
    progress = RichDownloadProgress(
        repo_id="mtplx/example",
        total_bytes=10_000_000,
    )
    progress.start(started_bytes=0)
    progress.update(2_500_000)
    progress.update(7_500_000)
    progress.complete(final_bytes=10_000_000)

    out = capsys.readouterr().out
    # Don't assert on rich's exact bar glyphs — it changes between rich
    # versions. Just confirm the repo identifier ended up in the output
    # somewhere, and the lifecycle didn't crash.
    assert "mtplx/example" in out


def test_progress_with_unknown_total_does_not_raise(capsys):
    progress = RichDownloadProgress(
        repo_id="mtplx/example",
        total_bytes=None,
    )
    progress.start(started_bytes=0)
    progress.update(1_234_567)
    progress.complete(final_bytes=12_345_678)
    out = capsys.readouterr().out
    assert "mtplx/example" in out


def test_progress_resume_path_does_not_raise():
    progress = RichDownloadProgress(
        repo_id="mtplx/example",
        total_bytes=10_000_000_000,
    )
    progress.start(started_bytes=4_000_000_000)
    progress.update(5_000_000_000)
    progress.complete(final_bytes=10_000_000_000)


def test_update_before_start_is_a_noop(capsys):
    progress = RichDownloadProgress(repo_id="mtplx/example", total_bytes=1024)
    progress.update(512)  # before start: silently ignored
    progress.update(700)
    out = capsys.readouterr().out
    assert out == ""


def test_stop_is_idempotent_and_safe_to_call_twice():
    progress = RichDownloadProgress(repo_id="mtplx/example", total_bytes=2048)
    progress.start(started_bytes=0)
    progress.stop(final_bytes=2048)
    # Subsequent stops must not raise. Calling stop without final_bytes
    # also must not raise after a previous stop with final_bytes.
    progress.stop()
    progress.stop(final_bytes=2048)


def test_update_after_stop_is_a_noop(capsys):
    progress = RichDownloadProgress(repo_id="mtplx/example", total_bytes=2048)
    progress.start(started_bytes=0)
    progress.stop(final_bytes=2048)
    capsys.readouterr()  # consume whatever start/stop printed
    progress.update(1024)
    out_after = capsys.readouterr().out
    assert out_after == ""  # update after stop is silently ignored


def test_progress_falls_back_when_rich_is_missing(monkeypatch, capsys):
    """When ``rich`` isn't importable at all, the class falls back to the
    plain stdlib path — same surface, same idempotency, no exceptions."""
    real_import = builtins.__import__

    def fail_rich(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("rich removed for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_rich)
    # Drop any cached rich submodules so the import-failure simulation
    # reaches RichDownloadProgress.__init__.
    for key in list(sys.modules):
        if key == "rich" or key.startswith("rich."):
            sys.modules.pop(key, None)

    progress = RichDownloadProgress(
        repo_id="mtplx/example",
        total_bytes=1_000_000_000,
    )
    progress.start(started_bytes=0)
    progress.update(500_000_000)
    progress.complete(final_bytes=1_000_000_000)

    out = capsys.readouterr().out
    # Fallback path emits explicit "downloading" and "download complete" text.
    assert "downloading mtplx/example" in out
    assert "download complete" in out


def test_event_adapter_routes_start_progress_complete():
    """The adapter that converts ``hf_loader`` events into bar calls must
    drive the bar's lifecycle correctly: start/resume → start, progress →
    update, complete → complete. ``start`` should also adopt a
    ``total_bytes`` from the event payload when the bar didn't have one."""
    progress = RichDownloadProgress(repo_id="mtplx/example", total_bytes=None)
    callback = from_progress_event_callback(progress=progress)

    callback({"event": "start", "size_bytes": 0, "total_bytes": 5_000_000_000})
    assert progress.total_bytes == 5_000_000_000

    callback({"event": "progress", "size_bytes": 1_000_000_000})
    callback({"event": "progress", "size_bytes": 4_500_000_000})
    callback({"event": "complete", "size_bytes": 5_000_000_000})


def test_event_adapter_swallows_exceptions():
    """A UX glitch in the bar must never break the underlying download."""

    class Boom(RichDownloadProgress):
        def update(self, current_bytes: int) -> None:
            raise RuntimeError("simulated rich crash")

    progress = Boom(repo_id="mtplx/example", total_bytes=1024)
    progress.start(started_bytes=0)

    captured: list[str] = []

    def capture_fallback(msg: str) -> None:
        captured.append(msg)

    callback = from_progress_event_callback(
        progress=progress, fallback_printer=capture_fallback
    )
    # Should not raise even though ``update`` blows up.
    callback({"event": "progress", "size_bytes": 512})
    assert any("simulated rich crash" in c for c in captured)


def test_event_adapter_silently_ignores_unknown_events():
    progress = RichDownloadProgress(repo_id="mtplx/example", total_bytes=1024)
    progress.start(started_bytes=0)
    callback = from_progress_event_callback(progress=progress)
    # Random / future event types should not raise.
    callback({"event": "metrics", "metric": "queue_depth", "value": 7})
    callback({})
    callback({"event": None})
