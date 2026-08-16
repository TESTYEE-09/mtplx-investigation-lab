"""Integration tests for the parent-process Max-mode idle watchdog.

The watchdog reads activity state from the server's ``/health`` endpoint,
drops fans to ``silent`` after ``idle_seconds`` of no foreground work, and
ramps back to ``performance`` as soon as the next request starts.

We exercise the state machine without real HTTP / real ThermalForge by
stubbing both ``_http_json`` and ``set_thermal_profile``.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def watchdog_module():
    from mtplx.commands import public

    return public


def _make_watchdog(public, *, idle_seconds: int = 1, poll_seconds: int = 1):
    return public._MaxIdleWatchdog(
        host="127.0.0.1",
        port=8000,
        idle_seconds=idle_seconds,
        poll_seconds=poll_seconds,
    )


def test_watchdog_drops_to_silent_after_idle_window(monkeypatch, watchdog_module):
    """No new ``requests_completed`` for ``idle_seconds`` → ``silent``."""

    health_responses = iter(
        [
            {"ok": True, "requests_completed": 5},
            {"ok": True, "requests_completed": 5},
            {"ok": True, "requests_completed": 5},
        ]
    )
    profile_calls: list[str] = []

    def fake_http_json(url, *, timeout=2.0):
        try:
            return next(health_responses)
        except StopIteration:
            return {"ok": True, "requests_completed": 5}

    def fake_set_profile(profile):
        profile_calls.append(profile)
        return {"ok": True, "detection": {"available": True}}

    monkeypatch.setattr(watchdog_module, "_http_json", fake_http_json)
    monkeypatch.setattr("mtplx.thermal.set_thermal_profile", fake_set_profile)

    watchdog = _make_watchdog(watchdog_module, idle_seconds=1, poll_seconds=1)
    watchdog.start()
    time.sleep(2.5)
    watchdog.stop()

    assert "silent" in profile_calls, profile_calls


def test_watchdog_ramps_back_to_performance_on_new_activity(monkeypatch, watchdog_module):
    """After dropping to silent, a new completed request must trigger a
    ramp back to ``performance`` so the next chat turn is fast."""

    state = {"completed": 5, "activity_after": 3.0}
    profile_calls: list[str] = []
    started_at = time.time()

    def fake_http_json(url, *, timeout=2.0):
        elapsed = time.time() - started_at
        if elapsed >= state["activity_after"]:
            state["completed"] = 6
        return {"ok": True, "requests_completed": state["completed"]}

    def fake_set_profile(profile):
        profile_calls.append(profile)
        return {"ok": True, "detection": {"available": True}}

    monkeypatch.setattr(watchdog_module, "_http_json", fake_http_json)
    monkeypatch.setattr("mtplx.thermal.set_thermal_profile", fake_set_profile)

    watchdog = _make_watchdog(watchdog_module, idle_seconds=1, poll_seconds=1)
    watchdog.start()
    time.sleep(6.0)
    watchdog.stop()

    silent_index = next((i for i, p in enumerate(profile_calls) if p == "silent"), None)
    assert silent_index is not None, profile_calls
    perf_index = next(
        (
            i
            for i, p in enumerate(profile_calls[silent_index + 1 :], silent_index + 1)
            if p == "performance"
        ),
        None,
    )
    assert perf_index is not None and perf_index > silent_index, profile_calls


def test_watchdog_ramps_before_request_completion(monkeypatch, watchdog_module):
    """If a long request starts while fans are idle, ramp on active state
    instead of waiting for ``requests_completed`` to change at the end."""

    profile_calls: list[str] = []
    started_at = time.time()

    def fake_http_json(url, *, timeout=2.0):
        elapsed = time.time() - started_at
        if elapsed >= 3.0:
            return {
                "ok": True,
                "requests_completed": 5,
                "foreground_active": 1,
                "last_request_started_at": 123.0,
            }
        return {
            "ok": True,
            "requests_completed": 5,
            "foreground_active": 0,
            "last_request_started_at": 0.0,
        }

    def fake_set_profile(profile):
        profile_calls.append(profile)
        return {"ok": True, "detection": {"available": True}}

    monkeypatch.setattr(watchdog_module, "_http_json", fake_http_json)
    monkeypatch.setattr("mtplx.thermal.set_thermal_profile", fake_set_profile)

    watchdog = _make_watchdog(watchdog_module, idle_seconds=1, poll_seconds=1)
    watchdog.start()
    time.sleep(5.0)
    watchdog.stop()

    silent_index = next((i for i, p in enumerate(profile_calls) if p == "silent"), None)
    assert silent_index is not None, profile_calls
    perf_index = next(
        (
            i
            for i, p in enumerate(profile_calls[silent_index + 1 :], silent_index + 1)
            if p == "performance"
        ),
        None,
    )
    assert perf_index is not None and perf_index > silent_index, profile_calls


def test_watchdog_does_not_thrash_when_continuously_active(monkeypatch, watchdog_module):
    """Continuous activity must not call set_thermal_profile every poll —
    only on ACTIVE → IDLE and IDLE → ACTIVE transitions."""

    counter = {"n": 5}

    def fake_http_json(url, *, timeout=2.0):
        counter["n"] += 1
        return {"ok": True, "requests_completed": counter["n"]}

    profile_calls: list[str] = []

    def fake_set_profile(profile):
        profile_calls.append(profile)
        return {"ok": True, "detection": {"available": True}}

    monkeypatch.setattr(watchdog_module, "_http_json", fake_http_json)
    monkeypatch.setattr("mtplx.thermal.set_thermal_profile", fake_set_profile)

    watchdog = _make_watchdog(watchdog_module, idle_seconds=10, poll_seconds=1)
    watchdog.start()
    time.sleep(3.0)
    watchdog.stop()

    # Activity is increasing every poll, so we never go idle and never
    # trigger any profile changes from the parent.
    assert profile_calls == [], profile_calls


def test_watchdog_stop_is_idempotent(watchdog_module):
    watchdog = _make_watchdog(watchdog_module, idle_seconds=1, poll_seconds=1)
    watchdog.stop()  # never started
    watchdog.start()
    watchdog.stop()
    watchdog.stop()  # twice in a row is fine
