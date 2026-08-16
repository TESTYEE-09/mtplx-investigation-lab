"""Tests for the bounded session-local wait for prior postcommit work.

The wait is the rebuild of closed PR #34 against the new model scheduler. It
sits on EngineSession.wait_for_pending_postcommit(), reads a Future stashed
when the previous turn's postcommit was scheduled, and is bounded so a stuck
postcommit never blocks the foreground request indefinitely.

Critical invariants exercised here:
  - HIT  : the wait completes when the postcommit lands within the timeout
  - SKIP : the wait short-circuits to "no_pending" when nothing was scheduled
  - BAIL : the wait reports "timeout" without raising and the request can
           proceed cold
  - ORDER: the wait must be done WITHOUT the session lock held - otherwise a
           same-session concurrent caller (or scheduler-bound work) can
           deadlock against us. We verify both that wait_for_pending_postcommit
           does not itself acquire the session lock and that the lock is
           reusable while a future is pending.
  - STATS: the wait emits a structured outcome dict the server can attach to
           request observability and surface in /metrics.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from threading import Thread

import pytest

from mtplx.engine_session import (
    EngineSession,
    _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S,
    _postcommit_wait_timeout_s,
)


def _new_session(sid: str = "sess-test") -> EngineSession:
    return EngineSession(sid)


def test_wait_no_pending_short_circuits_without_blocking() -> None:
    session = _new_session()
    t0 = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=5.0)
    elapsed = time.monotonic() - t0

    assert outcome["outcome"] == "no_pending"
    assert outcome["waited"] is False
    assert outcome["elapsed_s"] == 0.0
    # Should be immediate; "no_pending" must never block.
    assert elapsed < 0.05
    assert session.last_postcommit_wait == outcome


def test_wait_completes_when_postcommit_lands_inside_timeout() -> None:
    session = _new_session()
    future: Future = Future()
    session.set_pending_postcommit(future)

    def producer() -> None:
        time.sleep(0.05)
        future.set_result("committed")

    Thread(target=producer, daemon=True).start()
    outcome = session.wait_for_pending_postcommit(timeout_s=2.0)

    assert outcome["outcome"] == "completed"
    assert outcome["waited"] is True
    assert 0.04 < outcome["elapsed_s"] < 1.5
    # Reference must be cleared so a follow-up call does not double-wait.
    assert session.pending_postcommit is None


def test_wait_times_out_without_raising_and_request_can_proceed_cold() -> None:
    session = _new_session()
    future: Future = Future()  # never resolved
    session.set_pending_postcommit(future)

    t0 = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=0.1)
    elapsed = time.monotonic() - t0

    assert outcome["outcome"] == "timeout"
    assert outcome["waited"] is True
    assert outcome["abort_requested"] is True
    assert outcome["abort_reason"] == "foreground_preempted_postcommit"
    # Bounded: should not exceed ~3x the timeout under load.
    assert elapsed < 0.5
    # The future is dropped even on timeout so we do not re-wait next turn.
    assert session.pending_postcommit is None
    # Caller can now proceed; nothing in our state forces a retry.


def test_wait_swallows_postcommit_exceptions() -> None:
    session = _new_session()
    future: Future = Future()
    future.set_exception(RuntimeError("scheduler exploded"))
    session.set_pending_postcommit(future)

    outcome = session.wait_for_pending_postcommit(timeout_s=1.0)

    # We do not surface the underlying error to the foreground request - the
    # wait is a best-effort cache warmup, not a correctness dependency.
    assert outcome["outcome"].startswith("error:")
    assert "RuntimeError" in outcome["outcome"]
    assert outcome["waited"] is True


def test_wait_disabled_when_timeout_zero_or_negative() -> None:
    session = _new_session()
    future: Future = Future()
    session.set_pending_postcommit(future)

    outcome = session.wait_for_pending_postcommit(timeout_s=0.0)
    assert outcome["outcome"] == "disabled"
    assert outcome["waited"] is False
    # The future MUST stay visible on the session: "disabled" means we did
    # not observe the future, so we cannot claim ownership over clearing
    # it. If the operator re-enables waiting on the next request, that
    # request must still find the future and wait on it. (PR #37 review:
    # only clear `pending_postcommit` after a real resolve/timeout.)
    assert session.pending_postcommit is future


def test_wait_does_not_acquire_session_lock_so_no_foreground_deadlock() -> None:
    """The wait MUST run without holding the session lock.

    If `wait_for_pending_postcommit` ever acquired the session lock, a
    same-session concurrent foreground request that already holds the lock
    would deadlock against the wait, AND any scheduler-bound postcommit
    work that needs the lock to commit would deadlock against the waiter.

    We verify by holding the session lock from another thread and checking
    that the wait still progresses (it does not contend on the lock). We
    also verify the wait does NOT count as in-flight on the session.
    """
    session = _new_session()
    # Hold the lock from another thread to simulate a same-session
    # concurrent request that has already entered in_flight_generation().
    with session._lock:
        future: Future = Future()
        session.set_pending_postcommit(future)

        def land_postcommit() -> None:
            time.sleep(0.05)
            future.set_result(None)

        Thread(target=land_postcommit, daemon=True).start()
        # If the wait needed the lock, this would block until the outer
        # `with` releases it - we never do, so the call would hang and the
        # test would time out via pytest's outer timeout.
        outcome = session.wait_for_pending_postcommit(timeout_s=2.0)

    assert outcome["outcome"] == "completed"
    assert session.in_flight is False


def test_wait_records_outcome_in_admin_dict_for_metrics_endpoint() -> None:
    session = _new_session()
    future: Future = Future()
    future.set_result(None)
    session.set_pending_postcommit(future)
    session.wait_for_pending_postcommit(timeout_s=1.0)

    admin = session.to_admin_dict()
    assert admin["last_postcommit_wait"] is not None
    assert admin["last_postcommit_wait"]["outcome"] == "completed"
    # No future left dangling once we have observed it.
    assert admin["pending_postcommit"] is False


def test_set_pending_postcommit_overwrites_prior_reference() -> None:
    """A second commit on the same session supersedes the first.

    Only the most recent postcommit's bank entry matters for the next
    request's lookup, so we drop older references rather than chaining.
    """
    session = _new_session()
    first: Future = Future()
    second: Future = Future()
    second.set_result(None)
    session.set_pending_postcommit(first)
    session.set_pending_postcommit(second)

    outcome = session.wait_for_pending_postcommit(timeout_s=0.5)
    assert outcome["outcome"] == "completed"
    # First future is never observed by the wait, but is also no longer
    # referenced by the session.
    assert session.pending_postcommit is None


def test_postcommit_wait_timeout_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    assert _postcommit_wait_timeout_s() == _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S


@pytest.mark.parametrize("raw,expected", [
    ("0", 0.0),
    ("0.5", 0.5),
    ("3", 3.0),
    ("15", 15.0),
])
def test_postcommit_wait_timeout_env_override(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raw)
    assert _postcommit_wait_timeout_s() == expected


def test_postcommit_wait_timeout_env_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", "not-a-number")
    assert _postcommit_wait_timeout_s() == _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S


def test_postcommit_wait_timeout_env_negative_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", "-1")
    assert _postcommit_wait_timeout_s() == 0.0


def test_in_flight_generation_does_not_implicitly_wait_on_postcommit() -> None:
    """The wait is the chat handler's responsibility, not the session lock.

    The OLD PR #34 implementation wove the wait into the in_flight_generation
    context manager. That coupling is fragile because in_flight_generation is
    also used by code paths that legitimately do not have a session-bound
    postcommit (e.g. tests). We keep the wait at the call site and verify
    here that EngineSession's lock acquisition is independent.
    """
    session = _new_session()
    future: Future = Future()  # never resolves
    session.set_pending_postcommit(future)

    # Entering in_flight_generation must not block on the future.
    started = time.monotonic()
    with session.in_flight_generation():
        elapsed = time.monotonic() - started
        assert elapsed < 0.1
        assert session.in_flight is True
    assert session.in_flight is False
    # The future is still pending on the session for the next caller to
    # observe via wait_for_pending_postcommit().
    assert session.pending_postcommit is future


def test_wait_then_lock_then_wait_again_simulates_two_turn_sequence() -> None:
    """End-to-end: turn 1 schedules postcommit, turn 2 waits-then-locks."""
    session = _new_session()

    # Turn 1: simulate that a postcommit was scheduled at the end of the turn.
    turn1_future: Future = Future()
    session.set_pending_postcommit(turn1_future)

    def land_turn1_postcommit() -> None:
        time.sleep(0.05)
        turn1_future.set_result(None)

    Thread(target=land_turn1_postcommit, daemon=True).start()

    # Turn 2: BEFORE acquiring the session lock, wait for turn 1's commit.
    pre_lock = session.wait_for_pending_postcommit(timeout_s=2.0)
    assert pre_lock["outcome"] == "completed"

    # Now turn 2 enters in_flight_generation and runs.
    with session.in_flight_generation():
        assert session.in_flight is True

    # Turn 2 schedules its own postcommit at the end.
    turn2_future: Future = Future()
    session.set_pending_postcommit(turn2_future)
    assert session.pending_postcommit is turn2_future


def test_wait_for_pending_postcommit_concurrent_same_session() -> None:
    """Two concurrent same-session waiters must observe the same future.

    Regression for the PR #37 review bug: the original implementation cleared
    `self.pending_postcommit` BEFORE waiting on it. A second concurrent
    same-session caller saw the field empty and returned ``no_pending`` even
    while the postcommit was still in flight, defeating the cache-warmup
    purpose of the wait.

    The fix is:
      - keep the future visible on the session until resolve OR timeout,
      - clear it only if it is still the same future identity after the
        wait completes,
      - guard reads/writes with a per-session lock.

    This test exercises the three invariants the maintainer asked for:

    (a) on resolve, BOTH waiters see ``completed``;
    (b) ``no_pending`` is NEVER reported while the future is unresolved;
    (c) on timeout, BOTH waiters see ``timeout`` deterministically.
    """
    # --- (a) + (b): both waiters see "completed" on resolve, neither sees
    # "no_pending" while the future is still in flight.
    session = _new_session("concurrent-resolve")
    future: Future = Future()
    session.set_pending_postcommit(future)

    outcomes: dict[int, dict] = {}
    start_barrier = threading.Barrier(2)

    def waiter(idx: int) -> None:
        # Both waiters arrive at the wait call as close to simultaneously as
        # we can arrange in CPython, so they both observe the same active
        # future on the session.
        start_barrier.wait(timeout=2.0)
        outcomes[idx] = session.wait_for_pending_postcommit(timeout_s=2.0)

    threads = [Thread(target=waiter, args=(i,), daemon=True) for i in (0, 1)]
    for t in threads:
        t.start()

    # Give both waiters a chance to enter the wait. While they are waiting,
    # `pending_postcommit` MUST stay visible on the session - the original
    # bug was that the first waiter cleared it up front, causing the second
    # waiter (or a third concurrent peek) to see "no_pending".
    deadline = time.monotonic() + 0.5
    saw_pending_during_wait = False
    while time.monotonic() < deadline and not future.done():
        if session.pending_postcommit is future:
            saw_pending_during_wait = True
        time.sleep(0.005)
    # The wait should still be in flight at this point; the future has not
    # been resolved yet, so the field MUST still point at it.
    assert saw_pending_during_wait, (
        "pending_postcommit was cleared before the future resolved - this is "
        "exactly the race PR #37 review flagged"
    )
    assert session.pending_postcommit is future

    # Resolve the future and let both waiters complete.
    future.set_result("done")
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "waiter thread did not return in time"

    assert outcomes[0]["outcome"] == "completed", outcomes
    assert outcomes[1]["outcome"] == "completed", outcomes
    # Neither waiter must report no_pending while the future was active.
    assert outcomes[0]["outcome"] != "no_pending"
    assert outcomes[1]["outcome"] != "no_pending"
    # After both have observed the resolve, the field is cleared exactly
    # once (by whichever waiter re-acquired the postcommit lock first); the
    # other no-ops because identity no longer matches.
    assert session.pending_postcommit is None

    # --- (c): on timeout, both waiters observe "timeout" deterministically.
    timeout_session = _new_session("concurrent-timeout")
    stuck_future: Future = Future()  # never resolves
    timeout_session.set_pending_postcommit(stuck_future)

    timeout_outcomes: dict[int, dict] = {}
    timeout_barrier = threading.Barrier(2)

    def timeout_waiter(idx: int) -> None:
        timeout_barrier.wait(timeout=2.0)
        timeout_outcomes[idx] = timeout_session.wait_for_pending_postcommit(
            timeout_s=0.1
        )

    timeout_threads = [
        Thread(target=timeout_waiter, args=(i,), daemon=True) for i in (0, 1)
    ]
    started = time.monotonic()
    for t in timeout_threads:
        t.start()
    for t in timeout_threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "timeout waiter did not return"
    elapsed = time.monotonic() - started

    assert timeout_outcomes[0]["outcome"] == "timeout", timeout_outcomes
    assert timeout_outcomes[1]["outcome"] == "timeout", timeout_outcomes
    # Both must have actually waited (elapsed_s > 0) and reported the
    # configured timeout in their envelope - the metric shape is unchanged.
    for outcome in timeout_outcomes.values():
        assert outcome["waited"] is True
        assert outcome["timeout_s"] == 0.1
        assert outcome["abort_requested"] is True
        assert outcome["abort_reason"] == "foreground_preempted_postcommit"
        assert set(outcome.keys()) == {
            "waited",
            "elapsed_s",
            "outcome",
            "timeout_s",
            "abort_requested",
            "future_cancelled",
            "abort_reason",
        }
    # Bounded: both waiters must finish within a small multiple of the
    # timeout - if either had observed "no_pending" early it would have
    # returned ~immediately, but more importantly neither must hang past
    # the timeout.
    assert elapsed < 1.0, f"timeout waiters took too long: {elapsed:.3f}s"
    # After timeout, the field is cleared (one of the waiters won the
    # identity race; the other no-oped).
    assert timeout_session.pending_postcommit is None
