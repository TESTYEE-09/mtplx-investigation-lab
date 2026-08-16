"""Stability tests for the SessionBank policy/fingerprint compatibility logic.

These tests pin down the bug fix for the >=16K cache cliff: when the resolved
``mtp_history_policy`` flips from ``committed`` (sub-threshold) to
``last_window`` (>= ``MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD``), the previously
stored bank entry must remain reusable. Both policies share the same committed
mtp-history cache shape, so the entry must NOT be rejected as a policy mismatch.

The unit tests here exercise ``SessionBank.restore`` and the helper directly
without needing a real MLX runtime.
"""

from __future__ import annotations

from pathlib import Path

from mtplx.session_bank import (
    CacheMissReason,
    SessionBank,
    _mtp_history_policy_compatible,
)


class _StubRuntime:
    """Minimal runtime stub. ``put`` only reads ``model_path``/``mtp_enabled``;
    ``restore`` reads ``model_path`` and (for non-empty caches) calls ``make_cache``,
    which we never reach because our test caches are empty."""

    def __init__(self, model_path: str = "models/example") -> None:
        self.model_path = Path(model_path)
        self.mtp_enabled = True

    def make_cache(self) -> list:
        return []

    def make_mtp_cache(self) -> list:
        return []


def _seed_entry(bank: SessionBank, *, mtp_history_policy: str, tokens=(1, 2, 3, 4)) -> None:
    runtime = _StubRuntime()
    entry = bank.put(
        runtime=runtime,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        hidden_variant="post_norm",
        session_id="sess-1",
        template_hash="tmpl-abc",
        mtp_history_policy=mtp_history_policy,
        draft_head_identity="draft-xyz",
        policy_fingerprint="fp-stable",
    )
    assert entry is not None, "test setup: entry should be stored"


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_policy_compatible_equality_is_compatible():
    assert _mtp_history_policy_compatible("committed", "committed") is True
    assert _mtp_history_policy_compatible("last_window", "last_window") is True
    assert _mtp_history_policy_compatible("cycle", "cycle") is True
    assert _mtp_history_policy_compatible(None, None) is True


def test_policy_compatible_committed_and_last_window_are_interchangeable():
    # The whole point of the fix: sub-16K stores ``committed``, >=16K lookup
    # asks for ``last_window``. Both share the committed-cache shape.
    assert _mtp_history_policy_compatible("committed", "last_window") is True
    assert _mtp_history_policy_compatible("last_window", "committed") is True


def test_policy_compatible_cycle_is_not_committed_compatible():
    # ``cycle`` does NOT use the committed cache; it is structurally different
    # and must not be considered compatible.
    assert _mtp_history_policy_compatible("cycle", "committed") is False
    assert _mtp_history_policy_compatible("committed", "cycle") is False
    assert _mtp_history_policy_compatible("cycle", "last_window") is False


def test_policy_compatible_none_is_only_compatible_with_none():
    assert _mtp_history_policy_compatible(None, "committed") is False
    assert _mtp_history_policy_compatible("committed", None) is False


# ---------------------------------------------------------------------------
# Restore-level tests
# ---------------------------------------------------------------------------


def test_restore_accepts_lookup_last_window_against_stored_committed():
    """Reproduces the >=16K cliff: stored entry has ``committed`` (because the
    storing turn was sub-threshold); the next-turn lookup resolves to
    ``last_window`` (because prompt crossed the threshold). With the fix these
    are compatible and restore must succeed."""
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    _seed_entry(bank, mtp_history_policy="committed")

    restored = bank.restore(
        _StubRuntime(),
        [1, 2, 3, 4],
        hidden_variant="post_norm",
        template_hash="tmpl-abc",
        mtp_history_policy="last_window",
        draft_head_identity="draft-xyz",
        policy_fingerprint="fp-stable",
    )

    assert restored is not None, "policy mismatch cliff: lookup last_window must reuse stored committed entry"
    assert bank.last_miss_reason is None
    assert restored.entry.mtp_history_policy == "committed"


def test_restore_accepts_lookup_committed_against_stored_last_window():
    """Reverse direction: a turn at >=16K could store ``last_window`` (under
    the future fix that records the resolved policy on the entry); the next
    turn at <16K asks for ``committed``. Must still match."""
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    _seed_entry(bank, mtp_history_policy="last_window")

    restored = bank.restore(
        _StubRuntime(),
        [1, 2, 3, 4],
        hidden_variant="post_norm",
        template_hash="tmpl-abc",
        mtp_history_policy="committed",
        draft_head_identity="draft-xyz",
        policy_fingerprint="fp-stable",
    )

    assert restored is not None
    assert bank.last_miss_reason is None


def test_restore_still_rejects_truly_incompatible_policy():
    """``cycle`` must still be rejected as a policy mismatch when an entry was
    stored with ``committed`` (and vice versa). The fix relaxes ONLY the
    committed/last_window pair, never the cycle/committed boundary."""
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    _seed_entry(bank, mtp_history_policy="committed")

    restored = bank.restore(
        _StubRuntime(),
        [1, 2, 3, 4],
        hidden_variant="post_norm",
        template_hash="tmpl-abc",
        mtp_history_policy="cycle",
        draft_head_identity="draft-xyz",
        policy_fingerprint="fp-stable",
    )

    assert restored is None
    assert bank.last_miss_reason == CacheMissReason.POLICY_MISMATCH.value


# ---------------------------------------------------------------------------
# Resolved-policy stability test (against the integration boundary)
# ---------------------------------------------------------------------------


def test_resolve_mtp_history_policy_auto_flips_at_default_threshold(monkeypatch):
    """Sanity check that the diagnostic auto policy still flips near 16K."""
    monkeypatch.setenv("MTPLX_MTP_HISTORY_POLICY", "auto")
    monkeypatch.delenv("MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD", raising=False)

    # Re-import so module-level env reads are fresh (the function reads at
    # call time anyway, but this also guards against caching surprises).
    from mtplx.generation import _resolve_mtp_history_policy

    just_under = _resolve_mtp_history_policy("committed", 16_383)
    just_over = _resolve_mtp_history_policy("committed", 16_384)

    assert just_under == "committed"
    assert just_over == "last_window"

    # And the policy compat helper says these are interchangeable, which is
    # the load-bearing invariant for the SessionBank cliff fix.
    assert _mtp_history_policy_compatible(just_under, just_over) is True


def test_resolve_mtp_history_policy_env_override_applies_to_auto(monkeypatch):
    monkeypatch.setenv("MTPLX_MTP_HISTORY_POLICY", "committed")

    from mtplx.generation import _resolve_mtp_history_policy

    assert _resolve_mtp_history_policy("auto", 32_768) == "committed"


def test_resolve_mtp_history_policy_explicit_non_default_wins(monkeypatch):
    monkeypatch.setenv("MTPLX_MTP_HISTORY_POLICY", "committed")

    from mtplx.generation import _resolve_mtp_history_policy

    assert _resolve_mtp_history_policy("last_window", 1) == "last_window"
    assert _resolve_mtp_history_policy("cycle", 32_768) == "cycle"
