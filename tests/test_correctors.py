from __future__ import annotations

import numpy as np

from mtplx.correctors import (
    DiagonalAffineCorrector,
    NoOpCorrector,
    blend_with_identity,
    deterministic_train_mask,
    ensure_nonempty_split,
    fit_c1_diagonal,
    fit_c2_low_rank,
    load_corrector,
)


def test_noop_returns_identical_hidden_object() -> None:
    hidden = np.arange(6, dtype=np.float32).reshape(2, 3)
    corrected = NoOpCorrector().apply_numpy(hidden, depth=2)
    assert corrected is hidden
    np.testing.assert_array_equal(corrected, hidden)


def test_c1_fit_recovers_diagonal_affine() -> None:
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    depths = np.array([1, 1, 2, 2], dtype=np.int64)
    y = x.copy()
    y[depths == 1] = x[depths == 1] * np.array([2.0, 3.0, 4.0], dtype=np.float32) + 1.0
    y[depths == 2] = x[depths == 2] * np.array([0.5, 1.5, 2.5], dtype=np.float32) - 2.0

    corrector = fit_c1_diagonal(x, y, depths, depth_count=2, eps=1e-8)

    np.testing.assert_allclose(corrector.apply_numpy(x[0], depth=1), y[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(corrector.apply_numpy(x[2], depth=2), y[2], rtol=1e-5, atol=1e-5)


def test_c1_save_load_roundtrip(tmp_path) -> None:
    corrector = DiagonalAffineCorrector(
        scale=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        bias=np.array([[0.5, 0.25], [0.125, 0.0]], dtype=np.float32),
        kind="c1_diagonal_affine",
        hidden_variant="pre_norm",
        metadata={"source": "unit"},
    )
    path = tmp_path / "corrector.npz"
    corrector.save(path)

    loaded = DiagonalAffineCorrector.load(path)

    assert loaded.kind == corrector.kind
    assert loaded.hidden_variant == "pre_norm"
    assert loaded.metadata == {"source": "unit"}
    np.testing.assert_array_equal(loaded.scale, corrector.scale)
    np.testing.assert_array_equal(loaded.bias, corrector.bias)


def test_blend_with_identity_dampens_affine() -> None:
    corrector = DiagonalAffineCorrector(
        scale=np.array([[3.0, 5.0]], dtype=np.float32),
        bias=np.array([[2.0, -4.0]], dtype=np.float32),
    )
    blended = blend_with_identity(corrector, 0.25)

    np.testing.assert_allclose(blended.scale, np.array([[1.5, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(blended.bias, np.array([[0.5, -1.0]], dtype=np.float32))


def test_c2_low_rank_fit_and_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    direction = np.array([[1.0], [0.5], [-0.25], [0.0], [0.25], [0.75]], dtype=np.float32)
    out = np.array([[0.5, -0.5, 0.25, 0.0, 0.1, -0.1]], dtype=np.float32)
    y = x + ((x - x.mean(axis=0)) @ direction) @ out
    depths = np.ones(12, dtype=np.int64)

    corrector = fit_c2_low_rank(x, y, depths, depth_count=1, rank=1, ridge=1e-6)
    corrected = corrector.apply_numpy(x, depth=1)

    assert np.mean((corrected - y) ** 2) < np.mean((x - y) ** 2)

    path = tmp_path / "c2.npz"
    corrector.save(path)
    loaded = type(corrector).load(path)
    np.testing.assert_allclose(loaded.apply_numpy(x, depth=1), corrected, rtol=1e-6, atol=1e-6)


def test_generic_load_corrector_handles_low_rank(tmp_path) -> None:
    corrector = fit_c2_low_rank(
        np.eye(4, dtype=np.float32),
        np.eye(4, dtype=np.float32) * 1.1,
        np.ones(4, dtype=np.int64),
        depth_count=1,
        rank=1,
    )
    path = tmp_path / "low_rank.npz"
    corrector.save(path)

    loaded = load_corrector(path)

    assert type(loaded).__name__ == "LowRankResidualCorrector"


def test_deterministic_split_groups_by_prompt_window() -> None:
    prompt_ids = np.array(["a", "a", "a", "b", "b", "b"])
    windows = np.array([0, 0, 1, 0, 0, 1])

    mask_a = deterministic_train_mask(prompt_ids, windows, train_fraction=0.5, salt="same")
    mask_b = deterministic_train_mask(prompt_ids, windows, train_fraction=0.5, salt="same")
    np.testing.assert_array_equal(mask_a, mask_b)

    for prompt_id, window in {("a", 0), ("a", 1), ("b", 0), ("b", 1)}:
        group = (prompt_ids == prompt_id) & (windows == window)
        assert len(set(mask_a[group])) == 1


def test_ensure_nonempty_split_repairs_tiny_all_train_case() -> None:
    prompt_ids = np.array(["a", "a", "b", "b"])
    windows = np.array([0, 0, 1, 1])
    repaired = ensure_nonempty_split(np.ones(4, dtype=bool), prompt_ids, windows)

    assert repaired.any()
    assert (~repaired).any()
    for prompt_id, window in {("a", 0), ("b", 1)}:
        group = (prompt_ids == prompt_id) & (windows == window)
        assert len(set(repaired[group])) == 1
