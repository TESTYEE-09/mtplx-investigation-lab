from __future__ import annotations

import pytest

from mtplx.turboquant import CENTROIDS_3BIT, value_centroids


def test_value_centroids_preserves_upstream_q3_table() -> None:
    assert value_centroids(3) == CENTROIDS_3BIT


@pytest.mark.parametrize("bits", [4, 8])
def test_value_centroids_supports_q4_q8(bits: int) -> None:
    centroids = value_centroids(bits)

    assert len(centroids) == 1 << bits
    assert all(a < b for a, b in zip(centroids, centroids[1:]))
    for lo, hi in zip(centroids, reversed(centroids), strict=True):
        assert lo == pytest.approx(-hi, abs=1e-6)


def test_value_centroids_rejects_unsupported_width() -> None:
    with pytest.raises(ValueError, match=r"bits must be in \[1, 8\]"):
        value_centroids(9)
