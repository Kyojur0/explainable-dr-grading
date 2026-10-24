"""Tests for cosine pair search, nearest neighbours and clustering (brief 5.4, D03)."""

from __future__ import annotations

import numpy as np
import pytest

from drgrade.neardup import best_neighbour_similarity, connected_components, cosine_pairs


def _found(pairs: np.ndarray) -> dict[tuple[int, int], float]:
    return {(int(i), int(j)): float(s) for i, j, s in pairs}


def test_cosine_pairs_small_hand_made_set() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],  # identical to row 0
            [0.0, 1.0],  # orthogonal to rows 0 and 1
            [0.6, 0.8],  # cosine 0.6 with rows 0 and 1, 0.8 with row 2
        ]
    )
    pairs = cosine_pairs(vectors, threshold=0.6)
    found = _found(pairs)
    assert set(found) == {(0, 1), (0, 3), (1, 3), (2, 3)}
    assert found[(0, 1)] == pytest.approx(1.0, abs=1e-6)
    assert found[(0, 3)] == pytest.approx(0.6, abs=1e-6)
    assert found[(2, 3)] == pytest.approx(0.8, abs=1e-6)


def test_orthogonal_vectors_are_not_reported() -> None:
    vectors = np.eye(3)
    assert cosine_pairs(vectors, threshold=0.5).shape == (0, 3)


def test_threshold_is_inclusive() -> None:
    vectors = np.array([[1.0, 0.0], [0.5, 0.8660254037844386]])
    found = _found(cosine_pairs(vectors, threshold=0.5))
    assert (0, 1) in found
    assert found[(0, 1)] == pytest.approx(0.5, abs=1e-6)


def test_block_size_does_not_change_result() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 16))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    vectors = np.concatenate([base, base * 1.0])  # every row has a copy
    a = cosine_pairs(vectors, threshold=0.3, block_size=7)
    b = cosine_pairs(vectors, threshold=0.3, block_size=1000)
    assert np.array_equal(a, b)
    assert np.all(a[:, 0] < a[:, 1])


def test_cosine_pairs_rows_are_sorted_by_i_then_j() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(40, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = cosine_pairs(vectors, threshold=0.0, block_size=9)
    keys = [(int(i), int(j)) for i, j, _ in pairs]
    assert keys == sorted(keys)


def test_cosine_pairs_empty_input() -> None:
    assert cosine_pairs(np.zeros((0, 4)), threshold=0.9).shape == (0, 3)


def test_best_neighbour_excludes_self() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    best = best_neighbour_similarity(vectors, block_size=2)
    assert best[0] == pytest.approx(1.0, abs=1e-6)
    assert best[1] == pytest.approx(1.0, abs=1e-6)
    assert best[2] == pytest.approx(0.0, abs=1e-6)


def test_connected_components_labels_in_first_seen_order() -> None:
    edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int64)
    labels = connected_components(6, edges)
    assert list(labels) == [0, 0, 0, 1, 1, 2]


def test_connected_components_without_edges_are_singletons() -> None:
    labels = connected_components(3, np.empty((0, 2), dtype=np.int64))
    assert list(labels) == [0, 1, 2]
