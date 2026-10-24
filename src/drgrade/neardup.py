"""Near-duplicate search over unit-norm vessel-pattern signatures (brief 5.4, DECISIONS.md D03).

Vectors are unit-norm, so the dot product of two vectors is their cosine
similarity. All pairs are compared in blocks of rows with matrix products, so
that about 11k vectors (roughly 60 million pairs) need no Python-level loop over
pairs.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BLOCK_SIZE = 512


def _as_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the vectors as a float32 two-dimensional array, or raise if the shape is wrong."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"vectors must be two-dimensional, got shape {matrix.shape}")
    return matrix


def cosine_pairs(
    vectors: np.ndarray,
    threshold: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """Find all index pairs (i, j), i < j, whose cosine similarity is at least `threshold`.

    Args:
        vectors: array of shape (n, d). Rows should be unit-norm; zero rows have
            similarity 0 with every row.
        threshold: minimum similarity to report (inclusive).
        block_size: number of rows compared against all rows at once.

    Returns:
        float64 array of shape (k, 3) with columns (i, j, similarity), where i and j
        are row indices stored as floats (convert with astype(np.int64)). Rows are
        sorted by i, then j. Matrix products run in float32, so similarities are
        accurate to about 1e-6. An empty result has shape (0, 3).
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    matrix = _as_matrix(vectors)
    n = matrix.shape[0]
    blocks: list[np.ndarray] = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sims = matrix[start:stop] @ matrix.T
        rows, cols = np.nonzero(sims >= threshold)
        rows = rows + start
        keep = cols > rows
        rows, cols = rows[keep], cols[keep]
        values = sims[rows - start, cols]
        blocks.append(
            np.column_stack([rows, cols, values]).astype(np.float64)
        )
    if not blocks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(blocks, axis=0)


def best_neighbour_similarity(
    vectors: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """Return, for each row, the highest cosine similarity to any other row.

    Args:
        vectors: array of shape (n, d), rows as in `cosine_pairs`.
        block_size: number of rows compared against all rows at once.

    Returns:
        float64 array of length n. Empty when n is 0. For n equal to 1 the single
        value is -inf, because there is no other row.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    matrix = _as_matrix(vectors)
    n = matrix.shape[0]
    best = np.full(n, -np.inf, dtype=np.float64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sims = (matrix[start:stop] @ matrix.T).astype(np.float64)
        local = np.arange(stop - start)
        sims[local, start + local] = -np.inf  # a row is not its own neighbour
        best[start:stop] = sims.max(axis=1)
    return best


def connected_components(n: int, edges: np.ndarray) -> np.ndarray:
    """Label the connected components of an undirected graph with union-find.

    Args:
        n: number of nodes, labelled 0..n-1.
        edges: array of shape (k, 2) with node index pairs.

    Returns:
        int64 array of length n. Labels are 0..c-1, numbered in order of the
        smallest node index in each component.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist():
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    labels = np.empty(n, dtype=np.int64)
    label_of_root: dict[int, int] = {}
    for node in range(n):
        root = find(node)
        labels[node] = label_of_root.setdefault(root, len(label_of_root))
    return labels
