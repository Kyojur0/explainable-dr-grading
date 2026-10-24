"""Stage 5: test images that have a genuine near-duplicate in the official training data.

Brief 9.2 asks for the headline test metrics once more with near-duplicate test images
removed. This module builds that list from the stage 1 audit outputs:

- near_duplicate_pairs.csv gives the edges (repeat captures of the same eye);
- dup_clusters.csv gives the groups and the official split of every member.

Known false matches (splits/false_matches.csv, DECISIONS.md D07) are removed as edges
before anything is decided. A test image is listed when it is connected, through genuine
edges only, to at least one official training image (the train or validation subset).
The linked ids are every official training image in that connected component, and
max_similarity and exact describe the direct genuine edges from the test image to an
official training image.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from drgrade.utils import write_atomically

logger = logging.getLogger(__name__)

TEST_SPLIT = "test"
TRAIN_SPLIT = "train"  # official training split; contains the train and validation subsets
SUBSET_OF_OFFICIAL: dict[str, tuple[str, ...]] = {
    TEST_SPLIT: ("test",),
    TRAIN_SPLIT: ("train", "val"),
}
FALSE_MATCH_COLUMNS: tuple[str, ...] = ("image_id_a", "image_id_b", "reason")
PAIR_COLUMNS: tuple[str, ...] = (
    "image_id_a",
    "image_id_b",
    "similarity",
    "exact",
    "split_a",
    "split_b",
    "grade_a",
    "grade_b",
)
CLUSTER_COLUMNS: tuple[str, ...] = ("image_id", "cluster_id", "cluster_size", "official_split", "grade")
LEAKAGE_COLUMNS: tuple[str, ...] = ("image_id", "linked_train_ids", "max_similarity", "exact")
LINK_SEPARATOR = ";"


def read_false_matches(path: Path | str) -> set[frozenset[str]]:
    """Return the known false-match pairs as unordered sets of two image ids.

    The file needs the columns image_id_a, image_id_b and reason. A pair listed twice
    (in either order), a self-pair or an empty reason raises ValueError.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(frame, FALSE_MATCH_COLUMNS, str(path))
    pairs: set[frozenset[str]] = set()
    for row in frame.itertuples(index=False):
        a, b = str(row.image_id_a).strip(), str(row.image_id_b).strip()
        if not a or not b or a == b:
            raise ValueError(f"{path}: invalid false match {a!r}, {b!r}")
        if not str(row.reason).strip():
            raise ValueError(f"{path}: false match {a}, {b} has no reason")
        key = frozenset((a, b))
        if key in pairs:
            raise ValueError(f"{path}: false match {a}, {b} is listed twice")
        pairs.add(key)
    return pairs


def build_leakage_list(
    pairs: pd.DataFrame,
    clusters: pd.DataFrame,
    false_matches: Iterable[frozenset[str]],
) -> pd.DataFrame:
    """Return one row per official test image that has a genuine link to official training data.

    pairs: near_duplicate_pairs.csv (columns in PAIR_COLUMNS).
    clusters: dup_clusters.csv (columns in CLUSTER_COLUMNS).
    false_matches: unordered pairs to drop as edges before grouping.

    Raises ValueError when the two audit tables disagree (a pair whose members are not
    in one cluster, or a split that differs between the tables), so that a stale or
    mixed set of outputs cannot give a silent answer.
    """
    _require_columns(pairs, PAIR_COLUMNS, "near_duplicate_pairs")
    _require_columns(clusters, CLUSTER_COLUMNS, "dup_clusters")
    if clusters["image_id"].astype(str).duplicated().any():
        raise ValueError("dup_clusters.csv lists an image id more than once")

    split_of = dict(zip(clusters["image_id"].astype(str), clusters["official_split"].astype(str)))
    cluster_of = dict(zip(clusters["image_id"].astype(str), clusters["cluster_id"].astype(int)))
    false_set = set(false_matches)

    edges: list[tuple[str, str, float, bool]] = []
    dropped: list[frozenset[str]] = []
    seen_false: set[frozenset[str]] = set()
    for row in pairs.itertuples(index=False):
        a, b = str(row.image_id_a), str(row.image_id_b)
        for node in (a, b):
            if node not in split_of:
                raise ValueError(f"pair member {node} is missing from dup_clusters.csv")
        if cluster_of[a] != cluster_of[b]:
            raise ValueError(f"pair {a}, {b} lies in different clusters in dup_clusters.csv")
        if split_of[a] != str(row.split_a) or split_of[b] != str(row.split_b):
            raise ValueError(f"pair {a}, {b}: split labels differ between the audit tables")
        key = frozenset((a, b))
        if key in false_set:
            dropped.append(key)
            seen_false.add(key)
            continue
        edges.append((a, b, float(row.similarity), bool(row.exact)))

    missing_false = sorted(
        ", ".join(sorted(key)) for key in false_set if key not in seen_false
    )
    if missing_false:
        logger.warning("%d listed false matches are not pairs in this audit: %s", len(missing_false), missing_false)
    logger.info("false-match edges removed: %d", len(dropped))

    parent = {node: node for node in split_of}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b, _similarity, _exact in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    members: dict[str, list[str]] = {}
    for node in split_of:
        members.setdefault(find(node), []).append(node)

    direct: dict[str, list[tuple[str, float, bool]]] = {}
    for a, b, similarity, exact in edges:
        direct.setdefault(a, []).append((b, similarity, exact))
        direct.setdefault(b, []).append((a, similarity, exact))

    rows: list[dict[str, object]] = []
    for test_id in sorted(node for node, split in split_of.items() if split == TEST_SPLIT):
        component = members[find(test_id)]
        linked = sorted(node for node in component if split_of[node] == TRAIN_SPLIT)
        if not linked:
            continue
        links = [(sim, exact) for other, sim, exact in direct.get(test_id, []) if split_of[other] == TRAIN_SPLIT]
        if links:
            max_similarity = max(sim for sim, _ in links)
            exact_copy = any(exact for _, exact in links)
        else:
            max_similarity = math.nan
            exact_copy = False
        rows.append(
            {
                "image_id": test_id,
                "linked_train_ids": LINK_SEPARATOR.join(linked),
                "max_similarity": max_similarity,
                "exact": exact_copy,
            }
        )
    frame = pd.DataFrame(rows, columns=list(LEAKAGE_COLUMNS))
    n_indirect = int((frame["max_similarity"].isna()).sum()) if len(frame) else 0
    logger.info(
        "test images with a genuine copy in official training data: %d (%d reached only through a chain)",
        len(frame),
        n_indirect,
    )
    return frame


def check_against_split(clusters: pd.DataFrame, split: pd.DataFrame) -> None:
    """Raise ValueError when an official split label in dup_clusters.csv disagrees with the split file.

    Test ids must be in subset test. Official train ids must be in subset train or val.
    Ids missing from the split file are reported too.
    """
    _require_columns(split, ("image_id", "subset"), "split file")
    subset_of = dict(zip(split["image_id"].astype(str), split["subset"].astype(str)))
    problems: list[str] = []
    for row in clusters.itertuples(index=False):
        image_id = str(row.image_id)
        official = str(row.official_split)
        if image_id not in subset_of:
            problems.append(f"{image_id} is not in the split file")
        elif subset_of[image_id] not in SUBSET_OF_OFFICIAL.get(official, ()):
            problems.append(f"{image_id} is {official} in dup_clusters but {subset_of[image_id]} in the split file")
    if problems:
        raise ValueError(f"{len(problems)} split disagreements, first: {problems[:5]}")


def read_leakage_list(path: Path | str) -> pd.DataFrame:
    """Read a leakage list written by build_leakage_list, with the exact column names checked."""
    frame = pd.read_csv(path, dtype={"image_id": str, "linked_train_ids": str})
    _require_columns(frame, LEAKAGE_COLUMNS, str(path))
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{path} lists an image id more than once")
    return frame


def write_leakage_list(frame: pd.DataFrame, path: Path | str) -> None:
    """Write the leakage list atomically with the fixed column order."""
    ordered = frame.loc[:, list(LEAKAGE_COLUMNS)]
    write_atomically(Path(path), lambda p: ordered.to_csv(p, index=False))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise ValueError naming the file when a required column is missing."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks the columns {missing}")
