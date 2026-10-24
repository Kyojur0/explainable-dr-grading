"""Tests for the test-with-train-copy list (synthetic audit tables only)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from drgrade.leakage import (
    build_leakage_list,
    check_against_split,
    read_false_matches,
    read_leakage_list,
    write_leakage_list,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "make_leakage_list.py"

# Nodes and their official splits. Train here means the official train split (train or val subset).
SPLIT_OF: dict[str, str] = {
    "tr000001": "train",
    "tr000002": "train",
    "tr000003": "train",
    "tr000004": "train",
    "tr000005": "train",
    "tr000006": "train",
    "tr000007": "train",
    "ts000001": "test",
    "ts000002": "test",
    "ts000003": "test",
    "ts000004": "test",
    "ts000005": "test",
    "ts000006": "test",
    "ts000007": "test",
    "ts000008": "test",
    "ts000009": "test",
    "ts000010": "test",
}

# (a, b, similarity, exact). The false matches are the first, fourth and fifth of their kind below.
EDGES: list[tuple[str, str, float, bool]] = [
    ("tr000001", "ts000001", 0.99, True),  # genuine, direct
    ("tr000002", "ts000002", 0.93, False),  # FALSE MATCH (listed), so ts000002 must not be listed
    ("tr000003", "ts000003", 0.95, False),  # genuine
    ("ts000003", "ts000004", 0.94, False),  # genuine, so ts000004 is linked through ts000003
    ("tr000004", "ts000005", 0.93, False),  # FALSE MATCH (listed)
    ("ts000005", "ts000006", 0.94, False),  # genuine between two test images with no train link
    ("ts000007", "ts000008", 0.96, False),  # test only
    ("tr000005", "ts000009", 0.93, False),  # FALSE MATCH (listed)
    ("tr000006", "ts000009", 0.97, True),  # genuine, direct; the false edge must not add tr000005
    ("tr000007", "ts000010", 0.98, True),  # genuine, direct
]
FALSE_MATCHES: set[frozenset[str]] = {
    frozenset(("tr000002", "ts000002")),
    frozenset(("tr000004", "ts000005")),
    frozenset(("tr000005", "ts000009")),
}
EXPECTED_LISTED = ["ts000001", "ts000003", "ts000004", "ts000009", "ts000010"]


def _tables(split_of: dict[str, str], edges: list[tuple[str, str, float, bool]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build near_duplicate_pairs and dup_clusters tables from edges, as the audit does (components of size 2 or more)."""
    parent = {node: node for node in split_of}

    def find(node: str) -> str:
        while parent[node] != node:
            node = parent[node]
        return node

    for a, b, _s, _e in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    members: dict[str, list[str]] = {}
    for a, b, _s, _e in edges:
        for node in (a, b):
            members.setdefault(find(node), [])
            if node not in members[find(node)]:
                members[find(node)].append(node)
    cluster_rows: list[dict[str, object]] = []
    cluster_ids: dict[str, int] = {}
    for root in sorted(members, key=lambda r: min(members[r])):
        cluster_ids[root] = len(cluster_ids)
        for node in sorted(members[root]):
            cluster_rows.append(
                {
                    "image_id": node,
                    "cluster_id": cluster_ids[root],
                    "cluster_size": len(members[root]),
                    "official_split": split_of[node],
                    "grade": 0,
                }
            )
    pair_rows = [
        {
            "image_id_a": a,
            "image_id_b": b,
            "similarity": s,
            "exact": e,
            "split_a": split_of[a],
            "split_b": split_of[b],
            "grade_a": 0,
            "grade_b": 0,
        }
        for a, b, s, e in edges
    ]
    return pd.DataFrame(pair_rows), pd.DataFrame(cluster_rows)


def test_false_matches_are_removed_and_chains_are_kept() -> None:
    """A test image is listed only through a genuine path to official training data."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    frame = build_leakage_list(pairs, clusters, FALSE_MATCHES)
    assert list(frame["image_id"]) == EXPECTED_LISTED
    assert set(frame["image_id"]) & {"ts000002", "ts000005", "ts000006", "ts000007", "ts000008"} == set()


def test_direct_and_chained_links_are_reported_correctly() -> None:
    """Direct links carry the similarity and exact flag; a chained link has no direct similarity."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    frame = build_leakage_list(pairs, clusters, FALSE_MATCHES).set_index("image_id")

    assert frame.loc["ts000001", "linked_train_ids"] == "tr000001"
    assert frame.loc["ts000001", "max_similarity"] == pytest.approx(0.99)
    assert bool(frame.loc["ts000001", "exact"]) is True

    assert frame.loc["ts000003", "linked_train_ids"] == "tr000003"
    assert bool(frame.loc["ts000003", "exact"]) is False

    assert frame.loc["ts000004", "linked_train_ids"] == "tr000003"
    assert math.isnan(frame.loc["ts000004", "max_similarity"])
    assert bool(frame.loc["ts000004", "exact"]) is False

    # tr000005 is joined to ts000009 only by a false edge, so only tr000006 is linked.
    assert frame.loc["ts000009", "linked_train_ids"] == "tr000006"
    assert frame.loc["ts000009", "max_similarity"] == pytest.approx(0.97)


def test_without_the_false_list_the_false_matches_would_be_listed() -> None:
    """The same tables with no false-match list count the three false edges as links."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    frame = build_leakage_list(pairs, clusters, set())
    listed = set(frame["image_id"])
    assert {"ts000002", "ts000005", "ts000009"} <= listed
    assert "ts000006" in listed


def test_a_listed_false_match_absent_from_the_audit_is_not_an_error() -> None:
    """A false-match entry with no matching pair only logs a warning."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    extra = FALSE_MATCHES | {frozenset(("tr000001", "ts000010"))}
    frame = build_leakage_list(pairs, clusters, extra)
    assert list(frame["image_id"]) == EXPECTED_LISTED


def test_inconsistent_audit_tables_are_refused() -> None:
    """A pair whose members sit in different clusters, or whose split differs, raises ValueError."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    clusters.loc[clusters["image_id"] == "ts000001", "cluster_id"] = 99
    with pytest.raises(ValueError, match="different clusters"):
        build_leakage_list(pairs, clusters, set())

    pairs2, clusters2 = _tables(SPLIT_OF, EDGES)
    pairs2.loc[0, "split_a"] = "test"
    with pytest.raises(ValueError, match="split labels differ"):
        build_leakage_list(pairs2, clusters2, set())


def test_read_false_matches_reads_the_committed_file() -> None:
    """The committed false-match list has the one known pair, as an unordered set."""
    pairs = read_false_matches(REPO / "splits" / "false_matches.csv")
    assert pairs == {frozenset(("tr004850", "ts001140"))}


def test_read_false_matches_rejects_bad_rows(tmp_path: Path) -> None:
    """Self pairs, empty reasons and duplicates (in either order) raise ValueError."""
    cases = {
        "self.csv": "image_id_a,image_id_b,reason\ntr000001,tr000001,x\n",
        "reason.csv": "image_id_a,image_id_b,reason\ntr000001,ts000001,\n",
        "dup.csv": "image_id_a,image_id_b,reason\ntr000001,ts000001,x\nts000001,tr000001,y\n",
    }
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            read_false_matches(path)


def test_check_against_split_detects_a_disagreement() -> None:
    """A cluster label that contradicts the split file is refused."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    split = pd.DataFrame(
        {
            "image_id": list(SPLIT_OF),
            "subset": ["train" if s == "train" else "test" for s in SPLIT_OF.values()],
        }
    )
    check_against_split(clusters, split)
    split.loc[split["image_id"] == "ts000001", "subset"] = "val"
    with pytest.raises(ValueError, match="split disagreements"):
        check_against_split(clusters, split)


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    """The written list reads back with the same ids and columns."""
    pairs, clusters = _tables(SPLIT_OF, EDGES)
    frame = build_leakage_list(pairs, clusters, FALSE_MATCHES)
    path = tmp_path / "list.csv"
    write_leakage_list(frame, path)
    back = read_leakage_list(path)
    assert list(back["image_id"]) == EXPECTED_LISTED
    assert list(back.columns) == ["image_id", "linked_train_ids", "max_similarity", "exact"]


def test_cli_writes_the_list_and_refuses_inconsistent_splits(tmp_path: Path) -> None:
    """The script writes the list from CSV inputs and exits with status 2 on a split disagreement."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_leakage_list", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_leakage_list"] = module
    spec.loader.exec_module(module)

    pairs, clusters = _tables(SPLIT_OF, EDGES)
    pairs_path = tmp_path / "pairs.csv"
    clusters_path = tmp_path / "clusters.csv"
    pairs.to_csv(pairs_path, index=False)
    clusters.to_csv(clusters_path, index=False)
    split_path = tmp_path / "split.csv"
    pd.DataFrame({"image_id": list(SPLIT_OF), "subset": [("train" if s == "train" else "test") for s in SPLIT_OF.values()]}).to_csv(
        split_path, index=False
    )
    false_path = tmp_path / "false.csv"
    false_path.write_text('image_id_a,image_id_b,reason\ntr000002,ts000002,"x, y"\n', encoding="utf-8")
    out = tmp_path / "out.csv"
    args = [
        "--pairs", str(pairs_path), "--clusters", str(clusters_path), "--false-matches", str(false_path),
        "--split", str(split_path), "--out", str(out),
    ]
    assert module.main(args) == 0
    # Only the first false match is listed in this file, so the other two edges count as genuine:
    # ts000005 (and ts000006 through it) and ts000009 are listed as well.
    expected = ["ts000001", "ts000003", "ts000004", "ts000005", "ts000006", "ts000009", "ts000010"]
    assert list(read_leakage_list(out)["image_id"]) == expected

    bad = pd.read_csv(split_path)
    bad.loc[bad["image_id"] == "ts000001", "subset"] = "val"
    bad_path = tmp_path / "bad_split.csv"
    bad.to_csv(bad_path, index=False)
    assert module.main([*args[:-2], "--split", str(bad_path), "--out", str(tmp_path / "x.csv")]) == 2
