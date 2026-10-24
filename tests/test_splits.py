"""Tests for the validation split (brief 6.2). Synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drgrade.splits import grade_distribution, make_split


def _labels(n_train: int = 700, n_test: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade_p = [0.59, 0.12, 0.17, 0.06, 0.06]
    rows = [
        {"image_id": f"tr{i:06d}", "official_split": "train", "grade": int(rng.choice(5, p=grade_p))}
        for i in range(1, n_train + 1)
    ] + [
        {"image_id": f"ts{i:06d}", "official_split": "test", "grade": int(rng.choice(5, p=grade_p))}
        for i in range(1, n_test + 1)
    ]
    return pd.DataFrame(rows)


def _clusters() -> pd.DataFrame:
    # Three train-only groups, and one group spanning train and test.
    members = {
        0: ["tr000001", "tr000002", "tr000003"],
        1: ["tr000010", "tr000011"],
        2: ["tr000020", "tr000021", "tr000022", "tr000023"],
        3: ["tr000030", "ts000005"],
    }
    return pd.DataFrame(
        [{"image_id": i, "cluster_id": c} for c, ids in members.items() for i in ids]
    )


def test_every_image_once_and_test_untouched() -> None:
    labels = _labels()
    split = make_split(labels, _clusters(), n_folds=7, val_fold=0, seed=42)
    assert split["image_id"].is_unique
    assert set(split["image_id"]) == set(labels["image_id"])
    test_ids = set(labels.loc[labels["official_split"] == "test", "image_id"])
    assert set(split.loc[split["subset"] == "test", "image_id"]) == test_ids


def test_groups_stay_together() -> None:
    labels = _labels()
    clusters = _clusters()
    split = make_split(labels, clusters, n_folds=7, val_fold=0, seed=42).set_index("image_id")
    for _, members in clusters.groupby("cluster_id")["image_id"]:
        train_side = [split.loc[i, "subset"] for i in members if i.startswith("tr")]
        assert len(set(train_side)) == 1


def test_val_size_and_stratification() -> None:
    labels = _labels(n_train=2000)
    split = make_split(labels, _clusters(), n_folds=7, val_fold=0, seed=42)
    n_val = (split["subset"] == "val").sum()
    assert 0.11 * 2000 < n_val < 0.18 * 2000
    dist = grade_distribution(split, labels)
    assert (dist["train"] - dist["val"]).abs().max() < 2.5


def test_deterministic_for_seed() -> None:
    labels = _labels()
    a = make_split(labels, _clusters(), n_folds=7, val_fold=0, seed=42)
    b = make_split(labels, _clusters(), n_folds=7, val_fold=0, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_bad_val_fold_raises() -> None:
    with pytest.raises(ValueError):
        make_split(_labels(), _clusters(), n_folds=7, val_fold=7, seed=42)
