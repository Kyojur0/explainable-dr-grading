"""Validation split construction (brief 6.2).

The official MMRDR-CFP split has no validation set, so one is carved from the
official training images. Images of the same eye captured more than once
(near-duplicate groups from the audit) are kept on the same side of the
train/validation boundary, otherwise validation scores would be inflated in
the same way the official test set is (see DECISIONS.md, D03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SUBSETS: tuple[str, ...] = ("train", "val", "test")


def assign_groups(image_ids: pd.Series, clusters: pd.DataFrame) -> pd.Series:
    """Return a group label per image: its near-duplicate cluster, or itself.

    Args:
        image_ids: ids to label.
        clusters: table with columns image_id and cluster_id, one row per image
            that belongs to a near-duplicate group of size two or more.

    Returns:
        Series of string group labels aligned with `image_ids`. Images outside
        any cluster form a group of one, labelled by their own id.
    """
    cluster_of = dict(zip(clusters["image_id"], clusters["cluster_id"]))
    return image_ids.map(
        lambda image_id: f"cluster_{cluster_of[image_id]}" if image_id in cluster_of else f"image_{image_id}"
    )


def make_split(
    labels: pd.DataFrame,
    clusters: pd.DataFrame,
    n_folds: int,
    val_fold: int,
    seed: int,
) -> pd.DataFrame:
    """Assign every image to train, val or test.

    Official test images stay in test. Official training images are grouped by
    near-duplicate cluster, the groups are split with StratifiedKFold on their
    grade, and the groups in fold `val_fold` become the validation set.

    Args:
        labels: output of drgrade.io.load_fp_csv (needs image_id,
            official_split, grade).
        clusters: near-duplicate clusters from the audit (image_id, cluster_id).
        n_folds: number of folds; validation is roughly 1 / n_folds of train.
        val_fold: index of the fold used for validation, 0 <= val_fold < n_folds.
        seed: random state for the fold shuffle.

    Returns:
        DataFrame with columns image_id and subset, sorted by image_id.
    """
    if not 0 <= val_fold < n_folds:
        raise ValueError(f"val_fold must be in 0..{n_folds - 1}, got {val_fold}")

    train = labels.loc[labels["official_split"] == "train"].reset_index(drop=True)
    test = labels.loc[labels["official_split"] == "test"]
    groups = assign_groups(train["image_id"], clusters)

    # Stratify at group level: one row per near-duplicate group (most groups are
    # single images), labelled with the group's most common grade, ties to the
    # higher grade. StratifiedKFold on these rows keeps grade shares balanced and
    # gives the same folds across scikit-learn versions, unlike
    # StratifiedGroupKFold, whose greedy assignment changed between releases.
    group_table = pd.DataFrame({"group": groups.values, "grade": train["grade"].values})
    group_grade = group_table.groupby("group")["grade"].agg(
        lambda grades: max(grades.mode())
    )
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(group_grade)), group_grade.values))
    val_groups = set(group_grade.index[folds[val_fold][1]])
    val_index = np.flatnonzero(group_table["group"].isin(val_groups).values)

    subset = np.full(len(train), "train", dtype=object)
    subset[val_index] = "val"
    split = pd.concat(
        [
            pd.DataFrame({"image_id": train["image_id"], "subset": subset}),
            pd.DataFrame({"image_id": test["image_id"], "subset": "test"}),
        ],
        ignore_index=True,
    )
    split = split.sort_values("image_id").reset_index(drop=True)
    check_split(split, labels, groups=pd.Series(groups.values, index=train["image_id"]))
    return split


def check_split(split: pd.DataFrame, labels: pd.DataFrame, groups: pd.Series) -> None:
    """Assert the split invariants. Raises AssertionError on any violation.

    - every labelled image appears exactly once, with a known subset name;
    - official test images are all in test and nothing else is;
    - no near-duplicate group is divided between train and val.
    """
    assert split["image_id"].is_unique, "an image appears in more than one subset"
    assert set(split["subset"]) <= set(SUBSETS), f"unknown subset in {set(split['subset'])}"
    assert set(split["image_id"]) == set(labels["image_id"]), "split does not cover every image exactly"

    official = labels.set_index("image_id")["official_split"]
    subset = split.set_index("image_id")["subset"]
    assert ((official == "test") == (subset.loc[official.index] == "test")).all(), (
        "official test images must be exactly the test subset"
    )

    train_val = subset.loc[groups.index]
    sides_per_group = train_val.groupby(groups.values).nunique()
    assert (sides_per_group == 1).all(), "a near-duplicate group is split across train and val"


def grade_distribution(split: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Return the percentage of each grade within each subset (rows: grade)."""
    merged = split.merge(labels[["image_id", "grade"]], on="image_id")
    counts = pd.crosstab(merged["grade"], merged["subset"])
    return counts.div(counts.sum(axis=0), axis=1) * 100.0
