"""Stage 6: error analysis tables and representative cases.

Inputs are a predictions table (from predict.py), the manifest (lesion flags, quality
measures, border level), the audit flags and the leakage list. Everything is
deterministic: ties are broken by image_id.

Case quotas: the 6 most confident errors; up to 6 referable misses; 4
adjacent-grade errors and 4 errors two or more grades apart (each the most confident
ones); and the 2 most confident correct predictions for each grade. An image can appear
under more than one reason, and each row carries one reason.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from drgrade.io import GRADE_VALUES, LESION_COLS
from drgrade.metrics import PDR_GRADE, REFERABLE_GRADE, full_metrics

PROB_COLUMNS: tuple[str, ...] = tuple(f"prob_{g}" for g in GRADE_VALUES)
QUALITY_MEASURES: tuple[str, ...] = ("brightness", "contrast", "sharpness")
QUARTILE_LABELS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
DANGEROUS_LESIONS: tuple[str, ...] = ("les_nv", "les_vh")
LESION_SHORT: dict[str, str] = {
    "les_ma": "MA",
    "les_he": "HE",
    "les_irh": "IRH",
    "les_vb_irma": "VB/IRMA",
    "les_nv": "NV",
    "les_vh": "VH",
    "les_rd": "RD",
}
MANIFEST_COLUMNS: tuple[str, ...] = ("image_id", "border_level", *QUALITY_MEASURES, *LESION_COLS, "file")
CASE_MOST_CONFIDENT = "most_confident_error"
CASE_REFERABLE_MISS = "referable_miss"
CASE_ADJACENT = "adjacent_error"
CASE_DISTANT = "error_two_or_more_grades"
CASE_QUOTAS: dict[str, int] = {
    CASE_MOST_CONFIDENT: 6,
    CASE_REFERABLE_MISS: 6,
    CASE_ADJACENT: 4,
    CASE_DISTANT: 4,
}
CORRECT_PER_GRADE = 2
CASE_COLUMNS: tuple[str, ...] = (
    "reason",
    "image_id",
    "true_grade",
    "pred_grade",
    "confidence",
    "lesions_present",
)
ERROR_COLUMNS: tuple[str, ...] = (
    "image_id",
    "true_grade",
    "pred_grade",
    "confidence",
    "entropy",
    "grade_gap",
    "referable_miss",
    *PROB_COLUMNS,
    *LESION_COLS,
    *QUALITY_MEASURES,
    "border_level",
    "audit_flags",
    "in_leakage_list",
)


def join_predictions(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return the predictions with the manifest columns of MANIFEST_COLUMNS added by image_id.

    Raises ValueError for a duplicated id or an id that is missing from the manifest.
    """
    if predictions["image_id"].astype(str).duplicated().any():
        raise ValueError("predictions list an image id more than once")
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest lacks the columns {missing}")
    lookup = manifest.loc[:, list(MANIFEST_COLUMNS)].copy()
    lookup["image_id"] = lookup["image_id"].astype(str)
    frame = predictions.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    joined = frame.merge(lookup, on="image_id", how="left", validate="one_to_one", indicator=True)
    absent = joined.loc[joined["_merge"] != "both", "image_id"].tolist()
    if absent:
        raise ValueError(f"{len(absent)} prediction ids are missing from the manifest, first: {absent[:5]}")
    return joined.drop(columns=["_merge"])


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add confidence (largest probability), entropy (natural log), the grade gap and a correctness flag."""
    probs = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    out = frame.copy()
    out["confidence"] = probs.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0, np.log(probs), 0.0)
    out["entropy"] = -(probs * logs).sum(axis=1)
    out["grade_gap"] = (out["pred_grade"] - out["true_grade"]).abs().astype(int)
    out["correct"] = out["pred_grade"] == out["true_grade"]
    out["referable_miss"] = (out["true_grade"] >= REFERABLE_GRADE) & (out["pred_grade"] < REFERABLE_GRADE)
    return out


def audit_flags_by_image(flags: pd.DataFrame | None) -> dict[str, str]:
    """Return image id -> semicolon-joined audit flag names, from audit flagged_images.csv (columns image_id, flag)."""
    if flags is None:
        return {}
    if not {"image_id", "flag"} <= set(flags.columns):
        raise ValueError("flagged_images.csv needs the columns image_id and flag")
    grouped = flags.assign(image_id=flags["image_id"].astype(str)).groupby("image_id")["flag"]
    return {image_id: ";".join(sorted(str(f) for f in names)) for image_id, names in grouped}


def lesions_present(row: pd.Series) -> str:
    """Return the short names of the lesions present in a row, joined by commas ('' when none)."""
    return ", ".join(LESION_SHORT[c] for c in LESION_COLS if int(row[c]) == 1)


def error_table(
    frame: pd.DataFrame,
    audit_flags: dict[str, str],
    leakage_ids: Iterable[str],
) -> pd.DataFrame:
    """Return every misclassified image with confidence, entropy, lesions, quality, audit flags and leakage status."""
    leak = set(str(i) for i in leakage_ids)
    errors = frame.loc[~frame["correct"]].copy()
    errors["audit_flags"] = errors["image_id"].map(lambda i: audit_flags.get(i, ""))
    errors["in_leakage_list"] = errors["image_id"].map(lambda i: i in leak)
    errors = errors.sort_values(["confidence", "image_id"], ascending=[False, True])
    return errors.loc[:, list(ERROR_COLUMNS)].reset_index(drop=True)


def lesion_stratified(frame: pd.DataFrame) -> pd.DataFrame:
    """Return, per lesion, the underestimation and referable miss rates with the lesion present and absent."""
    rows: list[dict[str, Any]] = []
    under_all = frame["pred_grade"] < frame["true_grade"]
    referable = frame["true_grade"] >= REFERABLE_GRADE
    miss_all = frame["pred_grade"] < REFERABLE_GRADE
    for column in LESION_COLS:
        present = frame[column].astype(int) == 1
        row: dict[str, Any] = {"lesion": column, "name": LESION_SHORT[column]}
        for label, mask in (("present", present), ("absent", ~present)):
            n = int(mask.sum())
            n_ref = int((mask & referable).sum())
            row[f"n_{label}"] = n
            row[f"underestimation_{label}"] = metric_share(under_all[mask].sum(), n)
            row[f"n_referable_{label}"] = n_ref
            row[f"referable_miss_{label}"] = metric_share((mask & referable & miss_all).sum(), n_ref)
        rows.append(row)
    return pd.DataFrame(rows)


def dangerous_misses(frame: pd.DataFrame) -> pd.DataFrame:
    """Return images with neovascularisation or vitreous haemorrhage present that were predicted below grade 4."""
    present = np.zeros(len(frame), dtype=bool)
    for column in DANGEROUS_LESIONS:
        present |= frame[column].astype(int).to_numpy() == 1
    mask = present & (frame["pred_grade"].to_numpy() < PDR_GRADE)
    cols = ["image_id", "true_grade", "pred_grade", "confidence", *DANGEROUS_LESIONS]
    selected = frame.loc[mask, cols].copy()
    return selected.sort_values(["confidence", "image_id"], ascending=[False, True]).reset_index(drop=True)


def quality_stratified(frame: pd.DataFrame) -> pd.DataFrame:
    """Return headline metrics per quartile of each quality measure (quartiles by rank, so groups are equal in size)."""
    rows: list[dict[str, Any]] = []
    for measure in QUALITY_MEASURES:
        ranks = frame[measure].rank(method="first")
        quartile = pd.qcut(ranks, len(QUARTILE_LABELS), labels=list(QUARTILE_LABELS))
        for label in QUARTILE_LABELS:
            sub = frame.loc[quartile == label]
            if len(sub) == 0:
                continue
            metrics = full_metrics(sub, n_boot=0)
            rows.append(
                {
                    "measure": measure,
                    "quartile": label,
                    "value_min": float(sub[measure].min()),
                    "value_max": float(sub[measure].max()),
                    "n": metrics["n"],
                    "accuracy": _blank(metrics["accuracy"]),
                    "macro_f1": _blank(metrics["macro_f1"]),
                    "qwk": _blank(metrics["qwk"]),
                    "referable_sensitivity": _blank(metrics["referable"]["sensitivity"]),
                    "referable_miss_rate": _blank(metrics["errors"]["referable_miss_rate"]),
                    "underestimation_rate": _blank(metrics["errors"]["underestimation_rate"]),
                }
            )
    return pd.DataFrame(rows)


def select_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the representative cases, one row per (reason, image), in the order of the reasons."""
    errors = frame.loc[~frame["correct"]]
    ordered = _most_confident(errors)
    selections: list[tuple[str, pd.DataFrame]] = [
        (CASE_MOST_CONFIDENT, ordered.head(CASE_QUOTAS[CASE_MOST_CONFIDENT])),
        (CASE_REFERABLE_MISS, _most_confident(errors.loc[errors["referable_miss"]]).head(CASE_QUOTAS[CASE_REFERABLE_MISS])),
        (CASE_ADJACENT, _most_confident(errors.loc[errors["grade_gap"] == 1]).head(CASE_QUOTAS[CASE_ADJACENT])),
        (CASE_DISTANT, _most_confident(errors.loc[errors["grade_gap"] >= 2]).head(CASE_QUOTAS[CASE_DISTANT])),
    ]
    correct = frame.loc[frame["correct"]]
    for grade in GRADE_VALUES:
        chosen = _most_confident(correct.loc[correct["true_grade"] == grade]).head(CORRECT_PER_GRADE)
        selections.append((f"correct_grade_{grade}", chosen))
    rows: list[pd.DataFrame] = []
    for reason, chosen in selections:
        if len(chosen) == 0:
            continue
        block = chosen.assign(reason=reason)
        rows.append(block)
    if not rows:
        return pd.DataFrame(columns=list(CASE_COLUMNS))
    out = pd.concat(rows, ignore_index=True)
    out["lesions_present"] = out.apply(lesions_present, axis=1)
    return out.loc[:, list(CASE_COLUMNS)]


def case_groups(cases: pd.DataFrame) -> list[str]:
    """Return the distinct reasons in the order they appear in cases."""
    return list(dict.fromkeys(cases["reason"].tolist()))


def _most_confident(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by confidence (highest first), then image_id, for a deterministic choice."""
    return frame.sort_values(["confidence", "image_id"], ascending=[False, True])


def metric_share(numerator: float, denominator: float) -> float | None:
    """Return numerator / denominator, or None when the denominator is zero."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _blank(value: Any) -> float | None:
    """Return a finite float, or None for None and NaN."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value
