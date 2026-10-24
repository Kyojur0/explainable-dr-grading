"""Metrics for training (brief 8.4) and for evaluation (brief 9.2).

validation_metrics() is used during training for model selection and history.csv.
full_metrics() computes the complete set in brief 9.2 from a predictions table, and
bootstrap_ci() gives the percentile intervals for the headline metrics.

Everything is computed from one 5 by 5 confusion matrix (rows true, columns predicted),
so the bootstrap resamples and the point estimates use exactly the same code. Undefined
values (for example a sensitivity with no positive images) are reported as None with a
note, never as an error. Macro-F1 and weighted-F1 treat an undefined per-class F1 as 0,
which is the convention of validation_metrics (zero_division=0), so training and
evaluation numbers agree.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from drgrade.io import GRADE_VALUES

GRADE_COUNT = len(GRADE_VALUES)
REFERABLE_GRADE = 2  # referable DR: grade 2 or above (brief 9.2)
OFF_BY_TWO_OR_MORE = 2  # clinically weighted error: |pred - true| at least this
PDR_GRADE = 4
PROB_COLUMNS: tuple[str, ...] = tuple(f"prob_{g}" for g in GRADE_VALUES)
REFERABLE_SCORE_GRADES: tuple[int, ...] = (2, 3, 4)  # AUROC score is p(2) + p(3) + p(4)
BOOTSTRAP_METRICS: tuple[str, ...] = ("accuracy", "macro_f1", "qwk", "referable_sensitivity", "referable_miss_rate")
BOOTSTRAP_PERCENTILES: tuple[float, float] = (2.5, 97.5)
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 42
KAPPA_WEIGHT_DENOMINATOR = (GRADE_COUNT - 1) ** 2


def validation_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Return accuracy, macro-F1, weighted-F1 and quadratic weighted kappa (QWK).

    The labels are always the five grades 0 to 4. A grade absent from both the
    truth and the predictions therefore scores F1 = 0 in the macro average (the
    zero_division setting). Quadratic QWK is NaN when the expected agreement is 1
    (for example, every label and prediction is the same grade); callers must treat
    NaN as the worst value.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and of the same length")
    labels = list(GRADE_VALUES)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=labels, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(truth, pred, labels=labels, weights="quadratic")),
    }


def majority_grade(grades: Sequence[int]) -> int:
    """Return the most common grade. Ties go to the lower grade, so the result is deterministic."""
    values = np.asarray(list(grades), dtype=np.int64)
    if values.size == 0:
        raise ValueError("cannot take the majority grade of no grades")
    counts = np.bincount(values, minlength=GRADE_COUNT)[:GRADE_COUNT]
    return int(np.argmax(counts))


def confusion_counts(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return the 5 by 5 count matrix, rows true grade and columns predicted grade."""
    flat = true.astype(np.int64) * GRADE_COUNT + pred.astype(np.int64)
    return np.bincount(flat, minlength=GRADE_COUNT * GRADE_COUNT).reshape(GRADE_COUNT, GRADE_COUNT)


def _nan_div(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when the denominator is zero."""
    return float(numerator) / float(denominator) if denominator else math.nan


def _per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, F1 (NaN where undefined) and support for each class."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    precision = np.array([_nan_div(t, p) for t, p in zip(tp, predicted)])
    recall = np.array([_nan_div(t, s) for t, s in zip(tp, support)])
    f1 = np.array([_nan_div(2 * t, 2 * t + (p - t) + (s - t)) for t, p, s in zip(tp, predicted, support)])
    return precision, recall, f1, support


def _accuracy(cm: np.ndarray) -> float:
    """Return the share of images on the diagonal."""
    return _nan_div(np.trace(cm), cm.sum())


def _macro_f1(cm: np.ndarray) -> float:
    """Return the unweighted mean F1 over the five grades, with an undefined F1 counted as 0."""
    _, _, f1, _ = _per_class(cm)
    return float(np.mean(np.nan_to_num(f1, nan=0.0)))


def _weighted_f1(cm: np.ndarray) -> float:
    """Return the F1 mean weighted by true support, with an undefined F1 counted as 0."""
    _, _, f1, support = _per_class(cm)
    total = support.sum()
    return _nan_div(float(np.sum(np.nan_to_num(f1, nan=0.0) * support)), total)


def _qwk(cm: np.ndarray) -> float:
    """Return quadratic weighted kappa, as sklearn computes it with weights='quadratic'."""
    total = cm.sum()
    if total == 0:
        return math.nan
    idx = np.arange(GRADE_COUNT)
    weights = (idx[:, None] - idx[None, :]) ** 2 / KAPPA_WEIGHT_DENOMINATOR
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / total
    denominator = float(np.sum(weights * expected))
    if denominator == 0:
        return math.nan
    return 1.0 - float(np.sum(weights * cm)) / denominator


def _referable_counts(cm: np.ndarray) -> dict[str, int]:
    """Return the referable 2 by 2 counts from the five-class matrix (true or predicted grade 2 or above)."""
    positive = slice(REFERABLE_GRADE, None)
    negative = slice(0, REFERABLE_GRADE)
    return {
        "tp": int(cm[positive, positive].sum()),
        "fn": int(cm[positive, negative].sum()),
        "fp": int(cm[negative, positive].sum()),
        "tn": int(cm[negative, negative].sum()),
    }


def _referable_rates(cm: np.ndarray) -> dict[str, float]:
    """Return referable sensitivity, specificity, PPV, NPV and the miss rate (1 - sensitivity)."""
    c = _referable_counts(cm)
    return {
        "sensitivity": _nan_div(c["tp"], c["tp"] + c["fn"]),
        "specificity": _nan_div(c["tn"], c["tn"] + c["fp"]),
        "ppv": _nan_div(c["tp"], c["tp"] + c["fp"]),
        "npv": _nan_div(c["tn"], c["tn"] + c["fn"]),
    }


def _cm_metrics(cm: np.ndarray) -> dict[str, float]:
    """Return the five bootstrap metrics from one confusion matrix (NaN where undefined)."""
    rates = _referable_rates(cm)
    counts = _referable_counts(cm)
    return {
        "accuracy": _accuracy(cm),
        "macro_f1": _macro_f1(cm),
        "qwk": _qwk(cm),
        "referable_sensitivity": rates["sensitivity"],
        "referable_miss_rate": _nan_div(counts["fn"], counts["tp"] + counts["fn"]),
    }


def _auroc(true: np.ndarray, probs: np.ndarray | None) -> float | None:
    """Return the referable AUROC with score p(2) + p(3) + p(4), or None when undefined."""
    if probs is None:
        return None
    referable = true >= REFERABLE_GRADE
    if referable.all() or (~referable).all():
        return None
    score = probs[:, list(REFERABLE_SCORE_GRADES)].sum(axis=1)
    return float(roc_auc_score(referable.astype(int), score))


def _clean(value: float) -> float | None:
    """Map NaN to None so that the value serialises as null."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return true grades, predicted grades and (when present and complete) the probability matrix."""
    for column in ("true_grade", "pred_grade"):
        if column not in df.columns:
            raise ValueError(f"predictions table lacks the column {column!r}")
    if len(df) == 0:
        raise ValueError("predictions table is empty")
    true = df["true_grade"].to_numpy().astype(np.int64)
    pred = df["pred_grade"].to_numpy().astype(np.int64)
    for name, values in (("true_grade", true), ("pred_grade", pred)):
        if values.min() < 0 or values.max() >= GRADE_COUNT:
            raise ValueError(f"{name} must be in 0..{GRADE_COUNT - 1}")
    probs: np.ndarray | None = None
    if all(column in df.columns for column in PROB_COLUMNS):
        probs = df[list(PROB_COLUMNS)].to_numpy().astype(np.float64)
    return true, pred, probs


def full_metrics(df: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Return every metric in brief 9.2 for a predictions table.

    df needs true_grade and pred_grade. The prob_0 to prob_4 columns are optional; without
    them the referable AUROC is None (as for the majority baseline). n_boot 0 skips the
    bootstrap, which is used for subgroups.
    """
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")
    true, pred, probs = _prepare(df)
    n = int(true.size)
    cm = confusion_counts(true, pred)
    notes: list[str] = []

    precision, recall, f1, support = _per_class(cm)
    per_class = [
        {
            "grade": grade,
            "precision": _clean(precision[i]),
            "recall": _clean(recall[i]),
            "f1": _clean(f1[i]),
            "support": int(support[i]),
        }
        for i, grade in enumerate(GRADE_VALUES)
    ]
    for row in per_class:
        if row["f1"] is None:
            notes.append(f"grade {row['grade']}: precision, recall and F1 undefined (no true or predicted images)")

    row_sums = cm.sum(axis=1)
    confusion_norm = [
        [None if row_sums[i] == 0 else float(cm[i, j] / row_sums[i]) for j in range(GRADE_COUNT)]
        for i in range(GRADE_COUNT)
    ]
    for i, total in enumerate(row_sums):
        if total == 0:
            notes.append(f"confusion row for grade {i} is undefined (no true images)")

    rates = _referable_rates(cm)
    referable_counts = _referable_counts(cm)
    auroc = _auroc(true, probs)
    if probs is None:
        notes.append("referable AUROC is null: no probability columns")
    elif auroc is None:
        notes.append("referable AUROC is null: only one referable class present")
    for name, value in rates.items():
        if math.isnan(value):
            notes.append(f"referable {name} is undefined (no images in the required group)")

    n_referable_true = int((true >= REFERABLE_GRADE).sum())
    n_pdr_true = int((true == PDR_GRADE).sum())
    miss = _nan_div(referable_counts["fn"], n_referable_true)
    pdr_miss = _nan_div(int(((true == PDR_GRADE) & (pred <= REFERABLE_GRADE)).sum()), n_pdr_true)
    if math.isnan(pdr_miss):
        notes.append("PDR miss rate is undefined (no true grade 4 images)")
    if math.isnan(miss):
        notes.append("referable miss rate is undefined (no true referable images)")
    result: dict[str, Any] = {
        "n": n,
        "accuracy": _clean(_accuracy(cm)),
        "macro_f1": _clean(_macro_f1(cm)),
        "weighted_f1": _clean(_weighted_f1(cm)),
        "qwk": _clean(_qwk(cm)),
        "per_class": per_class,
        "confusion_counts": cm.astype(int).tolist(),
        "confusion_norm": confusion_norm,
        "referable": {
            **{k: _clean(v) for k, v in rates.items()},
            "auroc": auroc,
            "n_positive": n_referable_true,
            "n_negative": n - n_referable_true,
            **referable_counts,
        },
        "errors": {
            "underestimation_rate": float(np.mean(pred < true)),
            "off_by_two_or_more_rate": float(np.mean(np.abs(pred - true) >= OFF_BY_TWO_OR_MORE)),
            "referable_miss_rate": _clean(miss),
            "pdr_miss_rate": _clean(pdr_miss),
            "n_referable_true": n_referable_true,
            "n_pdr_true": n_pdr_true,
        },
        "notes": notes,
    }
    result["bootstrap_ci"] = bootstrap_ci(true, pred, n_boot=n_boot, seed=seed) if n_boot > 0 else {}
    return result


def bootstrap_ci(
    true: np.ndarray,
    pred: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Return 95% percentile bootstrap intervals over images for the headline metrics.

    Each of n_boot resamples draws len(true) images with replacement, using a generator
    seeded with seed. A resample where a metric is undefined is left out of that metric's
    percentiles, and n_valid says how many resamples remain. The point estimate is the
    value on the full sample.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    n = int(true.size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    values = {name: np.full(n_boot, np.nan) for name in BOOTSTRAP_METRICS}
    for b in range(n_boot):
        index = draws[b]
        cm = confusion_counts(true[index], pred[index])
        metrics = _cm_metrics(cm)
        for name in BOOTSTRAP_METRICS:
            values[name][b] = metrics[name]
    point = _cm_metrics(confusion_counts(true, pred))
    low_p, high_p = BOOTSTRAP_PERCENTILES
    out: dict[str, dict[str, Any]] = {}
    for name in BOOTSTRAP_METRICS:
        valid = values[name][~np.isnan(values[name])]
        if valid.size:
            low, high = (float(x) for x in np.percentile(valid, [low_p, high_p]))
        else:
            low = high = None
        out[name] = {
            "point": _clean(point[name]),
            "low": low,
            "high": high,
            "n_valid": int(valid.size),
            "n_boot": n_boot,
            "seed": seed,
            "method": "percentile, 95%",
        }
    return out


def majority_baseline_frame(df: pd.DataFrame, majority: int) -> pd.DataFrame:
    """Return the table with every prediction set to the majority grade and no probabilities (M0)."""
    if not 0 <= majority < GRADE_COUNT:
        raise ValueError(f"majority grade must be in 0..{GRADE_COUNT - 1}, got {majority}")
    baseline = df.drop(columns=[c for c in PROB_COLUMNS if c in df.columns]).copy()
    baseline["pred_grade"] = majority
    return baseline
