"""Stage 7: referral by confidence and by a referable score (DECISIONS.md D30 to D32).

Everything here works on calibrated probabilities (softmax of logits / T, with T from
calibration.json). Nothing is fitted here: thresholds and fractions are chosen on validation
by the rules below, and a fixed threshold is then applied to test.

Referral sets are defined by a confidence threshold: a case is referred when its calibrated
max probability is at or below the threshold. For a nominal fraction f the threshold is the
confidence of the k-th most uncertain validation case, k = round(f * n). Because
the sets are nested for increasing thresholds, the risk-coverage curve is monotone in the
share of referable misses caught.

Pre-declared rules (DECISIONS.md D30, decided before any test prediction):
- primary: refer the most uncertain 20% of cases (score = calibrated max probability);
- secondary: the lowest nominal fraction, in steps of 1%, at which the retained referable miss
  rate on validation is 5% or below; if no fraction up to 50% reaches it, report that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from drgrade.calibration import PROB_FLOOR
from drgrade.io import GRADE_VALUES
from drgrade.metrics import REFERABLE_GRADE, REFERABLE_SCORE_GRADES, full_metrics

PRIMARY_FRACTION = 0.20
SECONDARY_TARGET_MISS_RATE = 0.05
FRACTION_STEP = 0.01
MAX_FRACTION = 0.50
REFERABLE_SENSITIVITY_TARGET = 0.90  # used when no reference run is given
THRESHOLD_GRID_STEP = 0.01
SENSITIVITY_TOLERANCE = 1e-12
THRESHOLD_GRID_STEPS = round(1.0 / THRESHOLD_GRID_STEP)
ROUNDING_OFFSET = 0.5
REFERRAL_FRACTIONS: tuple[float, ...] = tuple(
    round(i * FRACTION_STEP, 2) for i in range(round(MAX_FRACTION / FRACTION_STEP) + 1)
)
RISK_COLUMNS: tuple[str, ...] = (
    "nominal_fraction",
    "referral_fraction",
    "threshold",
    "n_referred",
    "n_retained",
    "accuracy",
    "qwk",
    "referable_miss_rate_retained",
    "referable_misses_total",
    "referable_misses_referred",
    "referable_misses_retained",
    "share_of_misses_referred",
)


@dataclass(frozen=True)
class Scores:
    """Per-image scores from calibrated probabilities.

    confidence: largest probability (low means uncertain).
    entropy: natural-log predictive entropy (high means uncertain).
    referable: p(2) + p(3) + p(4) (the referable score).
    """

    confidence: np.ndarray
    entropy: np.ndarray
    referable: np.ndarray


def scores_from_probs(probs: np.ndarray) -> Scores:
    """Return confidence, entropy and the referable score for an N by 5 probability matrix."""
    if probs.ndim != 2 or probs.shape[1] != len(GRADE_VALUES):
        raise ValueError(f"probabilities must have shape (N, {len(GRADE_VALUES)})")
    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(np.clip(probs, PROB_FLOOR, None))).sum(axis=1)
    referable = probs[:, list(REFERABLE_SCORE_GRADES)].sum(axis=1)
    return Scores(confidence=confidence, entropy=entropy, referable=referable)


def referred_count(fraction: float, n: int) -> int:
    """Return round(fraction * n) with halves rounded up, so the count is the same on every platform."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return int(math.floor(fraction * n + ROUNDING_OFFSET))


def threshold_for_count(confidence: np.ndarray, count: int) -> float | None:
    """Return the confidence of the count-th most uncertain case (None when count is 0)."""
    if count <= 0:
        return None
    if count > confidence.size:
        raise ValueError("count is larger than the number of cases")
    return float(np.sort(confidence, kind="stable")[count - 1])


def referred_mask(confidence: np.ndarray, threshold: float | None) -> np.ndarray:
    """Return True for the cases referred by a confidence threshold: confidence at or below it."""
    if threshold is None:
        return np.zeros(confidence.shape, dtype=bool)
    return confidence <= threshold


def referable_miss_mask(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return True where the true grade is referable and the predicted grade is not."""
    return (true >= REFERABLE_GRADE) & (pred < REFERABLE_GRADE)


def auroc(uncertainty: np.ndarray, positive: np.ndarray) -> float | None:
    """Return the AUROC of an uncertainty score for the positive class, or None when one class is absent."""
    if positive.size == 0 or positive.all() or (~positive).all():
        return None
    return float(roc_auc_score(positive.astype(int), uncertainty))


def subset_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return n, accuracy, QWK (null unless two or more true grades are present) and referable miss rate.

    frame needs true_grade and pred_grade. An empty frame gives n 0 and null metrics.
    """
    if len(frame) == 0:
        return {"n": 0, "accuracy": None, "qwk": None, "referable_miss_rate": None}
    base = full_metrics(frame[["true_grade", "pred_grade"]], n_boot=0)
    qwk = base["qwk"] if frame["true_grade"].nunique() >= 2 else None
    return {
        "n": base["n"],
        "accuracy": base["accuracy"],
        "qwk": qwk,
        "referable_miss_rate": base["errors"]["referable_miss_rate"],
    }


def risk_coverage_table(frame: pd.DataFrame, confidence: np.ndarray) -> pd.DataFrame:
    """Return one row per nominal referral fraction (0, 0.01, ..., 0.50) with metrics on the retained cases.

    The threshold for each row is the confidence of the k-th most uncertain case, and the referred
    set is every case at or below it, so rows are nested and the share of misses caught never falls.
    """
    true = frame["true_grade"].to_numpy().astype(np.int64)
    pred = frame["pred_grade"].to_numpy().astype(np.int64)
    misses = referable_miss_mask(true, pred)
    total_misses = int(misses.sum())
    n = len(frame)
    rows: list[dict[str, Any]] = []
    for fraction in REFERRAL_FRACTIONS:
        threshold = threshold_for_count(confidence, referred_count(fraction, n))
        mask = referred_mask(confidence, threshold)
        retained = frame.loc[~mask]
        metrics = subset_metrics(retained)
        referred_misses = int((misses & mask).sum())
        rows.append(
            {
                "nominal_fraction": fraction,
                "referral_fraction": float(mask.mean()),
                "threshold": threshold,
                "n_referred": int(mask.sum()),
                "n_retained": metrics["n"],
                "accuracy": metrics["accuracy"],
                "qwk": metrics["qwk"],
                "referable_miss_rate_retained": metrics["referable_miss_rate"],
                "referable_misses_total": total_misses,
                "referable_misses_referred": referred_misses,
                "referable_misses_retained": int((misses & ~mask).sum()),
                "share_of_misses_referred": (referred_misses / total_misses) if total_misses else None,
            }
        )
    return pd.DataFrame(rows, columns=list(RISK_COLUMNS))


def secondary_row(table: pd.DataFrame) -> pd.Series | None:
    """Return the first row (lowest nominal fraction) whose retained referable miss rate is at or below the target."""
    rate = table["referable_miss_rate_retained"]
    reached = rate.notna() & (rate.astype(float) <= SECONDARY_TARGET_MISS_RATE)
    if not reached.any():
        return None
    return table.loc[reached.idxmax()]


def operating_point(frame: pd.DataFrame, confidence: np.ndarray, threshold: float | None) -> dict[str, Any]:
    """Return the referral outcome of a fixed confidence threshold on one set of cases."""
    true = frame["true_grade"].to_numpy().astype(np.int64)
    pred = frame["pred_grade"].to_numpy().astype(np.int64)
    misses = referable_miss_mask(true, pred)
    mask = referred_mask(confidence, threshold)
    total = int(misses.sum())
    referred = int((misses & mask).sum())
    return {
        "threshold": threshold,
        "n": int(len(frame)),
        "n_referred": int(mask.sum()),
        "referral_fraction": float(mask.mean()),
        "retained": subset_metrics(frame.loc[~mask]),
        "referable_misses_total": total,
        "referable_misses_referred": referred,
        "share_of_misses_referred": (referred / total) if total else None,
    }


def uncertainty_summary(frame: pd.DataFrame, scores: Scores) -> dict[str, Any]:
    """Return the AUROC of each uncertainty score for separating errors, and for separating referable misses.

    Both scores are oriented so that a higher value means more uncertain: 1 minus confidence, and
    entropy. The positive class is an incorrect prediction, or a referable miss.
    """
    true = frame["true_grade"].to_numpy().astype(np.int64)
    pred = frame["pred_grade"].to_numpy().astype(np.int64)
    incorrect = true != pred
    misses = referable_miss_mask(true, pred)
    uncertainty = {"confidence": 1.0 - scores.confidence, "entropy": scores.entropy}
    errors = {name: auroc(values, incorrect) for name, values in uncertainty.items()}
    referable = {name: auroc(values, misses) for name, values in uncertainty.items()}
    ranked = [(value, name) for name, value in errors.items() if value is not None]
    better = max(ranked, key=lambda item: (item[0], item[1] == "confidence"))[1] if ranked else None
    return {
        "n": int(len(frame)),
        "n_incorrect": int(incorrect.sum()),
        "n_referable_miss": int(misses.sum()),
        "auroc_incorrect": errors,
        "auroc_referable_miss": referable,
        "better_for_errors": better,
    }


def referable_sweep(score: np.ndarray, positive: np.ndarray) -> pd.DataFrame:
    """Return sensitivity and specificity at every distinct score, in descending threshold order.

    A case is flagged when its score is at or above the threshold. Sensitivity is non-decreasing
    down the table.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("the referable sweep needs both referable and non-referable cases")
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_pos = positive[order]
    ends = np.append(np.nonzero(np.diff(sorted_score))[0], sorted_score.size - 1)
    true_pos = np.cumsum(sorted_pos)[ends]
    false_pos = np.cumsum(~sorted_pos)[ends]
    return pd.DataFrame(
        {
            "threshold": sorted_score[ends],
            "flagged": ends + 1,
            "sensitivity": true_pos / n_pos,
            "specificity": 1.0 - false_pos / n_neg,
        }
    )


def highest_threshold_reaching(sweep: pd.DataFrame, target: float) -> pd.Series:
    """Return the highest threshold of the sweep whose sensitivity is at least target."""
    if not 0.0 < target <= 1.0:
        raise ValueError("the sensitivity target must lie in (0, 1]")
    reached = sweep["sensitivity"].to_numpy() >= target - SENSITIVITY_TOLERANCE
    if not reached.any():
        raise ValueError("no threshold reaches the target sensitivity")
    return sweep.iloc[int(np.argmax(reached))]


def threshold_grid(score: np.ndarray, positive: np.ndarray) -> list[dict[str, float]]:
    """Return sensitivity and specificity at thresholds 0.00, 0.01, ..., 1.00 (score at or above threshold is flagged)."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    grid: list[dict[str, float]] = []
    for step in range(THRESHOLD_GRID_STEPS + 1):
        threshold = round(step * THRESHOLD_GRID_STEP, 2)
        flagged = score >= threshold
        grid.append(
            {
                "threshold": threshold,
                "sensitivity": float((flagged & positive).sum()) / n_pos if n_pos else math.nan,
                "specificity": float((~flagged & ~positive).sum()) / n_neg if n_neg else math.nan,
            }
        )
    return grid


def fixed_threshold_outcome(score: np.ndarray, positive: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return sensitivity, specificity and counts for a fixed threshold on any set of cases."""
    flagged = score >= threshold
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    return {
        "threshold": float(threshold),
        "n": int(score.size),
        "n_referable": n_pos,
        "n_flagged": int(flagged.sum()),
        "sensitivity": (float((flagged & positive).sum()) / n_pos) if n_pos else None,
        "specificity": (float((~flagged & ~positive).sum()) / n_neg) if n_neg else None,
    }
