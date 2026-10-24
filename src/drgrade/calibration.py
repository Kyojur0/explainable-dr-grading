"""Stage 7: temperature scaling, calibration metrics and reliability bins (DECISIONS.md D28 and D29).

A single scalar temperature T divides the logits before the softmax. T is fitted on
validation logits only, by minimising the negative log-likelihood. Dividing every logit
of an image by the same T never changes the argmax, so predicted grades are unchanged.

Expected calibration error (ECE) uses 15 equal-width bins on the confidence (the largest
calibrated probability). Each bin is [lower, upper), and the last bin also holds 1.0.

Entry points:
- load_logit_table(path): reads a predictions CSV (plain or TTA) and checks it.
- fit_temperature(logits, labels): the one fitted parameter.
- apply_temperature(logits, T): calibrated probabilities.
- ece, nll, brier, reliability_bins, summarise: the metrics reported before and after.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from drgrade.io import GRADE_VALUES
from drgrade.predict import LOGIT_COLUMNS, PROB_COLUMNS, sha256_file

logger = logging.getLogger(__name__)

DEFAULT_N_BINS = 15
INITIAL_TEMPERATURE = 1.0  # the optimisation starts at T = 1 (no scaling)
LBFGS_MAX_ITER = 100
LBFGS_LEARNING_RATE = 1.0
LBFGS_LINE_SEARCH = "strong_wolfe"
PROB_FLOOR = 1e-12  # probabilities below this are clipped before the logarithm in NLL
PROB_TOLERANCE = 1e-6  # saved probabilities must equal softmax(saved logits) to this tolerance
TTA_SUM_TOLERANCE = 1e-6  # TTA logits are log(averaged probabilities): exp(logits) sums to 1
CALIBRATION_JSON = "calibration.json"
GRADE_MAX = max(GRADE_VALUES)


@dataclass(frozen=True)
class LogitTable:
    """A predictions file read for calibration and referral: ids, labels, logits and checks."""

    path: Path
    frame: pd.DataFrame
    ids: np.ndarray
    labels: np.ndarray
    pred: np.ndarray
    logits: np.ndarray
    tta: bool
    sha256: str


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Return row-wise softmax in float64 with the usual max subtraction for stability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    expd = np.exp(shifted)
    return expd / expd.sum(axis=1, keepdims=True)


def detect_tta(logits: np.ndarray) -> bool:
    """Return True when every row of logits exponentiates to probabilities summing to 1.

    TTA files store log(averaged probabilities) (DECISIONS.md D24), so their exponentials sum
    to one. Raw logits of a plain prediction do not, so this check tells the two apart.
    """
    sums = np.exp(logits).sum(axis=1)
    return bool(np.all(np.abs(sums - 1.0) <= TTA_SUM_TOLERANCE))


def load_logit_table(path: Path | str) -> LogitTable:
    """Read a predictions CSV written by predict.py or by training and check it.

    Checks: required columns, unique ids, grades in range, finite logits, pred_grade equal to
    the argmax of the logits, and probability columns equal to softmax(logits) within
    PROB_TOLERANCE. Raises ValueError with the reason when a check fails.
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, dtype={"image_id": str}, float_precision="round_trip")
    required = ("image_id", "true_grade", "pred_grade", *LOGIT_COLUMNS, *PROB_COLUMNS)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks the columns {missing}")
    if len(frame) == 0:
        raise ValueError(f"{csv_path} has no rows")
    ids = frame["image_id"].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{csv_path} lists an image id more than once")
    labels = frame["true_grade"].to_numpy().astype(np.int64)
    pred = frame["pred_grade"].to_numpy().astype(np.int64)
    for name, values in (("true_grade", labels), ("pred_grade", pred)):
        if values.min() < 0 or values.max() > GRADE_MAX:
            raise ValueError(f"{csv_path}: {name} must be in 0..{GRADE_MAX}")
    logits = frame[list(LOGIT_COLUMNS)].to_numpy().astype(np.float64)
    if not np.isfinite(logits).all():
        raise ValueError(f"{csv_path} has non-finite logits")
    if not np.array_equal(logits.argmax(axis=1), pred):
        raise ValueError(f"{csv_path}: pred_grade is not the argmax of the logits")
    probs_file = frame[list(PROB_COLUMNS)].to_numpy().astype(np.float64)
    if np.abs(_softmax(logits) - probs_file).max() > PROB_TOLERANCE:
        raise ValueError(f"{csv_path}: probability columns differ from softmax(logits)")
    tta = detect_tta(logits)
    logger.info("%s: %d rows, tta=%s", csv_path, len(frame), tta)
    return LogitTable(
        path=csv_path,
        frame=frame,
        ids=ids,
        labels=labels,
        pred=pred,
        logits=logits,
        tta=tta,
        sha256=sha256_file(csv_path),
    )


def _check_logits_and_labels(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 logits (N by K) and int64 labels (N), after shape and range checks."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError("logits must be a non-empty array of shape (N, K) with K >= 2")
    if y.shape != (z.shape[0],):
        raise ValueError("labels must have one entry per row of logits")
    if y.min() < 0 or y.max() >= z.shape[1]:
        raise ValueError("labels must index the columns of logits")
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    return z, y


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the single temperature T > 0 that minimises the NLL of softmax(logits / T).

    The optimisation variable is log T, so T stays positive. It uses LBFGS in float64 with a
    strong Wolfe line search, at most LBFGS_MAX_ITER iterations, starting from T = 1. Raises
    ValueError for empty or inconsistent input, and RuntimeError if the result is not a
    positive finite number.
    """
    z, y = _check_logits_and_labels(logits, labels)
    logits_t = torch.as_tensor(z, dtype=torch.float64)
    labels_t = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.full((1,), math.log(INITIAL_TEMPERATURE), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [log_t],
        lr=LBFGS_LEARNING_RATE,
        max_iter=LBFGS_MAX_ITER,
        line_search_fn=LBFGS_LINE_SEARCH,
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = F.cross_entropy(logits_t / torch.exp(log_t), labels_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(torch.exp(log_t.detach()).item())
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"temperature fit returned an invalid value: {temperature}")
    logger.info("fitted temperature %.6f on %d rows", temperature, z.shape[0])
    return temperature


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return softmax(logits / temperature) in float64. The argmax of each row is unchanged."""
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be a positive finite number, got {temperature}")
    z = np.asarray(logits, dtype=np.float64)
    return _softmax(z / temperature)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> list[dict[str, Any]]:
    """Return one row per equal-width confidence bin: edges, count, mean confidence and accuracy.

    Confidence is the largest probability of each row. A bin with no images has count 0 and
    null mean confidence and accuracy.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "mean_confidence": float(confidence[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return rows


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> tuple[float, list[dict[str, Any]]]:
    """Return the expected calibration error and the per-bin table.

    ECE is the sum over bins of (count / N) times |accuracy - mean confidence|, with empty bins
    contributing nothing.
    """
    bins = reliability_bins(probs, labels, n_bins)
    total = float(labels.size)
    value = 0.0
    for row in bins:
        if row["count"]:
            value += row["count"] / total * abs(row["accuracy"] - row["mean_confidence"])
    return float(value), bins


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean negative log-likelihood of the true grade (probabilities floored at PROB_FLOOR)."""
    chosen = probs[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(chosen, PROB_FLOOR, None))))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return the multi-class Brier score: the mean over images of the sum over classes of (p - onehot)^2."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.size), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def summarise(probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> dict[str, Any]:
    """Return n, accuracy, ECE, NLL, Brier and the reliability bins for one set of probabilities."""
    value, bins = ece(probs, labels, n_bins)
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(probs.argmax(axis=1) == labels)),
        "ece": value,
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "bins": bins,
    }
