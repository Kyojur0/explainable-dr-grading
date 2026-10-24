"""Tests for the validation metrics used during training (synthetic values only)."""

from __future__ import annotations

import pytest

from drgrade.metrics import validation_metrics


def test_perfect_predictions_give_one() -> None:
    """Perfect predictions over all five grades give 1.0 for every metric."""
    truth = [0, 1, 2, 3, 4, 2, 1, 0]
    metrics = validation_metrics(truth, truth)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)
    assert metrics["qwk"] == pytest.approx(1.0)


def test_small_example_matches_hand_calculation() -> None:
    """Hand calculation for y_true=[0,0,1,1], y_pred=[0,1,1,1] (grades 2 to 4 never occur).

    Accuracy is 3/4. Grade 0: precision 1, recall 1/2, F1 2/3. Grade 1: precision 2/3,
    recall 1, F1 4/5. Grades 2 to 4 have F1 0, so macro-F1 is (2/3 + 4/5) / 5. Weighted
    F1 uses the true supports (2 and 2), so it is (2/3 + 4/5) / 2. Quadratic kappa:
    observed weighted disagreement 1/16, expected 2/16, so kappa is 1 - 1/2 = 0.5.
    """
    metrics = validation_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 5)
    assert metrics["weighted_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert metrics["qwk"] == pytest.approx(0.5)


def test_qwk_penalises_off_by_two_more_than_off_by_one() -> None:
    """Two errors of equal count: one off by one grade, the other off by two grades."""
    truth = [0, 1, 2, 3, 4, 2]
    one_off = validation_metrics(truth, [0, 1, 2, 3, 4, 1])
    two_off = validation_metrics(truth, [0, 1, 2, 3, 4, 0])
    assert one_off["accuracy"] == pytest.approx(two_off["accuracy"])
    assert one_off["qwk"] > two_off["qwk"]


def test_rejects_empty_or_mismatched_inputs() -> None:
    """Empty inputs and length mismatches raise ValueError."""
    with pytest.raises(ValueError):
        validation_metrics([], [])
    with pytest.raises(ValueError):
        validation_metrics([0, 1], [0])


# ---------------------------------------------------------------------------
# Full metrics (brief 9.2), hand-checked on one small example
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from drgrade.metrics import (  # noqa: E402
    bootstrap_ci,
    full_metrics,
    majority_baseline_frame,
    majority_grade,
)

# true:  0 0 2 2 3 4 4 4      pred: 0 1 2 1 3 4 2 4
HAND_TRUE = [0, 0, 2, 2, 3, 4, 4, 4]
HAND_PRED = [0, 1, 2, 1, 3, 4, 2, 4]


def _with_probs(true: list[int], pred: list[int]) -> pd.DataFrame:
    """Return a predictions table whose referable score separates the two groups perfectly."""
    rows = []
    for t, p in zip(true, pred):
        referable = 0.9 if t >= 2 else 0.1
        probs = [1 - referable, 0.0, 0.0, 0.0, 0.0]
        probs[2] = referable / 3
        probs[3] = referable / 3
        probs[4] = referable / 3
        rows.append({"true_grade": t, "pred_grade": p, **{f"prob_{i}": probs[i] for i in range(5)}})
    return pd.DataFrame(rows)


def test_hand_checked_confusion_and_referable_numbers() -> None:
    """Accuracy, referable rates, error rates, macro and weighted F1 match the hand calculation."""
    frame = pd.DataFrame({"true_grade": HAND_TRUE, "pred_grade": HAND_PRED})
    m = full_metrics(frame, n_boot=0)
    assert m["confusion_counts"] == [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 2],
    ]
    assert m["accuracy"] == pytest.approx(5 / 8)
    ref = m["referable"]
    assert ref["tp"] == 5 and ref["fn"] == 1 and ref["tn"] == 2 and ref["fp"] == 0
    assert ref["sensitivity"] == pytest.approx(5 / 6)
    assert ref["specificity"] == pytest.approx(1.0)
    assert ref["ppv"] == pytest.approx(1.0)
    assert ref["npv"] == pytest.approx(2 / 3)
    assert m["errors"]["underestimation_rate"] == pytest.approx(2 / 8)
    assert m["errors"]["off_by_two_or_more_rate"] == pytest.approx(1 / 8)
    assert m["errors"]["referable_miss_rate"] == pytest.approx(1 / 6)
    assert m["errors"]["pdr_miss_rate"] == pytest.approx(1 / 3)
    # Per-class F1: grade 0 = 2/3, grade 1 = 0 (predicted twice, never true), grade 2 = 1/2, grade 3 = 1, grade 4 = 4/5.
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0 + 1 / 2 + 1 + 4 / 5) / 5)
    assert m["weighted_f1"] == pytest.approx((2 * 2 / 3 + 2 * 1 / 2 + 1 * 1 + 3 * 4 / 5) / 8)
    assert m["per_class"][1]["recall"] is None  # no true grade 1 images
    assert m["per_class"][1]["f1"] == pytest.approx(0.0)
    assert m["confusion_norm"][1] == [None] * 5  # undefined row
    assert m["n"] == 8


def test_full_metrics_agree_with_sklearn_on_random_data() -> None:
    """The confusion-matrix versions equal the scikit-learn values for accuracy, macro-F1 and QWK."""
    rng = np.random.default_rng(3)
    true = rng.integers(0, 5, 400)
    pred = np.clip(true + rng.integers(-1, 2, 400), 0, 4)
    frame = pd.DataFrame({"true_grade": true, "pred_grade": pred})
    m = full_metrics(frame, n_boot=0)
    v = validation_metrics(true, pred)
    assert m["accuracy"] == pytest.approx(v["accuracy"], abs=1e-12)
    assert m["macro_f1"] == pytest.approx(v["macro_f1"], abs=1e-12)
    assert m["weighted_f1"] == pytest.approx(v["weighted_f1"], abs=1e-12)
    assert m["qwk"] == pytest.approx(cohen_kappa_score(true, pred, weights="quadratic"), abs=1e-12)


def test_auroc_uses_the_referable_score() -> None:
    """A score that separates the groups gives AUROC 1; the majority-baseline frame gives None."""
    frame = _with_probs(HAND_TRUE, HAND_PRED)
    m = full_metrics(frame, n_boot=0)
    assert m["referable"]["auroc"] == pytest.approx(1.0)
    baseline = full_metrics(majority_baseline_frame(frame, 0), n_boot=0)
    assert baseline["referable"]["auroc"] is None
    assert any("AUROC is null" in note for note in baseline["notes"])


def test_bootstrap_is_deterministic_and_contains_the_point() -> None:
    """Repeated runs with the same seed agree, a different seed gives different draws, and the CI brackets the point."""
    rng = np.random.default_rng(11)
    true = rng.integers(0, 5, 300)
    pred = np.clip(true + rng.integers(-1, 2, 300), 0, 4)
    frame = pd.DataFrame({"true_grade": true, "pred_grade": pred})
    first = full_metrics(frame, n_boot=200, seed=42)["bootstrap_ci"]
    second = full_metrics(frame, n_boot=200, seed=42)["bootstrap_ci"]
    assert first == second
    other = full_metrics(frame, n_boot=200, seed=7)["bootstrap_ci"]
    assert other["accuracy"]["low"] != first["accuracy"]["low"]
    assert set(first) == {"accuracy", "macro_f1", "qwk", "referable_sensitivity", "referable_miss_rate"}
    for name, interval in first.items():
        assert interval["n_valid"] == 200
        assert interval["low"] <= interval["point"] <= interval["high"], name
    assert full_metrics(frame, n_boot=0)["bootstrap_ci"] == {}
    with pytest.raises(ValueError):
        bootstrap_ci(true, pred, n_boot=0)


def test_majority_grade_and_baseline() -> None:
    """The majority grade breaks ties towards the lower grade; M0 predicts it for every image."""
    assert majority_grade([0, 0, 1, 1, 2, 4]) == 0
    assert majority_grade([3, 3, 4]) == 3
    with pytest.raises(ValueError):
        majority_grade([])
    frame = pd.DataFrame({"true_grade": HAND_TRUE, "pred_grade": HAND_PRED})
    baseline = majority_baseline_frame(frame, 0)
    assert set(baseline["pred_grade"]) == {0}
    m0 = full_metrics(baseline, n_boot=0)
    assert m0["accuracy"] == pytest.approx(2 / 8)
    assert m0["referable"]["sensitivity"] == pytest.approx(0.0)
    assert m0["referable"]["specificity"] == pytest.approx(1.0)
    assert m0["referable"]["auroc"] is None


def test_undefined_values_are_null_not_errors() -> None:
    """No positives, no grade 4 and a constant prediction give None (with notes), never an exception."""
    no_positives = pd.DataFrame({"true_grade": [0, 1, 0, 1], "pred_grade": [0, 0, 1, 1]})
    m = full_metrics(no_positives, n_boot=20)
    assert m["referable"]["sensitivity"] is None
    assert m["errors"]["referable_miss_rate"] is None
    assert m["errors"]["pdr_miss_rate"] is None
    assert m["referable"]["specificity"] is not None
    assert m["bootstrap_ci"]["referable_sensitivity"]["n_valid"] == 0
    assert m["bootstrap_ci"]["referable_sensitivity"]["low"] is None
    assert any("referable sensitivity is undefined" in note for note in m["notes"])

    constant = pd.DataFrame({"true_grade": [0, 0], "pred_grade": [0, 0]})
    c = full_metrics(constant, n_boot=0)
    assert c["qwk"] is None
    assert c["accuracy"] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        full_metrics(pd.DataFrame({"true_grade": [], "pred_grade": []}), n_boot=0)
    with pytest.raises(ValueError):
        full_metrics(pd.DataFrame({"true_grade": [5], "pred_grade": [0]}), n_boot=0)
