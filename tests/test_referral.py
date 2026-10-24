"""Tests for referral by confidence, the risk-coverage table, fixed thresholds and the referable threshold (synthetic)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drgrade.calibration import apply_temperature
from drgrade.metrics import full_metrics
from drgrade.referral import (
    REFERRAL_FRACTIONS,
    auroc,
    highest_threshold_reaching,
    operating_point,
    referable_sweep,
    referred_count,
    risk_coverage_table,
    scores_from_probs,
    secondary_row,
    subset_metrics,
    threshold_for_count,
    uncertainty_summary,
)
from drgrade.predict import PREDICTION_COLUMNS

from _prediction_data import register_test, register_val, write_split_for

REPO = Path(__file__).resolve().parents[1]
CALIBRATE_SCRIPT = REPO / "scripts" / "calibrate.py"
REFERRAL_SCRIPT = REPO / "scripts" / "referral.py"
N_CLASSES = 5
N_CASES = 800
SEED = 5


def _world(n: int, seed: int, scale: float = 1.5, strength: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (frame of true and predicted grades, logits) for a synthetic model that is informative and overconfident."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(N_CLASSES, size=n, p=[0.45, 0.2, 0.15, 0.12, 0.08])
    logits = rng.normal(size=(n, N_CLASSES))
    logits[np.arange(n), labels] += strength
    logits *= scale
    frame = pd.DataFrame(
        {
            "image_id": [f"w{seed}{i:05d}" for i in range(n)],
            "true_grade": labels,
            "pred_grade": logits.argmax(axis=1),
        }
    )
    return frame, logits


def _probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return apply_temperature(logits, temperature)


def test_risk_coverage_at_zero_equals_the_full_set_metrics() -> None:
    """The zero-referral row is the full validation set: accuracy, QWK and referable miss rate match full_metrics."""
    frame, logits = _world(N_CASES, SEED)
    scores = scores_from_probs(_probs(logits))
    table = risk_coverage_table(frame, scores.confidence)
    assert list(table["nominal_fraction"]) == list(REFERRAL_FRACTIONS)
    first = table.iloc[0]
    full = full_metrics(frame, n_boot=0)
    assert first["n_referred"] == 0
    assert first["n_retained"] == full["n"]
    assert first["accuracy"] == pytest.approx(full["accuracy"])
    assert first["qwk"] == pytest.approx(full["qwk"])
    assert first["referable_miss_rate_retained"] == pytest.approx(full["errors"]["referable_miss_rate"])


def test_share_of_misses_caught_and_referred_count_never_decrease() -> None:
    """Referral sets are nested, so the number of referable misses referred can only rise with the fraction."""
    frame, logits = _world(N_CASES, SEED + 1)
    scores = scores_from_probs(_probs(logits))
    table = risk_coverage_table(frame, scores.confidence)
    referred = table["referable_misses_referred"].to_numpy()
    shares = table["share_of_misses_referred"].to_numpy().astype(float)
    counts = table["n_referred"].to_numpy()
    assert np.all(np.diff(referred) >= 0)
    assert np.all(np.diff(shares) >= -1e-12)
    assert np.all(np.diff(counts) >= 0)
    assert shares[0] == 0.0
    assert np.all((table["referable_misses_referred"] + table["referable_misses_retained"]) == table["referable_misses_total"])


def test_primary_fraction_refers_the_requested_count() -> None:
    """Referring 20% of validation refers round(0.2 * n) cases when no confidences tie at the threshold."""
    frame, logits = _world(N_CASES, SEED + 2)
    scores = scores_from_probs(_probs(logits))
    count = referred_count(0.20, len(frame))
    threshold = threshold_for_count(scores.confidence, count)
    result = operating_point(frame, scores.confidence, threshold)
    assert result["n_referred"] == count
    assert result["referral_fraction"] == pytest.approx(count / len(frame))


def test_referred_count_rounds_halves_up_and_handles_zero() -> None:
    """round(f * n) with halves rounded up: 0.25 of 10 gives 3, and zero gives 0."""
    assert referred_count(0.20, 10) == 2
    assert referred_count(0.25, 10) == 3
    assert referred_count(0.0, 10) == 0
    with pytest.raises(ValueError):
        referred_count(1.5, 10)


def test_fixed_validation_threshold_applies_to_test_without_refitting() -> None:
    """The validation threshold is used as it is on test: test referral is exactly the cases at or below it."""
    val_frame, val_logits = _world(N_CASES, SEED + 3)
    test_frame, test_logits = _world(N_CASES, SEED + 4)
    val_scores = scores_from_probs(_probs(val_logits))
    threshold = threshold_for_count(val_scores.confidence, referred_count(0.20, len(val_frame)))
    test_scores = scores_from_probs(_probs(test_logits))
    outcome = operating_point(test_frame, test_scores.confidence, threshold)
    assert outcome["threshold"] == threshold
    assert outcome["n_referred"] == int((test_scores.confidence <= threshold).sum())
    retained = test_frame.loc[test_scores.confidence > threshold]
    assert outcome["retained"]["n"] == len(retained)
    assert outcome["retained"]["accuracy"] == pytest.approx(subset_metrics(retained)["accuracy"])


def test_uncertainty_scores_rank_errors_above_chance() -> None:
    """On informative logits, 1 minus confidence and entropy both separate errors better than chance."""
    frame, logits = _world(N_CASES, SEED + 5)
    scores = scores_from_probs(_probs(logits))
    summary = uncertainty_summary(frame, scores)
    assert summary["auroc_incorrect"]["confidence"] > 0.5
    assert summary["auroc_incorrect"]["entropy"] > 0.5
    assert summary["better_for_errors"] in {"confidence", "entropy"}
    assert summary["n"] == len(frame)


def test_auroc_is_none_when_one_class_is_absent() -> None:
    """With no positive or no negative cases the AUROC is undefined and is reported as None."""
    assert auroc(np.array([0.1, 0.9]), np.array([True, True])) is None
    assert auroc(np.array([0.1, 0.9]), np.array([False, False])) is None


def test_highest_threshold_reaching_a_target_sensitivity() -> None:
    """Three positives and three negatives: the highest threshold with sensitivity 2/3 is 0.8, with specificity 1."""
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    positive = np.array([True, True, False, True, False, False])
    sweep = referable_sweep(score, positive)
    assert list(sweep["threshold"]) == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert np.all(np.diff(sweep["sensitivity"].to_numpy()) >= 0)
    chosen = highest_threshold_reaching(sweep, 2.0 / 3.0)
    assert chosen["threshold"] == pytest.approx(0.8)
    assert chosen["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert chosen["specificity"] == pytest.approx(1.0)
    full = highest_threshold_reaching(sweep, 0.9)
    assert full["threshold"] == pytest.approx(0.6)
    assert full["specificity"] == pytest.approx(2.0 / 3.0)


def test_secondary_rule_reports_not_reached_and_finds_the_first_fraction() -> None:
    """The secondary row is the lowest fraction whose retained miss rate is at most the target; None when no row qualifies."""
    table = pd.DataFrame(
        {
            "nominal_fraction": [0.0, 0.01, 0.02],
            "referable_miss_rate_retained": [0.2, 0.06, 0.05],
        }
    )
    assert secondary_row(table)["nominal_fraction"] == pytest.approx(0.02)
    never = table.assign(referable_miss_rate_retained=[0.2, 0.1, 0.07])
    assert secondary_row(never) is None


def _write_csv(path: Path, frame: pd.DataFrame, logits: np.ndarray) -> None:
    probs = _probs(logits)
    out = frame.copy()
    for k in range(N_CLASSES):
        out[f"prob_{k}"] = probs[:, k]
    for k in range(N_CLASSES):
        out[f"logit_{k}"] = logits[:, k]
    out[list(PREDICTION_COLUMNS)].to_csv(path, index=False)


def _load_script(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_shuffled_test_labels_change_neither_temperature_nor_thresholds(tmp_path: Path) -> None:
    """Calibration and referral with a shuffled test file give the same T and the same thresholds as without it."""
    val_frame, val_logits = _world(N_CASES, SEED + 6)
    test_frame, test_logits = _world(N_CASES, SEED + 7)
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    val_csv, test_csv = run / "predictions" / "val.csv", run / "predictions" / "test.csv"
    _write_csv(val_csv, val_frame, val_logits)
    test_shuffled = test_frame.copy()
    test_shuffled["true_grade"] = np.random.default_rng(2).permutation(test_frame["true_grade"].to_numpy())
    _write_csv(test_csv, test_shuffled, test_logits)
    register_val(val_csv)
    register_test(test_csv)
    split = str(tmp_path / "split.csv")
    write_split_for(split, val_csv, test_csv)

    calibrate = _load_script("calibrate_for_referral", CALIBRATE_SCRIPT)
    referral = _load_script("referral_script", REFERRAL_SCRIPT)
    assert calibrate.main(["--run", str(run), "--split", split]) == 0
    assert referral.main(["--run", str(run), "--out", str(tmp_path / "no_test"), "--split", split]) == 0
    assert calibrate.main(["--run", str(run), "--out", str(tmp_path / "cal_test"), "--test", str(test_csv), "--split", split]) == 0
    assert referral.main([
        "--run", str(run),
        "--calibration", str(tmp_path / "cal_test" / "calibration.json"),
        "--test", str(test_csv),
        "--out", str(tmp_path / "with_test"),
        "--split", split,
    ]) == 0

    cal_plain = json.loads((run / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    cal_test = json.loads((tmp_path / "cal_test" / "calibration.json").read_text(encoding="utf-8"))
    assert cal_test["temperature"] == cal_plain["temperature"]

    plain = json.loads((tmp_path / "no_test" / "referral.json").read_text(encoding="utf-8"))
    with_test = json.loads((tmp_path / "with_test" / "referral.json").read_text(encoding="utf-8"))
    assert with_test["primary"]["threshold"] == plain["primary"]["threshold"]
    assert with_test["secondary"]["threshold"] == plain["secondary"]["threshold"]
    assert with_test["primary"]["validation"] == plain["primary"]["validation"]
    assert with_test["primary"]["test"] is not None and plain["primary"]["test"] is None

    referable_plain = json.loads((tmp_path / "no_test" / "referable_threshold.json").read_text(encoding="utf-8"))
    referable_test = json.loads((tmp_path / "with_test" / "referable_threshold.json").read_text(encoding="utf-8"))
    assert referable_test["validation"]["threshold"] == referable_plain["validation"]["threshold"]
    assert referable_test["test"]["threshold"] == referable_plain["validation"]["threshold"]


def test_referable_threshold_script_matches_the_reference_sensitivity(tmp_path: Path) -> None:
    """With a reference file, the chosen threshold reaches the reference sensitivity on validation and is the highest such."""
    val_frame, val_logits = _world(N_CASES, SEED + 8)
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    val_csv = run / "predictions" / "val.csv"
    _write_csv(val_csv, val_frame, val_logits)
    register_val(val_csv)
    split = str(tmp_path / "split.csv")
    write_split_for(split, val_csv)
    reference_logits = val_logits + np.random.default_rng(9).normal(scale=0.6, size=val_logits.shape)
    reference_frame = val_frame.copy()
    reference_frame["pred_grade"] = reference_logits.argmax(axis=1)
    _write_csv(tmp_path / "reference.csv", reference_frame, reference_logits)
    register_val(tmp_path / "reference.csv", run_name="reference", checkpoint="c1")

    calibrate = _load_script("calibrate_for_referable", CALIBRATE_SCRIPT)
    referral = _load_script("referral_for_referable", REFERRAL_SCRIPT)
    assert calibrate.main(["--run", str(run), "--split", split]) == 0
    assert referral.main([
        "--run", str(run),
        "--match-sensitivity-of", str(tmp_path / "reference.csv"),
        "--split", split,
    ]) == 0
    payload = json.loads((run / "referral" / "referable_threshold.json").read_text(encoding="utf-8"))
    target = payload["reference"]["sensitivity"]
    assert payload["target_sensitivity"] == pytest.approx(target)
    assert payload["validation"]["sensitivity"] >= target - 1e-9
    higher = [row for row in payload["threshold_grid_validation"] if row["threshold"] > payload["validation"]["threshold"]]
    assert all(row["sensitivity"] <= payload["validation"]["sensitivity"] + 1e-9 for row in higher)
