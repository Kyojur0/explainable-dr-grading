"""Tests for temperature scaling, ECE, NLL, Brier and the calibration script (synthetic data, CPU, no model)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drgrade.calibration import (
    apply_temperature,
    brier,
    detect_tta,
    ece,
    fit_temperature,
    load_logit_table,
    nll,
    reliability_bins,
)
from drgrade.predict import PREDICTION_COLUMNS

from _prediction_data import register_test, register_val, write_split_for

REPO = Path(__file__).resolve().parents[1]
CALIBRATE_SCRIPT = REPO / "scripts" / "calibrate.py"
N_CLASSES = 5
N_LARGE = 20000
SEED = 7


def _calibrated_world(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return true class probabilities p and labels sampled from p, so that log(p) is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(N_CLASSES, 0.5), size=n)
    labels = np.array([rng.choice(N_CLASSES, p=row) for row in probs], dtype=np.int64)
    return probs, labels


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_perfectly_calibrated_data_gives_small_ece_and_unit_temperature() -> None:
    """With labels drawn from the stated probabilities, T is near 1 and ECE is near 0."""
    probs, labels = _calibrated_world(N_LARGE, SEED)
    logits = np.log(probs)
    temperature = fit_temperature(logits, labels)
    assert abs(temperature - 1.0) < 0.05
    value, _ = ece(apply_temperature(logits, temperature), labels)
    assert value < 0.02


def test_overconfident_logits_fit_temperature_near_three_and_lower_nll() -> None:
    """Logits multiplied by 3 are overconfident, so T is near 3 and the scaled NLL is lower."""
    probs, labels = _calibrated_world(N_LARGE, SEED + 1)
    overconfident = 3.0 * np.log(probs)
    before = nll(apply_temperature(overconfident, 1.0), labels)
    temperature = fit_temperature(overconfident, labels)
    assert abs(temperature - 3.0) < 0.15
    after = nll(apply_temperature(overconfident, temperature), labels)
    assert after < before


def test_temperature_never_changes_the_argmax() -> None:
    """Dividing every logit of an image by the same positive T keeps the predicted grade."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(500, N_CLASSES))
    for temperature in (0.05, 0.5, 1.0, 3.0, 50.0):
        probs = apply_temperature(logits, temperature)
        assert np.array_equal(probs.argmax(axis=1), logits.argmax(axis=1))
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_temperature_rejects_empty_and_inconsistent_input() -> None:
    """Empty logits, mismatched labels and out-of-range labels raise ValueError."""
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((0, N_CLASSES)), np.zeros(0, dtype=int))
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((4, N_CLASSES)), np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((4, N_CLASSES)), np.array([0, 1, 2, 9]))


def test_apply_temperature_rejects_non_positive_temperature() -> None:
    """A temperature must be a positive finite number."""
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            apply_temperature(np.zeros((2, 2)), bad)


def test_ece_matches_a_hand_calculation() -> None:
    """Four images in 10 bins: two at confidence 0.95 (one correct), two at 0.65 (both correct).

    Bin 9 has accuracy 0.5 and mean confidence 0.95, weight 2/4, so 0.225. Bin 6 has accuracy 1.0
    and mean confidence 0.65, weight 2/4, so 0.175. ECE is 0.4.
    """
    probs = np.array(
        [
            [0.95, 0.0125, 0.0125, 0.0125, 0.0125],
            [0.95, 0.0125, 0.0125, 0.0125, 0.0125],
            [0.65, 0.0875, 0.0875, 0.0875, 0.0875],
            [0.65, 0.0875, 0.0875, 0.0875, 0.0875],
        ]
    )
    labels = np.array([0, 1, 0, 0])
    value, bins = ece(probs, labels, n_bins=10)
    assert value == pytest.approx(0.4)
    filled = {row["bin"]: row for row in bins if row["count"]}
    assert set(filled) == {6, 9}
    assert filled[9]["count"] == 2 and filled[9]["accuracy"] == pytest.approx(0.5)
    assert filled[9]["mean_confidence"] == pytest.approx(0.95)
    assert filled[6]["accuracy"] == pytest.approx(1.0)
    assert filled[6]["mean_confidence"] == pytest.approx(0.65)
    assert sum(row["count"] for row in bins) == 4


def test_reliability_bins_cover_the_unit_interval_with_equal_width() -> None:
    """Bins are equal width, contiguous, and a confidence of exactly 1.0 falls in the last bin."""
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    bins = reliability_bins(probs, np.array([0, 1]), n_bins=4)
    assert [row["lower"] for row in bins] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert bins[-1]["upper"] == pytest.approx(1.0)
    assert bins[-1]["count"] == 1
    assert bins[2]["count"] == 1
    assert bins[0]["count"] == 0 and bins[0]["accuracy"] is None


def test_nll_and_brier_hand_example() -> None:
    """NLL is the mean of -log p(true); Brier sums (p - onehot)^2 over classes and averages over images."""
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    labels = np.array([0, 1])
    assert nll(probs, labels) == pytest.approx(-np.log(0.5) / 2.0)
    assert brier(probs, labels) == pytest.approx(0.25)


def test_detect_tta_recognises_log_of_averaged_probabilities() -> None:
    """Log of a probability vector sums to one after exponentiation; raw logits do not, in general."""
    rng = np.random.default_rng(11)
    probs = rng.dirichlet(np.ones(N_CLASSES), size=50)
    assert detect_tta(np.log(probs))
    assert not detect_tta(rng.normal(size=(50, N_CLASSES)) + 1.0)


def _write_predictions(path: Path, labels: np.ndarray, logits: np.ndarray, prefix: str) -> None:
    """Write a predictions CSV with the project's columns; probabilities are softmax of the logits."""
    probs = _softmax_rows(logits)
    frame = pd.DataFrame(
        {
            "image_id": [f"{prefix}{i:05d}" for i in range(labels.size)],
            "true_grade": labels,
            "pred_grade": logits.argmax(axis=1),
        }
    )
    for k in range(N_CLASSES):
        frame[f"prob_{k}"] = probs[:, k]
    for k in range(N_CLASSES):
        frame[f"logit_{k}"] = logits[:, k]
    frame[list(PREDICTION_COLUMNS)].to_csv(path, index=False)


def _load_script(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_calibrate_script_keeps_predictions_and_writes_outputs(tmp_path: Path) -> None:
    """The script fits T on validation, writes the calibrated files, keeps pred_grade, and reports the test file."""
    probs, labels = _calibrated_world(1500, SEED + 2)
    overconfident = 2.0 * np.log(probs)
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    val_csv, test_csv = run / "predictions" / "val.csv", run / "predictions" / "test.csv"
    _write_predictions(val_csv, labels, overconfident, "v")
    test_labels = np.roll(labels, 3)
    _write_predictions(test_csv, test_labels, overconfident, "t")
    register_val(val_csv)
    register_test(test_csv)
    write_split_for(tmp_path / "split.csv", val_csv, test_csv)
    script = _load_script("calibrate_script", CALIBRATE_SCRIPT)
    assert script.main(["--run", str(run), "--test", str(test_csv), "--split", str(tmp_path / "split.csv")]) == 0

    out = run / "calibration"
    summary = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert summary["fitted_on"] == "val"
    assert summary["temperature"] > 0
    assert abs(summary["temperature"] - 2.0) < 0.2
    assert summary["val"]["after"]["ece"] < summary["val"]["before"]["ece"]
    assert summary["test"] is not None and summary["test"]["file"].endswith("test.csv")
    assert summary["tta"] is False

    calibrated = pd.read_csv(out / "val_calibrated.csv", dtype={"image_id": str})
    original = pd.read_csv(val_csv, dtype={"image_id": str})
    assert np.array_equal(calibrated["pred_grade"].to_numpy(), original["pred_grade"].to_numpy())
    recomputed = _softmax_rows(calibrated[[f"logit_{k}" for k in range(N_CLASSES)]].to_numpy())
    assert np.allclose(recomputed, calibrated[[f"prob_{k}" for k in range(N_CLASSES)]].to_numpy(), atol=1e-9)
    assert (out / "test_calibrated.csv").is_file()
    assert (out / "reliability_val.png").is_file() and (out / "reliability_test.png").is_file()


def test_test_file_does_not_change_the_fitted_temperature(tmp_path: Path) -> None:
    """Shuffling the test labels leaves T and the validation metrics identical (the test file is never fitted on)."""
    probs, labels = _calibrated_world(1000, SEED + 3)
    logits = 2.5 * np.log(probs)
    run = tmp_path / "run"
    (run / "predictions").mkdir(parents=True)
    val_csv, test_csv = run / "predictions" / "val.csv", run / "predictions" / "test.csv"
    _write_predictions(val_csv, labels, logits, "v")
    _write_predictions(test_csv, np.random.default_rng(1).permutation(labels), logits, "t")
    register_val(val_csv)
    register_test(test_csv)
    split = str(tmp_path / "split.csv")
    write_split_for(split, val_csv, test_csv)
    script = _load_script("calibrate_script_shuffle", CALIBRATE_SCRIPT)
    assert script.main(["--run", str(run), "--split", split]) == 0
    without = json.loads((run / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    assert script.main(["--run", str(run), "--out", str(tmp_path / "with_test"), "--test", str(test_csv), "--split", split]) == 0
    with_test = json.loads((tmp_path / "with_test" / "calibration.json").read_text(encoding="utf-8"))
    assert with_test["temperature"] == without["temperature"]
    assert with_test["val"] == without["val"]


def test_load_logit_table_rejects_inconsistent_predictions(tmp_path: Path) -> None:
    """A pred_grade that is not the argmax of the logits is refused."""
    _, labels = _calibrated_world(20, SEED + 4)
    logits = np.log(_calibrated_world(20, SEED + 4)[0])
    path = tmp_path / "bad.csv"
    _write_predictions(path, labels, logits, "b")
    frame = pd.read_csv(path)
    frame["pred_grade"] = (frame["pred_grade"] + 1) % N_CLASSES
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_logit_table(path)
