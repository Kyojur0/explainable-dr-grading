"""Tests for the synthetic quality stress test (tiny model, 64 px synthetic images, CPU, no real data)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from _prediction_data import register_val
from _synthetic import make_fundus
from test_predict import Fixture
from drgrade.predict import build_subset, load_model, predict_subset
from drgrade.preprocess import apply_circle_mask

REPO = Path(__file__).resolve().parents[1]
STRESS_SCRIPT = REPO / "scripts" / "quality_stress.py"
CALIBRATE_SCRIPT = REPO / "scripts" / "calibrate.py"
REFERRAL_SCRIPT = REPO / "scripts" / "referral.py"
SIZE = 64
EXPECTED_LEVELS = ("none", "blur_r1", "blur_r2", "blur_r4", "darken_f0.8", "darken_f0.6", "darken_f0.4")
MAX_RESUME_CALLS = 80


def _load_script(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stress_argv(fx: Fixture, *extra: str) -> list[str]:
    return [
        "--run", str(fx.run_dir),
        "--data-config", str(fx.data_yaml),
        "--device", "cpu",
        "--batch-size", "2",
        "--num-workers", "0",
        *extra,
    ]


def _inside_mask(size: int) -> np.ndarray:
    white = Image.new("RGB", (size, size), (255, 255, 255))
    return np.asarray(apply_circle_mask(white))[..., 0] > 0


@pytest.fixture()
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_perturbations_change_pixels_and_keep_the_corners_black() -> None:
    """Blur and darkening change the pixels; blur keeps everything outside the inscribed circle black."""
    stress = _load_script("quality_stress_pixels", STRESS_SCRIPT)
    base = apply_circle_mask(make_fundus(seed=3, size=SIZE))
    pixels = np.asarray(base)
    outside = ~_inside_mask(SIZE)

    blurred = np.asarray(stress.Perturbation("blur", 2.0)(base))
    assert not np.array_equal(blurred, pixels)
    assert np.all(blurred[outside] == 0)

    darkened = np.asarray(stress.Perturbation("darken", 0.6)(base))
    assert not np.array_equal(darkened, pixels)
    assert np.array_equal(darkened, np.clip(np.rint(pixels.astype(np.float64) * 0.6), 0, 255).astype(np.uint8))
    assert np.all(darkened[outside] == 0)


def test_none_level_is_the_identity_and_names_are_stable() -> None:
    """The baseline returns the image unchanged, and the level names are the file names used in the outputs."""
    stress = _load_script("quality_stress_names", STRESS_SCRIPT)
    base = apply_circle_mask(make_fundus(seed=4, size=SIZE))
    assert np.array_equal(np.asarray(stress.Perturbation("none", 0.0)(base)), np.asarray(base))
    assert stress.Perturbation("none", 0.0).name == "none"
    assert stress.Perturbation("blur", 2.0).name == "blur_r2"
    assert stress.Perturbation("darken", 0.6).name == "darken_f0.6"
    with pytest.raises(ValueError):
        stress.Perturbation("darken", 1.5)
    with pytest.raises(ValueError):
        stress.Perturbation("blur", 0.0)


def test_the_none_level_equals_the_plain_predictions(fx: Fixture) -> None:
    """The baseline goes through the same code path and gives the same CSV as plain predict_subset."""
    device = torch.device("cpu")
    loaded = load_model(fx.run_dir, device)
    table, dataset = build_subset(loaded, fx.data_cfg, "val")
    plain = fx.run_dir / "plain_val.csv"
    assert predict_subset(loaded, table, dataset, plain, device, tta=False, batch_size=2)

    stress = _load_script("quality_stress_baseline", STRESS_SCRIPT)
    out = fx.run_dir / "stress_plain"
    assert stress.main(_stress_argv(fx, "--out", str(out))) == 0
    assert (out / "none.csv").read_bytes() == plain.read_bytes()


def test_resumed_levels_give_identical_csvs(fx: Fixture) -> None:
    """Running with --max-seconds 0 one batch at a time, until done, gives the same level files as one run."""
    stress = _load_script("quality_stress_resume", STRESS_SCRIPT)
    resumed = fx.run_dir / "stress_resumed"
    statuses: list[int] = []
    for _ in range(MAX_RESUME_CALLS):
        status = stress.main(_stress_argv(fx, "--out", str(resumed), "--max-seconds", "0"))
        statuses.append(status)
        if status == 0:
            break
    assert statuses[-1] == 0
    assert 3 in statuses

    once = fx.run_dir / "stress_once"
    assert stress.main(_stress_argv(fx, "--out", str(once))) == 0
    for name in EXPECTED_LEVELS:
        assert (resumed / f"{name}.csv").read_bytes() == (once / f"{name}.csv").read_bytes()
        assert not (resumed / f"{name}.csv.partial.csv").exists()


def test_summary_has_every_level_and_the_perturbations_change_predictions(fx: Fixture) -> None:
    """summary.csv has one row per level in order, with the uncalibrated labels, and the figure is written."""
    stress = _load_script("quality_stress_summary", STRESS_SCRIPT)
    out = fx.run_dir / "stress_summary"
    assert stress.main(_stress_argv(fx, "--out", str(out))) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["level"]) == list(EXPECTED_LEVELS)
    assert set(summary["confidence_scale"]) == {"uncalibrated"}
    assert summary["referral_fraction"].isna().all()
    assert summary.loc[summary["level"] == "none", "severity"].item() == 0.0
    assert (out / "quality_stress.png").is_file()
    none = pd.read_csv(out / "none.csv", dtype={"image_id": str})
    blur = pd.read_csv(out / "blur_r4.csv", dtype={"image_id": str})
    assert not np.allclose(none["logit_0"].to_numpy(), blur["logit_0"].to_numpy())


def test_calibrated_summary_uses_the_fitted_temperature_and_the_referral_threshold(fx: Fixture) -> None:
    """With calibration.json and referral.json present, the confidence is calibrated and the referral fraction is reported."""
    device = torch.device("cpu")
    loaded = load_model(fx.run_dir, device)
    table, dataset = build_subset(loaded, fx.data_cfg, "val")
    val_csv = fx.run_dir / "predictions" / "val.csv"
    val_csv.parent.mkdir(parents=True, exist_ok=True)
    assert predict_subset(loaded, table, dataset, val_csv, device, tta=False, batch_size=2)
    register_val(val_csv)  # the sidecar scripts/predict.py writes (DECISIONS.md D38)
    split = str(fx.dataset.split)

    calibrate = _load_script("calibrate_for_stress", CALIBRATE_SCRIPT)
    referral = _load_script("referral_for_stress", REFERRAL_SCRIPT)
    assert calibrate.main(["--run", str(fx.run_dir), "--split", split]) == 0
    assert referral.main(["--run", str(fx.run_dir), "--split", split]) == 0

    stress = _load_script("quality_stress_calibrated", STRESS_SCRIPT)
    out = fx.run_dir / "stress_calibrated"
    assert stress.main(_stress_argv(fx, "--out", str(out))) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["confidence_scale"]) == {"calibrated"}
    temperature = summary["temperature"].iloc[0]
    assert temperature > 0
    threshold = summary["referral_threshold"].iloc[0]
    referral_payload = json.loads((fx.run_dir / "referral" / "referral.json").read_text(encoding="utf-8"))
    assert threshold == pytest.approx(referral_payload["primary"]["threshold"])
    assert summary["referral_fraction"].between(0.0, 1.0).all()
