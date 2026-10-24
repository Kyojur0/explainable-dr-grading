"""Smoke tests for scripts/explain.py on a tiny random-weight run (EfficientNet-B0, 64 px, CPU).

The run is a fixture built the same way as in tests/test_predict.py. Its predictions, locked
test output and access log come from scripts/predict.py, so the explain gates see real files.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from _synthetic import make_cfg
from _training_data import write_tiny_dataset
from PIL import Image

from drgrade.config import load_data_config
from drgrade.data import build_transforms
from drgrade.explain_checks import (
    blurred,
    cascade_stages,
    composite,
    random_fraction_mask,
)
from drgrade.gradcam import fundus_mask
from drgrade.models import build_model
from drgrade.predict import image_folder, load_model
from drgrade.provenance import ENV_ALLOWLIST, read_access_log
from drgrade.run_config import (
    AugConfig,
    DataSection,
    ModelConfig,
    OptimiserConfig,
    RunConfig,
    ScheduleConfig,
)

REPO = Path(__file__).resolve().parents[1]
EXPLAIN_SCRIPT = REPO / "scripts" / "explain.py"
PREDICT_SCRIPT = REPO / "scripts" / "predict.py"
TARGET_SIZE = 64
INPUT_SIZE = 64
BATCH = 2
RUN_NAME = "tiny_predict"  # tests/conftest.py allowlists this name
DELETION_COLUMNS = ("image_id", "condition", "fraction", "prob", "drop", "replaced_share")
DISC_CHECK_MAX_SIDE = 1800
EXPLAIN_CONFIG = {
    "seed": 42,
    "n_randomisation": 2,
    "n_deletion": 3,
    "deletion_fractions": [0.1, 0.5],
    "deletion_blur_radius": 10.0,
    "compare_size": 32,
    "overlay_alpha": 0.4,
    "colormap": "jet",
    "edge_frac": 0.05,
    "disc_radius_frac": 0.10,
    "disc_smooth_frac": 0.05,
    "min_group": 2,
}


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_cfg(split: Path) -> RunConfig:
    aug = AugConfig(180, 0.5, 0.5, 0.85, 1.0, 0.95, 1.05, 0.2, 0.2, 0.8, 1.2)
    return RunConfig(
        run_name=RUN_NAME,
        model=ModelConfig("M2", "efficientnet_b0", "efficientnet_b0.ra_in1k", INPUT_SIZE, 0.2),
        loss="ce",
        label_smoothing=0.0,
        optimiser=OptimiserConfig("adamw", 3e-4, 1e-4),
        schedule=ScheduleConfig("cosine", 0),
        max_epochs=1,
        early_stopping_patience=5,
        batch_size=BATCH,
        accumulation_steps=1,
        grad_clip_norm=1.0,
        selection_metric="qwk",
        seed=42,
        num_workers=0,
        aug=aug,
        data=DataSection(str(split)),
    )


class Tiny:
    """A tiny processed dataset, a random-weight run folder with best.pt, and the predictions made by predict.py."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = write_tiny_dataset(root / "data", target_size=TARGET_SIZE, seed=5)
        data_cfg = make_cfg(processed_dir=str(self.dataset.processed_dir), target_size=TARGET_SIZE)
        self.run_cfg = _run_cfg(self.dataset.split)
        self.run_dir = root / "runs" / RUN_NAME
        self.run_dir.mkdir(parents=True)
        torch.manual_seed(0)
        model = build_model(self.run_cfg, pretrained=False)
        torch.save(
            {"model_state": model.state_dict(), "epoch": 1, "metrics": {}, "config": self.run_cfg.to_dict()},
            self.run_dir / "best.pt",
        )
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump({"run": self.run_cfg.to_dict(), "data": {"target_size": TARGET_SIZE}}, sort_keys=False),
            encoding="utf-8",
        )
        self.data_yaml = root / "data.yaml"
        self.data_yaml.write_text(yaml.safe_dump(dataclasses.asdict(data_cfg), sort_keys=False), encoding="utf-8")
        self.explain_yaml = root / "explain.yaml"
        self.explain_yaml.write_text(yaml.safe_dump(EXPLAIN_CONFIG, sort_keys=False), encoding="utf-8")
        frame = pd.read_csv(self.dataset.split, dtype={"image_id": str})
        self.val_ids = sorted(frame.loc[frame["subset"] == "val", "image_id"])
        self.test_ids = sorted(frame.loc[frame["subset"] == "test", "image_id"])

        predict = _load_script("predict_script", PREDICT_SCRIPT)
        assert predict.main(self.predict_cli()) == 0
        assert predict.main(self.predict_cli("--subset", "test", "--final")) == 0

    def predict_cli(self, *extra: str) -> list[str]:
        return ["--run", str(self.run_dir), "--data-config", str(self.data_yaml), "--device", "cpu",
                "--batch-size", str(BATCH), "--num-workers", "0", *extra]

    def explain(self, step: str, *extra: str, script: ModuleType | None = None) -> int:
        module = script if script is not None else _load_script("explain_script", EXPLAIN_SCRIPT)
        argv = [step, "--run", str(self.run_dir), "--data-config", str(self.data_yaml),
                "--explain-config", str(self.explain_yaml), "--device", "cpu", "--batch-size", str(BATCH), *extra]
        return module.main(argv)

    def ids_file(self, name: str, ids: list[str]) -> Path:
        path = self.root / name
        pd.DataFrame({"image_id": ids}).to_csv(path, index=False)
        return path


@pytest.fixture()
def tiny(tmp_path: Path) -> Tiny:
    return Tiny(tmp_path)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _explain_events() -> list[dict]:  # type: ignore[type-arg]
    return [event for event in read_access_log() if event.get("event") == "explain"]


def test_maps_val_is_complete_and_resumes_identically(tiny: Tiny) -> None:
    """maps val covers every val id with a logit check under 1e-3; a stopped run resumes to the same npz."""
    full = tiny.root / "maps_full"
    assert tiny.explain("maps", "--subset", "val", "--out", str(full)) == 0
    arrays = _arrays(full / "maps_val.npz")
    assert sorted(arrays["image_id"].tolist()) == tiny.val_ids
    meta = json.loads((full / "maps_val.json").read_text(encoding="utf-8"))
    assert meta["max_abs_logit_difference"] < 1e-3
    assert meta["target_layer"] == "bn2"

    resumed = tiny.root / "maps_resumed"
    assert tiny.explain("maps", "--subset", "val", "--out", str(resumed), "--max-seconds", "0") == 3
    assert (resumed / "maps_val.partial.npz").is_file()
    assert not (resumed / "maps_val.npz").exists()
    assert tiny.explain("maps", "--subset", "val", "--out", str(resumed)) == 0
    again = _arrays(resumed / "maps_val.npz")
    for key in ("image_id", "cams", "logits", "target"):
        np.testing.assert_array_equal(arrays[key], again[key])


def test_maps_test_needs_ids_logs_access_and_refuses_other_runs(tiny: Tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maps need --ids-from, log an explain event before reading images, and refuse a run off the allowlist."""
    out = tiny.root / "maps_test"
    assert tiny.explain("maps", "--subset", "test", "--out", str(out)) == 2

    ids = tiny.ids_file("test_ids.csv", tiny.test_ids[:2])
    assert tiny.explain("maps", "--subset", "test", "--ids-from", str(ids), "--out", str(out)) == 0
    assert sorted(_arrays(out / "maps_test.npz")["image_id"].tolist()) == sorted(tiny.test_ids[:2])
    events = _explain_events()
    assert events and events[-1]["run_name"] == RUN_NAME and events[-1]["n_images"] == 2

    other = tiny.root / "other_allowlist.yaml"
    other.write_text("allowed_runs:\n  - some_other_run\n", encoding="utf-8")
    monkeypatch.setenv(ENV_ALLOWLIST, str(other))
    refused = tiny.root / "maps_refused"
    before = len(_explain_events())
    assert tiny.explain("maps", "--subset", "test", "--ids-from", str(ids), "--out", str(refused)) == 2
    assert not (refused / "maps_test.npz").exists()
    assert len(_explain_events()) == before


def test_panels_val_adds_true_grade_panel_for_wrong_cases_only(tiny: Tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong case gets a third, true-grade panel and a finite Spearman; a correct case keeps two panels and empty columns."""
    script = _load_script("explain_script", EXPLAIN_SCRIPT)
    shown: list[list[tuple[bool, str, bool]]] = []
    save_figure = script._save_figure

    def record(fig: Any, path: Path, *args: Any, **kwargs: Any) -> None:
        """Note each axis (visible, title, has an image) before the figure is saved."""
        shown.append([(bool(ax.axison), ax.get_title(), bool(ax.images)) for ax in fig.axes])
        save_figure(fig, path, *args, **kwargs)

    monkeypatch.setattr(script, "_save_figure", record)

    out = tiny.root / "panels"
    assert tiny.explain("maps", "--subset", "val", "--out", str(out)) == 0
    first, second = tiny.val_ids[0], tiny.val_ids[1]
    cases = pd.DataFrame(
        [
            {"reason": "confident_error", "image_id": first, "true_grade": 0, "pred_grade": 3,
             "confidence": 0.9, "lesions_present": "none annotated"},
            {"reason": "uncertain", "image_id": second, "true_grade": 2, "pred_grade": 2,
             "confidence": 0.3, "lesions_present": "none annotated"},
        ]
    )
    cases_csv = tiny.root / "cases.csv"
    cases.to_csv(cases_csv, index=False)
    (tiny.run_dir / "calibration").mkdir()
    shutil.copy(tiny.run_dir / "predictions" / "val.csv", tiny.run_dir / "calibration" / "val_calibrated.csv")
    (tiny.run_dir / "referral").mkdir()
    (tiny.run_dir / "referral" / "referral.json").write_text(json.dumps({"primary": {"threshold": 0.25}}), encoding="utf-8")

    assert tiny.explain("panels", "--subset", "val", "--out", str(out), "--cases", str(cases_csv), script=script) == 0
    panels = out / "panels_val"
    assert (panels / "confident_error" / f"{first}.png").is_file()
    assert (panels / "uncertain" / f"{second}.png").is_file()
    assert (panels / "confident_error.png").is_file()
    assert (panels / "uncertain.png").is_file()

    # shown[0] and shown[1] are the per-image figures of the wrong and the correct case, in that order.
    wrong_axes, correct_axes = shown[0], shown[1]
    assert sum(1 for _, _, has_image in wrong_axes if has_image) == 3
    assert [title for _, title, has_image in wrong_axes if has_image][2] == "Grad-CAM for the true grade (0)"
    assert sum(1 for _, _, has_image in correct_axes if has_image) == 2
    assert all(not visible for visible, _, has_image in correct_axes if not has_image)

    table = pd.read_csv(panels / "panels.csv", dtype={"image_id": str})
    assert "reviewer_note" in table.columns
    assert table["reviewer_note"].isna().all()
    assert list(table["image_id"]) == [first, second]
    wrong_row, correct_row = table.iloc[0], table.iloc[1]
    assert wrong_row["true_class_map"] == "yes"
    assert math.isfinite(float(wrong_row["spearman_pred_vs_true_map"]))
    assert correct_row["true_class_map"] == "no"
    assert pd.isna(correct_row["spearman_pred_vs_true_map"])


def test_randomisation_outputs_reference_row_and_resume(tiny: Tiny) -> None:
    """Stage rows cover every cascade stage, summary.csv has the reference row, and a stop resumes to the same CSV."""
    full = tiny.root / "rand_full"
    assert tiny.explain("randomisation", "--out", str(full)) == 0
    rdir = full / "randomisation"
    n_images = EXPLAIN_CONFIG["n_randomisation"]
    n_stages = len(cascade_stages(build_model(tiny.run_cfg, pretrained=False), "efficientnet_b0"))
    frame = pd.read_csv(rdir / "randomisation.csv", dtype={"image_id": str})
    stages = frame[frame["stage_index"] >= 0]
    assert len(stages) == n_images * n_stages
    assert sorted(stages["stage_index"].unique()) == list(range(n_stages))
    reference = frame[frame["stage_index"] == -1]
    assert len(reference) == n_images
    assert set(reference["stage"]) == {"reference_other_image"}
    summary = pd.read_csv(rdir / "summary.csv")
    reference_row = summary[summary["stage_index"] == -1]
    assert list(reference_row["stage"]) == ["reference_other_image"]
    assert int(reference_row["n"].iloc[0]) == n_images
    for name in ("summary.csv", "randomisation.png", "examples.png"):
        assert (rdir / name).is_file()

    resumed = tiny.root / "rand_resumed"
    assert tiny.explain("randomisation", "--out", str(resumed), "--max-seconds", "0") == 3
    assert (resumed / "randomisation" / "randomisation.partial.csv").is_file()
    assert tiny.explain("randomisation", "--out", str(resumed)) == 0
    assert (rdir / "randomisation.csv").read_bytes() == (resumed / "randomisation" / "randomisation.csv").read_bytes()


def test_randomisation_rebuilds_outputs_without_recomputing_stages(tiny: Tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    """With randomisation.csv complete, a second run rebuilds summary and plot without any stage (no randomised copies)."""
    out = tiny.root / "rand_rebuild"
    assert tiny.explain("randomisation", "--out", str(out)) == 0
    rdir = out / "randomisation"
    before = (rdir / "randomisation.csv").read_bytes()
    (rdir / "summary.csv").unlink()
    (rdir / "randomisation.png").unlink()

    script = _load_script("explain_script", EXPLAIN_SCRIPT)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a stage was recomputed although randomisation.csv is complete")

    monkeypatch.setattr(script, "randomised_copies", refuse)
    assert tiny.explain("randomisation", "--out", str(out), script=script) == 0
    assert (rdir / "summary.csv").is_file()
    assert (rdir / "randomisation.png").is_file()
    assert (rdir / "randomisation.csv").read_bytes() == before


def test_deletion_rows_conditions_and_replaced_share(tiny: Tiny) -> None:
    """Deletion writes 3 conditions x (1 + fractions) rows per image, zero drop and zero share at fraction 0, and the summary columns."""
    out = tiny.root / "all"
    assert tiny.explain("maps", "--subset", "val", "--out", str(out)) == 0
    assert tiny.explain("deletion", "--out", str(out)) == 0
    ddir = out / "deletion"
    frame = pd.read_csv(ddir / "deletion.csv", dtype={"image_id": str})
    n_images = EXPLAIN_CONFIG["n_deletion"]
    n_fractions = len(EXPLAIN_CONFIG["deletion_fractions"])
    assert tuple(frame.columns) == DELETION_COLUMNS
    assert set(frame["condition"]) == {"salient", "random", "rotated"}
    assert len(frame) == 3 * (1 + n_fractions) * n_images
    assert (frame.groupby("image_id").size() == 3 * (1 + n_fractions)).all()
    zero = frame[frame["fraction"] == 0.0]
    assert len(zero) == 3 * n_images
    assert (zero["drop"] == 0.0).all()
    assert (zero["replaced_share"] == 0.0).all()
    moved = frame[frame["fraction"] > 0.0]
    assert ((moved["replaced_share"] > 0.0) & (moved["replaced_share"] <= 1.0)).all()

    summary = pd.read_csv(ddir / "summary.csv")
    assert len(summary) == 3 * (1 + n_fractions)
    for column in (
        "mean_diff_salient_minus_random",
        "se_diff_salient_minus_random",
        "mean_diff_salient_minus_rotated",
        "se_diff_salient_minus_rotated",
        "mean_replaced_share",
    ):
        assert column in summary.columns
    assert summary["mean_diff_salient_minus_rotated"].notna().all()
    assert (ddir / "deletion.png").is_file()


def test_deletion_random_rows_are_unchanged_by_the_rotated_condition(tiny: Tiny) -> None:
    """The random rows equal the probabilities of the random masks drawn from the seed + position generator.

    The rotation angles come from their own generator, so the random masks are the same draws as before
    the rotated condition existed. The check recomputes the random masks in the test and compares probabilities.
    """
    out = tiny.root / "random_check"
    assert tiny.explain("maps", "--subset", "val", "--out", str(out)) == 0
    assert tiny.explain("deletion", "--out", str(out)) == 0
    frame = pd.read_csv(out / "deletion" / "deletion.csv", dtype={"image_id": str})

    loaded = load_model(tiny.run_dir, torch.device("cpu"))
    data_cfg = load_data_config(tiny.data_yaml)
    transform = build_transforms(loaded.run.model.input_size, False, loaded.run.aug, data_cfg.target_size)
    script = _load_script("explain_script", EXPLAIN_SCRIPT)
    order = np.random.default_rng(EXPLAIN_CONFIG["seed"]).permutation(len(tiny.val_ids))
    sample = [tiny.val_ids[int(i)] for i in order][: EXPLAIN_CONFIG["n_deletion"]]
    for position, image_id in enumerate(sample):
        image = script._read_rgb(image_folder(data_cfg), image_id)
        region = fundus_mask(image.shape[0])
        replacement = blurred(image, EXPLAIN_CONFIG["deletion_blur_radius"])
        rng = np.random.default_rng(EXPLAIN_CONFIG["seed"] + position)
        variants = [image]
        for fraction in EXPLAIN_CONFIG["deletion_fractions"]:
            variants.append(composite(image, replacement, random_fraction_mask(region, fraction, rng)))
        with torch.inference_mode():
            logits = loaded.model(torch.stack([transform(Image.fromarray(v)) for v in variants]))
        probs = torch.softmax(logits.to(torch.float64), dim=1).numpy()
        target = int(np.argmax(probs[0]))
        for k, fraction in enumerate(EXPLAIN_CONFIG["deletion_fractions"]):
            row = frame[
                (frame["image_id"] == image_id)
                & (frame["condition"] == "random")
                & np.isclose(frame["fraction"], fraction)
            ]
            assert len(row) == 1
            assert abs(float(row["prob"].iloc[0]) - probs[k + 1, target]) < 1e-4


def test_deletion_refuses_partial_files_without_rotated_or_replaced_share(tiny: Tiny) -> None:
    """A partial file in the old format, or one without the rotated condition, makes deletion exit 2 and is kept."""
    out = tiny.root / "stale_partial"
    assert tiny.explain("maps", "--subset", "val", "--out", str(out)) == 0
    ddir = out / "deletion"
    ddir.mkdir(parents=True)
    partial = ddir / "deletion.partial.csv"
    old_format = pd.DataFrame(
        [{"image_id": tiny.val_ids[0], "condition": "salient", "fraction": 0.1, "prob": 0.5, "drop": 0.0}]
    )
    old_format.to_csv(partial, index=False)
    assert tiny.explain("deletion", "--out", str(out)) == 2
    assert partial.is_file()

    without_rotated = pd.DataFrame(
        [
            {"image_id": tiny.val_ids[0], "condition": "salient", "fraction": 0.1, "prob": 0.5, "drop": 0.0,
             "replaced_share": 0.1},
            {"image_id": tiny.val_ids[0], "condition": "random", "fraction": 0.1, "prob": 0.5, "drop": 0.0,
             "replaced_share": 0.1},
        ],
        columns=list(DELETION_COLUMNS),
    )
    without_rotated.to_csv(partial, index=False)
    assert tiny.explain("deletion", "--out", str(out)) == 2
    assert partial.is_file()


def test_stats_outputs(tiny: Tiny) -> None:
    """Stats writes one heatmap row per val image, the shortcut summary, both lesion scopes and a disc check within the pixel limit."""
    out = tiny.root / "all_stats"
    assert tiny.explain("maps", "--subset", "val", "--out", str(out)) == 0
    assert tiny.explain("stats", "--out", str(out)) == 0
    stats = out / "stats"
    heat = pd.read_csv(stats / "heatmap_stats.csv", dtype={"image_id": str})
    assert len(heat) == len(tiny.val_ids)
    assert set(heat["image_id"]) == set(tiny.val_ids)
    summary = json.loads((stats / "shortcut_summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == len(tiny.val_ids)
    comparison = pd.read_csv(stats / "lesion_comparison.csv")
    assert set(comparison["scope"]) == {"all", "referable"}
    disc = stats / "disc_check.png"
    assert disc.is_file()
    with Image.open(disc) as picture:
        assert max(picture.size) <= DISC_CHECK_MAX_SIDE


def test_save_figure_leaves_no_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved figure is a complete PNG and no .tmp file remains; a failed save leaves neither file."""
    script = _load_script("explain_script", EXPLAIN_SCRIPT)
    folder = tmp_path / "figures"
    folder.mkdir()
    fig, ax = script.plt.subplots()
    ax.plot([0, 1])
    target = folder / "figure.png"
    script._save_figure(fig, target)
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(path.name for path in folder.iterdir()) == ["figure.png"]

    failing, failing_ax = script.plt.subplots()
    failing_ax.plot([0, 1])

    def broken_savefig(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(failing, "savefig", broken_savefig)
    failed_target = folder / "failed.png"
    with pytest.raises(OSError):
        script._save_figure(failing, failed_target)
    assert not failed_target.exists()
    assert sorted(path.name for path in folder.iterdir()) == ["figure.png"]


def test_deletion_without_maps_is_refused(tiny: Tiny) -> None:
    """The deletion test needs maps_val.npz; without it the command exits 2."""
    assert tiny.explain("deletion", "--out", str(tiny.root / "empty")) == 2
