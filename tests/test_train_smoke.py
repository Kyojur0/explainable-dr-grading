"""End-to-end tests of train() on a tiny synthetic dataset (CPU, no weights, a few seconds each).

Checks the outputs, the probabilities, that no test id reaches the predictions, that
the smoke path checkpoints and resumes, and that an interrupted run resumed to the same
number of epochs ends with the same weights as an uninterrupted run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from _synthetic import make_cfg
from _training_data import write_tiny_dataset
from drgrade.run_config import (
    AugConfig,
    DataSection,
    ModelConfig,
    OptimiserConfig,
    RunConfig,
    ScheduleConfig,
)
from drgrade.train_loop import RUN_FILES, train

TARGET_SIZE = 64
INPUT_SIZE = 64


def _run_cfg(split: Path, max_epochs: int = 2) -> RunConfig:
    aug = AugConfig(180, 0.5, 0.5, 0.85, 1.0, 0.95, 1.05, 0.2, 0.2, 0.8, 1.2)
    return RunConfig(
        run_name="tiny_e3",
        model=ModelConfig("M2", "efficientnet_b0", "efficientnet_b0.ra_in1k", INPUT_SIZE, 0.2),
        loss="wce",
        label_smoothing=0.0,
        optimiser=OptimiserConfig("adamw", 3e-4, 1e-4),
        schedule=ScheduleConfig("cosine", 1),
        max_epochs=max_epochs,
        early_stopping_patience=5,
        batch_size=4,
        accumulation_steps=1,
        grad_clip_norm=1.0,
        selection_metric="qwk",
        seed=42,
        num_workers=0,
        aug=aug,
        data=DataSection(str(split)),
    )


def _data(tmp_path: Path):  # type: ignore[no-untyped-def]
    dataset = write_tiny_dataset(tmp_path / "data", target_size=TARGET_SIZE)
    data_cfg = make_cfg(processed_dir=str(dataset.processed_dir), target_size=TARGET_SIZE)
    return dataset, data_cfg


def test_two_epoch_run_writes_all_outputs(tmp_path: Path) -> None:
    """A two-epoch run writes every output, valid probabilities, and only validation ids."""
    dataset, data_cfg = _data(tmp_path)
    out = tmp_path / "run"
    result = train(_run_cfg(dataset.split), data_cfg, out, torch.device("cpu"), pretrained=False)

    assert result["finished"] is True
    assert result["epochs_run"] == 2
    for name in RUN_FILES:
        assert (out / name).is_file(), name

    history = pd.read_csv(out / "history.csv")
    assert len(history) == 2
    assert list(history["epoch"]) == [1, 2]

    predictions = pd.read_csv(out / "val_predictions.csv")
    prob_cols = [f"prob_{k}" for k in range(5)]
    logit_cols = [f"logit_{k}" for k in range(5)]
    assert len(predictions) == 5
    assert list(predictions.columns[:3]) == ["image_id", "true_grade", "pred_grade"]
    assert predictions[prob_cols].sum(axis=1).round(6).eq(1.0).all()
    assert predictions[logit_cols].shape == (5, 5)
    assert list(predictions["image_id"]) == sorted(predictions["image_id"])
    test_ids = set(pd.read_csv(dataset.split, dtype=str).query("subset == 'test'")["image_id"])
    assert not test_ids & set(predictions["image_id"])

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_run"] == 2
    assert summary["best_epoch"] in (1, 2)

    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["seed"] == 42 and env["pretrained"] is False
    assert env["end_utc"] is not None
    assert (out / "config.yaml").read_text(encoding="utf-8").count("tiny_e3") == 1


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    """Interrupting after epoch 1 and resuming gives exactly the weights of an uninterrupted run."""
    dataset, data_cfg = _data(tmp_path)
    device = torch.device("cpu")
    straight = tmp_path / "straight"
    train(_run_cfg(dataset.split, max_epochs=2), data_cfg, straight, device, pretrained=False)

    resumed = tmp_path / "resumed"
    partial = train(
        _run_cfg(dataset.split, max_epochs=2),
        data_cfg,
        resumed,
        device,
        pretrained=False,
        stop_after_epoch=1,
    )
    assert partial["finished"] is False
    assert not (resumed / "val_predictions.csv").exists()
    final = train(
        _run_cfg(dataset.split, max_epochs=2),
        data_cfg,
        resumed,
        device,
        pretrained=False,
        resume=True,
    )
    assert final["finished"] is True

    expected = torch.load(straight / "last.pt", map_location="cpu", weights_only=False)["model_state"]
    actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)["model_state"]
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    assert len(pd.read_csv(resumed / "history.csv")) == 2


def test_existing_run_is_not_overwritten_without_resume(tmp_path: Path) -> None:
    """A second call without resume on a folder with last.pt raises FileExistsError."""
    dataset, data_cfg = _data(tmp_path)
    out = tmp_path / "run"
    train(
        _run_cfg(dataset.split, max_epochs=2),
        data_cfg,
        out,
        torch.device("cpu"),
        pretrained=False,
        stop_after_epoch=1,
    )
    with pytest.raises(FileExistsError):
        train(_run_cfg(dataset.split, max_epochs=2), data_cfg, out, torch.device("cpu"), pretrained=False)


def test_resume_refuses_a_changed_learning_rate(tmp_path: Path) -> None:
    """Changing a setting other than max_epochs or num_workers stops the resume."""
    dataset, data_cfg = _data(tmp_path)
    out = tmp_path / "run"
    train(_run_cfg(dataset.split, max_epochs=2), data_cfg, out, torch.device("cpu"), pretrained=False, stop_after_epoch=1)
    changed = _run_cfg(dataset.split, max_epochs=2)
    changed = replace(changed, optimiser=OptimiserConfig("adamw", 1e-3, 1e-4))
    with pytest.raises(ValueError, match="optimiser"):
        train(changed, data_cfg, out, torch.device("cpu"), pretrained=False, resume=True)


def test_missing_processed_data_names_the_preprocessing_script(tmp_path: Path) -> None:
    """Missing processed images raise FileNotFoundError that tells the user to run preprocess.py."""
    dataset, _ = _data(tmp_path)
    data_cfg = make_cfg(processed_dir=str(tmp_path / "absent"), target_size=TARGET_SIZE)
    with pytest.raises(FileNotFoundError, match="preprocess.py"):
        train(_run_cfg(dataset.split), data_cfg, tmp_path / "run", torch.device("cpu"), pretrained=False)


def test_smoke_mode_checkpoints_and_resumes(tmp_path: Path) -> None:
    """The smoke path runs epoch 1, resumes for epoch 2 in-process and leaves two history rows."""
    dataset, data_cfg = _data(tmp_path)
    out = tmp_path / "smoke"
    result = train(
        _run_cfg(dataset.split, max_epochs=5),
        data_cfg,
        out,
        torch.device("cpu"),
        pretrained=False,
        smoke=True,
    )
    assert result["epochs_run"] == 2
    assert len(pd.read_csv(out / "history.csv")) == 2
    config_text = (out / "config.yaml").read_text(encoding="utf-8")
    assert "max_epochs: 2" in config_text
