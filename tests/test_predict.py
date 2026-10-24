"""Tests for prediction, resume, TTA and the test-set lock (tiny model, synthetic 64 px images, CPU)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from _synthetic import make_cfg
from _training_data import write_tiny_dataset
from drgrade.config import DataConfig
from drgrade.models import build_model
from drgrade.predict import (
    LOCK_CLOSED,
    LOCK_OPEN,
    PREDICTION_COLUMNS,
    PredictionRefused,
    batch_outputs,
    build_subset,
    close_test_lock,
    decide_test_access,
    default_output,
    load_model,
    predict_subset,
    read_lock,
    record_test_output,
    sha256_file,
)
from drgrade.run_config import AugConfig, DataSection, ModelConfig, OptimiserConfig, RunConfig, ScheduleConfig

REPO = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = REPO / "scripts" / "predict.py"
TARGET_SIZE = 64
INPUT_SIZE = 64
BATCH = 2


def _run_cfg(split: Path) -> RunConfig:
    aug = AugConfig(180, 0.5, 0.5, 0.85, 1.0, 0.95, 1.05, 0.2, 0.2, 0.8, 1.2)
    return RunConfig(
        run_name="tiny_predict",
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


class Fixture:
    """A tiny processed dataset, a fake trained run (random weights, saved as best.pt) and the data config."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = write_tiny_dataset(root / "data", target_size=TARGET_SIZE, seed=5)
        self.data_cfg: DataConfig = make_cfg(processed_dir=str(self.dataset.processed_dir), target_size=TARGET_SIZE)
        self.run = _run_cfg(self.dataset.split)
        self.run_dir = root / "runs" / "tiny_predict"
        self.run_dir.mkdir(parents=True)
        torch.manual_seed(0)
        model = build_model(self.run, pretrained=False)
        torch.save(
            {"model_state": model.state_dict(), "epoch": 1, "metrics": {}, "config": self.run.to_dict()},
            self.run_dir / "best.pt",
        )
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump({"run": self.run.to_dict(), "data": {"target_size": TARGET_SIZE}}, sort_keys=False),
            encoding="utf-8",
        )
        self.data_yaml = root / "data.yaml"
        self.data_yaml.write_text(yaml.safe_dump(dataclasses.asdict(self.data_cfg), sort_keys=False), encoding="utf-8")

    def cli(self, *extra: str) -> list[str]:
        return ["--run", str(self.run_dir), "--data-config", str(self.data_yaml), "--device", "cpu",
                "--batch-size", str(BATCH), "--num-workers", "0", *extra]


@pytest.fixture()
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def _predict_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("predict_script", PREDICT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict_script"] = module
    spec.loader.exec_module(module)
    return module


def _loaded_and_subset(fx: Fixture, subset: str):  # type: ignore[no-untyped-def]
    device = torch.device("cpu")
    loaded = load_model(fx.run_dir, device)
    table, dataset = build_subset(loaded, fx.data_cfg, subset)
    return loaded, table, dataset


def test_predictions_have_the_columns_and_probabilities_sum_to_one(fx: Fixture) -> None:
    """A complete plain prediction has the output columns, and each row of probabilities sums to 1."""
    loaded, table, dataset = _loaded_and_subset(fx, "val")
    out = fx.run_dir / "predictions" / "val.csv"
    assert predict_subset(loaded, table, dataset, out, torch.device("cpu"), tta=False, batch_size=BATCH) is True
    frame = pd.read_csv(out)
    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    assert len(frame) == len(table)
    assert list(frame["image_id"]) == sorted(frame["image_id"])
    probs = frame[[f"prob_{g}" for g in range(5)]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    logits = frame[[f"logit_{g}" for g in range(5)]].to_numpy()
    np.testing.assert_allclose(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True), probs, atol=1e-9)
    assert not (fx.run_dir / "predictions" / "val.csv.partial.csv").exists()


def test_tta_has_the_same_shape_differs_and_logits_are_log_probabilities(fx: Fixture) -> None:
    """TTA keeps the shape, changes the probabilities, and its logit columns are the log of the probabilities."""
    loaded, table, dataset = _loaded_and_subset(fx, "val")
    plain_out = fx.run_dir / "predictions" / "val.csv"
    tta_out = fx.run_dir / "predictions" / "val_tta.csv"
    predict_subset(loaded, table, dataset, plain_out, torch.device("cpu"), tta=False, batch_size=BATCH)
    predict_subset(loaded, table, dataset, tta_out, torch.device("cpu"), tta=True, batch_size=BATCH)
    plain = pd.read_csv(plain_out)
    tta = pd.read_csv(tta_out)
    assert plain.shape == tta.shape
    assert list(tta.columns) == list(PREDICTION_COLUMNS)
    prob_cols = [f"prob_{g}" for g in range(5)]
    assert not np.allclose(plain[prob_cols].to_numpy(), tta[prob_cols].to_numpy())
    np.testing.assert_allclose(tta[prob_cols].to_numpy().sum(axis=1), 1.0, atol=1e-12)
    logit_cols = [f"logit_{g}" for g in range(5)]
    np.testing.assert_allclose(np.log(tta[prob_cols].to_numpy()), tta[logit_cols].to_numpy(), atol=1e-12)


def test_tta_averages_the_four_flip_views(fx: Fixture) -> None:
    """batch_outputs with tta equals the mean softmax of the four flipped views, computed by hand."""
    loaded, table, dataset = _loaded_and_subset(fx, "val")
    images = torch.stack([dataset[i][0] for i in range(2)])
    model = loaded.model
    with torch.inference_mode():
        views = [images, images.flip(-1), images.flip(-2), images.flip(-1).flip(-2)]
        expected = np.mean(
            [torch.softmax(model(v).double(), dim=1).numpy() for v in views], axis=0
        )
    probs, logits = batch_outputs(model, images, tta=True)
    np.testing.assert_allclose(probs, expected, atol=1e-6)
    np.testing.assert_allclose(logits, np.log(probs))


def test_resume_gives_an_identical_csv(fx: Fixture) -> None:
    """Stopping after every batch (max_seconds 0) and resuming gives a byte-identical file to one uninterrupted run."""
    loaded, table, dataset = _loaded_and_subset(fx, "val")
    device = torch.device("cpu")
    whole = fx.run_dir / "predictions" / "whole.csv"
    assert predict_subset(loaded, table, dataset, whole, device, tta=False, batch_size=BATCH) is True

    resumed = fx.run_dir / "predictions" / "resumed.csv"
    calls = 0
    done = False
    while not done:
        calls += 1
        done = predict_subset(loaded, table, dataset, resumed, device, tta=False, batch_size=BATCH, max_seconds=0.0)
        assert calls < 50, "resume did not finish"
    assert calls > 1
    assert whole.read_bytes() == resumed.read_bytes()


def test_partial_file_is_removed_only_when_complete(fx: Fixture) -> None:
    """After an interrupted call the partial file exists and the final file does not."""
    loaded, table, dataset = _loaded_and_subset(fx, "val")
    out = fx.run_dir / "predictions" / "val.csv"
    finished = predict_subset(loaded, table, dataset, out, torch.device("cpu"), tta=False, batch_size=BATCH, max_seconds=0.0)
    assert finished is False
    assert not out.exists()
    partial = out.with_name("val.csv.partial.csv")
    assert partial.is_file()
    assert len(pd.read_csv(partial)) == BATCH


def test_test_subset_is_refused_by_the_loader_without_the_lock_path(fx: Fixture) -> None:
    """build_subset allows the test subset only through the evaluation path; the plain loader still refuses it."""
    from drgrade.data import load_run_table

    with pytest.raises(ValueError, match="test subset"):
        load_run_table(fx.dataset.manifest, fx.dataset.split, ("test",), fx.dataset.image_dir)


def test_load_model_refuses_a_config_yaml_that_disagrees(fx: Fixture) -> None:
    """A config.yaml with a different augmentation setting from best.pt is refused."""
    raw = yaml.safe_load((fx.run_dir / "config.yaml").read_text(encoding="utf-8"))
    raw["run"]["aug"]["brightness"] = 0.9
    (fx.run_dir / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_model(fx.run_dir, torch.device("cpu"))


def test_lock_decision_rules() -> None:
    """decide_test_access follows the rules of DECISIONS.md D23 for each lock state."""
    sha = "a" * 64
    with pytest.raises(PredictionRefused, match="--final"):
        decide_test_access(None, sha, False, final=False, force=False, reason=None)
    with pytest.raises(PredictionRefused, match="together"):
        decide_test_access(None, sha, False, final=True, force=True, reason=None)
    assert decide_test_access(None, sha, False, final=True, force=False, reason=None) is False
    assert decide_test_access(None, sha, False, final=True, force=True, reason="reproduction") is True

    open_lock = {"status": LOCK_OPEN, "checkpoint": {"sha256": sha}, "outputs": [{"variant": "plain"}]}
    assert decide_test_access(open_lock, sha, True, final=True, force=False, reason=None) is False
    with pytest.raises(PredictionRefused, match="already"):
        decide_test_access(open_lock, sha, False, final=True, force=False, reason=None)
    assert decide_test_access(open_lock, sha, False, final=True, force=True, reason="repeat") is True

    closed = {"status": LOCK_CLOSED, "checkpoint": {"sha256": sha}, "outputs": []}
    with pytest.raises(PredictionRefused, match="closed"):
        decide_test_access(closed, sha, False, final=True, force=False, reason=None)
    assert decide_test_access(closed, sha, False, final=True, force=True, reason="why") is True

    with pytest.raises(PredictionRefused, match="differs"):
        decide_test_access(open_lock, "b" * 64, True, final=True, force=False, reason=None)


def test_cli_test_lock_flow(fx: Fixture) -> None:
    """The CLI: test needs --final; the first final run writes the lock; plain and TTA share one open lock;
    closing refuses a plain rerun; a forced rerun is recorded; validation needs no lock."""
    predict = _predict_script()
    lock_path = fx.run_dir / "TEST_LOCK.json"

    assert predict.main(fx.cli("--subset", "test")) == 2
    assert not lock_path.exists()
    assert not (fx.run_dir / "predictions" / "test.csv").exists()

    assert predict.main(fx.cli("--subset", "test", "--final")) == 0
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == LOCK_OPEN
    assert lock["checkpoint"]["sha256"] == sha256_file(fx.run_dir / "best.pt")
    assert [o["variant"] for o in lock["outputs"]] == ["plain"]
    assert lock["tta"] is False and lock["closed_utc"] is None and lock["git"] is not None

    assert predict.main(fx.cli("--subset", "test", "--final", "--tta")) == 0
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert [o["variant"] for o in lock["outputs"]] == ["plain", "tta"]
    assert (fx.run_dir / "predictions" / "test_tta.csv").is_file()

    assert predict.main(fx.cli("--subset", "test", "--final")) == 2  # plain already recorded while open

    assert predict.main(fx.cli("--close-lock")) == 0
    assert json.loads(lock_path.read_text(encoding="utf-8"))["status"] == LOCK_CLOSED
    before = lock_path.read_text(encoding="utf-8")

    assert predict.main(fx.cli("--subset", "test", "--final")) == 2
    assert predict.main(fx.cli("--subset", "test", "--final", "--force")) == 2  # --force needs --reason
    assert lock_path.read_text(encoding="utf-8") == before

    assert predict.main(fx.cli("--subset", "test", "--final", "--force", "--reason", "re-check the crop")) == 0
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == LOCK_CLOSED
    assert len(lock["reruns"]) == 1
    assert lock["reruns"][0]["reason"] == "re-check the crop"
    assert lock["reruns"][0]["variant"] == "plain"
    assert "--reason" in lock["reruns"][0]["command"]

    assert predict.main(fx.cli("--subset", "val")) == 0
    assert (fx.run_dir / "predictions" / "val.csv").is_file()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["status"] == LOCK_CLOSED


def test_cli_partial_test_run_writes_no_lock_until_complete(fx: Fixture) -> None:
    """An interrupted test prediction exits 3 and leaves no lock; the rerun completes and writes it."""
    predict = _predict_script()
    lock_path = fx.run_dir / "TEST_LOCK.json"
    status = 3
    attempts = 0
    while status == 3:
        attempts += 1
        assert attempts < 50
        status = predict.main(fx.cli("--subset", "test", "--final", "--max-seconds", "0"))
        if status == 3:
            assert not lock_path.exists()
    assert status == 0
    assert read_lock(fx.run_dir)["status"] == LOCK_OPEN
    assert len(pd.read_csv(fx.run_dir / "predictions" / "test.csv")) == 4


def test_record_and_close_helpers(fx: Fixture) -> None:
    """record_test_output creates the lock; close_test_lock closes it; closing with no lock is refused."""
    with pytest.raises(PredictionRefused):
        close_test_lock(fx.run_dir)
    out = default_output(fx.run_dir, "test", tta=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("x", encoding="utf-8")
    sha = sha256_file(fx.run_dir / "best.pt")
    lock = record_test_output(fx.run_dir, "tiny", sha, False, out, 4, "cmd", False, None)
    assert lock["status"] == LOCK_OPEN and lock["outputs"][0]["file"] == "predictions/test.csv"
    closed = close_test_lock(fx.run_dir)
    assert closed["status"] == LOCK_CLOSED and closed["closed_utc"]
    assert close_test_lock(fx.run_dir)["closed_utc"] == closed["closed_utc"]


def test_cli_refuses_validation_with_final_and_test_outside_run(fx: Fixture) -> None:
    """--final is for test only, and a test output outside the run folder is refused before anything runs."""
    predict = _predict_script()
    assert predict.main(fx.cli("--subset", "val", "--final")) == 2
    outside = fx.root / "elsewhere.csv"
    assert predict.main(fx.cli("--subset", "test", "--final", "--out", str(outside))) == 2
    assert not outside.exists()


def test_cli_test_access_is_logged_and_hashed(fx: Fixture) -> None:
    """A test prediction logs an attempt before it starts and a complete event with the CSV hash;
    the hash is in the lock and the sidecar, and a second folder holding the same run cannot repeat it."""
    from drgrade.provenance import read_access_log, read_meta, sha256_of

    predict = _predict_script()
    assert predict.main(fx.cli("--subset", "test", "--final")) == 0
    events = read_access_log()
    assert [e["event"] for e in events] == ["attempt", "complete"]
    out = fx.run_dir / "predictions" / "test.csv"
    digest = sha256_of(out)
    assert events[1]["csv_sha256"] == digest
    assert read_lock(fx.run_dir)["outputs"][0]["csv_sha256"] == digest
    meta = read_meta(out)
    assert meta["subset"] == "test" and meta["csv_sha256"] == digest and meta["rows"] == 4

    # A copy of the run folder has no lock of its own, but the central log still refuses it.
    import shutil

    copy = fx.root / "runs" / "copy_of_run"
    shutil.copytree(fx.run_dir, copy)
    (copy / "TEST_LOCK.json").unlink()
    (copy / "predictions" / "test.csv").unlink()
    args = fx.cli("--subset", "test", "--final")
    args[args.index("--run") + 1] = str(copy)
    assert predict.main(args) == 2
    assert not (copy / "predictions" / "test.csv").exists()


def test_cli_refuses_a_run_outside_the_allowlist(fx: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that is not in the allowlist is refused before any test image is read, and nothing is logged."""
    from drgrade.provenance import ENV_ALLOWLIST, read_access_log

    allowlist = fx.root / "only_others.yaml"
    allowlist.write_text("allowed_runs:\n  - some_other_run\n", encoding="utf-8")
    monkeypatch.setenv(ENV_ALLOWLIST, str(allowlist))
    predict = _predict_script()
    assert predict.main(fx.cli("--subset", "test", "--final")) == 2
    assert not (fx.run_dir / "predictions" / "test.csv").exists()
    assert read_access_log() == []


def test_unknown_lock_status_is_refused(fx: Fixture) -> None:
    """A lock with a status other than open or closed is refused, not treated as open."""
    lock_path = fx.run_dir / "TEST_LOCK.json"
    lock_path.write_text(
        json.dumps({"status": "opened", "checkpoint": {"sha256": sha256_file(fx.run_dir / "best.pt")}, "outputs": []}),
        encoding="utf-8",
    )
    with pytest.raises(PredictionRefused, match="unknown status"):
        decide_test_access(read_lock(fx.run_dir), sha256_file(fx.run_dir / "best.pt"), False, True, False, None)


def test_validation_prediction_writes_a_sidecar(fx: Fixture) -> None:
    """Validation output gets a sidecar too, and no test access event."""
    from drgrade.provenance import read_access_log, read_meta

    predict = _predict_script()
    assert predict.main(fx.cli("--subset", "val")) == 0
    meta = read_meta(fx.run_dir / "predictions" / "val.csv")
    assert meta["subset"] == "val" and meta["run_name"] == "tiny_predict" and meta["tta"] is False
    assert read_access_log() == []
