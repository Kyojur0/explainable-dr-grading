"""Provenance checks on predictions files and the test access log (DECISIONS.md D38). No torch needed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from _prediction_data import make_predictions, register_test, register_val, write_split_for
from drgrade.provenance import (
    EVENT_COMPLETE,
    ProvenanceError,
    append_access_event,
    check_same_checkpoint,
    check_test_access,
    load_allowlist,
    read_access_log,
    verify_for_use,
    verify_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path, n: int = 6, prefix: str = "ts") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [f"{prefix}{i:06d}" for i in range(n)]
    make_predictions(ids, [i % 5 for i in range(n)], [i % 5 for i in range(n)], [0.8] * n).to_csv(path, index=False)
    return path


def test_the_committed_allowlist_names_only_e2_and_e3() -> None:
    """The real allowlist holds the selected model and the declared secondary model, nothing else (D35)."""
    assert load_allowlist(ROOT / "configs" / "test_allowlist.yaml") == {"e2_effb0_512_ce", "e3_effb0_512_wce"}


def test_allowlist_and_repeat_rules() -> None:
    """Allowed runs pass once per variant; a repeat needs the override; other runs never pass."""
    check_test_access("tiny_predict", "plain", forced=False)
    with pytest.raises(ProvenanceError, match="not in"):
        check_test_access("e9_anything", "plain", forced=True)
    append_access_event({"event": EVENT_COMPLETE, "run_name": "tiny_predict", "variant": "plain", "csv_sha256": "x"})
    with pytest.raises(ProvenanceError, match="already completed"):
        check_test_access("tiny_predict", "plain", forced=False)
    check_test_access("tiny_predict", "tta", forced=False)
    check_test_access("tiny_predict", "plain", forced=True)


def test_access_log_is_append_only() -> None:
    """Each event is one more line; earlier lines are unchanged."""
    append_access_event({"event": "attempt", "run_name": "a"})
    first = read_access_log()
    append_access_event({"event": "attempt", "run_name": "b"})
    second = read_access_log()
    assert second[:1] == first and len(second) == 2 and second[1]["run_name"] == "b"
    assert "time_utc" in second[0]


def test_verify_validation_file(tmp_path: Path) -> None:
    """A registered validation file passes; no sidecar, wrong subset, an edit or a wrong row count are refused."""
    csv = _csv(tmp_path / "run" / "predictions" / "val.csv", prefix="tr")
    with pytest.raises(ProvenanceError, match="sidecar"):
        verify_predictions(csv, "val")
    register_val(csv)
    assert verify_predictions(csv, "val", expected_rows=6)["rows"] == 6
    with pytest.raises(ProvenanceError, match="subset"):
        verify_predictions(csv, "test")
    with pytest.raises(ProvenanceError, match="incomplete"):
        verify_predictions(csv, "val", expected_rows=7)
    frame = pd.read_csv(csv)
    frame.loc[0, "pred_grade"] = 4
    frame.to_csv(csv, index=False)
    with pytest.raises(ProvenanceError, match="changed"):
        verify_predictions(csv, "val")


def test_verify_test_file_needs_log_and_lock(tmp_path: Path) -> None:
    """A test file with a correct sidecar is still refused unless its hash is in the log and in the lock."""
    from drgrade.provenance import write_meta

    csv = _csv(tmp_path / "run" / "predictions" / "test.csv")
    write_meta(csv, {"subset": "test", "rows": 6})
    with pytest.raises(ProvenanceError, match="not recorded"):
        verify_predictions(csv, "test")
    register_test(csv)
    verify_predictions(csv, "test", expected_rows=6)
    (tmp_path / "run" / "TEST_LOCK.json").write_text(json.dumps({"outputs": [{"csv_sha256": "other"}]}), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not in"):
        verify_predictions(csv, "test")


def test_verify_for_use_accepts_unregistered_validation_only_when_allowed(tmp_path: Path) -> None:
    """Training-time validation files have no sidecar: accepted for comparison, refused for calibration."""
    csv = _csv(tmp_path / "run" / "val_predictions.csv", prefix="tr")
    assert verify_for_use(csv, "val", 6, require_meta=False) is None
    with pytest.raises(ProvenanceError):
        verify_for_use(csv, "val", 6, require_meta=True)
    test_csv = _csv(tmp_path / "run2" / "predictions" / "test.csv")
    with pytest.raises(ProvenanceError):
        verify_for_use(test_csv, "test", 6, require_meta=False)


def test_same_checkpoint_rule() -> None:
    check_same_checkpoint({"run_name": "a", "checkpoint_sha256": "1"}, {"run_name": "a", "checkpoint_sha256": "1"})
    with pytest.raises(ProvenanceError):
        check_same_checkpoint({"run_name": "a", "checkpoint_sha256": "1"}, {"run_name": "a", "checkpoint_sha256": "2"})


def _script(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(f"{name}_prov", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_prov"] = module
    spec.loader.exec_module(module)
    return module


def test_calibrate_refuses_unregistered_or_mismatched_inputs(tmp_path: Path) -> None:
    """calibrate.py refuses a validation file without a sidecar, and val and test from different checkpoints."""
    pytest.importorskip("torch")
    run = tmp_path / "run"
    val_csv = _csv(run / "predictions" / "val.csv", n=40, prefix="tr")
    test_csv = _csv(run / "predictions" / "test.csv", n=40)
    split = tmp_path / "split.csv"
    write_split_for(split, val_csv, test_csv)
    calibrate = _script("calibrate")
    assert calibrate.main(["--run", str(run), "--split", str(split)]) == 2
    register_val(val_csv, checkpoint="c0")
    register_test(test_csv, checkpoint="c9")
    assert calibrate.main(["--run", str(run), "--test", str(test_csv), "--split", str(split)]) == 2
    assert not (run / "calibration" / "calibration.json").exists()
