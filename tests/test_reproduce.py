"""Tests for the reproduction helpers (no dataset needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from _prediction_data import make_predictions

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(f"t_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"t_{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_compare_rounds_to_the_printed_decimals() -> None:
    """A recomputed value matches when it rounds to the printed value; percentages use a scale."""
    rep = _load("reproduce_report")
    expected = [
        {"name": "a", "label": "a", "value": 0.918, "decimals": 3},
        {"name": "b", "label": "b", "value": 7.9, "decimals": 1, "scale": 100},
        {"name": "c", "label": "c", "value": 0.919, "decimals": 3},
        {"name": "d", "label": "d", "value": 1.0, "decimals": 1},
    ]
    rows = rep.compare({"a": 0.918465, "b": 0.0788, "c": 0.918465}, expected)
    assert [r["match"] for r in rows] == [True, True, False, False]


def test_check_predictions_detects_identical_and_changed_files(tmp_path: Path) -> None:
    """Identical files match; a changed grade or logit is reported."""
    chk = _load("check_predictions")
    ids = [f"ts{i:06d}" for i in range(5)]
    frame = make_predictions(ids, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0.9] * 5)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    frame.to_csv(a, index=False)
    frame.sample(frac=1, random_state=0).to_csv(b, index=False)
    assert chk.compare(a, b) == (5, 0, 0.0)
    changed = frame.copy()
    changed.loc[0, "pred_grade"] = 3
    changed.loc[1, "logit_0"] += 0.5
    changed.to_csv(b, index=False)
    rows, grades, diff = chk.compare(a, b)
    assert grades == 1 and abs(diff - 0.5) < 1e-9
