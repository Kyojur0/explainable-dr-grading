"""Smoke test for scripts/plot_training.py on two synthetic run folders."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = REPO / "scripts" / "plot_training.py"


def _run(root: Path, name: str, best_epoch: int) -> Path:
    run = root / name
    run.mkdir(parents=True)
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.2, 1.0, 0.9],
            "val_loss": [1.1, 0.95, 0.97],
            "val_accuracy": [0.5, 0.6, 0.62],
            "val_macro_f1": [0.2, 0.3, 0.35],
            "val_qwk": [0.4, 0.7, 0.75],
            "lr": [3e-4, 2e-4, 1e-4],
            "epoch_seconds": [10.0, 10.0, 10.0],
        }
    )
    history.to_csv(run / "history.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"best_epoch": best_epoch}), encoding="utf-8")
    return run


def test_plot_writes_png_and_csv_with_selected_epochs(tmp_path: Path) -> None:
    """Both runs are plotted, the selected epoch is flagged, smoke folders are skipped."""
    spec = importlib.util.spec_from_file_location("plot_script", PLOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_script"] = module
    spec.loader.exec_module(module)

    runs = tmp_path / "runs"
    _run(runs, "e1_run", best_epoch=3)
    _run(runs, "e3_run", best_epoch=2)
    _run(runs, "smoke_e3", best_epoch=1)
    out = tmp_path / "compare"
    assert module.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    assert (out / "training_curves.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    table = pd.read_csv(out / "training_curves.csv")
    assert set(table["run"]) == {"e1_run", "e3_run"}
    selected = table.loc[table["selected"]].groupby("run")["epoch"].first().to_dict()
    assert selected == {"e1_run": 3, "e3_run": 2}


def test_plot_without_runs_is_refused(tmp_path: Path) -> None:
    """An empty runs folder is a bad input (status 2)."""
    spec = importlib.util.spec_from_file_location("plot_script2", PLOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_script2"] = module
    spec.loader.exec_module(module)
    assert module.main(["--runs-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o")]) == 2
