"""Stage 6 support: training curves for every run in a runs folder (or the runs given).

Usage:
    PYTHONPATH=src python scripts/plot_training.py [--runs-dir runs] [--run runs/e1 runs/e2 ...] [--out-dir outputs/compare]

Left panel: validation QWK (solid) and validation macro-F1 (dashed). Right panel: training
loss (dashed) and validation loss (solid). One colour per run; the selected epoch (from
summary.json) is marked with a circle. Validation loss of weighted runs is a weighted loss
and is not comparable with unweighted runs (DECISIONS.md D15), which the figure states.

Writes training_curves.png (150 dpi) and training_curves.csv with the plotted values.
Runs whose folder name starts with smoke_ are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from drgrade.reporting import FIGURE_DPI
from drgrade.utils import setup_logging, write_atomically

logger = logging.getLogger(__name__)

HISTORY_NAME = "history.csv"
SUMMARY_NAME = "summary.json"
SMOKE_PREFIX = "smoke_"
COLOURS: tuple[str, ...] = ("#1f77b4", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e")
CURVE_COLUMNS: tuple[str, ...] = ("run", "epoch", "train_loss", "val_loss", "val_qwk", "val_macro_f1", "selected")
STATUS_OK = 0
STATUS_BAD_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Plot training curves for runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run", nargs="+", type=Path, default=None, help="run folders (overrides --runs-dir)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/compare"))
    return parser


def find_runs(runs_dir: Path) -> list[Path]:
    """Return the run folders under runs_dir that have a history.csv and are not smoke runs."""
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if (p / HISTORY_NAME).is_file() and not p.name.startswith(SMOKE_PREFIX))


def selected_epoch(run_dir: Path, history: pd.DataFrame) -> int | None:
    """Return the selected epoch from summary.json, or None when the run is unfinished and has no summary."""
    summary_path = run_dir / SUMMARY_NAME
    if summary_path.is_file():
        summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        epoch = summary.get("best_epoch")
        return int(epoch) if epoch is not None else None
    return None


def curves_frame(run_dir: Path) -> tuple[pd.DataFrame, int | None]:
    """Return the plotted values of one run and its selected epoch."""
    history = pd.read_csv(run_dir / HISTORY_NAME)
    chosen = selected_epoch(run_dir, history)
    frame = pd.DataFrame(
        {
            "run": run_dir.name,
            "epoch": history["epoch"].astype(int),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_qwk": history["val_qwk"],
            "val_macro_f1": history["val_macro_f1"],
            "selected": history["epoch"].astype(int) == (chosen if chosen is not None else -1),
        }
    )
    return frame, chosen


def main(argv: list[str] | None = None) -> int:
    """Write the training-curve figure and its CSV. Returns the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    run_dirs = list(args.run) if args.run else find_runs(args.runs_dir)
    if not run_dirs:
        logger.error("no runs with %s found", HISTORY_NAME)
        return STATUS_BAD_INPUT
    frames: list[pd.DataFrame] = []
    try:
        for run_dir in run_dirs:
            frame, _ = curves_frame(run_dir)
            frames.append(frame)
    except (OSError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    table = pd.concat(frames, ignore_index=True).loc[:, list(CURVE_COLUMNS)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_atomically(args.out_dir / "training_curves.csv", lambda p: table.to_csv(p, index=False))
    _figure(frames, args.out_dir / "training_curves.png")
    logger.info("wrote training curves for %d runs to %s", len(frames), args.out_dir)
    return STATUS_OK


def _figure(frames: list[pd.DataFrame], path: Path) -> None:
    """Draw the two panels and write the PNG."""
    fig = Figure(figsize=(11.0, 4.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    left = fig.add_subplot(1, 2, 1)
    right = fig.add_subplot(1, 2, 2)
    for index, frame in enumerate(frames):
        colour = COLOURS[index % len(COLOURS)]
        name = str(frame["run"].iloc[0])
        left.plot(frame["epoch"], frame["val_qwk"], color=colour, linestyle="-", label=f"{name}")
        left.plot(frame["epoch"], frame["val_macro_f1"], color=colour, linestyle="--")
        right.plot(frame["epoch"], frame["val_loss"], color=colour, linestyle="-", label=f"{name}")
        right.plot(frame["epoch"], frame["train_loss"], color=colour, linestyle="--")
        chosen = frame.loc[frame["selected"]]
        if len(chosen):
            left.scatter(chosen["epoch"], chosen["val_qwk"], color=colour, edgecolor="black", zorder=3, s=60)
            right.scatter(chosen["epoch"], chosen["val_loss"], color=colour, edgecolor="black", zorder=3, s=60)
    left.set_title("Validation QWK (solid) and macro-F1 (dashed)")
    left.set_xlabel("epoch")
    left.set_ylabel("score")
    left.legend(loc="lower right", fontsize=8)
    right.set_title("Validation loss (solid) and training loss (dashed)")
    right.set_xlabel("epoch")
    right.set_ylabel("loss")
    right.legend(loc="upper right", fontsize=8)
    fig.text(
        0.5,
        -0.02,
        "Circles mark the selected epoch. Weighted-loss runs have weighted losses, so compare losses only between runs with the same loss (DECISIONS.md D15).",
        ha="center",
        fontsize=8,
    )
    write_atomically(path, lambda p: fig.savefig(p, dpi=FIGURE_DPI, format="png", bbox_inches="tight"))


if __name__ == "__main__":
    sys.exit(main())
