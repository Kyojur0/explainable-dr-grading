"""Stage 7: temperature scaling fitted on validation logits, and calibration metrics.

Usage:
    PYTHONPATH=src python scripts/calibrate.py --run runs/e2_effb0_512_ce
    PYTHONPATH=src python scripts/calibrate.py --run runs/e2_effb0_512_ce \
        --val runs/e2_effb0_512_ce/val_predictions.csv --test runs/e2_effb0_512_ce/predictions/test.csv

The temperature is fitted on the validation file only. A test file, if given, is read only to
report calibration metrics before and after scaling; it is never fitted on (DECISIONS.md D28).
Outputs go to <run>/calibration/ (or --out):
    calibration.json, reliability_bins.csv, val_calibrated.csv, reliability_val.png,
    and, when --test is given, test_calibrated.csv and reliability_test.png.
Calibrated files keep the columns of the input. Their probabilities are recalibrated, their
logit columns are the input logits divided by T, and pred_grade is unchanged.

Exit status: 0 on success, 2 on bad input.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drgrade.calibration import (
    CALIBRATION_JSON,
    DEFAULT_N_BINS,
    LogitTable,
    apply_temperature,
    fit_temperature,
    load_logit_table,
    summarise,
)
from drgrade.io import GRADE_VALUES
from drgrade.predict import LOGIT_COLUMNS, PROB_COLUMNS
from drgrade.provenance import ProvenanceError, check_same_checkpoint, subset_size, verify_for_use
from drgrade.reliability_figures import save_reliability
from drgrade.utils import setup_logging, utc_now_iso, write_atomically

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_BAD_INPUT = 2
CALIBRATION_DIR = "calibration"
VAL_FILE_NAME = "predictions/val.csv"  # CPU fp32, same path as test (DECISIONS.md D38)
BINS_CSV = "reliability_bins.csv"
VAL_CALIBRATED_CSV = "val_calibrated.csv"
TEST_CALIBRATED_CSV = "test_calibrated.csv"
METHOD = (
    "temperature scaling: one scalar T dividing the logits; LBFGS over log T in float64, "
    "strong Wolfe line search, at most 100 iterations, starting at T = 1"
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Stage 7 calibration.")
    parser.add_argument("--run", required=True, type=Path, help="run folder, for example runs/e2_effb0_512_ce")
    parser.add_argument("--val", type=Path, default=None, help="validation predictions (default <run>/predictions/val.csv)")
    parser.add_argument("--test", type=Path, default=None, help="test predictions; read for reporting only")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default <run>/calibration/)")
    parser.add_argument("--split", type=Path, default=Path("splits/split_v1.csv"), help="split, for row-count checks")
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS, help="equal-width bins for ECE (DECISIONS.md D29)")
    return parser


def _calibrated_frame(table: LogitTable, probs: np.ndarray, temperature: float) -> pd.DataFrame:
    """Return the input rows with recalibrated probabilities and logits divided by T; pred_grade checked unchanged."""
    if not np.array_equal(probs.argmax(axis=1), table.pred):
        raise RuntimeError("temperature scaling changed a predicted grade; this must not happen")
    out = table.frame.copy()
    for index, grade in enumerate(GRADE_VALUES):
        out[PROB_COLUMNS[index]] = probs[:, index]
        out[LOGIT_COLUMNS[index]] = table.logits[:, index] / temperature
    return out


def _metrics_block(table: LogitTable, temperature: float, n_bins: int) -> dict[str, Any]:
    """Return the before and after metrics (no bins) and the bins of both, for one labelled file."""
    before = summarise(apply_temperature(table.logits, 1.0), table.labels, n_bins)
    after = summarise(apply_temperature(table.logits, temperature), table.labels, n_bins)
    return {
        "file": table.path.as_posix(),
        "sha256": table.sha256,
        "n": before["n"],
        "tta": table.tta,
        "before": {k: before[k] for k in ("accuracy", "ece", "nll", "brier")},
        "after": {k: after[k] for k in ("accuracy", "ece", "nll", "brier")},
        "_bins": {"before": before["bins"], "after": after["bins"]},
    }


def run_calibration(
    run_name: str,
    val: LogitTable,
    test: LogitTable | None,
    out_dir: Path,
    n_bins: int,
) -> dict[str, Any]:
    """Fit T on validation, write every calibration output and return calibration.json as a dict."""
    temperature = fit_temperature(val.logits, val.labels)
    val_block = _metrics_block(val, temperature, n_bins)
    val_probs = apply_temperature(val.logits, temperature)
    rows_for_csv: list[dict[str, Any]] = []
    panels: dict[str, dict[str, Any]] = {"val": val_block}

    write_atomically(
        out_dir / VAL_CALIBRATED_CSV,
        lambda p: _calibrated_frame(val, val_probs, temperature).to_csv(p, index=False),
    )
    if test is not None:
        test_block = _metrics_block(test, temperature, n_bins)
        test_probs = apply_temperature(test.logits, temperature)
        write_atomically(
            out_dir / TEST_CALIBRATED_CSV,
            lambda p: _calibrated_frame(test, test_probs, temperature).to_csv(p, index=False),
        )
        panels["test"] = test_block

    for split, block in panels.items():
        for stage in ("before", "after"):
            for row in block["_bins"][stage]:
                rows_for_csv.append({"split": split, "stage": stage, **row})
    write_atomically(out_dir / BINS_CSV, lambda p: pd.DataFrame(rows_for_csv).to_csv(p, index=False))

    for split, block in panels.items():
        save_reliability(
            [
                (f"{split}, before scaling (ECE {block['before']['ece']:.4f})", block["_bins"]["before"]),
                (f"{split}, after scaling (ECE {block['after']['ece']:.4f})", block["_bins"]["after"]),
            ],
            out_dir / f"reliability_{split}.png",
            f"{run_name}: reliability on {split} (T = {temperature:.4f})",
        )

    summary: dict[str, Any] = {
        "run": run_name,
        "method": METHOD,
        "fitted_on": "val",
        "temperature": temperature,
        "tta": val.tta,
        "n_bins": n_bins,
        "n_val": val_block["n"],
        "val": {k: v for k, v in val_block.items() if not k.startswith("_")},
        "test": None,
        "written_utc": utc_now_iso(),
        "notes": [
            "calibrated files: probabilities are softmax(logits / T); logit columns are logits / T",
            "pred_grade is unchanged by temperature scaling (checked)",
        ],
    }
    if test is not None:
        summary["test"] = {k: v for k, v in panels["test"].items() if not k.startswith("_")}
        summary["notes"].append("the test file is reported only; T was fitted on validation logits")
    if val.tta:
        summary["notes"].append(
            "validation file is a TTA file: T rescales log of averaged probabilities, still a monotone "
            "recalibration (DECISIONS.md D28)"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run calibration for one run folder and return the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run
    val_path = args.val if args.val is not None else run_dir / VAL_FILE_NAME
    out_dir = args.out if args.out is not None else run_dir / CALIBRATION_DIR
    try:
        metas = [verify_for_use(val_path, "val", subset_size(args.split, "val"), require_meta=True)]
        if args.test is not None:
            metas.append(verify_for_use(args.test, "test", subset_size(args.split, "test"), require_meta=True))
        check_same_checkpoint(*metas)
        val = load_logit_table(val_path)
        test = load_logit_table(args.test) if args.test is not None else None
        if args.n_bins < 1:
            raise ValueError("--n-bins must be at least 1")
        summary = run_calibration(run_dir.name, val, test, out_dir, args.n_bins)
    except (OSError, ValueError, ProvenanceError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    write_atomically(
        out_dir / CALIBRATION_JSON,
        lambda p: p.write_text(_dump(summary), encoding="utf-8"),
    )
    logger.info(
        "T = %.6f; val ECE %.4f -> %.4f; outputs in %s",
        summary["temperature"],
        summary["val"]["before"]["ece"],
        summary["val"]["after"]["ece"],
        out_dir,
    )
    return STATUS_OK


def _dump(payload: dict[str, Any]) -> str:
    """Return the JSON text with sorted keys and two-space indentation, ending in a newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    sys.exit(main())
