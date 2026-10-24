"""Quantify how the training-time validation file differs from the CPU prediction of the same model.

`<run>/val_predictions.csv` is written on the GPU during training (mixed precision);
`<run>/predictions/val.csv` is written by scripts/predict.py on the CPU, the path the test
set uses. Selection used the first; calibration, referral and test use the second
(DECISIONS.md D38). This writes, per run, the number of images whose predicted grade
differs, the largest absolute probability difference, and the headline metrics on each.

Usage:
    python scripts/compare_val_paths.py --runs runs/e2_effb0_512_ce runs/e3_effb0_512_wce \
        --out results/val_path_differences.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from drgrade.metrics import full_metrics
from drgrade.predict import PROB_COLUMNS
from drgrade.utils import setup_logging, utc_now_iso

logger = logging.getLogger(__name__)


def compare(run_dir: Path) -> dict:
    """Return the differences between the two validation files of one run."""
    gpu = pd.read_csv(run_dir / "val_predictions.csv", dtype={"image_id": str}, float_precision="round_trip")
    cpu = pd.read_csv(run_dir / "predictions" / "val.csv", dtype={"image_id": str}, float_precision="round_trip")
    merged = gpu.merge(cpu, on="image_id", suffixes=("_gpu", "_cpu"), validate="one_to_one")
    if len(merged) != len(gpu) or len(merged) != len(cpu):
        raise ValueError(f"{run_dir}: the two validation files hold different images")
    if not (merged["true_grade_gpu"] == merged["true_grade_cpu"]).all():
        raise ValueError(f"{run_dir}: true grades differ between the files")
    flips = merged["pred_grade_gpu"] != merged["pred_grade_cpu"]
    prob_diff = np.abs(
        merged[[f"{c}_gpu" for c in PROB_COLUMNS]].to_numpy() - merged[[f"{c}_cpu" for c in PROB_COLUMNS]].to_numpy()
    )

    def headline(frame: pd.DataFrame) -> dict:
        m = full_metrics(frame, n_boot=0)
        return {"qwk": m["qwk"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "referable_sensitivity": m["referable"]["sensitivity"], "referable_auroc": m["referable"]["auroc"]}

    return {
        "n": int(len(merged)),
        "pred_grade_differs": int(flips.sum()),
        "differing_images": [
            {"image_id": r.image_id, "true": int(r.true_grade_gpu), "gpu": int(r.pred_grade_gpu), "cpu": int(r.pred_grade_cpu)}
            for r in merged.loc[flips].itertuples()
        ],
        "max_abs_prob_difference": float(prob_diff.max()),
        "training_time_gpu": headline(gpu),
        "cpu_predict": headline(cpu),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = {"written_utc": utc_now_iso(), "runs": {r.name: compare(r) for r in args.runs}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, block in payload["runs"].items():
        logger.info("%s: %d of %d grades differ; max |dp| %.2e", name, block["pred_grade_differs"], block["n"], block["max_abs_prob_difference"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
