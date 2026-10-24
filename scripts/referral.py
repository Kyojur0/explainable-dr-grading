"""Stage 7: referral by confidence and the referable-threshold analysis (DECISIONS.md D30 to D32).

Usage:
    PYTHONPATH=src python scripts/referral.py --run runs/e2_effb0_512_ce \
        --match-sensitivity-of runs/e3_effb0_512_wce/val_predictions.csv [--test runs/e2_effb0_512_ce/predictions/test.csv]

Reads the temperature from <run>/calibration/calibration.json (fitted on validation by
scripts/calibrate.py) and never refits it. The validation file must be the one T was fitted on
(checked by SHA-256). Every operating point is chosen on validation. A test file, if given, only
receives the fixed thresholds and is reported.

Outputs go to <run>/referral/ (or --out): referral.json, referable_threshold.json,
risk_coverage_val.csv and .png, and risk_coverage_test.csv and .png when --test is given.
The test risk-coverage curve is descriptive; the test decision uses the fixed thresholds only.

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

from drgrade.calibration import CALIBRATION_JSON, LogitTable, apply_temperature, load_logit_table
from drgrade.metrics import full_metrics
from drgrade.predict import sha256_file
from drgrade.provenance import ProvenanceError, check_same_checkpoint, subset_size, verify_for_use
from drgrade.reliability_figures import save_risk_coverage
from drgrade.referral import (
    FRACTION_STEP,
    MAX_FRACTION,
    PRIMARY_FRACTION,
    REFERABLE_SENSITIVITY_TARGET,
    SECONDARY_TARGET_MISS_RATE,
    fixed_threshold_outcome,
    highest_threshold_reaching,
    operating_point,
    referable_sweep,
    referred_count,
    risk_coverage_table,
    scores_from_probs,
    secondary_row,
    auroc,
    threshold_for_count,
    threshold_grid,
    uncertainty_summary,
    Scores,
)
from drgrade.utils import setup_logging, utc_now_iso, write_atomically

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_BAD_INPUT = 2
REFERRAL_DIR = "referral"
REFERRAL_JSON = "referral.json"
REFERABLE_JSON = "referable_threshold.json"
VAL_FILE_NAME = "predictions/val.csv"  # CPU fp32, same path as test (DECISIONS.md D38)
REFERABLE_LEVEL = 2  # referable score uses grades 2 and above; the hard decision is pred >= 2
REFERENCE_COLUMNS = ("image_id", "true_grade", "pred_grade")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Stage 7 referral by confidence.")
    parser.add_argument("--run", required=True, type=Path, help="run folder, for example runs/e2_effb0_512_ce")
    parser.add_argument("--calibration", type=Path, default=None, help="calibration.json (default <run>/calibration/)")
    parser.add_argument("--val", type=Path, default=None, help="validation predictions (default <run>/predictions/val.csv)")
    parser.add_argument("--test", type=Path, default=None, help="test predictions; receives fixed thresholds only")
    parser.add_argument(
        "--match-sensitivity-of",
        type=Path,
        default=None,
        help="validation predictions of a reference run; its referable sensitivity is the target (DECISIONS.md D32)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output folder (default <run>/referral/)")
    parser.add_argument("--split", type=Path, default=Path("splits/split_v1.csv"), help="split, for row-count checks")
    return parser


def _read_calibration(path: Path) -> dict[str, Any]:
    """Return calibration.json, checking the keys that referral needs."""
    if not path.is_file():
        raise ValueError(f"{path} is missing: run scripts/calibrate.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "temperature" not in data or "val" not in data or "sha256" not in data.get("val", {}):
        raise ValueError(f"{path} lacks the temperature or the validation hash")
    return data


def _scored_frame(table: LogitTable, temperature: float) -> tuple[pd.DataFrame, Scores]:
    """Return the ids, labels and predictions, and the scores of the calibrated probabilities."""
    probs = apply_temperature(table.logits, temperature)
    frame = pd.DataFrame({"image_id": table.ids, "true_grade": table.labels, "pred_grade": table.pred})
    return frame, scores_from_probs(probs)


def _reference_frame(path: Path, val_ids: np.ndarray, val_labels: np.ndarray) -> pd.DataFrame:
    """Return the reference run's hard predictions aligned with the validation ids, after checking they cover the same images."""
    ref = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in REFERENCE_COLUMNS if c not in ref.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    if ref["image_id"].duplicated().any():
        raise ValueError(f"{path} lists an image id more than once")
    if set(ref["image_id"]) != set(val_ids.tolist()):
        raise ValueError(f"{path} does not cover the same images as the validation file")
    aligned = pd.DataFrame({"image_id": val_ids}).merge(ref[list(REFERENCE_COLUMNS)], on="image_id", how="left")
    if not np.array_equal(aligned["true_grade"].to_numpy().astype(np.int64), val_labels):
        raise ValueError(f"{path} disagrees with the validation labels")
    return aligned[list(REFERENCE_COLUMNS)]


def _plain(value: Any) -> Any:
    """Convert numpy scalars to Python values for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _json_text(payload: dict[str, Any]) -> str:
    """Return the JSON text with sorted keys, two-space indentation and a final newline."""
    return json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n"


def _threshold_or_none(value: Any) -> float | None:
    """Return a float threshold, or None for a missing value (the zero-fraction row)."""
    return None if value is None or pd.isna(value) else float(value)


def main(argv: list[str] | None = None) -> int:
    """Run the referral analysis and write its outputs; return the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run
    cal_path = args.calibration if args.calibration is not None else run_dir / "calibration" / CALIBRATION_JSON
    val_path = args.val if args.val is not None else run_dir / VAL_FILE_NAME
    out_dir = args.out if args.out is not None else run_dir / REFERRAL_DIR
    try:
        metas = [verify_for_use(val_path, "val", subset_size(args.split, "val"), require_meta=True)]
        if args.test is not None:
            metas.append(verify_for_use(args.test, "test", subset_size(args.split, "test"), require_meta=True))
        check_same_checkpoint(*metas)
        cal = _read_calibration(cal_path)
        temperature = float(cal["temperature"])
        val = load_logit_table(val_path)
        if val.sha256 != cal["val"]["sha256"]:
            raise ValueError(f"{val_path} differs from the file T was fitted on (see {cal_path})")
        val_frame, val_scores = _scored_frame(val, temperature)
        val_positive = val_frame["true_grade"].to_numpy() >= REFERABLE_LEVEL
        test_table = load_logit_table(args.test) if args.test is not None else None
        test_frame: pd.DataFrame | None = None
        test_scores: Scores | None = None
        test_positive: np.ndarray | None = None
        if test_table is not None:
            test_frame, test_scores = _scored_frame(test_table, temperature)
            test_positive = test_frame["true_grade"].to_numpy() >= REFERABLE_LEVEL
        reference_block: dict[str, Any] | None = None
        if args.match_sensitivity_of is not None:
            verify_for_use(args.match_sensitivity_of, "val", subset_size(args.split, "val"), require_meta=True)
            ref_frame = _reference_frame(args.match_sensitivity_of, val.ids, val.labels)
            ref_referable = full_metrics(ref_frame, n_boot=0)["referable"]
            reference_block = {
                "file": args.match_sensitivity_of.as_posix(),
                "sha256": sha256_file(args.match_sensitivity_of),
                "sensitivity": ref_referable["sensitivity"],
                "specificity": ref_referable["specificity"],
            }
        _analyse(
            run_name=run_dir.name,
            temperature=temperature,
            val=val,
            val_frame=val_frame,
            val_scores=val_scores,
            val_positive=val_positive,
            test_table=test_table,
            test_frame=test_frame,
            test_scores=test_scores,
            test_positive=test_positive,
            reference_block=reference_block,
            out_dir=out_dir,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError, ProvenanceError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    logger.info("referral outputs written to %s", out_dir)
    return STATUS_OK


def _analyse(
    run_name: str,
    temperature: float,
    val: LogitTable,
    val_frame: pd.DataFrame,
    val_scores: Scores,
    val_positive: np.ndarray,
    test_table: LogitTable | None,
    test_frame: pd.DataFrame | None,
    test_scores: Scores | None,
    test_positive: np.ndarray | None,
    reference_block: dict[str, Any] | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Choose every operating point on validation, apply the fixed thresholds to test, and write the outputs."""
    confidence_val = val_scores.confidence
    risk_val = risk_coverage_table(val_frame, confidence_val)
    uncertainty = uncertainty_summary(val_frame, val_scores)

    primary_threshold = threshold_for_count(confidence_val, referred_count(PRIMARY_FRACTION, len(val_frame)))
    secondary = secondary_row(risk_val)
    secondary_threshold = None if secondary is None else _threshold_or_none(secondary["threshold"])

    primary_val = operating_point(val_frame, confidence_val, primary_threshold)
    secondary_val = None if secondary is None else operating_point(val_frame, confidence_val, secondary_threshold)

    if test_frame is not None and test_scores is not None:
        confidence_test = test_scores.confidence
        risk_test = risk_coverage_table(test_frame, confidence_test)
        primary_test = operating_point(test_frame, confidence_test, primary_threshold)
        secondary_test = (
            None if secondary is None else operating_point(test_frame, confidence_test, secondary_threshold)
        )
    else:
        risk_test = None
        primary_test = None
        secondary_test = None

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_risk(risk_val, out_dir, "val", run_name)
    if risk_test is not None:
        _write_risk(risk_test, out_dir, "test", run_name)

    referable = _referable_analysis(val_scores, val_positive, val_frame, reference_block, test_scores, test_positive)

    primary_block: dict[str, Any] = {
        "rule": f"refer the most uncertain {PRIMARY_FRACTION:.0%} of cases (score = calibrated max probability)",
        "nominal_fraction": PRIMARY_FRACTION,
        "threshold": primary_threshold,
        "validation": primary_val,
        "test": primary_test,
    }
    secondary_block: dict[str, Any] = {
        "rule": (
            "lowest nominal fraction, in steps of "
            f"{FRACTION_STEP:.0%}, at which the retained referable miss rate on validation is at most "
            f"{SECONDARY_TARGET_MISS_RATE:.0%}; if none up to {MAX_FRACTION:.0%}, not reached"
        ),
        "target_retained_referable_miss_rate": SECONDARY_TARGET_MISS_RATE,
        "reached": secondary is not None,
        "nominal_fraction": None if secondary is None else float(secondary["nominal_fraction"]),
        "threshold": secondary_threshold,
        "validation": secondary_val,
        "test": secondary_test,
    }
    referral_payload: dict[str, Any] = {
        "run": run_name,
        "temperature": temperature,
        "tta": val.tta,
        "validation_file": val.path.as_posix(),
        "validation_sha256": val.sha256,
        "test_file": None if test_table is None else test_table.path.as_posix(),
        "test_sha256": None if test_table is None else test_table.sha256,
        "pre_declared_rules": {
            "decided_before_any_test_prediction": "2026-10-20 (DECISIONS.md D30)",
            "primary": PRIMARY_FRACTION,
            "secondary_target_miss_rate": SECONDARY_TARGET_MISS_RATE,
        },
        "uncertainty_validation": uncertainty,
        "primary": primary_block,
        "secondary": secondary_block,
        "risk_coverage_files": {
            "validation": "risk_coverage_val.csv",
            "test": None if risk_test is None else "risk_coverage_test.csv",
        },
        "test_curve_note": "risk_coverage_test.csv is descriptive; test decisions use the fixed thresholds only",
        "referable_threshold_file": REFERABLE_JSON,
        "written_utc": utc_now_iso(),
    }
    write_atomically(out_dir / REFERRAL_JSON, lambda p: p.write_text(_json_text(referral_payload), encoding="utf-8"))
    write_atomically(out_dir / REFERABLE_JSON, lambda p: p.write_text(_json_text(referable), encoding="utf-8"))
    return referral_payload


def _write_risk(table: pd.DataFrame, out_dir: Path, split: str, run_name: str) -> None:
    """Write the risk-coverage CSV and its two-panel PNG for one split."""
    write_atomically(out_dir / f"risk_coverage_{split}.csv", lambda p: table.to_csv(p, index=False))
    save_risk_coverage(
        table,
        out_dir / f"risk_coverage_{split}.png",
        f"{run_name}: risk-coverage on {split} (confidence-ranked referral)",
    )


def _referable_analysis(
    val_scores: Scores,
    val_positive: np.ndarray,
    val_frame: pd.DataFrame,
    reference_block: dict[str, Any] | None,
    test_scores: Scores | None,
    test_positive: np.ndarray | None,
) -> dict[str, Any]:
    """Return referable_threshold.json: the threshold chosen on validation and its outcome on test."""
    sweep = referable_sweep(val_scores.referable, val_positive)
    if reference_block is not None:
        target = float(reference_block["sensitivity"])
        target_source = "sensitivity of the reference run's validation hard predictions"
    else:
        target = REFERABLE_SENSITIVITY_TARGET
        target_source = "fixed target (no reference run given)"
    chosen = highest_threshold_reaching(sweep, target)
    threshold = float(chosen["threshold"])
    hard = full_metrics(val_frame, n_boot=0)["referable"]
    val_block = fixed_threshold_outcome(val_scores.referable, val_positive, threshold)
    val_block["auroc"] = auroc(val_scores.referable, val_positive)
    payload: dict[str, Any] = {
        "score": "calibrated p(2) + p(3) + p(4)",
        "rule": "highest threshold on validation whose sensitivity reaches the target (DECISIONS.md D32)",
        "target_sensitivity": target,
        "target_source": target_source,
        "reference": reference_block,
        "model_hard_decision_validation": {"sensitivity": hard["sensitivity"], "specificity": hard["specificity"]},
        "validation": val_block,
        "threshold_grid_validation": threshold_grid(val_scores.referable, val_positive),
        "test": None,
        "written_utc": utc_now_iso(),
    }
    if test_scores is not None and test_positive is not None:
        test_block = fixed_threshold_outcome(test_scores.referable, test_positive, threshold)
        test_block["auroc"] = auroc(test_scores.referable, test_positive)
        payload["test"] = test_block
    return payload


if __name__ == "__main__":
    sys.exit(main())
