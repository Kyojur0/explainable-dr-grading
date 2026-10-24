"""Stage 5: evaluate predictions tables, or compare validation runs (brief 9.2 and 9.3).

Usage:
    PYTHONPATH=src python scripts/evaluate.py --pred runs/<run>/predictions/val.csv
    PYTHONPATH=src python scripts/evaluate.py --pred runs/<run>/predictions/test.csv [--out DIR]
    PYTHONPATH=src python scripts/evaluate.py --compare --pred runs/<a>/val_predictions.csv runs/<b>/val_predictions.csv ...

Outputs for each predictions file go to <pred dir>/eval_<stem>/ (or --out, with one file):
metrics.json, per_class.csv, confusion_counts.csv, confusion_norm.csv, confusion_norm.png, bootstrap_ci.json.
--compare writes outputs/compare/val_comparison.{csv,md} (validation only) and states the
pre-declared selection (DECISIONS.md D16).

Exit status: 0 on success, 2 on bad or inconsistent input.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drgrade.provenance import verify_for_use
from drgrade.config import ConfigError, load_data_config
from drgrade.errors import join_predictions
from drgrade.io import GRADE_VALUES
from drgrade.leakage import read_leakage_list
from drgrade.metrics import DEFAULT_N_BOOT, DEFAULT_SEED, full_metrics, majority_baseline_frame, majority_grade
from drgrade.predict import PREDICTION_COLUMNS, SUBSET_TEST, SUBSET_VAL, image_folder
from drgrade.reporting import (
    headline_row,
    json_safe,
    subset_of_ids,
    write_comparison,
    write_confusion_figure,
)
from drgrade.utils import setup_logging, write_atomically

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_BAD_INPUT = 2
BLACK_BORDER_MAX_LEVEL = 8  # border_level below this is a black background; at or above is not (DECISIONS.md D13)
PER_CLASS_COLUMNS: tuple[str, ...] = ("grade", "precision", "recall", "f1", "support")
RUN_SUMMARY_NAME = "summary.json"
QWK_CROSSCHECK_TOLERANCE = 1e-6
TRAIN_SUBSET = "train"
COMPARE_DIR = Path("outputs/compare")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Evaluate predictions (brief 9.2) or compare validation runs.")
    parser.add_argument("--pred", nargs="+", required=True, type=Path, help="predictions CSV file(s)")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.csv (default: from the data config)")
    parser.add_argument("--split", type=Path, default=Path("splits/split_v1.csv"))
    parser.add_argument("--leakage-list", type=Path, default=Path("splits/test_with_train_copy.csv"))
    parser.add_argument("--train-majority-grade", type=int, default=None, help="M0 grade (default: from the training subset)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (only with one --pred)")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--compare", action="store_true", help="compare validation runs instead")
    parser.add_argument("--compare-dir", type=Path, default=COMPARE_DIR)
    parser.add_argument(
        "--no-selection",
        action="store_true",
        help="compare without applying the selection rule (stability runs and ablations, DECISIONS.md D34)",
    )
    parser.add_argument("--note", default=None, help="footer text for a --no-selection table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation or the comparison and return the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        split = pd.read_csv(args.split, dtype={"image_id": str, "subset": str})
        if args.compare:
            return _compare(args, split)
        manifest_path = args.manifest
        if manifest_path is None:
            manifest_path = image_folder(load_data_config(args.data_config)) / "manifest.csv"
        manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
        majority = args.train_majority_grade
        if majority is None:
            train_ids = split.loc[split["subset"] == TRAIN_SUBSET, "image_id"].astype(str)
            train_grades = manifest.loc[manifest["image_id"].astype(str).isin(set(train_ids)), "grade"].astype(int)
            majority = majority_grade(train_grades.tolist())
        logger.info("M0 majority grade (training subset): %d", majority)
        if args.out is not None and len(args.pred) != 1:
            raise ValueError("--out can be used with one --pred only")
        leakage_ids = _leakage_ids(args.leakage_list)
        for pred_path in args.pred:
            out_dir = args.out if args.out is not None else _default_out(pred_path)
            evaluate_one(pred_path, out_dir, manifest, split, majority, leakage_ids, args.n_boot, args.seed)
    except (OSError, ConfigError, ValueError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    return STATUS_OK


def _default_out(pred_path: Path) -> Path:
    """Return <pred dir>/eval_<stem>/."""
    return pred_path.parent / f"eval_{pred_path.stem}"


def _leakage_ids(path: Path) -> set[str] | None:
    """Return the ids of the leakage list, or None when the file is absent (only test evaluation needs it)."""
    if not path.is_file():
        return None
    return set(read_leakage_list(path)["image_id"].astype(str))


def _read_predictions(path: Path) -> pd.DataFrame:
    """Read a predictions CSV with round-trip float precision and check its columns."""
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks the prediction columns {missing}")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{path} lists an image id more than once")
    return frame


def evaluate_one(
    pred_path: Path,
    out_dir: Path,
    manifest: pd.DataFrame,
    split: pd.DataFrame,
    majority: int,
    leakage_ids: set[str] | None,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Compute and write every evaluation output for one predictions file. Returns metrics.json's content."""
    preds = _read_predictions(pred_path)
    ids = preds["image_id"].astype(str).tolist()
    subset = subset_of_ids(ids, split)
    if subset not in (SUBSET_VAL, SUBSET_TEST):
        raise ValueError(f"{pred_path} holds {subset} images; only validation and test are evaluated")
    expected = int((split["subset"] == subset).sum())
    if len(preds) != expected:
        raise ValueError(f"{pred_path} has {len(preds)} rows but the split has {expected} {subset} images (incomplete)")
    # Test files must carry a sidecar whose hash is in the access log and the lock (D38).
    verify_for_use(pred_path, subset, expected, require_meta=False)
    joined = join_predictions(preds, manifest)
    headline = full_metrics(preds, n_boot=n_boot, seed=seed)

    if subset == SUBSET_TEST:
        if leakage_ids is None:
            raise ValueError("test evaluation needs the leakage list (scripts/make_leakage_list.py)")
        in_leak = preds["image_id"].astype(str).isin(leakage_ids).to_numpy()
    else:
        in_leak = np.zeros(len(preds), dtype=bool)
    kept = preds.loc[~in_leak]
    excluding = full_metrics(kept, n_boot=n_boot, seed=seed) if len(kept) else None

    baseline_frame = majority_baseline_frame(preds, majority)
    baseline = full_metrics(baseline_frame, n_boot=n_boot, seed=seed)

    groups: dict[str, Any] = {}
    black = joined["border_level"].to_numpy() < BLACK_BORDER_MAX_LEVEL
    for name, mask in (("black", black), ("non_black", ~black)):
        sub = preds.loc[mask]
        groups[name] = {
            "border_level_rule": f"< {BLACK_BORDER_MAX_LEVEL}" if name == "black" else f">= {BLACK_BORDER_MAX_LEVEL}",
            "n": int(mask.sum()),
            "metrics": full_metrics(sub, n_boot=0) if len(sub) else None,
        }

    bootstrap_headline = headline.pop("bootstrap_ci")
    baseline_bootstrap = baseline.pop("bootstrap_ci")
    headline_without_ci = dict(headline)
    if excluding is not None:
        excluding.pop("bootstrap_ci", None)
    baseline_metrics = dict(baseline)
    metrics = {
        "name": pred_path.stem,
        "predictions": str(pred_path),
        "subset": subset,
        "n": headline["n"],
        "headline": headline_without_ci,
        "m0_majority_baseline": {"train_majority_grade": majority, **baseline_metrics},
        "excluding_train_copies": {
            "n_excluded": int(in_leak.sum()),
            "metrics": excluding,
        },
        "by_border_group": groups,
        "sensitivity_rule": "excluding_train_copies uses splits/test_with_train_copy.csv; zero excluded on validation",
        "bootstrap": {"n_boot": n_boot, "seed": seed},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(
        out_dir / "bootstrap_ci.json",
        {"headline": bootstrap_headline, "m0_majority_baseline": baseline_bootstrap},
    )
    per_class = pd.DataFrame(headline["per_class"], columns=list(PER_CLASS_COLUMNS))
    write_atomically(out_dir / "per_class.csv", lambda p: per_class.to_csv(p, index=False))
    counts = pd.DataFrame(
        headline["confusion_counts"],
        index=[f"true_{g}" for g in GRADE_VALUES],
        columns=[f"pred_{g}" for g in GRADE_VALUES],
    )
    write_atomically(out_dir / "confusion_counts.csv", lambda p: counts.to_csv(p))
    norm = pd.DataFrame(
        headline["confusion_norm"],
        index=[f"true_{g}" for g in GRADE_VALUES],
        columns=[f"pred_{g}" for g in GRADE_VALUES],
    )
    write_atomically(out_dir / "confusion_norm.csv", lambda p: norm.to_csv(p))
    write_confusion_figure(
        np.array(headline["confusion_counts"]),
        headline["confusion_norm"],
        f"Row-normalised confusion, {pred_path.stem} (n={headline['n']})",
        out_dir / "confusion_norm.png",
    )
    logger.info("wrote evaluation for %s to %s (subset %s, n %d)", pred_path, out_dir, subset, headline["n"])
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write sorted, indented JSON atomically, with NaN replaced by null."""
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
    write_atomically(path, lambda p: p.write_text(text, encoding="utf-8"))


def _run_folder(pred_path: Path) -> Path:
    """Return the run folder of a predictions file: the parent, or the grandparent for predictions/."""
    if pred_path.parent.name == "predictions":
        return pred_path.parent.parent
    return pred_path.parent


def _compare(args: argparse.Namespace, split: pd.DataFrame) -> int:
    """Write the validation comparison table for several runs. Returns the exit status."""
    rows: list[dict[str, Any]] = []
    for pred_path in args.pred:
        preds = _read_predictions(pred_path)
        subset = subset_of_ids(preds["image_id"].astype(str).tolist(), split)
        if subset != SUBSET_VAL:
            raise ValueError(f"{pred_path} is not a validation predictions file (found {subset})")
        metrics = full_metrics(preds, n_boot=0)
        run_name = _run_folder(pred_path).name
        _cross_check_qwk(_run_folder(pred_path), run_name, metrics["qwk"])
        rows.append(headline_row(run_name, metrics))
    if len({row["run"] for row in rows}) != len(rows):
        raise ValueError("comparison needs one predictions file per run (run names repeat)")
    compare_dir = args.compare_dir
    compare_dir.mkdir(parents=True, exist_ok=True)
    selected = write_comparison(
        rows,
        compare_dir / "val_comparison.csv",
        compare_dir / "val_comparison.md",
        select=not args.no_selection,
        note=args.note,
    )
    if selected is None:
        logger.info("comparison without selection written to %s", compare_dir)
    else:
        logger.info("pre-declared rule selects %s; table written to %s", selected, compare_dir)
    return STATUS_OK


def _cross_check_qwk(run_dir: Path, run_name: str, qwk: float | None) -> None:
    """Warn when the QWK from the predictions differs from summary.json's best validation QWK."""
    summary_path = run_dir / RUN_SUMMARY_NAME
    if not summary_path.is_file():
        logger.warning("%s: no %s to cross-check QWK against", run_name, RUN_SUMMARY_NAME)
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded = (summary.get("best_val") or {}).get("qwk")
    if recorded is None or qwk is None or math.isnan(qwk):
        logger.warning("%s: QWK cannot be cross-checked (recorded %s, computed %s)", run_name, recorded, qwk)
        return
    if abs(float(recorded) - float(qwk)) > QWK_CROSSCHECK_TOLERANCE:
        logger.warning(
            "%s: QWK from predictions %.10f differs from summary.json %.10f", run_name, qwk, recorded
        )


if __name__ == "__main__":
    sys.exit(main())
