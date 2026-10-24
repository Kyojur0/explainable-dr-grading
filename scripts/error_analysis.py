"""Stage 6: error analysis of one predictions file.

Usage:
    PYTHONPATH=src python scripts/error_analysis.py --pred runs/<run>/predictions/val.csv \
        [--manifest DIR/manifest.csv] [--flags outputs/audit/flagged_images.csv] \
        [--leakage-list splits/test_with_train_copy.csv] [--out DIR]

Outputs go to <pred dir>/errors_<stem>/ (or --out): errors.csv, lesion_stratified.csv,
dangerous_misses.csv, quality_stratified.csv, cases.csv and one contact sheet per case
reason (contact_<reason>.png). The test file needs the leakage list.

Exit status: 0 on success, 2 on bad or inconsistent input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from drgrade.provenance import verify_for_use
from drgrade.config import ConfigError, load_data_config
from drgrade.errors import (
    add_scores,
    audit_flags_by_image,
    case_groups,
    dangerous_misses,
    error_table,
    join_predictions,
    lesion_stratified,
    quality_stratified,
    select_cases,
)
from drgrade.predict import PREDICTION_COLUMNS, SUBSET_TEST, SUBSET_VAL, image_folder
from drgrade.leakage import read_leakage_list
from drgrade.reporting import contact_sheet, subset_of_ids
from drgrade.utils import setup_logging, write_atomically

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_BAD_INPUT = 2
ERRORS_CSV = "errors.csv"
LESIONS_CSV = "lesion_stratified.csv"
DANGEROUS_CSV = "dangerous_misses.csv"
QUALITY_CSV = "quality_stratified.csv"
CASES_CSV = "cases.csv"
SHEET_PREFIX = "contact_"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Stage 6 error analysis.")
    parser.add_argument("--pred", required=True, type=Path, help="predictions CSV")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.csv (default: from the data config)")
    parser.add_argument("--flags", type=Path, default=Path("outputs/audit/flagged_images.csv"), help="audit flags (optional)")
    parser.add_argument("--split", type=Path, default=Path("splits/split_v1.csv"))
    parser.add_argument("--leakage-list", type=Path, default=Path("splits/test_with_train_copy.csv"))
    parser.add_argument("--out", type=Path, default=None, help="output folder (default <pred dir>/errors_<stem>/)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the error analysis for one predictions file and return the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    pred_path: Path = args.pred
    out_dir = args.out if args.out is not None else pred_path.parent / f"errors_{pred_path.stem}"
    try:
        manifest_path = args.manifest
        if manifest_path is None:
            manifest_path = image_folder(load_data_config(args.data_config)) / "manifest.csv"
        manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
        split = pd.read_csv(args.split, dtype={"image_id": str, "subset": str})
        preds = pd.read_csv(pred_path, dtype={"image_id": str}, float_precision="round_trip")
        missing = [c for c in PREDICTION_COLUMNS if c not in preds.columns]
        if missing:
            raise ValueError(f"{pred_path} lacks the prediction columns {missing}")
        subset = subset_of_ids(preds["image_id"].astype(str).tolist(), split)
        if subset not in (SUBSET_VAL, SUBSET_TEST):
            raise ValueError(f"{pred_path} holds {subset} images; only validation and test are analysed")
        expected = int((split["subset"] == subset).sum())
        if len(preds) != expected:
            raise ValueError(f"{pred_path} has {len(preds)} rows but the split has {expected} {subset} images (incomplete)")
        verify_for_use(pred_path, subset, expected, require_meta=False)
        if subset == SUBSET_TEST and not args.leakage_list.is_file():
            raise ValueError("test error analysis needs the leakage list (scripts/make_leakage_list.py)")
        leakage_ids = set(read_leakage_list(args.leakage_list)["image_id"].astype(str)) if args.leakage_list.is_file() else set()
        flags = _read_flags(args.flags)
        frame = add_scores(join_predictions(preds, manifest))
        errors = error_table(frame, audit_flags_by_image(flags), leakage_ids)
        lesions = lesion_stratified(frame)
        dangerous = dangerous_misses(frame)
        quality = quality_stratified(frame)
        cases = select_cases(frame)
        if not len(frame):
            raise ValueError("no predictions to analyse")
    except (OSError, ConfigError, ValueError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT

    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomically(out_dir / ERRORS_CSV, lambda p: errors.to_csv(p, index=False))
    write_atomically(out_dir / LESIONS_CSV, lambda p: lesions.to_csv(p, index=False))
    write_atomically(out_dir / DANGEROUS_CSV, lambda p: dangerous.to_csv(p, index=False))
    write_atomically(out_dir / QUALITY_CSV, lambda p: quality.to_csv(p, index=False))
    write_atomically(out_dir / CASES_CSV, lambda p: cases.to_csv(p, index=False))
    image_dir = manifest_path.parent
    for reason in case_groups(cases):
        group = cases.loc[cases["reason"] == reason]
        contact_sheet(
            group,
            image_dir,
            out_dir / f"{SHEET_PREFIX}{reason}.png",
            f"{reason} ({pred_path.stem}, {len(group)} images)",
        )
    logger.info(
        "errors %d of %d; dangerous misses %d; cases %d; written to %s",
        len(errors),
        len(frame),
        len(dangerous),
        len(cases),
        out_dir,
    )
    return STATUS_OK


def _read_flags(path: Path) -> pd.DataFrame | None:
    """Read the audit flagged-images table, or return None (with a warning) when it is absent."""
    if not path.is_file():
        logger.warning("audit flags not found at %s; audit_flags columns will be empty", path)
        return None
    return pd.read_csv(path, dtype={"image_id": str})


if __name__ == "__main__":
    sys.exit(main())
