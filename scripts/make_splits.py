"""Stage 2: build the fixed train/val/test split (brief 6.2).

Usage:
    python scripts/make_splits.py --config configs/data.yaml \
        --clusters outputs/audit/dup_clusters.csv --out splits/split_v1.csv

Writes the split (ids and subset names only, committed to git) and a JSON
summary next to it with counts and the grade distribution per subset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from drgrade.config import load_data_config, resolve_mmrdr_root
from drgrade.io import load_fp_csv
from drgrade.splits import grade_distribution, make_split
from drgrade.utils import get_git_info, setup_logging

logger = logging.getLogger(__name__)

N_FOLDS = 7  # validation is about 1/7 (14%) of the official training images
VAL_FOLD = 0
MAX_GRADE_GAP_PP = 1.5  # brief 6.2: per-grade share in train and val within 1.5 points


def _write_atomic(path: Path, text: str) -> None:
    """Write text to a temporary file beside `path`, then move it into place.

    A reader never sees a half-written split, and the replace also works when the
    existing file cannot be opened for writing (seen with synced macOS folders).
    """
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Create the fixed train/val/test split.")
    parser.add_argument("--config", type=Path, required=True, help="path to configs/data.yaml")
    parser.add_argument("--clusters", type=Path, required=True, help="dup_clusters.csv from the audit")
    parser.add_argument("--out", type=Path, required=True, help="output CSV, e.g. splits/split_v1.csv")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--val-fold", type=int, default=VAL_FOLD)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing split file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.out.exists() and not args.overwrite:
        logger.error("%s exists. The split is meant to be fixed; pass --overwrite to replace it.", args.out)
        return 1

    cfg = load_data_config(args.config)
    labels = load_fp_csv(resolve_mmrdr_root(cfg.mmrdr_root))
    clusters = pd.read_csv(args.clusters, dtype={"image_id": str})
    split = make_split(labels, clusters, n_folds=args.n_folds, val_fold=args.val_fold, seed=cfg.seed)

    dist = grade_distribution(split, labels)
    gap = (dist["train"] - dist["val"]).abs()
    logger.info("Grade share (%%) by subset:\n%s", dist.round(2).to_string())
    if (gap > MAX_GRADE_GAP_PP).any():
        logger.error("Train/val grade shares differ by more than %.1f points:\n%s", MAX_GRADE_GAP_PP, gap)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(args.out, split.to_csv(index=False))
    summary = {
        "split_file": args.out.name,
        "method": "StratifiedKFold over near-duplicate groups of official train images, stratified by group grade",
        "n_folds": args.n_folds,
        "val_fold": args.val_fold,
        "seed": cfg.seed,
        "clusters_file": args.clusters.name,
        "counts": split["subset"].value_counts().sort_index().to_dict(),
        "grade_share_pct": {subset: dist[subset].round(3).to_dict() for subset in dist.columns},
        "max_train_val_gap_pp": round(float(gap.max()), 3),
        "git": get_git_info(),
    }
    summary_path = args.out.with_suffix(".summary.json")
    _write_atomic(summary_path, json.dumps(summary, indent=2, default=str) + "\n")
    logger.info("Wrote %s (%s) and %s", args.out, summary["counts"], summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
