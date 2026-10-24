"""Stage 5: list test images that have a genuine near-duplicate in official training data.

Usage:
    PYTHONPATH=src python scripts/make_leakage_list.py \
        [--pairs outputs/audit/near_duplicate_pairs.csv] \
        [--clusters outputs/audit/dup_clusters.csv] \
        [--false-matches splits/false_matches.csv] \
        [--split splits/split_v1.csv] \
        [--out splits/test_with_train_copy.csv]

The output is committed to git (it contains ids only). The count is logged, not fixed
in code (DECISIONS.md D22).

Exit status: 0 on success, 2 when an input is missing or inconsistent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from drgrade.leakage import (
    build_leakage_list,
    check_against_split,
    read_false_matches,
    write_leakage_list,
)
from drgrade.utils import setup_logging

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_BAD_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Build the test-with-train-copy list.")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/audit/near_duplicate_pairs.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("outputs/audit/dup_clusters.csv"))
    parser.add_argument("--false-matches", type=Path, default=Path("splits/false_matches.csv"))
    parser.add_argument("--split", type=Path, default=Path("splits/split_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("splits/test_with_train_copy.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the list and write it. Returns the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        pairs = pd.read_csv(args.pairs, dtype={"image_id_a": str, "image_id_b": str, "split_a": str, "split_b": str})
        clusters = pd.read_csv(args.clusters, dtype={"image_id": str, "official_split": str})
        split = pd.read_csv(args.split, dtype={"image_id": str, "subset": str})
        false_matches = read_false_matches(args.false_matches)
        check_against_split(clusters, split)
        frame = build_leakage_list(pairs, clusters, false_matches)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    write_leakage_list(frame, args.out)
    logger.info("wrote %s with %d rows", args.out, len(frame))
    return STATUS_OK


if __name__ == "__main__":
    sys.exit(main())
