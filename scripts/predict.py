"""Stage 5: predict a validation or the locked test subset with one trained run (brief 9.1).

Usage:
    PYTHONPATH=src python scripts/predict.py --run runs/e3_effb0_512_wce --subset val [--tta]
    PYTHONPATH=src python scripts/predict.py --run runs/e3_effb0_512_wce --subset test --final [--tta]
    PYTHONPATH=src python scripts/predict.py --run runs/e3_effb0_512_wce --close-lock
    PYTHONPATH=src python scripts/predict.py --run runs/e3_effb0_512_wce --subset test --final \
        --force --reason "TEXT"          (only after the lock is closed, or to repeat an output)

Outputs go to <run>/predictions/<subset>[_tta].csv. Interrupted runs are resumed from
<output>.partial.csv: run the same command again until it exits with status 0.

Exit status: 0 done (or lock closed), 2 refused or bad input (including the test lock),
3 stopped at --max-seconds with work left (run the same command again), 1 unexpected error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from drgrade.config import ConfigError, load_data_config
from drgrade.predict import (
    PredictionRefused,
    SUBSET_TEST,
    SUBSET_VAL,
    SUBSETS,
    build_subset,
    check_data_settings,
    close_test_lock,
    decide_test_access,
    default_output,
    format_command,
    load_model,
    predict_subset,
    read_lock,
    record_test_output,
    sha256_file,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)
from drgrade.run_config import RunConfigError
from drgrade.provenance import (
    EVENT_ATTEMPT,
    EVENT_COMPLETE,
    ProvenanceError,
    append_access_event,
    check_test_access,
    write_meta,
)
from drgrade.utils import select_device, setup_logging

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_UNEXPECTED = 1
STATUS_REFUSED = 2
STATUS_INCOMPLETE = 3


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Predict a subset with one trained run (brief 9.1).")
    parser.add_argument("--run", required=True, type=Path, help="run folder with best.pt and config.yaml")
    parser.add_argument("--subset", choices=SUBSETS, default=SUBSET_VAL, help="val (no lock) or test (locked)")
    parser.add_argument("--tta", action="store_true", help="average softmax over four flip views (DECISIONS.md D24)")
    parser.add_argument("--final", action="store_true", help="required for --subset test")
    parser.add_argument("--force", action="store_true", help="override the test lock; needs --reason")
    parser.add_argument("--reason", default=None, help="why the locked test output is produced again (recorded)")
    parser.add_argument("--close-lock", action="store_true", help="mark TEST_LOCK.json closed and exit")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=None, help="overrides processed_dir from the data config")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda, then mps, then cpu)")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop after the first batch past this; exit 3")
    parser.add_argument("--out", type=Path, default=None, help="output CSV (default <run>/predictions/<subset>[_tta].csv)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one prediction (or close the lock) and return the exit status."""
    setup_logging()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run

    if args.close_lock:
        try:
            lock = close_test_lock(run_dir)
        except PredictionRefused as exc:
            logger.error("%s", exc)
            return STATUS_REFUSED
        logger.info("%s status: %s", run_dir, lock["status"])
        return STATUS_OK

    if args.subset == SUBSET_TEST and not args.final:
        logger.error("test predictions need --final")
        return STATUS_REFUSED
    if args.subset != SUBSET_TEST and args.final:
        logger.error("--final applies to --subset test only")
        return STATUS_REFUSED
    if args.batch_size < 1 or args.num_workers < 0:
        logger.error("--batch-size must be at least 1 and --num-workers non-negative")
        return STATUS_REFUSED

    out_path = args.out if args.out is not None else default_output(run_dir, args.subset, args.tta)
    if args.subset == SUBSET_TEST and not _inside(out_path, run_dir):
        logger.error("test outputs must be inside the run folder so that the lock can name them")
        return STATUS_REFUSED

    try:
        data_cfg = load_data_config(args.data_config)
        if args.processed_dir is not None:
            data_cfg = replace(data_cfg, processed_dir=str(args.processed_dir))
        device = select_device(args.device)
        loaded = load_model(run_dir, device)
        check_data_settings(loaded, data_cfg)
        forced = False
        if args.subset == SUBSET_TEST:
            forced = decide_test_access(
                read_lock(run_dir),
                loaded.checkpoint_sha256,
                args.tta,
                args.final,
                args.force,
                args.reason,
            )
        variant = "tta" if args.tta else "plain"
        if args.subset == SUBSET_TEST:
            # Allowlist and repeat check, then an "attempt" event BEFORE any test image is
            # loaded, so even an interrupted test run leaves a trace (DECISIONS.md D38).
            check_test_access(loaded.run.run_name, variant, forced)
            append_access_event(
                {
                    "event": EVENT_ATTEMPT,
                    "run_name": loaded.run.run_name,
                    "variant": variant,
                    "folder": run_dir.as_posix(),
                    "checkpoint_sha256": loaded.checkpoint_sha256,
                    "command": format_command(raw_args),
                    "forced": forced,
                    "reason": args.reason,
                }
            )
        table, dataset = build_subset(loaded, data_cfg, args.subset)
        complete = predict_subset(
            loaded,
            table,
            dataset,
            out_path,
            device,
            tta=args.tta,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_seconds=args.max_seconds,
        )
    except (PredictionRefused, ProvenanceError) as exc:
        logger.error("%s", exc)
        return STATUS_REFUSED
    except (ConfigError, RunConfigError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return STATUS_REFUSED
    except Exception:
        logger.exception("prediction failed")
        return STATUS_UNEXPECTED

    if not complete:
        return STATUS_INCOMPLETE
    meta = write_meta(
        out_path,
        {
            "subset": args.subset,
            "run_name": loaded.run.run_name,
            "folder": run_dir.as_posix(),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "tta": args.tta,
            "rows": len(table),
            "device": device.type,
            "command": format_command(raw_args),
        },
    )
    if args.subset == SUBSET_TEST:
        append_access_event(
            {
                "event": EVENT_COMPLETE,
                "run_name": loaded.run.run_name,
                "variant": "tta" if args.tta else "plain",
                "folder": run_dir.as_posix(),
                "checkpoint_sha256": loaded.checkpoint_sha256,
                "file": out_path.as_posix(),
                "rows": len(table),
                "csv_sha256": meta["csv_sha256"],
            }
        )
        record_test_output(
            run_dir,
            loaded.run.run_name,
            loaded.checkpoint_sha256,
            args.tta,
            out_path,
            len(table),
            format_command(raw_args),
            forced,
            args.reason,
            csv_sha256=meta["csv_sha256"],
        )
        logger.info("TEST_LOCK.json updated in %s", run_dir)
    return STATUS_OK


def _inside(path: Path, folder: Path) -> bool:
    """Return True when path is inside folder (both resolved)."""
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
