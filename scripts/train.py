"""Stage 4: train one run (brief 8, 13.1).

Usage:
    python scripts/train.py --config configs/e3_effb0_512_wce.yaml \
        --data-config configs/data.yaml [--out DIR] [--processed-dir DIR] \
        [--device cuda|mps|cpu] [--resume] [--no-pretrained] [--smoke] \
        [--max-epochs N] [--num-workers N]

Outputs go to runs/<run_name>/ (or --out). With --smoke the default is
runs/smoke_<run_name>/ and the run uses 64 train and 32 validation images for two
epochs, including a checkpoint and a resume (brief 14.3).

Exit status: 0 on success, 1 when training or the smoke check fails, 2 when the
configuration or the processed data is missing or invalid.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from drgrade.config import ConfigError, load_data_config
from drgrade.run_config import RunConfigError, load_run_config
from drgrade.train_loop import SMOKE_OUTPUT_PREFIX, train
from drgrade.utils import select_device, setup_logging

logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
STATUS_OK = 0
STATUS_FAILED = 1
STATUS_BAD_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Train one DR grading run (brief section 8).")
    parser.add_argument("--config", required=True, type=Path, help="run config, for example configs/e3_effb0_512_wce.yaml")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"), help="data config (default configs/data.yaml)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default runs/<run_name>, or runs/smoke_<run_name>)")
    parser.add_argument("--processed-dir", type=Path, default=None, help="overrides processed_dir from the data config")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda, then mps, then cpu)")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the output folder if it exists")
    parser.add_argument("--no-pretrained", action="store_true", help="build the architecture with random weights (no download)")
    parser.add_argument("--smoke", action="store_true", help="smoke test: 64 train and 32 val images, 2 epochs, checkpoint and resume")
    parser.add_argument("--max-epochs", type=int, default=None, help="override max_epochs from the run config")
    parser.add_argument("--num-workers", type=int, default=None, help="override num_workers from the run config")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override batch_size (for example 2 for the smoke test on a 3 GB machine); recorded in config.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run training and return the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        run = load_run_config(args.config)
        data = load_data_config(args.data_config)
    except (RunConfigError, ConfigError, OSError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT

    if args.processed_dir is not None:
        data = replace(data, processed_dir=str(args.processed_dir))
    if args.max_epochs is not None:
        run = run.with_overrides(max_epochs=args.max_epochs)
    if args.num_workers is not None:
        run = run.with_overrides(num_workers=args.num_workers)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise SystemExit("--batch-size must be at least 1")
        run = run.with_overrides(batch_size=args.batch_size)

    if args.out is not None:
        out_dir = args.out
    elif args.smoke:
        out_dir = RUNS_DIR / f"{SMOKE_OUTPUT_PREFIX}{run.run_name}"
    else:
        out_dir = RUNS_DIR / run.run_name

    try:
        device = select_device(args.device)
        logger.info("device: %s; output: %s", device, out_dir)
        result = train(
            run,
            data,
            out_dir,
            device,
            pretrained=not args.no_pretrained,
            smoke=args.smoke,
            resume=args.resume,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    except (FileExistsError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return STATUS_FAILED
    except Exception:
        logger.exception("training failed")
        return STATUS_FAILED

    logger.info(
        "finished=%s epochs_run=%d best_epoch=%s best_val=%s",
        result["finished"],
        result["epochs_run"],
        result["best_epoch"],
        result["best_val"],
    )
    return STATUS_OK


if __name__ == "__main__":
    sys.exit(main())
