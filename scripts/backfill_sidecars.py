"""Write provenance sidecars for validation predictions made before sidecars existed (DECISIONS.md D38).

Validation only: a test file can never be backfilled, because its provenance must come
from the access log written at prediction time. The sidecar is marked `backfilled: true`
and records the checkpoint hash of `best.pt` as it is now, the run name from the run's
config.yaml, the row count, and the CSV hash. The script refuses a file whose rows do
not match the validation subset of the split, and never overwrites an existing sidecar.

Usage:
    python scripts/backfill_sidecars.py runs/e2_effb0_512_ce/predictions/val.csv \
        runs/e3_effb0_512_wce/predictions/val.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from drgrade.predict import CHECKPOINT_NAME, sha256_file
from drgrade.provenance import read_meta, write_meta
from drgrade.utils import setup_logging

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_REFUSED = 2


def backfill(csv_path: Path, split: pd.DataFrame) -> dict:
    """Write the sidecar for one validation predictions file and return it. Raises ValueError on refusal."""
    if csv_path.parent.name != "predictions":
        raise ValueError(f"{csv_path} is not in a run's predictions/ folder")
    if not csv_path.name.startswith("val"):
        raise ValueError(f"{csv_path}: only validation files can be backfilled")
    if read_meta(csv_path) is not None:
        raise ValueError(f"{csv_path} already has a sidecar; not overwritten")
    run_dir = csv_path.parent.parent
    run_name = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["run"]["run_name"]
    frame = pd.read_csv(csv_path, dtype={"image_id": str})
    val_ids = set(split.loc[split["subset"] == "val", "image_id"].astype(str))
    if set(frame["image_id"].astype(str)) != val_ids or len(frame) != len(val_ids):
        raise ValueError(f"{csv_path} does not hold exactly the validation images of the split")
    return write_meta(
        csv_path,
        {
            "subset": "val",
            "run_name": run_name,
            "folder": run_dir.as_posix(),
            "checkpoint_sha256": sha256_file(run_dir / CHECKPOINT_NAME),
            "tta": "tta" in csv_path.stem,
            "rows": len(frame),
            "device": "cpu",
            "backfilled": True,
            "note": "sidecar written after the file, from best.pt as it is now (DECISIONS.md D38)",
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--split", type=Path, default=Path("splits/split_v1.csv"))
    args = parser.parse_args(argv)
    split = pd.read_csv(args.split, dtype={"image_id": str, "subset": str})
    status = STATUS_OK
    for path in args.files:
        try:
            meta = backfill(path, split)
            logger.info("%s: sidecar written (run %s, %d rows)", path, meta["run_name"], meta["rows"])
        except (OSError, KeyError, ValueError) as exc:
            logger.error("%s", exc)
            status = STATUS_REFUSED
    return status


if __name__ == "__main__":
    sys.exit(main())
