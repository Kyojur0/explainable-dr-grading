"""Stage 2: preprocessing of the fundus photographs (brief 6.1).

Usage:
    python scripts/preprocess.py --config configs/data.yaml [--workers 4]
        [--limit N] [--overwrite] [--max-seconds S] [--zip]

Outputs go to <processed_dir>/s<target_size>/: one JPEG per image, manifest.csv,
preprocess_info.json and, when --zip is given and the run is complete,
<processed_dir>/s<target_size>.zip. A --limit run writes to s<target_size>_limit<N>/
and never writes the zip, the full manifest or the figure.

Resuming: progress is appended to manifest_partial.csv after every batch. Rerunning
the same command skips images that have a row there and an existing JPEG. With
--max-seconds S the run stops after the first batch that ends at or past S seconds
and exits with status 3; run the same command again until it exits 0. Use
--overwrite to rebuild everything. If the settings in the config differ from those
recorded in preprocess_info.json, the run stops (status 2) and asks for --overwrite.

Quality measures (brightness, contrast, sharpness) are computed on the resized
image before Ben Graham normalisation (DECISIONS.md D10).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import time
import zipfile
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from drgrade.config import ConfigError, DataConfig, load_data_config, resolve_mmrdr_root
from drgrade.io import LESION_COLS, image_path, load_fp_csv
from drgrade.preprocess import border_level, ben_graham, fundus_mask, quality_measures, resize_fundus
from drgrade.utils import get_git_info, setup_logging
from drgrade.viz import save_contact_sheet

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 4
MAP_CHUNKSIZE = 32
RESUME_BATCH_SIZE = 400  # images per batch; progress is saved after each batch
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3
JPEG_SUFFIX = ".jpg"
TMP_SUFFIX = ".tmp"
PARTIAL_NAME = "manifest_partial.csv"
MANIFEST_NAME = "manifest.csv"
INFO_NAME = "preprocess_info.json"
FIGURE_NAME = "contact_processed.png"
FIGURE_SUBDIR = "preprocess"
FIGURE_COLUMNS = 6
FIGURE_QUOTAS = {"random": 8, "aspect": 4, "dark": 4, "blurred": 4, "sharp": 4}

# Settings that change the saved pixels or the measures. Stored in preprocess_info.json
# and compared on resume.
SETTING_KEYS: tuple[str, ...] = (
    "target_size",
    "circle_mask",
    "jpeg_quality",
    "crop_threshold",
    "crop_margin_frac",
    "crop_min_area_frac",
    "bg_detect_min",
    "bg_tolerance",
    "bg_corner_spread",
    "ben_graham",
    "ben_graham_sigma",
)

# Fields returned by a worker for each image. Labels are joined in the main process.
WORKER_FIELDS: tuple[str, ...] = (
    "image_id",
    "orig_width",
    "orig_height",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "crop_failed",
    "border_level",
    "brightness",
    "contrast",
    "sharpness",
    "file",
)
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "official_split",
    "grade",
    "lr",
    *LESION_COLS,
    "orig_width",
    "orig_height",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "crop_failed",
    "border_level",
    "brightness",
    "contrast",
    "sharpness",
    "file",
)


class PreprocessIncomplete(RuntimeError):
    """Raised when --max-seconds stops the run before every image is done."""


# ---------------------------------------------------------------------------
# Per-image work (runs in worker processes; returns only small primitives)
# ---------------------------------------------------------------------------


def _process_image(task: tuple[str, str], cfg: DataConfig, out_dir: str) -> dict[str, Any]:
    """Preprocess one original file, save its JPEG and return the manifest fields.

    The JPEG is written to a temporary name and moved into place with os.replace,
    so an interrupted run never leaves a truncated file under the final name.
    """
    image_id, path_str = task
    try:
        with Image.open(path_str) as opened:
            opened.load()
            orig_width, orig_height = opened.size
            original_border, _ = border_level(opened)
            resized, box, crop_failed = resize_fundus(opened, cfg)
    except Exception as exc:  # Pillow raises several exception types for corrupt files
        raise RuntimeError(f"{image_id}: cannot preprocess {path_str}: {exc}") from exc

    measures = quality_measures(resized, fundus_mask(resized, cfg.crop_threshold))
    final_image = resized
    if cfg.ben_graham:
        final_image = ben_graham(resized, cfg.ben_graham_sigma, cfg.crop_threshold)

    folder = Path(out_dir)
    final_path = folder / f"{image_id}{JPEG_SUFFIX}"
    tmp_path = folder / f"{image_id}{JPEG_SUFFIX}{TMP_SUFFIX}"
    final_image.save(tmp_path, format="JPEG", quality=cfg.jpeg_quality, subsampling=0)
    os.replace(tmp_path, final_path)

    return {
        "image_id": image_id,
        "orig_width": orig_width,
        "orig_height": orig_height,
        "crop_left": box[0],
        "crop_top": box[1],
        "crop_right": box[2],
        "crop_bottom": box[3],
        "crop_failed": crop_failed,
        "border_level": original_border,
        "brightness": measures["brightness"],
        "contrast": measures["contrast"],
        "sharpness": measures["sharpness"],
        "file": final_path.name,
    }


def _iter_batches(
    tasks: list[tuple[str, str]],
    cfg: DataConfig,
    out_dir: Path,
    workers: int,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Yield worker records batch by batch, in task order.

    Batches let the caller save progress and stop between batches. Workers run in a
    ProcessPoolExecutor when workers > 1, otherwise in this process.
    """
    worker = functools.partial(_process_image, cfg=cfg, out_dir=str(out_dir))
    starts = range(0, len(tasks), batch_size)
    if workers == 1:
        for start in starts:
            yield [worker(task) for task in tasks[start : start + batch_size]]
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in starts:
            yield list(pool.map(worker, tasks[start : start + batch_size], chunksize=MAP_CHUNKSIZE))


# ---------------------------------------------------------------------------
# Progress files and settings
# ---------------------------------------------------------------------------


def _settings(cfg: DataConfig) -> dict[str, Any]:
    """Return the config values that determine the saved outputs."""
    return {key: getattr(cfg, key) for key in SETTING_KEYS}


def _check_settings(info_path: Path, settings: dict[str, Any]) -> None:
    """Refuse to resume into a folder built with different settings."""
    if not info_path.is_file():
        return
    stored = json.loads(info_path.read_text(encoding="utf-8")).get("config")
    if stored != settings:
        raise ConfigError(
            f"{info_path} was written with different settings ({stored}); "
            "rerun with --overwrite to rebuild with the current configuration"
        )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV to a temporary name, then move it into place."""
    tmp_path = path.with_name(path.name + TMP_SUFFIX)
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _append_partial(partial: Path, batch: list[dict[str, Any]]) -> None:
    """Append one batch of worker records to the partial manifest (header already present)."""
    pd.DataFrame.from_records(batch, columns=list(WORKER_FIELDS)).to_csv(
        partial, mode="a", header=False, index=False
    )


def _read_partial(partial: Path, out_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """Return the finished images: rows in the partial manifest whose JPEG exists.

    When there is no partial manifest but a finished manifest.csv exists (an earlier
    run completed, with the same settings, checked by the caller), its rows are used
    instead, so rerunning the command, for example with --zip, does no image work.
    """
    source = partial
    if not partial.is_file():
        source = out_dir / MANIFEST_NAME
        if not source.is_file():
            return {}
    # round_trip parsing makes resumed floats identical to freshly computed ones.
    frame = pd.read_csv(
        source, dtype={"image_id": str, "file": str}, float_precision="round_trip"
    )[list(WORKER_FIELDS)]
    done: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        image_id = str(record["image_id"])
        if image_id in wanted and (out_dir / f"{image_id}{JPEG_SUFFIX}").is_file():
            done[image_id] = record
    return done


def _write_info(
    info_path: Path,
    settings: dict[str, Any],
    status: str,
    n_images: int,
    n_done: int,
    n_crop_failed: int | None,
    run_seconds: float,
) -> None:
    """Write preprocess_info.json: settings used, counts, git state and UTC time."""
    payload = {
        "status": status,
        "config": settings,
        "n_images": n_images,
        "n_done": n_done,
        "n_crop_failed": n_crop_failed,
        "git": get_git_info(),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_seconds_this_invocation": round(run_seconds, 1),
    }
    tmp_path = info_path.with_name(info_path.name + TMP_SUFFIX)
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, info_path)


def _build_manifest(labels: pd.DataFrame, done: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Join labels with the worker records, sorted by image_id, in MANIFEST_COLUMNS order."""
    label_cols = ["image_id", "official_split", "grade", "lr", *LESION_COLS]
    label_frame = labels[label_cols].copy()
    for col in LESION_COLS:
        label_frame[col] = label_frame[col].astype("int64")
    records = pd.DataFrame.from_records(list(done.values()), columns=list(WORKER_FIELDS))
    merged = label_frame.merge(records, on="image_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        raise RuntimeError(f"manifest incomplete: {len(merged)} of {len(labels)} images")
    return merged.sort_values("image_id").reset_index(drop=True)[list(MANIFEST_COLUMNS)]


# ---------------------------------------------------------------------------
# Finishing steps: figure and zip
# ---------------------------------------------------------------------------


def _write_figure(manifest: pd.DataFrame, out_dir: Path, figure_path: Path, seed: int) -> None:
    """Draw 24 processed images chosen for a visual check (seeded random plus extremes).

    Order of selection: 8 random, 4 most extreme original aspect ratio, 4 lowest
    brightness, 4 lowest sharpness, 4 highest sharpness. An image already chosen is
    skipped, so the sheet has no repeats.
    """
    rng = np.random.default_rng(seed)
    frame = manifest.copy()
    frame["_aspect"] = frame["orig_width"] / frame["orig_height"]
    frame["_aspect_dev"] = np.abs(np.log(frame["_aspect"]))

    random_order = frame.iloc[rng.permutation(len(frame))]
    aspect_order = frame.sort_values(["_aspect_dev", "image_id"], ascending=[False, True])
    dark_order = frame.sort_values(["brightness", "image_id"], ascending=[True, True])
    blurred_order = frame.sort_values(["sharpness", "image_id"], ascending=[True, True])
    sharp_order = frame.sort_values(["sharpness", "image_id"], ascending=[False, True])

    plan: list[tuple[pd.DataFrame, int, Callable[[pd.Series], str]]] = [
        (random_order, FIGURE_QUOTAS["random"], lambda row: "random sample"),
        (aspect_order, FIGURE_QUOTAS["aspect"], lambda row: f"original aspect {row['_aspect']:.3f}"),
        (dark_order, FIGURE_QUOTAS["dark"], lambda row: f"brightness {row['brightness']:.1f}"),
        (blurred_order, FIGURE_QUOTAS["blurred"], lambda row: f"sharpness {row['sharpness']:.1f} (low)"),
        (sharp_order, FIGURE_QUOTAS["sharp"], lambda row: f"sharpness {row['sharpness']:.1f} (high)"),
    ]

    chosen: set[str] = set()
    panels: list[tuple[Image.Image, str]] = []
    for order, quota, reason in plan:
        taken = 0
        for _, row in order.iterrows():
            if taken >= quota:
                break
            image_id = str(row["image_id"])
            if image_id in chosen:
                continue
            chosen.add(image_id)
            taken += 1
            with Image.open(out_dir / str(row["file"])) as opened:
                panel = opened.convert("RGB")
            caption = f"{image_id}, grade {int(row['grade'])}\n{reason(row)}"
            panels.append((panel, caption))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Processed images, {out_dir.name} (random seed {seed})"
    if save_contact_sheet(panels, figure_path, title, ncols=FIGURE_COLUMNS):
        logger.info("Wrote %s (%d panels)", figure_path, len(panels))


def _write_zip(out_dir: Path, zip_path: Path) -> None:
    """Store the folder (JPEGs, manifest, info) in a zip without compression."""
    started = time.monotonic()
    members = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file()
        and (path.suffix == JPEG_SUFFIX or path.name in (MANIFEST_NAME, INFO_NAME))
    )
    tmp_path = zip_path.with_name(zip_path.name + TMP_SUFFIX)
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members:
            archive.write(path, arcname=f"{out_dir.name}/{path.name}")
    os.replace(tmp_path, zip_path)
    logger.info(
        "Wrote %s with %d files in %.1f s", zip_path, len(members), time.monotonic() - started
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run_preprocess(
    cfg: DataConfig,
    *,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    overwrite: bool = False,
    max_seconds: float | None = None,
    zip_output: bool = False,
) -> Path:
    """Preprocess every image (or the first `limit` rows) and write the outputs.

    Returns the output folder. Raises PreprocessIncomplete when `max_seconds` stops
    the run with work left, and ConfigError when the settings differ from the
    folder's recorded settings and `overwrite` is not set.
    """
    started = time.monotonic()
    root = resolve_mmrdr_root(cfg.mmrdr_root)
    labels = load_fp_csv(root)
    processed_root = Path(cfg.processed_dir)
    if limit is not None:
        labels = labels.head(limit).reset_index(drop=True)
        out_dir = processed_root / f"s{cfg.target_size}_limit{limit}"
    else:
        out_dir = processed_root / f"s{cfg.target_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    partial = out_dir / PARTIAL_NAME
    info_path = out_dir / INFO_NAME
    settings = _settings(cfg)
    if overwrite:
        partial.unlink(missing_ok=True)
    else:
        _check_settings(info_path, settings)
    for stale in out_dir.glob(f"*{TMP_SUFFIX}"):
        stale.unlink()

    total = len(labels)
    done = {} if overwrite else _read_partial(partial, out_dir, set(labels["image_id"]))
    _write_csv_atomic(pd.DataFrame.from_records(list(done.values()), columns=list(WORKER_FIELDS)), partial)
    _write_info(info_path, settings, "in progress", total, len(done), None, time.monotonic() - started)
    logger.info("%d of %d images already done in %s", len(done), total, out_dir)

    tasks = [
        (image_id, str(image_path(root, rel)))
        for image_id, rel in zip(labels["image_id"], labels["rel_path"])
        if image_id not in done
    ]
    for batch in _iter_batches(tasks, cfg, out_dir, workers, RESUME_BATCH_SIZE):
        for record in batch:
            done[str(record["image_id"])] = record
        _append_partial(partial, batch)
        logger.info("Processed %d of %d images", len(done), total)
        elapsed = time.monotonic() - started
        if max_seconds is not None and len(done) < total and elapsed >= max_seconds:
            _write_info(info_path, settings, "in progress", total, len(done), None, elapsed)
            raise PreprocessIncomplete(
                f"stopped after {max_seconds} s with {total - len(done)} images left; "
                "run the same command again to continue"
            )

    manifest = _build_manifest(labels, done)
    _write_csv_atomic(manifest, out_dir / MANIFEST_NAME)
    partial.unlink(missing_ok=True)
    n_crop_failed = int(manifest["crop_failed"].sum())
    _write_info(
        info_path, settings, "complete", total, total, n_crop_failed, time.monotonic() - started
    )
    logger.info("Manifest written: %d images, %d with crop_failed", total, n_crop_failed)

    if limit is None:
        figure_path = Path(cfg.audit_out).parent / FIGURE_SUBDIR / FIGURE_NAME
        _write_figure(manifest, out_dir, figure_path, cfg.seed)
        if zip_output:
            _write_zip(out_dir, processed_root / f"{out_dir.name}.zip")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the preprocessing stage."""
    parser = argparse.ArgumentParser(description="Stage 2 preprocessing for MMRDR-CFP.")
    parser.add_argument("--config", type=Path, required=True, help="path to configs/data.yaml")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="worker processes")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows (smoke tests)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild every output from scratch")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop after the first batch that ends at or past this many seconds; exit status 3",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="after a complete run, write <processed_dir>/s<target_size>.zip (not with --limit)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit status (0, 2 or 3)."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.limit is not None and args.zip:
        parser.error("--zip cannot be combined with --limit")
    try:
        cfg = load_data_config(args.config)
        out_dir = run_preprocess(
            cfg,
            workers=args.workers,
            limit=args.limit,
            overwrite=args.overwrite,
            max_seconds=args.max_seconds,
            zip_output=args.zip,
        )
    except PreprocessIncomplete as exc:
        logger.warning("%s", exc)
        return EXIT_INCOMPLETE
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    logger.info("Preprocessing complete: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
