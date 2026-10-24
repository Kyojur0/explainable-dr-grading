"""Stage 1: data audit (brief section 5).

Reads labels for both official splits and opens every image file. No model is
loaded and no model statistic is computed.

Usage:
    python scripts/audit.py --config configs/data.yaml [--workers 4] [--limit N]
        [--reuse-cache] [--resume] [--max-seconds S]

Outputs go to the audit_out folder from the config (summary.json, CSV tables,
PNG figures, vessel_signatures.npz). A --limit run writes to a limit_<N>
subfolder and does not touch the full-run outputs or the per-image cache.

Near-duplicates use the vessel-pattern signature of drgrade.vessel_signature
(DECISIONS.md D03). The per-image pass is resumable; the signature pass is not.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from drgrade.audit_rules import FLAG_NAMES, compute_flags
from drgrade.config import DataConfig, load_data_config, resolve_mmrdr_root
from drgrade.io import (
    GRADE_VALUES,
    LESION_COLS,
    LESION_NAMES,
    LR_VALUES,
    SPLIT_NAMES,
    image_path,
    load_fp_csv,
)
from drgrade.neardup import best_neighbour_similarity, connected_components, cosine_pairs
from drgrade.preprocess import crop_fundus
from drgrade.utils import get_git_info, setup_logging
from drgrade.viz import save_contact_sheet, save_grouped_bar, save_heatmap
from drgrade.vessel_signature import vessel_signature

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 4
MAP_CHUNKSIZE = 64
MD5_CHUNK_BYTES = 1 << 20
TOP_RESOLUTIONS = 10
CONTACT_PAIRS = 12
CONTACT_THUMB_SIDE = 256
CONTACT_GAP_PX = 8
CONTACT_DECODE_SCALE = 4  # JPEG draft reduction for thumbnails (speed only)
BOUNDARY_BAND = 0.05  # boundary sheet shows similarities from threshold to threshold + band
SIMILARITY_BIN_EDGES = (0.95, 0.97, 0.99, 1.0001)  # histogram edges above the threshold
NN_QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)  # nearest-neighbour quantiles, "typical unrelated"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
CACHE_NAME = "images.csv"
PARTIAL_CACHE_NAME = "images_partial.csv"
SIGNATURE_CACHE_NAME = "vessel_signatures.npz"
RESUME_BATCH_SIZE = 400
EXIT_INCOMPLETE = 3

IMAGE_FIELDS: tuple[str, ...] = (
    "file_bytes",
    "md5",
    "readable",
    "error",
    "width",
    "height",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "crop_failed",
)


# ---------------------------------------------------------------------------
# Per-image pass (runs in worker processes; returns only small primitives)
# ---------------------------------------------------------------------------


def _file_md5(path: Path) -> str:
    """Return the MD5 hex digest of a file's bytes, read in chunks.

    MD5 is used only to find byte-identical files, not for security.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MD5_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analyse_image(task: tuple[str, str], cfg: DataConfig) -> dict[str, Any]:
    """Compute the per-image properties for one (image_id, path) task.

    Unreadable files are recorded with readable=False and the error text. They
    are not dropped from the label counts.
    """
    image_id, path_str = task
    path = Path(path_str)
    record: dict[str, Any] = {"image_id": image_id, "readable": False, "error": ""}
    record.update({field: None for field in IMAGE_FIELDS if field not in record})
    try:
        record["file_bytes"] = path.stat().st_size
        record["md5"] = _file_md5(path)
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    try:
        with Image.open(path) as opened:
            opened.load()
            width, height = opened.size
            rgb = opened.convert("RGB")
    except Exception as exc:  # Pillow raises several exception types for corrupt files
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    cropped, box, crop_failed = crop_fundus(
        rgb,
        cfg.crop_threshold,
        cfg.crop_margin_frac,
        cfg.crop_min_area_frac,
        cfg.bg_detect_min,
        cfg.bg_tolerance,
        cfg.bg_corner_spread,
    )
    record.update(
        readable=True,
        width=width,
        height=height,
        crop_left=box[0],
        crop_top=box[1],
        crop_right=box[2],
        crop_bottom=box[3],
        crop_failed=crop_failed,
    )
    return record


def _iter_batches(
    tasks: list[tuple[str, str]], cfg: DataConfig, workers: int, batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    """Yield per-image records batch by batch, in task order.

    Batches let the caller save progress and stop between batches without
    waiting for work that has not started yet.
    """
    worker = functools.partial(_analyse_image, cfg=cfg)
    starts = range(0, len(tasks), batch_size)
    if workers == 1:
        for start in starts:
            yield [worker(task) for task in tasks[start : start + batch_size]]
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in starts:
            batch = tasks[start : start + batch_size]
            yield list(pool.map(worker, batch, chunksize=MAP_CHUNKSIZE))


def _read_cache(path: Path) -> pd.DataFrame:
    """Read a cached per-image table written by an earlier full run."""
    cached = pd.read_csv(path, dtype={"image_id": str, "md5": str, "error": str})
    cached["error"] = cached["error"].fillna("")
    cached["readable"] = cached["readable"].astype(bool)
    return cached


def _read_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise column types so fresh and resumed records merge identically."""
    frame = frame.copy()
    frame["error"] = frame["error"].fillna("").astype(str)
    frame["readable"] = frame["readable"].astype(bool)
    return frame


class AuditIncomplete(RuntimeError):
    """Raised when --max-seconds stops the per-image pass before it finishes."""


def _image_table(
    labels: pd.DataFrame,
    root: Path,
    cfg: DataConfig,
    workers: int,
    cache_path: Path,
    reuse_cache: bool,
    partial_path: Path,
    resume: bool = False,
    max_seconds: float | None = None,
) -> pd.DataFrame:
    """Join labels with per-image properties, reusing the cache when it covers every id."""
    if reuse_cache and cache_path.is_file():
        cached = _read_cache(cache_path)
        if set(labels["image_id"]).issubset(set(cached["image_id"])):
            logger.info("Reusing per-image cache %s", cache_path)
            images = cached[["image_id", *IMAGE_FIELDS]]
            return labels.merge(images, on="image_id", how="left")
        logger.info("Cache %s does not cover every image id; recomputing", cache_path)

    # Progress file: lets a long pass be split across several invocations
    # (the audit VM stops any command after a few minutes). Rows are appended
    # batch by batch; with resume=True, ids already in the file are skipped.
    done_records: list[dict[str, Any]] = []
    if resume and partial_path.is_file():
        done_records = _read_cache(partial_path).to_dict(orient="records")
        logger.info("Resuming: %d images already in %s", len(done_records), partial_path)
    elif partial_path.is_file():
        partial_path.unlink()
    done_ids = {record["image_id"] for record in done_records}

    tasks = [
        (image_id, str(image_path(root, rel)))
        for image_id, rel in zip(labels["image_id"], labels["rel_path"])
        if image_id not in done_ids
    ]
    started = time.monotonic()
    columns = ["image_id", *IMAGE_FIELDS]
    for batch in _iter_batches(tasks, cfg, workers, RESUME_BATCH_SIZE):
        done_records.extend(batch)
        pd.DataFrame.from_records(batch, columns=columns).to_csv(
            partial_path, mode="a", header=not partial_path.is_file(), index=False
        )
        logger.info("Processed %d of %d images", len(done_records), len(labels))
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            remaining = len(labels) - len(done_records)
            if remaining > 0:
                raise AuditIncomplete(
                    f"Stopped after {max_seconds}s with {remaining} images left. "
                    "Rerun with --resume to continue."
                )
    logger.info("Processed %d images", len(done_records))
    images = pd.DataFrame.from_records(done_records, columns=columns)
    images = _read_cache_frame(images)
    partial_path.unlink(missing_ok=True)
    return labels.merge(images, on="image_id", how="left")


# ---------------------------------------------------------------------------
# Helpers for summaries
# ---------------------------------------------------------------------------


def _nested_counts(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Convert a two-way count table to a JSON-safe nested dict with string keys."""
    return {
        str(index): {str(col): int(value) for col, value in row.items()}
        for index, row in table.iterrows()
    }


def _describe(series: pd.Series) -> dict[str, float | None]:
    """Return min, median and max of a numeric series, or None values when empty."""
    values = series.dropna().astype(float)
    if values.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and arrays for json.dumps."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Section 5.1: counts
# ---------------------------------------------------------------------------


def counts_section(table: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Counts per split and grade, laterality, and the training imbalance ratio."""
    grades = list(GRADE_VALUES)
    by_split_grade = pd.crosstab(table["official_split"], table["grade"]).reindex(
        index=list(SPLIT_NAMES), columns=grades, fill_value=0
    )
    by_split_grade.index.name = "split"
    by_split_grade.columns.name = "grade"

    with_totals = by_split_grade.copy()
    with_totals["total"] = by_split_grade.sum(axis=1)
    with_totals.to_csv(out_dir / "counts_split_grade.csv")

    chart = by_split_grade.T
    chart.index = [f"grade {grade}" for grade in grades]
    chart.columns.name = "split"
    save_grouped_bar(
        chart,
        out_dir / "fig_counts_by_grade.png",
        title="Images per grade and split",
        ylabel="images",
    )

    eyes = {0: "left", 1: "right"}
    lateral_index = pd.MultiIndex.from_product(
        [list(SPLIT_NAMES), grades, list(LR_VALUES)], names=["split", "grade", "lr"]
    )
    laterality = table.groupby(["official_split", "grade", "lr"]).size()
    laterality.index.names = ["split", "grade", "lr"]
    laterality = laterality.reindex(lateral_index, fill_value=0).rename("n_images").reset_index()
    laterality["eye"] = laterality["lr"].map(eyes)
    laterality.to_csv(out_dir / "laterality_by_split_grade.csv", index=False)
    by_eye = laterality.groupby(["split", "eye"])["n_images"].sum()
    laterality_summary = {
        split: {eye: int(by_eye.loc[(split, eye)]) for eye in eyes.values()} for split in SPLIT_NAMES
    }

    train_counts = by_split_grade.loc[TRAIN_SPLIT, grades]
    smallest = int(train_counts.min())
    imbalance: float | None = None
    if smallest > 0:
        imbalance = float(train_counts.max()) / smallest
    else:
        logger.warning("Training split has a grade with no images; imbalance ratio is undefined")

    return {
        "per_split_total": {split: int(by_split_grade.loc[split].sum()) for split in SPLIT_NAMES},
        "per_split_grade": _nested_counts(by_split_grade),
        "laterality_by_split": laterality_summary,
        "train_grade_counts": {str(grade): int(train_counts[grade]) for grade in grades},
        "imbalance_ratio_train": imbalance,
    }


# ---------------------------------------------------------------------------
# Section 5.2: image properties
# ---------------------------------------------------------------------------


def image_section(table: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Resolution, file size and readability summaries, with unreadable files listed."""
    readable = table[table["readable"].astype(bool)]
    unreadable = table[~table["readable"].astype(bool)]

    resolutions = (
        readable.groupby(["width", "height"])
        .size()
        .rename("n_images")
        .reset_index()
        .sort_values(["n_images", "width", "height"], ascending=[False, True, True])
    )
    resolutions.to_csv(out_dir / "resolution_counts.csv", index=False)

    unreadable_table = unreadable[["image_id", "official_split", "file_bytes", "error"]]
    unreadable_table.to_csv(out_dir / "unreadable.csv", index=False)
    for row in unreadable_table.itertuples(index=False):
        logger.warning("Unreadable image %s (%s): %s", row.image_id, row.official_split, row.error)

    top = resolutions.head(TOP_RESOLUTIONS)
    return {
        "n_readable": int(len(readable)),
        "n_unreadable": int(len(unreadable)),
        "unreadable_ids": unreadable["image_id"].tolist(),
        "distinct_resolutions": int(len(resolutions)),
        "top_resolutions": [
            {"width": int(row.width), "height": int(row.height), "n_images": int(row.n_images)}
            for row in top.itertuples(index=False)
        ],
        "width": _describe(readable["width"]),
        "height": _describe(readable["height"]),
        "file_bytes": _describe(table["file_bytes"]),
    }


# ---------------------------------------------------------------------------
# Section 5.3: lesion prevalence and consistency flags
# ---------------------------------------------------------------------------


def label_section(table: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Lesion prevalence by split and grade, plus the consistency flags of brief 5.3."""
    rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        for grade in GRADE_VALUES:
            subset = table[(table["official_split"] == split) & (table["grade"] == grade)]
            n_images = len(subset)
            for col in LESION_COLS:
                count = int(subset[col].sum())
                rows.append(
                    {
                        "split": split,
                        "grade": grade,
                        "n_images": n_images,
                        "lesion": col,
                        "lesion_name": LESION_NAMES[col],
                        "count": count,
                        "prevalence": count / n_images if n_images else np.nan,
                    }
                )
    prevalence = pd.DataFrame(rows)
    prevalence.to_csv(out_dir / "lesion_prevalence.csv", index=False)

    train = prevalence[prevalence["split"] == TRAIN_SPLIT]
    matrix = train.pivot(index="grade", columns="lesion_name", values="prevalence")
    matrix = matrix.reindex(columns=[LESION_NAMES[col] for col in LESION_COLS])
    n_by_grade = train.groupby("grade")["n_images"].first()
    save_heatmap(
        matrix.to_numpy(dtype=float),
        [f"grade {grade} (n={n_by_grade.loc[grade]})" for grade in matrix.index],
        list(matrix.columns),
        title="Lesion prevalence by grade (training split)",
        path=out_dir / "fig_lesion_prevalence_train.png",
    )

    flags = compute_flags(table)
    flag_rows: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        mask = table["official_split"] == split
        n_images = int(mask.sum())
        for flag in FLAG_NAMES:
            hits = flags.loc[mask, flag]
            count = int(hits.sum())
            flag_rows.append(
                {
                    "split": split,
                    "flag": flag,
                    "n_images": n_images,
                    "count": count,
                    "share": count / n_images if n_images else np.nan,
                }
            )
            for _, row in table.loc[hits[hits].index].iterrows():
                flagged.append(
                    {
                        "image_id": row["image_id"],
                        "split": row["official_split"],
                        "grade": int(row["grade"]),
                        "flag": flag,
                    }
                )
    pd.DataFrame(flag_rows).to_csv(out_dir / "flag_counts.csv", index=False)
    pd.DataFrame(flagged, columns=["image_id", "split", "grade", "flag"]).to_csv(
        out_dir / "flagged_images.csv", index=False
    )

    flag_counts = pd.DataFrame(flag_rows)
    return {
        "note": (
            "Lesion labels are image-level presence flags with no location. A flag may reflect "
            "incomplete lesion annotation rather than a wrong grade. Flags are informational only."
        ),
        "flag_counts": {
            split: {
                record["flag"]: int(record["count"])
                for record in flag_counts[flag_counts["split"] == split].to_dict("records")
            }
            for split in SPLIT_NAMES
        },
    }


# ---------------------------------------------------------------------------
# Section 5.4: exact and near duplicates
# ---------------------------------------------------------------------------


def _exact_duplicates(table: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Group byte-identical files by MD5 and record split and grade agreement."""
    hashed = table[table["md5"].notna()]
    group_sizes = hashed.groupby("md5")["image_id"].transform("size")
    exact = hashed[group_sizes >= 2].sort_values(["md5", "image_id"])
    exact[["md5", "image_id", "official_split", "grade"]].to_csv(
        out_dir / "exact_duplicates.csv", index=False
    )
    groups = exact.groupby("md5")
    return {
        "n_groups": int(groups.ngroups),
        "n_images": int(len(exact)),
        "n_cross_split_groups": int((groups["official_split"].nunique() > 1).sum()),
        "n_groups_with_mixed_grades": int((groups["grade"].nunique() > 1).sum()),
    }


def _signature_params(cfg: DataConfig) -> dict[str, Any]:
    """Settings that determine the signature vectors. They are part of the cache key."""
    return {
        "vsig_decode_scale": cfg.vsig_decode_scale,
        "vsig_canvas": cfg.vsig_canvas,
        "vsig_blur_radius": cfg.vsig_blur_radius,
        "vsig_ring_frac": cfg.vsig_ring_frac,
        "vsig_grid": cfg.vsig_grid,
        "vsig_threshold": cfg.vsig_threshold,
        "crop_threshold": cfg.crop_threshold,
        "crop_margin_frac": cfg.crop_margin_frac,
        "crop_min_area_frac": cfg.crop_min_area_frac,
        "bg_detect_min": cfg.bg_detect_min,
        "bg_tolerance": cfg.bg_tolerance,
        "bg_corner_spread": cfg.bg_corner_spread,
    }


def _signature_for_task(task: tuple[str, str], cfg: DataConfig) -> np.ndarray:
    """Return the vessel signature of one (image_id, path) task. Runs in worker processes."""
    _, path_str = task
    return vessel_signature(Path(path_str), cfg)


def _renormalise(vectors: np.ndarray) -> np.ndarray:
    """Return float32 rows scaled to unit L2 norm. Zero rows stay zero."""
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return (matrix / safe).astype(np.float32)


def _read_signature_cache(path: Path, ids: list[str], params_json: str) -> np.ndarray | None:
    """Return the cached vectors only if the cache covers exactly these ids with these settings."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        cached_ids = [str(value) for value in data["image_ids"]]
        cached_params = str(data["params"])
        if cached_ids != ids or cached_params != params_json:
            return None
        return _renormalise(data["vectors"].astype(np.float32))


def _vessel_vectors(
    readable: pd.DataFrame,
    root: Path,
    cfg: DataConfig,
    workers: int,
    out_dir: Path,
    reuse_cache: bool,
) -> tuple[np.ndarray, bool]:
    """Return unit-norm vessel signatures for the readable images, in CSV row order.

    Vectors are always used after a float16 round trip, so a fresh run and a
    cached run give identical pairs. The cache is reused only with --reuse-cache
    and only when its image ids and settings match exactly.

    Returns:
        (vectors of shape (n, vsig_grid**2), whether the cache was reused)
    """
    ids = [str(value) for value in readable["image_id"]]
    params_json = json.dumps(_signature_params(cfg), sort_keys=True)
    cache_path = out_dir / SIGNATURE_CACHE_NAME
    if reuse_cache:
        cached = _read_signature_cache(cache_path, ids, params_json)
        if cached is not None:
            logger.info("Reusing vessel signatures %s", cache_path)
            return cached, True
        logger.info("Vessel signature cache %s does not match; recomputing", cache_path)

    tasks = [
        (image_id, str(image_path(root, rel)))
        for image_id, rel in zip(readable["image_id"], readable["rel_path"])
    ]
    started = time.monotonic()
    worker = functools.partial(_signature_for_task, cfg=cfg)
    if workers == 1:
        signatures = [worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            signatures = list(pool.map(worker, tasks, chunksize=MAP_CHUNKSIZE))
    dimension = cfg.vsig_grid * cfg.vsig_grid
    if signatures:
        stored = np.stack(signatures).astype(np.float16)
    else:
        stored = np.zeros((0, dimension), dtype=np.float16)
    np.savez(
        cache_path,
        image_ids=np.array(ids, dtype=str),
        vectors=stored,
        params=np.array(params_json),
    )
    logger.info("Computed %d vessel signatures in %.1f s", len(ids), time.monotonic() - started)
    return _renormalise(stored.astype(np.float32)), False


def _thumbnail(root: Path, record: pd.Series) -> Image.Image:
    """Decode an image at reduced size, crop it with the stored crop box, and shrink it.

    JPEG draft mode makes the decode fast. The stored box is in full-resolution
    pixels, so it is scaled to the reduced decode before cropping.
    """
    path = image_path(root, record["rel_path"])
    box = [int(record[key]) for key in ("crop_left", "crop_top", "crop_right", "crop_bottom")]
    with Image.open(path) as opened:
        full_width, full_height = opened.size
        opened.draft(
            "RGB",
            (max(1, full_width // CONTACT_DECODE_SCALE), max(1, full_height // CONTACT_DECODE_SCALE)),
        )
        decoded = opened.convert("RGB")
    scale_x = decoded.width / full_width
    scale_y = decoded.height / full_height
    scaled_box = (
        round(box[0] * scale_x),
        round(box[1] * scale_y),
        round(box[2] * scale_x),
        round(box[3] * scale_y),
    )
    cropped = decoded.crop(scaled_box)
    cropped.thumbnail((CONTACT_THUMB_SIDE, CONTACT_THUMB_SIDE))
    return cropped


def _pair_panel(row: Any, root: Path, info: pd.DataFrame) -> tuple[Image.Image, str]:
    """Build one contact-sheet panel: two cropped images side by side with a caption."""
    left = _thumbnail(root, info.loc[row.image_id_a])
    right = _thumbnail(root, info.loc[row.image_id_b])
    canvas = Image.new(
        "RGB",
        (left.width + right.width + CONTACT_GAP_PX, max(left.height, right.height)),
        (255, 255, 255),
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + CONTACT_GAP_PX, 0))
    tag = " (byte-identical copy)" if bool(row.exact) else ""
    caption = (
        f"{row.image_id_a} (grade {row.grade_a}) | {row.image_id_b} (grade {row.grade_b})\n"
        f"similarity {row.similarity:.3f}{tag}"
    )
    return canvas, caption


def _near_duplicates(
    readable: pd.DataFrame,
    vectors: np.ndarray,
    root: Path,
    out_dir: Path,
    cfg: DataConfig,
) -> dict[str, Any]:
    """Cosine near-duplicate analysis on vessel signatures: pairs, groups, grades and sheets."""
    threshold = cfg.vsig_threshold
    n = len(readable)
    ids = readable["image_id"].astype(str).to_numpy()
    splits = readable["official_split"].astype(str).to_numpy()
    grades = readable["grade"].astype(np.int64).to_numpy()
    md5 = readable["md5"].to_numpy()

    raw = cosine_pairs(vectors, threshold)
    left = raw[:, 0].astype(np.int64)
    right = raw[:, 1].astype(np.int64)
    pair_table = pd.DataFrame(
        {
            "image_id_a": ids[left],
            "image_id_b": ids[right],
            "similarity": raw[:, 2],
            "exact": np.asarray(md5[left] == md5[right], dtype=bool),
            "split_a": splits[left],
            "split_b": splits[right],
            "grade_a": grades[left],
            "grade_b": grades[right],
        }
    )
    pair_table = pair_table.sort_values(
        ["similarity", "image_id_a", "image_id_b"], ascending=[False, True, True]
    ).reset_index(drop=True)
    pair_table.to_csv(out_dir / "near_duplicate_pairs.csv", index=False)

    pair_table["pair_type"] = np.where(
        pair_table["split_a"] == pair_table["split_b"],
        "within_" + pair_table["split_a"],
        "cross_split",
    )
    pair_table["grade_match"] = pair_table["grade_a"] == pair_table["grade_b"]
    pair_types = [f"within_{split}" for split in SPLIT_NAMES] + ["cross_split"]
    pairs_by_type: dict[str, dict[str, int]] = {}
    for ptype in pair_types:
        subset = pair_table[pair_table["pair_type"] == ptype]
        pairs_by_type[ptype] = {
            "pairs": int(len(subset)),
            "grade_match": int(subset["grade_match"].sum()),
            "exact": int(subset["exact"].sum()),
        }

    similarities = pair_table["similarity"].to_numpy(dtype=np.float64)
    edges = [threshold] + [edge for edge in SIMILARITY_BIN_EDGES if edge > threshold]
    counts, _ = np.histogram(similarities, bins=edges)
    histogram = {
        f"{low:g} to {high:g}": int(count)
        for low, high, count in zip(edges[:-1], edges[1:], counts)
    }
    grade_range = range(max(GRADE_VALUES) - min(GRADE_VALUES) + 1)
    grade_gap = (pair_table["grade_a"] - pair_table["grade_b"]).abs()
    grade_difference = {str(gap): int((grade_gap == gap).sum()) for gap in grade_range}

    # Groups: connected components of the pair graph, with singletons left out.
    labels = connected_components(n, raw[:, :2].astype(np.int64))
    sizes = np.bincount(labels, minlength=n) if n else np.zeros(0, dtype=np.int64)
    members = np.flatnonzero(sizes[labels] >= 2) if n else np.zeros(0, dtype=np.int64)
    codes, _ = pd.factorize(labels[members])
    clusters = pd.DataFrame(
        {
            "image_id": ids[members],
            "cluster_id": codes,
            "cluster_size": sizes[labels[members]],
            "official_split": splits[members],
            "grade": grades[members],
        }
    )
    clusters.to_csv(out_dir / "dup_clusters.csv", index=False)
    by_group = clusters.groupby("cluster_id")
    group_size_counts = clusters.drop_duplicates("cluster_id")["cluster_size"].value_counts()
    spanning = int((by_group["official_split"].nunique() > 1).sum())
    train_groups = set(clusters.loc[clusters["official_split"] == TRAIN_SPLIT, "cluster_id"])
    test_in_train_groups = int(
        (
            (clusters["official_split"] == TEST_SPLIT)
            & clusters["cluster_id"].isin(train_groups)
        ).sum()
    )

    grade_spread = by_group["grade"].nunique()
    disagreeing = grade_spread[grade_spread > 1].index
    disagree_rows = clusters[clusters["cluster_id"].isin(disagreeing)]
    disagree_rows.to_csv(out_dir / "grade_disagreement_groups.csv", index=False)
    disagreement_groups = [
        {
            "cluster_id": int(cluster_id),
            "image_ids": group["image_id"].tolist(),
            "splits": group["official_split"].tolist(),
            "grades": [int(grade) for grade in group["grade"]],
        }
        for cluster_id, group in disagree_rows.groupby("cluster_id")
    ]

    if n >= 2:
        best = best_neighbour_similarity(vectors)
        neighbour_quantiles: dict[str, float | None] = {
            str(q): float(np.quantile(best, q)) for q in NN_QUANTILES
        }
    else:
        neighbour_quantiles = {str(q): None for q in NN_QUANTILES}

    # Contact sheets. Thumbnails are cropped with the stored crop boxes.
    info = readable.set_index("image_id")
    non_exact = pair_table[~pair_table["exact"]]
    cross_rows = non_exact[non_exact["split_a"] != non_exact["split_b"]].head(CONTACT_PAIRS)
    boundary_rows = (
        non_exact[
            (non_exact["similarity"] >= threshold)
            & (non_exact["similarity"] < threshold + BOUNDARY_BAND)
        ]
        .sort_values("similarity", ascending=True)
        .head(CONTACT_PAIRS)
    )
    disagree_pairs = pair_table[pair_table["grade_a"] != pair_table["grade_b"]].head(CONTACT_PAIRS)
    panel_sets = {
        "cross_split": (
            [_pair_panel(row, root, info) for row in cross_rows.itertuples(index=False)],
            "contact_cross_split_pairs.png",
            f"Cross-split pairs that are not byte-identical, most similar first (up to {CONTACT_PAIRS})",
        ),
        "threshold_boundary": (
            [_pair_panel(row, root, info) for row in boundary_rows.itertuples(index=False)],
            "contact_threshold_boundary_pairs.png",
            f"Non-identical pairs with similarity from {threshold:.2f} to "
            f"{threshold + BOUNDARY_BAND:.2f}, least similar first",
        ),
        "grade_disagreement": (
            [_pair_panel(row, root, info) for row in disagree_pairs.itertuples(index=False)],
            "contact_grade_disagreement_pairs.png",
            f"Pairs whose grades disagree, most similar first (up to {CONTACT_PAIRS})",
        ),
    }
    for panels, filename, title in panel_sets.values():
        if not save_contact_sheet(panels, out_dir / filename, title=title):
            logger.info("No pairs to draw; %s not written", filename)

    return {
        "threshold": threshold,
        "n_images_compared": int(n),
        "n_pairs": int(len(pair_table)),
        "n_exact_pairs": int(pair_table["exact"].sum()),
        "pairs_by_type": pairs_by_type,
        "similarity_histogram": histogram,
        "pair_grade_difference_counts": grade_difference,
        "n_groups": int(by_group.ngroups),
        "group_size_counts": {str(size): int(count) for size, count in group_size_counts.items()},
        "n_groups_spanning_splits": spanning,
        "test_images_in_groups_with_train": test_in_train_groups,
        "n_groups_with_grade_disagreement": int(len(disagreeing)),
        "grade_disagreement_groups": disagreement_groups,
        "nearest_neighbour_similarity": neighbour_quantiles,
        "contact_sheet_panels": {name: len(panels) for name, (panels, _, _) in panel_sets.items()},
    }


def duplicate_section(
    table: pd.DataFrame,
    readable: pd.DataFrame,
    vectors: np.ndarray,
    root: Path,
    out_dir: Path,
    cfg: DataConfig,
) -> dict[str, Any]:
    """Exact (MD5) and near (vessel-pattern cosine) duplicate analysis, as brief section 5.4."""
    return {
        "exact": _exact_duplicates(table, out_dir),
        "near": _near_duplicates(readable, vectors, root, out_dir, cfg),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_audit(
    cfg: DataConfig,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    reuse_cache: bool = False,
    resume: bool = False,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the full audit and write summary.json, the tables and the figures.

    Args:
        cfg: validated data configuration.
        workers: processes for the per-image pass (1 runs in-process).
        limit: if set, audit only the first `limit` CSV rows, into a limit_<N> subfolder.
        reuse_cache: reuse <audit_out>/images.csv when it covers every audited id.
        resume: continue the per-image pass from <out_dir>/images_partial.csv.
        max_seconds: stop the per-image pass after this many seconds (raises
            AuditIncomplete); progress is kept for a later resume.

    Returns:
        The summary dictionary that was written to summary.json.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    root = resolve_mmrdr_root(cfg.mmrdr_root)
    labels_all = load_fp_csv(root)
    labels = labels_all if limit is None else labels_all.head(limit).reset_index(drop=True)

    base_out = Path(cfg.audit_out)
    out_dir = base_out if limit is None else base_out / f"limit_{limit}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        logger.warning("Partial run: first %d of %d rows. Outputs go to %s", len(labels), len(labels_all), out_dir)

    cache_path = base_out / CACHE_NAME
    table = _image_table(
        labels,
        root,
        cfg,
        workers,
        cache_path,
        reuse_cache,
        partial_path=out_dir / PARTIAL_CACHE_NAME,
        resume=resume,
        max_seconds=max_seconds,
    )
    if limit is None:
        table.to_csv(cache_path, index=False)
    table.to_csv(out_dir / "images.csv", index=False)

    counts = counts_section(table, out_dir)
    image_props = image_section(table, out_dir)
    label_summary = label_section(table, out_dir)
    readable = table[table["readable"].astype(bool)].reset_index(drop=True)
    vectors, signatures_reused = _vessel_vectors(readable, root, cfg, workers, out_dir, reuse_cache)
    duplicates = duplicate_section(table, readable, vectors, root, out_dir, cfg)

    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": get_git_info(),
        "config": {key: value for key, value in dataclasses.asdict(cfg).items() if key != "mmrdr_root"},
        "run": {
            "workers": workers,
            "limit": limit,
            "partial_run": limit is not None,
            "reuse_cache": reuse_cache,
            "resumed": resume,
            "vessel_signature_cache_reused": signatures_reused,
            "n_csv_rows": int(len(labels_all)),
            "n_rows_audited": int(len(labels)),
        },
        "counts": counts,
        "image_properties": image_props,
        "labels": label_summary,
        "duplicates": duplicates,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Audit complete. Summary written to %s", out_dir / "summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the audit."""
    parser = argparse.ArgumentParser(description="Stage 1 data audit for MMRDR-CFP.")
    parser.add_argument("--config", type=Path, required=True, help="path to configs/data.yaml")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="worker processes")
    parser.add_argument("--limit", type=int, default=None, help="audit only the first N rows (smoke tests)")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="reuse outputs/audit/images.csv if it covers every image id",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted per-image pass from images_partial.csv",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop the per-image pass after this many seconds and exit with status 3",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    cfg = load_data_config(args.config)
    try:
        run_audit(
            cfg,
            workers=args.workers,
            limit=args.limit,
            reuse_cache=args.reuse_cache,
            resume=args.resume,
            max_seconds=args.max_seconds,
        )
    except AuditIncomplete as exc:
        logger.warning("%s", exc)
        return EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    sys.exit(main())
