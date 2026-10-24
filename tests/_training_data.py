"""Tiny synthetic processed dataset for the stage 3 and 4 tests (no real images, no weights).

Writes JPEGs, a manifest with the real column names and a split file with the same
layout as splits/split_v1.csv, all under a temporary folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from drgrade.io import LESION_COLS

MANIFEST_COLUMNS = (
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


@dataclass(frozen=True)
class TinyDataset:
    """Paths of a tiny processed dataset."""

    processed_dir: Path
    image_dir: Path
    manifest: Path
    split: Path


def write_tiny_dataset(
    root: Path,
    target_size: int = 64,
    n_train_per_grade: int = 3,
    n_val_per_grade: int = 1,
    n_test: int = 4,
    seed: int = 0,
) -> TinyDataset:
    """Create processed JPEGs, a manifest and a split under root. Every grade appears in train and val."""
    processed_dir = root / "processed"
    image_dir = processed_dir / f"s{target_size}"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, str]] = []

    def add(image_id: str, subset: str, grade: int, official: str) -> None:
        pixels = rng.integers(0, 256, size=(target_size, target_size, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / f"{image_id}.jpg", quality=95)
        row: dict[str, object] = {column: 0 for column in MANIFEST_COLUMNS}
        row.update(
            image_id=image_id,
            official_split=official,
            grade=grade,
            lr=0,
            orig_width=target_size,
            orig_height=target_size,
            crop_left=0,
            crop_top=0,
            crop_right=target_size,
            crop_bottom=target_size,
            crop_failed=False,
            border_level=0,
            brightness=100.0,
            contrast=20.0,
            sharpness=5.0,
            file=f"{image_id}.jpg",
        )
        rows.append(row)
        split_rows.append({"image_id": image_id, "subset": subset})

    index = 0
    for grade in range(5):
        for _ in range(n_train_per_grade):
            index += 1
            add(f"tr{index:06d}", "train", grade, "train")
    for grade in range(5):
        for _ in range(n_val_per_grade):
            index += 1
            add(f"tr{index:06d}", "val", grade, "train")
    for n in range(n_test):
        add(f"ts{n + 1:06d}", "test", n % 5, "test")

    manifest = image_dir / "manifest.csv"
    pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(manifest, index=False)
    split = root / "split.csv"
    pd.DataFrame(split_rows, columns=["image_id", "subset"]).to_csv(split, index=False)
    return TinyDataset(processed_dir=processed_dir, image_dir=image_dir, manifest=manifest, split=split)
