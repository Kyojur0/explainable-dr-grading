"""Stage 3: run table, dataset, transforms and data loaders (brief section 7).

The run table is the manifest (labels and quality measures, one row per image)
joined with the committed split file (splits/split_v1.csv). Only the subsets a run
asks for are kept. Training code never requests the test subset; that is enforced
here, not by convention (brief 8.4).

Transforms use torchvision.transforms.v2, which ships with torchvision, so no extra
dependency is needed (DECISIONS.md D14).
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

from drgrade.run_config import AugConfig

logger = logging.getLogger(__name__)

SUBSETS: tuple[str, ...] = ("train", "val", "test")
TEST_SUBSET = "test"
IMAGE_SUFFIX = ".jpg"
GRADE_MIN = 0
GRADE_MAX = 4
# ImageNet statistics for the pretrained backbones (brief 7).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
# DataLoader worker seeds are reduced modulo 2**32 to fit NumPy's seed range.
NUMPY_SEED_MODULUS = 2**32


def load_run_table(
    manifest_path: Path | str,
    split_path: Path | str,
    subsets: Iterable[str],
    image_dir: Path | str | None = None,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Return the manifest rows of the requested subsets, with a 'subset' column.

    Checks made here, each of which raises ValueError with the offending ids:
    - the subset names are known, and the test subset is refused unless allow_test is True
      (only the final evaluation passes it; training never does);
    - no image id appears twice in the manifest or in the split file;
    - every requested id is present in the manifest;
    - every requested id has a JPEG at <image_dir>/<image_id>.jpg (image_dir defaults to
      the folder that holds the manifest);
    - grades are integers from 0 to 4.

    The result is sorted by image_id so that smoke-test subsets are reproducible.
    """
    requested = tuple(subsets)
    if not requested:
        raise ValueError("at least one subset must be requested")
    if len(set(requested)) != len(requested):
        raise ValueError(f"subsets requested twice: {requested}")
    unknown = sorted(set(requested) - set(SUBSETS))
    if unknown:
        raise ValueError(f"unknown subsets {unknown}; expected a subset of {SUBSETS}")
    if TEST_SUBSET in requested and not allow_test:
        raise ValueError(
            "the test subset is not available to training; it is used once, by evaluation only (brief 9.1)"
        )

    manifest_path = Path(manifest_path)
    image_root = Path(image_dir) if image_dir is not None else manifest_path.parent
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
    for column in ("image_id", "grade"):
        if column not in manifest.columns:
            raise ValueError(f"manifest {manifest_path} lacks the column {column!r}")
    duplicated = manifest.loc[manifest["image_id"].duplicated(), "image_id"].tolist()
    if duplicated:
        raise ValueError(f"duplicate image ids in manifest: {duplicated[:10]}")

    split = pd.read_csv(split_path, dtype={"image_id": str, "subset": str})
    if list(split.columns[:2]) != ["image_id", "subset"]:
        raise ValueError(f"split file {split_path} must start with columns image_id, subset")
    dup_split = split.loc[split["image_id"].duplicated(), "image_id"].tolist()
    if dup_split:
        raise ValueError(f"image ids in more than one subset of {split_path}: {dup_split[:10]}")
    bad_subsets = sorted(set(split["subset"]) - set(SUBSETS))
    if bad_subsets:
        raise ValueError(f"split file contains unknown subsets {bad_subsets}")

    wanted = split.loc[split["subset"].isin(requested)]
    missing_in_manifest = sorted(set(wanted["image_id"]) - set(manifest["image_id"]))
    if missing_in_manifest:
        raise ValueError(f"{len(missing_in_manifest)} split ids are missing from the manifest: {missing_in_manifest[:10]}")

    table = wanted.merge(manifest, on="image_id", how="inner", validate="one_to_one")
    if len(table) != len(wanted):
        raise ValueError("joining the split with the manifest changed the number of rows")
    if not table["image_id"].is_unique:
        raise ValueError("an image id appears in more than one requested subset")

    missing_files = [i for i in table["image_id"] if not (image_root / f"{i}{IMAGE_SUFFIX}").is_file()]
    if missing_files:
        raise ValueError(f"{len(missing_files)} images have no JPEG in {image_root}: {missing_files[:10]}")

    grades = table["grade"]
    if not grades.map(lambda g: float(g).is_integer()).all():
        raise ValueError("grades must be integers")
    if not grades.between(GRADE_MIN, GRADE_MAX).all():
        raise ValueError(f"grades must lie in {GRADE_MIN}..{GRADE_MAX}")
    table["grade"] = grades.astype("int64")

    table = table.sort_values("image_id").reset_index(drop=True)
    logger.info("run table: %s", ", ".join(f"{s}={int((table['subset'] == s).sum())}" for s in requested))
    return table


class FundusDataset(Dataset):
    """Processed fundus JPEGs with their grades. Returns (tensor, grade, image_id)."""

    def __init__(self, table: pd.DataFrame, image_dir: Path | str, transform: Callable[[Image.Image], Any]) -> None:
        """Store the table rows, the folder of JPEGs and the transform to apply to each image."""
        self._image_ids: list[str] = table["image_id"].astype(str).tolist()
        self._grades: list[int] = table["grade"].astype(int).tolist()
        self._image_dir = Path(image_dir)
        self._transform = transform

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self._image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Load one JPEG as RGB, transform it and return it with its grade and id."""
        image_id = self._image_ids[index]
        with Image.open(self._image_dir / f"{image_id}{IMAGE_SUFFIX}") as handle:
            rgb = handle.convert("RGB")
        return self._transform(rgb), self._grades[index], image_id


class RandomGamma(nn.Module):
    """Gamma correction with gamma drawn uniformly from [gamma_min, gamma_max].

    Works on float images in [0, 1]. The draw uses torch's global generator, so it
    follows the run seed and is saved with the checkpoint.
    """

    def __init__(self, gamma_min: float, gamma_max: float) -> None:
        """Store the gamma range; both ends must be positive."""
        super().__init__()
        if not 0.0 < gamma_min <= gamma_max:
            raise ValueError(f"gamma range must satisfy 0 < min <= max, got {gamma_min}, {gamma_max}")
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction with a freshly drawn gamma."""
        gamma = float(torch.empty(()).uniform_(self.gamma_min, self.gamma_max))
        return F.adjust_gamma(image, gamma)


def build_transforms(input_size: int, train: bool, aug: AugConfig, source_size: int) -> T.Compose:
    """Return the image pipeline for training or evaluation.

    Training (brief 7): rotation over the full circle with black fill, horizontal and
    vertical flips, random resized crop, brightness and contrast jitter, gamma, then
    ImageNet normalisation. There is no hue or saturation change, because colour
    carries clinical meaning.

    Evaluation: resize to input_size only when it differs from source_size (the
    stored processed size), then ImageNet normalisation. Nothing else, so the
    output is deterministic.
    """
    steps: list[nn.Module | Callable[[Any], Any]] = [T.ToImage()]
    if train:
        steps += [
            T.RandomRotation(degrees=aug.rotation_degrees, fill=0),
            T.RandomHorizontalFlip(p=aug.hflip_p),
            T.RandomVerticalFlip(p=aug.vflip_p),
            T.RandomResizedCrop(
                input_size,
                scale=(aug.scale_min, aug.scale_max),
                ratio=(aug.ratio_min, aug.ratio_max),
                antialias=True,
            ),
            T.ToDtype(torch.float32, scale=True),
            T.ColorJitter(brightness=aug.brightness, contrast=aug.contrast),
            RandomGamma(aug.gamma_min, aug.gamma_max),
        ]
    else:
        if input_size != source_size:
            steps.append(T.Resize(input_size, antialias=True))
        steps.append(T.ToDtype(torch.float32, scale=True))
    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(steps)


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python in each DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % NUMPY_SEED_MODULUS
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device_type: str,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Return a DataLoader with the project's settings (brief 7).

    Shuffling is for training only (the caller decides). pin_memory is on for CUDA
    only. Persistent workers are used whenever there are workers. The generator
    drives shuffling; pass one in to checkpoint it, or leave it out to get a fresh
    generator seeded from the run seed.
    """
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )


def steps_per_epoch(n_images: int, batch_size: int, accumulation_steps: int) -> int:
    """Return the number of optimiser steps in one epoch, counting a final partial group."""
    if n_images < 1 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError("sizes must be positive")
    batches = math.ceil(n_images / batch_size)
    return math.ceil(batches / accumulation_steps)


def grades_of(table: pd.DataFrame, subset: str) -> Sequence[int]:
    """Return the grades of one subset as a list, in table order."""
    return table.loc[table["subset"] == subset, "grade"].astype(int).tolist()
