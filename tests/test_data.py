"""Tests for the run table, dataset, transforms and loaders (synthetic data only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

from _synthetic import make_fundus
from drgrade.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FundusDataset,
    build_transforms,
    load_run_table,
    make_loader,
)
from drgrade.run_config import AugConfig
from _training_data import write_tiny_dataset

AUG = AugConfig(
    rotation_degrees=180,
    hflip_p=0.5,
    vflip_p=0.5,
    scale_min=0.85,
    scale_max=1.0,
    ratio_min=0.95,
    ratio_max=1.05,
    brightness=0.2,
    contrast=0.2,
    gamma_min=0.8,
    gamma_max=1.2,
)
NO_JITTER = AugConfig(
    rotation_degrees=180,
    hflip_p=0.5,
    vflip_p=0.5,
    scale_min=1.0,
    scale_max=1.0,
    ratio_min=1.0,
    ratio_max=1.0,
    brightness=0.0,
    contrast=0.0,
    gamma_min=1.0,
    gamma_max=1.0,
)


def test_load_run_table_joins_and_keeps_requested_subsets(tmp_path: Path) -> None:
    """Train and val are joined with the manifest; test rows are not returned."""
    data = write_tiny_dataset(tmp_path)
    table = load_run_table(data.manifest, data.split, ("train", "val"), data.image_dir)
    assert set(table["subset"]) == {"train", "val"}
    assert len(table) == 15 + 5
    assert table["image_id"].is_unique
    assert list(table["image_id"]) == sorted(table["image_id"])


def test_test_subset_is_refused_unless_allowed(tmp_path: Path) -> None:
    """Requesting the test subset raises, and passes only with allow_test=True."""
    data = write_tiny_dataset(tmp_path)
    with pytest.raises(ValueError, match="test subset"):
        load_run_table(data.manifest, data.split, ("train", "test"), data.image_dir)
    table = load_run_table(data.manifest, data.split, ("test",), data.image_dir, allow_test=True)
    assert len(table) == 4


def test_missing_manifest_row_raises(tmp_path: Path) -> None:
    """A split id absent from the manifest is an error that names the id."""
    data = write_tiny_dataset(tmp_path)
    manifest = pd.read_csv(data.manifest, dtype={"image_id": str})
    manifest[manifest["image_id"] != "tr000001"].to_csv(data.manifest, index=False)
    with pytest.raises(ValueError, match="tr000001"):
        load_run_table(data.manifest, data.split, ("train",), data.image_dir)


def test_missing_jpeg_raises(tmp_path: Path) -> None:
    """A requested image without a JPEG is an error that names the file."""
    data = write_tiny_dataset(tmp_path)
    (data.image_dir / "tr000002.jpg").unlink()
    with pytest.raises(ValueError, match="tr000002"):
        load_run_table(data.manifest, data.split, ("train",), data.image_dir)


def test_id_in_two_subsets_raises(tmp_path: Path) -> None:
    """An id listed twice in the split file is an overlap error."""
    data = write_tiny_dataset(tmp_path)
    split = pd.read_csv(data.split, dtype=str)
    extra = pd.DataFrame([{"image_id": "tr000001", "subset": "val"}])
    pd.concat([split, extra]).to_csv(data.split, index=False)
    with pytest.raises(ValueError, match="more than one subset"):
        load_run_table(data.manifest, data.split, ("train", "val"), data.image_dir)


def test_train_transform_shape_dtype_and_range(tmp_path: Path) -> None:
    """Training output is float32 at the input size, normalised with ImageNet statistics."""
    image = make_fundus(seed=1, size=96)
    transform = build_transforms(input_size=48, train=True, aug=AUG, source_size=96)
    out = transform(image)
    assert out.shape == (3, 48, 48)
    assert out.dtype == torch.float32
    low = (0.0 - max(IMAGENET_MEAN)) / min(IMAGENET_STD)
    high = (1.0 - min(IMAGENET_MEAN)) / max(IMAGENET_STD)
    assert float(out.min()) >= low - 1e-4 and float(out.max()) <= high + 1e-4


def test_eval_transform_is_deterministic_and_normalised() -> None:
    """Evaluation output is identical across calls, and a grey image maps to the normalised grey value."""
    grey = Image.new("RGB", (64, 64), (128, 128, 128))
    transform = build_transforms(input_size=64, train=False, aug=AUG, source_size=64)
    first = transform(grey)
    second = transform(grey)
    assert torch.equal(first, second)
    expected = (128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert float(first[0].mean()) == pytest.approx(expected, abs=1e-3)


def test_eval_transform_resizes_only_when_size_differs() -> None:
    """Resize happens for a different input size and not for the stored size."""
    image = make_fundus(seed=2, size=64)
    same = build_transforms(input_size=64, train=False, aug=AUG, source_size=64)(image)
    smaller = build_transforms(input_size=32, train=False, aug=AUG, source_size=64)(image)
    assert same.shape == (3, 64, 64)
    assert smaller.shape == (3, 32, 32)


def test_rotation_keeps_black_corners_black() -> None:
    """With geometry and colour fixed, a rotated disc on black keeps its corners black."""
    size = 64
    disc = Image.new("RGB", (size, size), (0, 0, 0))
    centre = size / 2
    radius = size * 0.42
    ImageDraw.Draw(disc).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius], fill=(255, 255, 255)
    )
    transform = build_transforms(input_size=size, train=True, aug=NO_JITTER, source_size=size)
    for seed in range(5):
        torch.manual_seed(seed)
        out = transform(disc)
        corners = torch.cat([out[:, :4, :4], out[:, -4:, -4:]], dim=2)
        for channel in range(3):
            black = (0.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            assert float(corners[channel].mean()) == pytest.approx(black, abs=1e-3)


def test_dataset_returns_tensor_grade_and_id(tmp_path: Path) -> None:
    """FundusDataset yields (tensor, int grade, str id) in table order."""
    data = write_tiny_dataset(tmp_path)
    table = load_run_table(data.manifest, data.split, ("val",), data.image_dir)
    dataset = FundusDataset(table, data.image_dir, build_transforms(64, False, AUG, 64))
    tensor, grade, image_id = dataset[0]
    assert isinstance(tensor, torch.Tensor) and tensor.shape == (3, 64, 64)
    assert isinstance(grade, int) and isinstance(image_id, str)
    assert image_id == table.iloc[0]["image_id"]


def test_loader_batches_and_shuffle_flag(tmp_path: Path) -> None:
    """Training shuffles with the seeded generator; evaluation keeps table order; batches cover all rows."""
    data = write_tiny_dataset(tmp_path)
    table = load_run_table(data.manifest, data.split, ("train",), data.image_dir)
    dataset = FundusDataset(table, data.image_dir, build_transforms(64, False, AUG, 64))
    ordered = make_loader(dataset, 4, shuffle=False, num_workers=0, seed=42, device_type="cpu")
    ids_in_order = [i for _, _, batch in ordered for i in batch]
    assert ids_in_order == list(table["image_id"])
    shuffled_a = make_loader(dataset, 4, shuffle=True, num_workers=0, seed=42, device_type="cpu")
    shuffled_b = make_loader(dataset, 4, shuffle=True, num_workers=0, seed=42, device_type="cpu")
    ids_a = [i for _, _, batch in shuffled_a for i in batch]
    ids_b = [i for _, _, batch in shuffled_b for i in batch]
    assert ids_a == ids_b
    assert sorted(ids_a) == sorted(table["image_id"])
    assert ordered.pin_memory is False
    assert ordered.persistent_workers is False
