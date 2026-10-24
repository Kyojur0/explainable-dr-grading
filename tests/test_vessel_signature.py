"""Tests for the vessel-pattern signature on synthetic fundi (brief 5.4, D03)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from _synthetic import make_cfg, make_fundus
from drgrade.vessel_signature import vessel_signature


def _save_jpeg(img: Image.Image, path: Path, quality: int) -> Path:
    img.save(path, "JPEG", quality=quality)
    return path


def test_signature_has_grid_length_zero_mean_and_unit_norm(tmp_path: Path) -> None:
    cfg = make_cfg()
    path = _save_jpeg(make_fundus(seed=1), tmp_path / "a.jpg", quality=95)
    vector = vessel_signature(path, cfg)
    assert vector.shape == (cfg.vsig_grid * cfg.vsig_grid,)
    assert vector.dtype == np.float32
    assert abs(float(vector.mean())) < 1e-5
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_reencoded_and_brightened_copy_is_the_same_eye(tmp_path: Path) -> None:
    cfg = make_cfg()
    original = make_fundus(seed=1)
    a = _save_jpeg(original, tmp_path / "a.jpg", quality=95)
    brightened = ImageEnhance.Brightness(original).enhance(1.1)
    b = _save_jpeg(brightened, tmp_path / "b.jpg", quality=85)
    similarity = float(vessel_signature(a, cfg) @ vessel_signature(b, cfg))
    assert similarity >= 0.95


def test_different_vessel_pattern_is_a_different_eye(tmp_path: Path) -> None:
    cfg = make_cfg()
    a = _save_jpeg(make_fundus(seed=1), tmp_path / "a.jpg", quality=95)
    b = _save_jpeg(make_fundus(seed=2), tmp_path / "b.jpg", quality=95)
    similarity = float(vessel_signature(a, cfg) @ vessel_signature(b, cfg))
    assert similarity < 0.6


def test_featureless_image_gives_zero_vector(tmp_path: Path) -> None:
    cfg = make_cfg()
    # A uniform frame has no internal structure. Its high-pass signal is exactly zero.
    path = tmp_path / "flat.png"
    Image.new("RGB", (512, 512), (200, 120, 80)).save(path)
    vector = vessel_signature(path, cfg)
    assert vector.shape == (cfg.vsig_grid * cfg.vsig_grid,)
    assert not np.any(vector)
