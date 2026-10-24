"""Tests for configuration loading and dataset root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from drgrade.config import ConfigError, load_data_config, resolve_mmrdr_root

VALID = """
mmrdr_root: null
crop_threshold: 10
crop_margin_frac: 0.02
crop_min_area_frac: 0.30
bg_detect_min: 8
bg_tolerance: 10
bg_corner_spread: 12
target_size: 512
circle_mask: true
jpeg_quality: 95
processed_dir: data/processed
ben_graham: false
ben_graham_sigma: 10
vsig_decode_scale: 4
vsig_canvas: 256
vsig_blur_radius: 6
vsig_ring_frac: 0.9
vsig_grid: 64
vsig_threshold: 0.93
audit_out: outputs/audit
seed: 42
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path: Path) -> None:
    cfg = load_data_config(_write(tmp_path, VALID))
    assert cfg.vsig_threshold == 0.93
    assert cfg.vsig_grid == 64
    assert cfg.mmrdr_root is None


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    text = VALID.replace("seed: 42\n", "")
    with pytest.raises(ConfigError):
        load_data_config(_write(tmp_path, text))


def test_unsupported_decode_scale_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="vsig_decode_scale"):
        load_data_config(_write(tmp_path, VALID.replace("vsig_decode_scale: 4", "vsig_decode_scale: 3")))


def test_out_of_range_ring_fraction_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="vsig_ring_frac"):
        load_data_config(_write(tmp_path, VALID.replace("vsig_ring_frac: 0.9", "vsig_ring_frac: 0")))


def test_grid_larger_than_canvas_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="vsig_grid"):
        load_data_config(_write(tmp_path, VALID.replace("vsig_grid: 64", "vsig_grid: 512")))


def test_preprocessing_settings_load(tmp_path: Path) -> None:
    cfg = load_data_config(_write(tmp_path, VALID))
    assert cfg.target_size == 512
    assert cfg.jpeg_quality == 95
    assert cfg.processed_dir == "data/processed"
    assert cfg.ben_graham is False
    assert cfg.ben_graham_sigma == 10.0


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("target_size: 512", "target_size: 4", "target_size"),
        ("jpeg_quality: 95", "jpeg_quality: 0", "jpeg_quality"),
        ("jpeg_quality: 95", "jpeg_quality: 101", "jpeg_quality"),
        ("processed_dir: data/processed", 'processed_dir: "  "', "processed_dir"),
        ("ben_graham: false", "ben_graham: 1", "ben_graham"),
        ("ben_graham_sigma: 10", "ben_graham_sigma: 0", "ben_graham_sigma"),
    ],
)
def test_invalid_preprocessing_settings_are_rejected(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        load_data_config(_write(tmp_path, VALID.replace(old, new)))


def test_root_from_config_wins_over_environment(tmp_path: Path) -> None:
    root = resolve_mmrdr_root(str(tmp_path), environ={"MMRDR_ROOT": "/does/not/matter"})
    assert root == tmp_path


def test_root_falls_back_to_environment(tmp_path: Path) -> None:
    assert resolve_mmrdr_root(None, environ={"MMRDR_ROOT": str(tmp_path)}) == tmp_path


def test_root_unset_raises() -> None:
    with pytest.raises(ConfigError, match="MMRDR_ROOT"):
        resolve_mmrdr_root(None, environ={})


def test_root_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a directory"):
        resolve_mmrdr_root(str(tmp_path / "absent"), environ={})
