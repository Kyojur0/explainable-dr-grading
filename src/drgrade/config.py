"""Configuration loading and dataset root resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MMRDR_ROOT_ENV = "MMRDR_ROOT"
SUPPORTED_DECODE_SCALES = (1, 2, 4, 8)  # JPEG draft-mode reduction factors (PIL)
MIN_TARGET_SIZE = 8  # smallest square side the preprocessing will produce
MAX_JPEG_QUALITY = 100  # Pillow accepts 1 to 100


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    """Settings for the data audit and preprocessing, loaded from configs/data.yaml."""

    mmrdr_root: str | None
    crop_threshold: int
    crop_margin_frac: float
    crop_min_area_frac: float
    bg_detect_min: int
    bg_tolerance: int
    bg_corner_spread: int
    target_size: int
    circle_mask: bool
    jpeg_quality: int
    processed_dir: str
    ben_graham: bool
    ben_graham_sigma: float
    vsig_decode_scale: int
    vsig_canvas: int
    vsig_blur_radius: float
    vsig_ring_frac: float
    vsig_grid: int
    vsig_threshold: float
    audit_out: str
    seed: int

    def __post_init__(self) -> None:
        """Validate ranges so that bad values fail at load time, not mid-run."""
        if not 0 <= self.crop_threshold <= 255:
            raise ConfigError(f"crop_threshold must be in 0..255, got {self.crop_threshold}")
        if not 0.0 <= self.crop_margin_frac < 1.0:
            raise ConfigError(f"crop_margin_frac must be in [0, 1), got {self.crop_margin_frac}")
        if not 1 <= self.bg_detect_min <= 255:
            raise ConfigError(f"bg_detect_min must be in 1..255, got {self.bg_detect_min}")
        if not 0 <= self.bg_tolerance <= 255:
            raise ConfigError(f"bg_tolerance must be in 0..255, got {self.bg_tolerance}")
        if not 0 <= self.bg_corner_spread <= 255:
            raise ConfigError(f"bg_corner_spread must be in 0..255, got {self.bg_corner_spread}")
        if not 0.0 < self.crop_min_area_frac <= 1.0:
            raise ConfigError(f"crop_min_area_frac must be in (0, 1], got {self.crop_min_area_frac}")
        if self.target_size < MIN_TARGET_SIZE:
            raise ConfigError(f"target_size must be at least {MIN_TARGET_SIZE}, got {self.target_size}")
        if not 1 <= self.jpeg_quality <= MAX_JPEG_QUALITY:
            raise ConfigError(f"jpeg_quality must be in 1..{MAX_JPEG_QUALITY}, got {self.jpeg_quality}")
        if not self.processed_dir.strip():
            raise ConfigError("processed_dir must be a non-empty path")
        if not isinstance(self.ben_graham, bool):
            raise ConfigError(f"ben_graham must be true or false, got {self.ben_graham!r}")
        if self.ben_graham_sigma <= 0:
            raise ConfigError(f"ben_graham_sigma must be positive, got {self.ben_graham_sigma}")
        if self.vsig_decode_scale not in SUPPORTED_DECODE_SCALES:
            raise ConfigError(
                f"vsig_decode_scale must be one of {SUPPORTED_DECODE_SCALES}, "
                f"got {self.vsig_decode_scale}"
            )
        if self.vsig_canvas < 8:
            raise ConfigError(f"vsig_canvas must be at least 8, got {self.vsig_canvas}")
        if not 1 <= self.vsig_grid <= self.vsig_canvas:
            raise ConfigError(
                f"vsig_grid must be in 1..vsig_canvas ({self.vsig_canvas}), got {self.vsig_grid}"
            )
        if self.vsig_blur_radius < 0:
            raise ConfigError(f"vsig_blur_radius must be non-negative, got {self.vsig_blur_radius}")
        if not 0.0 < self.vsig_ring_frac <= 1.0:
            raise ConfigError(f"vsig_ring_frac must be in (0, 1], got {self.vsig_ring_frac}")
        if not 0.0 < self.vsig_threshold <= 1.0:
            raise ConfigError(f"vsig_threshold must be in (0, 1], got {self.vsig_threshold}")
        # Store floats as floats so that the cached parameter string is stable.
        object.__setattr__(self, "crop_margin_frac", float(self.crop_margin_frac))
        object.__setattr__(self, "crop_min_area_frac", float(self.crop_min_area_frac))
        object.__setattr__(self, "ben_graham_sigma", float(self.ben_graham_sigma))
        object.__setattr__(self, "vsig_blur_radius", float(self.vsig_blur_radius))
        object.__setattr__(self, "vsig_ring_frac", float(self.vsig_ring_frac))
        object.__setattr__(self, "vsig_threshold", float(self.vsig_threshold))


def load_data_config(path: Path | str) -> DataConfig:
    """Read a YAML file into a validated DataConfig.

    Every key is required. Unknown or missing keys raise ConfigError so that a
    typo cannot silently fall back to a default.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping")
    try:
        return DataConfig(**raw)
    except TypeError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc


def resolve_mmrdr_root(configured: str | None, environ: Mapping[str, str] | None = None) -> Path:
    """Return the dataset root (the folder that contains MMRDR-CFP/).

    The configured value wins. If it is null, the MMRDR_ROOT environment variable
    is used. Raises ConfigError if neither is set or the folder does not exist.
    """
    env = os.environ if environ is None else environ
    value = configured or env.get(MMRDR_ROOT_ENV)
    if not value:
        raise ConfigError(
            f"dataset root not set: set mmrdr_root in configs/data.yaml or export {MMRDR_ROOT_ENV}"
        )
    root = Path(value).expanduser()
    if not root.is_dir():
        raise ConfigError(f"dataset root is not a directory: {root}")
    return root
