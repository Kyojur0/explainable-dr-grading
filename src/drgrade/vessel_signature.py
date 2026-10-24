"""Vessel-pattern signature of a fundus photograph (DECISIONS.md D03).

Fundus photographs share the same global shape (a bright disc on black), so a
perceptual hash of the whole image mostly measures that shape. The signature
below keeps only the fine structure: the high-pass (fine detail) part of the
green channel, inside a central disc, downsampled to a small grid. The retinal
vessels dominate that signal, so two photographs of the same eye have similar
signatures and photographs of different eyes do not.

Pipeline (each step is also described in the docstring of `vessel_signature`):
    decode (JPEG draft) -> crop black border -> pad square -> resize (BOX)
    -> green channel high-pass inside the fundus mask and a central ring
    -> downsample (BOX) to grid x grid -> zero mean, unit L2 norm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from drgrade.config import DataConfig
from drgrade.preprocess import crop_fundus, pad_to_square


def _high_pass_map(canvas: Image.Image, cfg: DataConfig) -> np.ndarray:
    """Return the masked, ring-limited high-pass green channel on the canvas grid."""
    size = canvas.width
    green = np.asarray(canvas, dtype=np.float32)[:, :, 1]
    grey = np.asarray(canvas.convert("L"))
    mask = (grey > cfg.crop_threshold).astype(np.float32)

    # PIL cannot blur float images, so blur the 8-bit green channel and convert back.
    blurred = Image.fromarray(green.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=float(cfg.vsig_blur_radius))
    )
    blur = np.asarray(blurred, dtype=np.float32)

    high_pass = (green - blur) * mask

    centre = (size - 1) / 2.0
    limit = (cfg.vsig_ring_frac * size / 2.0) ** 2
    rows, cols = np.indices((size, size), dtype=np.float32)
    ring = ((rows - centre) ** 2 + (cols - centre) ** 2) < limit
    return high_pass * ring.astype(np.float32)


def vessel_signature(path: Path, cfg: DataConfig) -> np.ndarray:
    """Return the vessel-pattern signature of one fundus image.

    Args:
        path: image file (any format Pillow can read; JPEG is decoded at reduced size).
        cfg: data configuration holding the crop and vsig_* settings.

    Returns:
        float32 vector of length vsig_grid * vsig_grid with zero mean and unit L2
        norm. The zero vector is returned when the high-pass signal is empty (for
        example a featureless image), so callers must not treat it as a match.
    """
    with Image.open(path) as opened:
        opened.draft(
            "RGB",
            (max(1, opened.width // cfg.vsig_decode_scale), max(1, opened.height // cfg.vsig_decode_scale)),
        )
        rgb = opened.convert("RGB")

    cropped, _, _ = crop_fundus(
        rgb,
        cfg.crop_threshold,
        cfg.crop_margin_frac,
        cfg.crop_min_area_frac,
        cfg.bg_detect_min,
        cfg.bg_tolerance,
        cfg.bg_corner_spread,
    )
    canvas = pad_to_square(cropped).resize(
        (cfg.vsig_canvas, cfg.vsig_canvas), Image.Resampling.BOX
    )

    high_pass = _high_pass_map(canvas, cfg)
    grid_image = Image.fromarray(high_pass.astype(np.float32)).resize(
        (cfg.vsig_grid, cfg.vsig_grid), Image.Resampling.BOX
    )
    vector = np.asarray(grid_image, dtype=np.float32).ravel().copy()
    vector = vector - vector.mean()

    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm == 0.0:
        return np.zeros(cfg.vsig_grid * cfg.vsig_grid, dtype=np.float32)
    return (vector / norm).astype(np.float32)
