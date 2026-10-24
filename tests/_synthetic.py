"""Synthetic fundus images for tests (no dataset needed).

A fundus is a bright disc on a black background, with dark curved lines inside
the disc standing in for retinal vessels. The lines come from a seeded RNG so
that a given seed always gives the same picture.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from drgrade.config import DataConfig

DISC_COLOUR = (200, 120, 80)
VESSEL_COLOUR = (90, 40, 20)


def make_cfg(**overrides: object) -> DataConfig:
    """Return a DataConfig with the default settings, optionally overridden."""
    values: dict[str, object] = {
        "mmrdr_root": None,
        "crop_threshold": 10,
        "crop_margin_frac": 0.02,
        "crop_min_area_frac": 0.30,
        "bg_detect_min": 8,
        "bg_tolerance": 10,
        "bg_corner_spread": 12,
        "target_size": 512,
        "circle_mask": True,
        "jpeg_quality": 95,
        "processed_dir": "data/processed",
        "ben_graham": False,
        "ben_graham_sigma": 10,
        "vsig_decode_scale": 4,
        "vsig_canvas": 256,
        "vsig_blur_radius": 6,
        "vsig_ring_frac": 0.9,
        "vsig_grid": 64,
        "vsig_threshold": 0.93,
        "audit_out": "outputs/audit",
        "seed": 42,
    }
    values.update(overrides)
    return DataConfig(**values)  # type: ignore[arg-type]


def make_fundus(seed: int, size: int = 1024, n_vessels: int = 40) -> Image.Image:
    """Draw a synthetic fundus: a bright disc with seeded dark random-walk lines."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    centre = size / 2.0
    radius = size * 0.42
    draw.ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        fill=DISC_COLOUR,
    )
    for _ in range(n_vessels):
        start_angle = rng.uniform(0.0, 2.0 * math.pi)
        start_radius = radius * math.sqrt(rng.uniform(0.05, 0.85))
        x = centre + start_radius * math.cos(start_angle)
        y = centre + start_radius * math.sin(start_angle)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        points = [(x, y)]
        for _ in range(8):
            heading += rng.normal(0.0, 0.35)
            step = rng.uniform(size * 0.015, size * 0.03)
            x += step * math.cos(heading)
            y += step * math.sin(heading)
            if math.hypot(x - centre, y - centre) > radius * 0.95:
                break
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=VESSEL_COLOUR, width=max(2, size // 200), joint="curve")
    return img


def make_featureless_disc(size: int = 1024) -> Image.Image:
    """Draw a uniform bright disc on black, with no internal structure."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    centre = size / 2.0
    radius = size * 0.42
    ImageDraw.Draw(img).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        fill=DISC_COLOUR,
    )
    return img
