"""Tests for non-black camera backgrounds (DECISIONS.md D13). Synthetic data only.

Each case mirrors something seen in the real data: dark-grey borders (level 16 to
20), white borders, fundus discs that touch the frame edges, and dark findings
inside the disc that must never be changed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from drgrade.preprocess import blacken_background, border_level, crop_fundus

THRESHOLD, MARGIN, MIN_AREA = 10, 0.02, 0.30
BG_MIN, BG_TOL, BG_SPREAD = 8, 10, 12
FUNDUS = (180, 90, 40)
DARK_SPOT = (20, 5, 5)


def _fundus(background: int, size: tuple[int, int] = (400, 300), radius_frac: float = 0.45) -> Image.Image:
    """A frame with a fundus disc and a dark 'haemorrhage' at its centre."""
    img = Image.new("RGB", size, (background,) * 3)
    draw = ImageDraw.Draw(img)
    w, h = size
    r = int(h * radius_frac)
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=FUNDUS)
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=DARK_SPOT)
    return img


def _crop(img: Image.Image):
    return crop_fundus(img, THRESHOLD, MARGIN, MIN_AREA, BG_MIN, BG_TOL, BG_SPREAD)


def test_border_level_reads_corners() -> None:
    assert border_level(_fundus(0)) == (0, 0)
    assert border_level(_fundus(16)) == (16, 0)
    assert border_level(_fundus(255)) == (255, 0)


def test_black_background_is_untouched() -> None:
    img = _fundus(0)
    out, changed = blacken_background(img, BG_MIN, BG_TOL, BG_SPREAD)
    assert not changed and out is img


def test_grey_background_becomes_black_and_inner_spot_survives() -> None:
    out, changed = blacken_background(_fundus(16), BG_MIN, BG_TOL, BG_SPREAD)
    arr = np.asarray(out)
    assert changed
    assert arr[0, 0].tolist() == [0, 0, 0]
    assert arr[150, 5].tolist() == [0, 0, 0]
    assert arr[150, 200].tolist() == list(DARK_SPOT)  # darker than the border, inside the disc
    assert arr[150, 120].tolist() == list(FUNDUS)


def test_white_background_becomes_black() -> None:
    out, changed = blacken_background(_fundus(255), BG_MIN, BG_TOL, BG_SPREAD)
    arr = np.asarray(out)
    assert changed
    assert arr[0, 0].tolist() == [0, 0, 0]
    assert arr[150, 120].tolist() == list(FUNDUS)


def test_crops_agree_across_black_grey_and_white_backgrounds() -> None:
    boxes = []
    for background in (0, 16, 255):
        _, box, failed = _crop(_fundus(background))
        assert not failed
        boxes.append(box)
    assert boxes[0] == boxes[1] == boxes[2]
    assert boxes[0][2] - boxes[0][0] < 300  # the 400 px wide frame was cropped to the disc


def test_disc_touching_the_edges_is_not_eaten() -> None:
    """A disc wider than the frame touches the top and bottom edges. With a grey
    border the fundus must survive along those edges (the real-data failure)."""
    img = _fundus(16, size=(400, 300), radius_frac=0.6)
    cropped, _, failed = _crop(img)
    arr = np.asarray(img)
    out, _ = blacken_background(img, BG_MIN, BG_TOL, BG_SPREAD)
    out_arr = np.asarray(out)
    assert not failed
    assert out_arr[0, 200].tolist() == arr[0, 200].tolist() == list(FUNDUS)  # edge-touching fundus kept
    fundus_pixels = (arr == FUNDUS).all(axis=2)
    assert (out_arr[fundus_pixels] == FUNDUS).all()


def test_non_uniform_corners_are_left_alone() -> None:
    img = _fundus(16)
    ImageDraw.Draw(img).rectangle((0, 0, 30, 30), fill=(120, 120, 120))  # e.g. a burned-in label
    out, changed = blacken_background(img, BG_MIN, BG_TOL, BG_SPREAD)
    assert not changed and out is img


def test_without_background_params_grey_is_not_cropped() -> None:
    """Documents the original failure: a fixed threshold keeps the whole grey frame."""
    _, box, _ = crop_fundus(_fundus(16), THRESHOLD, MARGIN, MIN_AREA)
    assert box == (0, 0, 400, 300)


def test_circle_mask_blacks_out_corners_only() -> None:
    from drgrade.preprocess import apply_circle_mask

    img = Image.new("RGB", (64, 64), (30, 30, 30))
    out = np.asarray(apply_circle_mask(img))
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[63, 63].tolist() == [0, 0, 0]
    assert out[32, 32].tolist() == [30, 30, 30]
    assert out[32, 1].tolist() == [30, 30, 30]  # edge midpoint is inside the circle
