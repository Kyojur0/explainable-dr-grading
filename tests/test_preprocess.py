"""Tests for the stage 2 image functions on synthetic fundus images (brief 6.1, D10, D11)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from _synthetic import DISC_COLOUR, make_cfg, make_featureless_disc, make_fundus
from drgrade.preprocess import (
    ben_graham,
    fundus_mask,
    preprocess_image,
    quality_measures,
)

TARGET = 128
THRESHOLD = 10


def _bright_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box (left, top, right, bottom) of pixels above the fundus threshold."""
    mask = fundus_mask(img, THRESHOLD)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _flat_topped_disc(size: int = 400) -> Image.Image:
    """A disc on black whose top is cut flat, so the crop is not square."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    centre = size // 2
    radius = 150
    ImageDraw.Draw(img).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius], fill=DISC_COLOUR
    )
    ImageDraw.Draw(img).rectangle([0, 0, size, 80], fill=(0, 0, 0))
    return img


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------


def test_preprocess_output_size_and_mode() -> None:
    cfg = make_cfg(target_size=TARGET)
    out, box, crop_failed = preprocess_image(make_fundus(seed=1, size=300), cfg)
    assert out.size == (TARGET, TARGET)
    assert out.mode == "RGB"
    assert not crop_failed
    assert box[2] > box[0] and box[3] > box[1]


def test_preprocess_accepts_greyscale_input() -> None:
    cfg = make_cfg(target_size=TARGET)
    grey = make_fundus(seed=2, size=300).convert("L")
    out, _, _ = preprocess_image(grey, cfg)
    assert out.size == (TARGET, TARGET)
    assert out.mode == "RGB"


def test_non_square_fundus_is_padded_and_centred() -> None:
    cfg = make_cfg(target_size=TARGET)
    out, box, _ = preprocess_image(_flat_topped_disc(), cfg)
    assert out.size == (TARGET, TARGET)
    # The crop is wider than it is tall, so the padding is at the top and bottom.
    assert box[2] - box[0] > box[3] - box[1]
    assert out.getpixel((0, 0)) == (0, 0, 0)
    # Symmetric margins around the bright region mean the disc is centred.
    left, top, right, bottom = _bright_bbox(out)
    assert abs((left + right) / 2 - TARGET / 2) <= 2
    assert abs((top + bottom) / 2 - TARGET / 2) <= 2


def test_ben_graham_flag_applies_normalisation() -> None:
    plain = preprocess_image(make_fundus(seed=3, size=300), make_cfg(target_size=TARGET))[0]
    normalised = preprocess_image(
        make_fundus(seed=3, size=300), make_cfg(target_size=TARGET, ben_graham=True)
    )[0]
    assert normalised.size == plain.size
    assert np.asarray(normalised).tobytes() != np.asarray(plain).tobytes()


# ---------------------------------------------------------------------------
# fundus_mask
# ---------------------------------------------------------------------------


def test_mask_threshold_is_strict() -> None:
    img = Image.new("L", (4, 4), THRESHOLD)
    assert not fundus_mask(img, THRESHOLD).any()
    img.putpixel((1, 1), THRESHOLD + 1)
    mask = fundus_mask(img, THRESHOLD)
    assert mask.sum() == 1 and mask[1, 1]


# ---------------------------------------------------------------------------
# quality_measures
# ---------------------------------------------------------------------------


def test_blurred_copy_is_less_sharp() -> None:
    original = make_fundus(seed=4, size=256).resize((128, 128), Image.Resampling.BOX)
    blurred = original.filter(ImageFilter.GaussianBlur(radius=2))
    sharp = quality_measures(original, fundus_mask(original, THRESHOLD))["sharpness"]
    soft = quality_measures(blurred, fundus_mask(blurred, THRESHOLD))["sharpness"]
    assert sharp > 0
    assert soft < sharp


def test_darkened_copy_is_less_bright() -> None:
    original = make_fundus(seed=5, size=256).resize((128, 128), Image.Resampling.BOX)
    darker = ImageEnhance.Brightness(original).enhance(0.5)
    bright = quality_measures(original, fundus_mask(original, THRESHOLD))["brightness"]
    dim = quality_measures(darker, fundus_mask(darker, THRESHOLD))["brightness"]
    assert dim < bright
    assert dim == pytest.approx(bright * 0.5, rel=0.05)


def test_flat_disc_has_zero_contrast() -> None:
    img = make_featureless_disc(size=128)
    measures = quality_measures(img, fundus_mask(img, THRESHOLD))
    assert measures["contrast"] == pytest.approx(0.0, abs=1e-9)


def test_flat_disc_edge_gives_no_sharpness() -> None:
    # The disc has one hard edge. Without the erosion the Laplacian would be large
    # on that edge; with the erosion it must not contribute.
    img = make_featureless_disc(size=128)
    mask = fundus_mask(img, THRESHOLD)
    grey = np.asarray(img.convert("L"), dtype=np.float64)
    raw_laplacian = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4 * grey[1:-1, 1:-1]
    )
    assert raw_laplacian.var() > 1.0
    assert quality_measures(img, mask)["sharpness"] == pytest.approx(0.0, abs=1e-9)


def test_empty_mask_gives_zero_measures() -> None:
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    assert quality_measures(img, fundus_mask(img, THRESHOLD)) == {
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
    }


def test_mask_shape_mismatch_is_rejected() -> None:
    img = Image.new("RGB", (32, 16), (255, 255, 255))
    with pytest.raises(ValueError, match="mask shape"):
        quality_measures(img, np.ones((32, 32), dtype=bool))


# ---------------------------------------------------------------------------
# ben_graham
# ---------------------------------------------------------------------------


def test_ben_graham_keeps_shape_and_zeroes_outside_mask() -> None:
    img = make_fundus(seed=6, size=96).resize((64, 64), Image.Resampling.BOX)
    out = ben_graham(img, sigma=3.0, threshold=THRESHOLD)
    assert out.size == img.size
    assert out.mode == "RGB"
    outside = ~fundus_mask(img, THRESHOLD)
    assert outside.any()
    assert np.all(np.asarray(out)[outside] == 0)


def test_ben_graham_flat_region_maps_to_mid_grey() -> None:
    # A flat region has image minus blur equal to zero, so the output is the offset 128.
    img = make_featureless_disc(size=128)
    out = np.asarray(ben_graham(img, sigma=3.0, threshold=THRESHOLD)).astype(int)
    centre = out[64, 64]
    assert np.all(np.abs(centre - 128) <= 1)


def test_ben_graham_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        ben_graham(make_fundus(seed=7, size=32), sigma=0.0, threshold=THRESHOLD)
