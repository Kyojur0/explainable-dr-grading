"""Tests for black-border cropping and square padding (brief 6.1)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from drgrade.preprocess import crop_fundus, fundus_bbox, pad_to_square

THRESHOLD = 10
MARGIN = 0.02
MIN_AREA = 0.30


def _disc_canvas(size: int, centre: int, radius: int) -> Image.Image:
    """Black square canvas with a bright filled disc."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        fill=(255, 255, 255),
    )
    return img


def test_disc_bbox_is_close_to_disc() -> None:
    img = _disc_canvas(size=300, centre=150, radius=100)
    box, crop_failed = fundus_bbox(img, THRESHOLD, MARGIN, MIN_AREA)
    assert not crop_failed
    # Disc spans 50..250 on both axes; the margin adds about 4 px.
    for coord, expected in zip(box, (50, 50, 250, 250)):
        assert abs(coord - expected) <= 6, (box, expected)


def test_crop_fundus_returns_box_sized_image() -> None:
    img = _disc_canvas(size=300, centre=150, radius=100)
    cropped, box, crop_failed = crop_fundus(img, THRESHOLD, MARGIN, MIN_AREA)
    assert not crop_failed
    assert cropped.size == (box[2] - box[0], box[3] - box[1])


def test_all_black_is_crop_failed() -> None:
    img = Image.new("RGB", (120, 80), (0, 0, 0))
    box, crop_failed = fundus_bbox(img, THRESHOLD, MARGIN, MIN_AREA)
    assert crop_failed
    assert box == (0, 0, 120, 80)


def test_tiny_bright_speck_is_crop_failed() -> None:
    img = Image.new("RGB", (300, 300), (0, 0, 0))
    ImageDraw.Draw(img).rectangle([10, 10, 12, 12], fill=(255, 255, 255))
    box, crop_failed = fundus_bbox(img, THRESHOLD, MARGIN, MIN_AREA)
    assert crop_failed
    assert box == (0, 0, 300, 300)


def test_threshold_is_strict() -> None:
    # A plateau exactly at the threshold value must not count as fundus.
    img = Image.new("RGB", (300, 300), (THRESHOLD, THRESHOLD, THRESHOLD))
    _, crop_failed = fundus_bbox(img, THRESHOLD, MARGIN, MIN_AREA)
    assert crop_failed


def test_pad_to_square_landscape() -> None:
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    padded = pad_to_square(img)
    assert padded.size == (200, 200)
    assert padded.getpixel((0, 0)) == (0, 0, 0)
    assert padded.getpixel((100, 100)) == (255, 0, 0)


def test_pad_to_square_portrait() -> None:
    img = Image.new("RGB", (100, 200), (0, 255, 0))
    padded = pad_to_square(img)
    assert padded.size == (200, 200)
    assert padded.getpixel((0, 100)) == (0, 0, 0)
    assert padded.getpixel((100, 100)) == (0, 255, 0)
