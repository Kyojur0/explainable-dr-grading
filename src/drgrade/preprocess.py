"""Image preprocessing for the fundus photographs (brief 6.1 and 15).

Per image, `preprocess_image` converts to RGB, crops the black border, pads to a
square, resizes with area (BOX) interpolation and, only when the config flag is
set, applies Ben Graham normalisation (off by default, DECISIONS.md D11).

`fundus_mask`, `quality_measures` and `ben_graham` work on the processed image.
Their definitions are in DECISIONS.md D10 and D11.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from drgrade.config import DataConfig

BBox = tuple[int, int, int, int]  # (left, top, right, bottom), PIL convention

BEN_GRAHAM_GAIN = 4.0  # weight of the image minus its blur (Ben Graham 2015 method)
BEN_GRAHAM_OFFSET = 128.0  # grey level added so that flat regions map to mid-grey
LAPLACIAN_CENTRE_WEIGHT = 4.0  # 3x3 Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]


def fundus_bbox(
    img: Image.Image,
    threshold: int,
    margin_frac: float,
    min_area_frac: float,
) -> tuple[BBox, bool]:
    """Find the bounding box of the fundus, ignoring the black border.

    A pixel belongs to the fundus when its greyscale value is strictly above
    `threshold`. The tight box is padded by `margin_frac` of its longer side and
    clipped to the image. If no pixel exceeds the threshold, or the tight box covers
    less than `min_area_frac` of the image, the full image is returned with
    crop_failed=True.
    """
    grey = np.asarray(img.convert("L"))
    height, width = grey.shape
    full_box: BBox = (0, 0, width, height)

    mask = grey > threshold
    if not mask.any():
        return full_box, True

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    tight_area = (right - left) * (bottom - top)
    if tight_area < min_area_frac * width * height:
        return full_box, True

    margin = int(round(margin_frac * max(right - left, bottom - top)))
    box: BBox = (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )
    return box, False


def border_level(img: Image.Image, corner_frac: float = 0.02) -> tuple[int, int]:
    """Return (level, spread) of the background read from the four image corners.

    Each corner patch is a square of side max(4, corner_frac * shorter side). Level
    is the median of the four patch medians (greyscale); spread is the largest minus
    the smallest patch median. Corners lie outside the fundus disc in fundus
    photographs, so they measure the camera background even when the disc touches
    the image edges (DECISIONS.md D13).
    """
    grey = np.asarray(img.convert("L"))
    height, width = grey.shape
    side = max(4, int(corner_frac * min(height, width)))
    patches = (
        grey[:side, :side],
        grey[:side, width - side :],
        grey[height - side :, :side],
        grey[height - side :, width - side :],
    )
    medians = [int(np.median(patch)) for patch in patches]
    return int(np.median(medians)), max(medians) - min(medians)


def blacken_background(
    img: Image.Image,
    bg_detect_min: int,
    bg_tolerance: int,
    bg_corner_spread: int,
) -> tuple[Image.Image, bool]:
    """Set a uniform non-black camera background (dark grey or white) to black.

    Applied only when the four corners agree (spread at most `bg_corner_spread`) on a
    level of at least `bg_detect_min`. A pixel is background when its greyscale value
    is within `bg_tolerance` of that level AND it can be reached in a straight line
    (left, right, up or down) from the image edge through such pixels. For a convex
    fundus disc this is its exterior, so findings inside the disc are never changed.

    Returns (image, changed). The input object is returned unchanged when no
    uniform background is detected.
    """
    level, spread = border_level(img)
    if level < bg_detect_min or spread > bg_corner_spread:
        return img, False
    rgb = img.convert("RGB")
    grey = np.asarray(rgb.convert("L")).astype(np.int16)
    near = np.abs(grey - level) <= bg_tolerance
    reach = (
        np.logical_and.accumulate(near, axis=1)
        | np.logical_and.accumulate(near[:, ::-1], axis=1)[:, ::-1]
        | np.logical_and.accumulate(near, axis=0)
        | np.logical_and.accumulate(near[::-1, :], axis=0)[::-1, :]
    )
    pixels = np.asarray(rgb).copy()
    pixels[reach] = 0
    return Image.fromarray(pixels), True


def crop_fundus(
    img: Image.Image,
    threshold: int,
    margin_frac: float,
    min_area_frac: float,
    bg_detect_min: int | None = None,
    bg_tolerance: int | None = None,
    bg_corner_spread: int | None = None,
) -> tuple[Image.Image, BBox, bool]:
    """Crop the black border from a fundus image.

    When the three bg_* settings are given, a uniform grey or white camera
    background is first set to black with `blacken_background`, so the same
    `threshold` works for every camera. If the crop on the blackened image fails,
    the original image is used instead (safety net). Returns (cropped image, box
    used, crop_failed). See `fundus_bbox`.
    """
    if None not in (bg_detect_min, bg_tolerance, bg_corner_spread):
        cleaned, changed = blacken_background(img, bg_detect_min, bg_tolerance, bg_corner_spread)
        if changed:
            box, crop_failed = fundus_bbox(cleaned, threshold, margin_frac, min_area_frac)
            if not crop_failed:
                return cleaned.crop(box), box, False
    box, crop_failed = fundus_bbox(img, threshold, margin_frac, min_area_frac)
    return img.crop(box), box, crop_failed


def pad_to_square(img: Image.Image) -> Image.Image:
    """Pad an image with black to a square canvas, centring the original."""
    width, height = img.size
    side = max(width, height)
    canvas = Image.new(img.mode, (side, side))  # new canvases default to black
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas


def apply_circle_mask(img: Image.Image) -> Image.Image:
    """Set every pixel outside the circle inscribed in a square image to black.

    The crop box carries a small margin around the fundus, so the inscribed circle
    is slightly larger than the disc and removes only the corners, where camera
    backgrounds, halos and labels differ between devices (DECISIONS.md D13).
    """
    width, height = img.size
    if width != height:
        raise ValueError(f"apply_circle_mask needs a square image, got {img.size}")
    centre = (width - 1) / 2.0
    yy, xx = np.mgrid[:height, :width]
    outside = (yy - centre) ** 2 + (xx - centre) ** 2 > (width / 2.0) ** 2
    pixels = np.asarray(img.convert("RGB")).copy()
    pixels[outside] = 0
    return Image.fromarray(pixels)


def resize_fundus(img: Image.Image, cfg: DataConfig) -> tuple[Image.Image, BBox, bool]:
    """Crop the black border, pad to square and resize to `cfg.target_size` (no normalisation).

    Steps: RGB, `crop_fundus` with the config settings, `pad_to_square`, then resize
    to (target_size, target_size) with Image.Resampling.BOX (area averaging), then,
    when `cfg.circle_mask` is true, black out the corners outside the inscribed circle.

    Returns (resized image, crop box in original pixel coordinates, crop_failed).
    """
    rgb = img.convert("RGB")
    cropped, box, crop_failed = crop_fundus(
        rgb,
        cfg.crop_threshold,
        cfg.crop_margin_frac,
        cfg.crop_min_area_frac,
        cfg.bg_detect_min,
        cfg.bg_tolerance,
        cfg.bg_corner_spread,
    )
    square = pad_to_square(cropped)
    resized = square.resize((cfg.target_size, cfg.target_size), Image.Resampling.BOX)
    if cfg.circle_mask:
        resized = apply_circle_mask(resized)
    return resized, box, crop_failed


def preprocess_image(img: Image.Image, cfg: DataConfig) -> tuple[Image.Image, BBox, bool]:
    """Turn one original photograph into the image that is saved to disk.

    `resize_fundus`, then `ben_graham` only when `cfg.ben_graham` is true.
    Returns (processed image, crop box in original pixel coordinates, crop_failed).
    The pipeline script computes quality measures on the image before normalisation
    (DECISIONS.md D10), so it calls `resize_fundus` directly.
    """
    resized, box, crop_failed = resize_fundus(img, cfg)
    if cfg.ben_graham:
        resized = ben_graham(resized, cfg.ben_graham_sigma, cfg.crop_threshold)
    return resized, box, crop_failed


def fundus_mask(img: Image.Image, threshold: int) -> np.ndarray:
    """Return a boolean mask of the fundus: greyscale value strictly above `threshold`.

    The mask has the image's (height, width) shape. The pipeline calls it on the
    processed image (after resizing, before any Ben Graham normalisation).
    """
    grey = np.asarray(img.convert("L"))
    return grey > threshold


def _erode_3x3(mask: np.ndarray) -> np.ndarray:
    """Return True only where the full 3x3 neighbourhood is True in `mask`.

    Pixels on the image border are never kept, because their neighbourhood is
    cut off by the image edge.
    """
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return eroded


def quality_measures(img: Image.Image, mask: np.ndarray) -> dict[str, float]:
    """Compute three image-quality measures inside the fundus mask.

    * ``brightness``: mean of the HSV V channel (V = max of R, G, B), 0 to 255.
    * ``contrast``: RMS contrast, the standard deviation of greyscale values
      (ITU-R 601 greyscale from Pillow), 0 to 255.
    * ``sharpness``: variance of a 3x3 Laplacian of the greyscale image
      (kernel [[0,1,0],[1,-4,1],[0,1,0]], computed with numpy slicing). The
      Laplacian is evaluated only where the full 3x3 neighbourhood lies inside
      the mask, so the hard black-to-fundus edge of the disc does not dominate.

    Brightness and contrast use every masked pixel. If the mask is empty, all
    three values are 0.0. The sharpness value is 0.0 when no pixel survives the
    erosion, for example for a flat disc.
    """
    if mask.shape != (img.size[1], img.size[0]):
        raise ValueError(f"mask shape {mask.shape} does not match image size {img.size}")
    if not mask.any():
        return {"brightness": 0.0, "contrast": 0.0, "sharpness": 0.0}

    value = np.asarray(img.convert("HSV"))[..., 2].astype(np.float64)
    grey = np.asarray(img.convert("L")).astype(np.float64)

    brightness = float(value[mask].mean())
    contrast = float(grey[mask].std())

    valid = _erode_3x3(mask)[1:-1, 1:-1]
    laplacian = (
        grey[:-2, 1:-1]
        + grey[2:, 1:-1]
        + grey[1:-1, :-2]
        + grey[1:-1, 2:]
        - LAPLACIAN_CENTRE_WEIGHT * grey[1:-1, 1:-1]
    )
    sharpness = float(laplacian[valid].var()) if valid.any() else 0.0
    return {"brightness": brightness, "contrast": contrast, "sharpness": sharpness}


def ben_graham(img: Image.Image, sigma: float, threshold: int) -> Image.Image:
    """Apply Ben Graham local colour normalisation inside the fundus mask.

    Output = 4 * image - 4 * GaussianBlur(image, sigma) + 128, clipped to 0 to 255.
    Pixels outside the fundus mask (greyscale not above `threshold`) are set to 0.
    `sigma` is the Gaussian standard deviation in pixels, as Pillow's blur radius.

    Off by default (config `ben_graham: false`). Not used unless an ablation is run
    (DECISIONS.md D11).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    rgb = img.convert("RGB")
    source = np.asarray(rgb, dtype=np.float64)
    smooth = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float64)
    normalised = np.clip(BEN_GRAHAM_GAIN * (source - smooth) + BEN_GRAHAM_OFFSET, 0.0, 255.0)
    normalised[~fundus_mask(rgb, threshold)] = 0.0
    return Image.fromarray(np.rint(normalised).astype(np.uint8))
