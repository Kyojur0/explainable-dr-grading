"""Checks that turn saliency maps into evidence.

Three groups of helpers:

* Similarity for the model randomisation test (Adebayo et al., 2018): masked Spearman
  rank correlation, SSIM (Wang et al., 2004), and the cascading re-initialisation of a
  copy of the network from the top layers down.
* Deletion test: choose the most salient or a random share of fundus pixels, blur the
  image, and composite the replacement back in.
* Shortcut check and lesion relation: the fundus edge band, a brightness proxy for the
  optic disc, mass and area shares, the spread of the saliency mass, and a per-lesion
  comparison of heatmap statistics.

Maps are 2-D float arrays (H, W). Masks are boolean arrays of the same shape. Images are
uint8 RGB arrays (H, W, 3).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu, spearmanr
from torch import nn

from drgrade.preprocess import apply_circle_mask

SSIM_K1 = 0.01
SSIM_K2 = 0.03
SSIM_SIGMA = 1.5
SSIM_TRUNCATE = 3.5
SSIM_MODE = "reflect"

LUMA_RED = 0.299
LUMA_GREEN = 0.587
LUMA_BLUE = 0.114
RGB_CHANNELS = 3
SPREAD_TARGET = 0.5  # share of the saliency mass that defines the spread
EFFICIENTNET_PREFIX = "efficientnet"
RESNET_PREFIX = "resnet"
RESNET_LAYERS: tuple[str, ...] = ("layer4", "layer3", "layer2", "layer1")
EFFICIENTNET_STEM_CONV = "conv_stem"
EFFICIENTNET_STEM_BN = "bn1"
EFFICIENTNET_HEAD_CONV = "conv_head"
EFFICIENTNET_HEAD_BN = "bn2"
RESNET_STEM_CONV = "conv1"
RESNET_STEM_BN = "bn1"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def masked_spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Spearman rank correlation between two maps over the pixels inside `mask`.

    Returns nan when fewer than two pixels lie inside the mask, or when either map is
    constant inside it (the rank correlation is then undefined).
    """
    if not (a.shape == b.shape == mask.shape):
        raise ValueError(f"shape mismatch: a {a.shape}, b {b.shape}, mask {mask.shape}")
    x = np.asarray(a, dtype=np.float64)[mask]
    y = np.asarray(b, dtype=np.float64)[mask]
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y)[0])


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Mean structural similarity between two 2-D maps (Wang et al., 2004).

    Local statistics come from a Gaussian window (sigma 1.5, truncated at 3.5 sigma,
    reflected at the borders) with K1 = 0.01 and K2 = 0.03. Covariances are population
    (not sample) covariances. The result is the mean of the SSIM map over the pixels where the whole window fits (a border of
    5 pixels is left out, as skimage does).

    Matches (checked numerically in tests/test_explain_checks.py when scikit-image is
    installed)
    `skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=1)`.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2

    def window(z: np.ndarray) -> np.ndarray:
        return gaussian_filter(z, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode=SSIM_MODE)

    mu_x = window(x)
    mu_y = window(y)
    var_x = window(x * x) - mu_x * mu_x
    var_y = window(y * y) - mu_y * mu_y
    cov = window(x * y) - mu_x * mu_y
    luminance = (2.0 * mu_x * mu_y + c1) / (mu_x * mu_x + mu_y * mu_y + c1)
    contrast_structure = (2.0 * cov + c2) / (var_x + var_y + c2)
    ssim_map = luminance * contrast_structure
    # Like skimage, average only where the full window fits: drop (win - 1) / 2 pixels at
    # each border, with win = 2 * int(truncate * sigma + 0.5) + 1 (11 for sigma 1.5).
    pad = int(SSIM_TRUNCATE * SSIM_SIGMA + 0.5)
    if min(x.shape) > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]
    return float(ssim_map.mean())


def cascade_stages(model: nn.Module, arch: str) -> list[tuple[str, list[nn.Module]]]:
    """Return the named layer groups of a timm EfficientNet or ResNet, top layers first.

    Each entry is (stage name, modules). Randomising a stage means re-initialising the
    modules it lists. Raises ValueError for any other architecture name.
    """
    if arch.startswith(EFFICIENTNET_PREFIX):
        blocks = model.blocks
        stages: list[tuple[str, list[nn.Module]]] = [
            ("classifier", [model.classifier]),
            ("conv_head", [getattr(model, EFFICIENTNET_HEAD_CONV), getattr(model, EFFICIENTNET_HEAD_BN)]),
        ]
        for index in reversed(range(len(blocks))):
            stages.append((f"blocks.{index}", [blocks[index]]))
        stages.append(
            ("stem", [getattr(model, EFFICIENTNET_STEM_CONV), getattr(model, EFFICIENTNET_STEM_BN)])
        )
        return stages
    if arch.startswith(RESNET_PREFIX):
        stages = [("fc", [model.fc])]
        for name in RESNET_LAYERS:
            stages.append((name, [getattr(model, name)]))
        stages.append(("stem", [getattr(model, RESNET_STEM_CONV), getattr(model, RESNET_STEM_BN)]))
        return stages
    raise ValueError(f"no cascade definition for architecture {arch!r}")


def randomised_copies(model: nn.Module, arch: str, seed: int) -> Iterator[tuple[str, nn.Module]]:
    """Yield copies of `model` with stages re-initialised from the top layers down.

    The first copy has only the top stage re-initialised, the next copy the top two, and
    so on, so each stage is randomised on top of the earlier ones. Stage k is initialised
    under `torch.manual_seed(seed + k)`, so the same seed gives the same copies. The input
    model is never modified. Each yielded copy is a new deep copy in eval mode, so callers
    may keep it.
    """
    current = copy.deepcopy(model)
    for index, (name, modules) in enumerate(cascade_stages(current, arch)):
        torch.manual_seed(seed + index)
        for module in modules:
            for sub in module.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
                if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                    sub.reset_running_stats()
        yield name, copy.deepcopy(current).eval()


# ---------------------------------------------------------------------------
# Deletion test
# ---------------------------------------------------------------------------


def _count_in_region(region: np.ndarray, fraction: float) -> int:
    """Return the number of pixels to select: round(fraction * region size)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return int(round(fraction * int(region.sum())))


def top_fraction_mask(saliency: np.ndarray, region: np.ndarray, fraction: float) -> np.ndarray:
    """Select the most salient `fraction` of the pixels inside `region`.

    Exactly round(fraction * region pixels) pixels are chosen. Ties are broken by a stable
    sort over the region pixels in row-major order, so the result is deterministic.
    """
    if saliency.shape != region.shape:
        raise ValueError(f"shape mismatch: saliency {saliency.shape}, region {region.shape}")
    count = _count_in_region(region, fraction)
    flat_index = np.flatnonzero(region)
    values = np.asarray(saliency, dtype=np.float64).ravel()[flat_index]
    order = np.argsort(-values, kind="stable")
    mask = np.zeros(region.size, dtype=bool)
    mask[flat_index[order[:count]]] = True
    return mask.reshape(region.shape)


def random_fraction_mask(region: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Select `round(fraction * region pixels)` pixels uniformly at random inside `region`."""
    count = _count_in_region(region, fraction)
    flat_index = np.flatnonzero(region)
    chosen = rng.permutation(flat_index)[:count]
    mask = np.zeros(region.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(region.shape)


def blurred(image: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian-blur a uint8 RGB fundus image, then black out the outside of the fundus.

    Uses PIL `ImageFilter.GaussianBlur(radius)` and `apply_circle_mask`, so the corners
    stay black as they are in the preprocessed images. Returns a uint8 array.
    """
    _check_rgb(image)
    soft = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius))
    return np.array(apply_circle_mask(soft), dtype=np.uint8)


def composite(image: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a uint8 copy of `image` with `replacement` pixels where `mask` is True."""
    if image.shape != replacement.shape:
        raise ValueError(f"shape mismatch: image {image.shape}, replacement {replacement.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")
    out = np.array(image, dtype=np.uint8, copy=True)
    out[mask] = replacement[mask]
    return out


def _check_rgb(image: np.ndarray) -> None:
    """Raise ValueError unless `image` is a uint8 (H, W, 3) array."""
    if image.ndim != 3 or image.shape[2] != RGB_CHANNELS or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB array (H, W, 3), got {image.dtype} {image.shape}")


# ---------------------------------------------------------------------------
# Shortcut checks
# ---------------------------------------------------------------------------


def _fundus_geometry(size: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (inside, distance, radius) for the fundus circle of a square image.

    Same circle as `apply_circle_mask`: centre (size - 1) / 2, radius size / 2, inside when
    the squared distance is at most the squared radius.
    """
    centre = (size - 1) / 2.0
    radius = size / 2.0
    yy, xx = np.mgrid[:size, :size]
    dist_sq = (yy - centre) ** 2 + (xx - centre) ** 2
    inside = dist_sq <= radius * radius
    return inside, np.sqrt(dist_sq), radius


def edge_band(size: int, frac: float) -> np.ndarray:
    """Pixels inside the fundus circle that lie within `frac` of its radius from the edge.

    A pixel is in the band when it is inside the circle and its distance from the centre
    is greater than (1 - frac) * R. The band's area is about 1 - (1 - frac)^2 of the disc.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    inside, distance, radius = _fundus_geometry(size)
    return inside & (distance > (1.0 - frac) * radius)


def brightest_region(
    image: np.ndarray,
    radius_frac: float,
    smooth_frac: float,
    edge_frac: float,
) -> tuple[int, int, np.ndarray]:
    """Locate the brightest structure away from the fundus edge.

    A proxy for the optic disc (the brightest structure in a fundus photograph), not a
    detector. Luminance (0.299 R + 0.587 G + 0.114 B) is smoothed with a Gaussian of sigma
    smooth_frac * R. The argmax is taken over fundus pixels outside `edge_band(size, edge_frac)`,
    because rim glare is not the optic disc. Returns (row, col, mask), where mask is a disc of
    radius radius_frac * R around that point, intersected with the fundus.
    """
    _check_rgb(image)
    height, width = image.shape[:2]
    if height != width:
        raise ValueError(f"brightest_region needs a square image, got {image.shape[:2]}")
    pixels = image.astype(np.float64)
    luminance = LUMA_RED * pixels[..., 0] + LUMA_GREEN * pixels[..., 1] + LUMA_BLUE * pixels[..., 2]
    inside, _, radius = _fundus_geometry(height)
    smooth = gaussian_filter(luminance, sigma=smooth_frac * radius)
    candidate = inside & ~edge_band(height, edge_frac)
    if not candidate.any():
        raise ValueError("no fundus pixels remain outside the edge band")
    scores = np.where(candidate, smooth, -np.inf)
    row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
    row, col = int(row), int(col)
    yy, xx = np.mgrid[:height, :width]
    disc = (yy - row) ** 2 + (xx - col) ** 2 <= (radius_frac * radius) ** 2
    return row, col, disc & inside


def mass_share(saliency: np.ndarray, region: np.ndarray, within: np.ndarray) -> float:
    """Share of the saliency mass inside `region` that also lies in `within`.

    Returns nan when the saliency summed over `region` is 0.
    """
    total = float(saliency[region].sum())
    if total == 0.0:
        return float("nan")
    return float(saliency[region & within].sum()) / total


def area_share(region: np.ndarray, within: np.ndarray) -> float:
    """Share of the pixels in `region` that also lie in `within`. Nan for an empty region."""
    size = int(region.sum())
    if size == 0:
        return float("nan")
    return float((region & within).sum()) / size


# ---------------------------------------------------------------------------
# Spread and peak
# ---------------------------------------------------------------------------


def spread_50(saliency: np.ndarray, region: np.ndarray) -> float:
    """Smallest share of the region's pixels that holds 50% of its summed saliency.

    The values are sorted in descending order, and the first index whose cumulative sum
    reaches 50% of the total gives the share (k + 1) / n. Nan when the total is 0.
    """
    values = np.asarray(saliency, dtype=np.float64)[region]
    total = float(values.sum())
    if values.size == 0 or total == 0.0:
        return float("nan")
    cumulative = np.cumsum(np.sort(values)[::-1])
    k = int(np.argmax(cumulative >= SPREAD_TARGET * total))
    return (k + 1) / values.size


def lesion_comparison(
    frame: pd.DataFrame,
    lesion_cols: Sequence[str],
    value_cols: Sequence[str],
    min_group: int = 5,
) -> pd.DataFrame:
    """Compare heatmap statistics between images with and without each lesion.

    Lesion columns hold 0 or 1. For each (lesion, measure) pair the non-missing values
    are split by the lesion flag. The result has one row per pair with columns lesion,
    measure, n_with, n_without, median_with, median_without and mann_whitney_p (two-sided
    `scipy.stats.mannwhitneyu`). Medians and p are nan when either group has fewer than
    `min_group` values.
    """
    rows: list[dict[str, object]] = []
    for lesion in lesion_cols:
        flags = frame[lesion]
        if not set(flags.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"lesion column {lesion!r} must hold only 0 or 1")
        present = flags == 1
        absent = flags == 0
        for measure in value_cols:
            with_values = frame.loc[present, measure].dropna().to_numpy(dtype=np.float64)
            without_values = frame.loc[absent, measure].dropna().to_numpy(dtype=np.float64)
            enough = with_values.size >= min_group and without_values.size >= min_group
            p_value = float("nan")
            if enough:
                p_value = float(mannwhitneyu(with_values, without_values, alternative="two-sided").pvalue)
            rows.append(
                {
                    "lesion": lesion,
                    "measure": measure,
                    "n_with": int(with_values.size),
                    "n_without": int(without_values.size),
                    "median_with": float(np.median(with_values)) if with_values.size >= min_group else float("nan"),
                    "median_without": (
                        float(np.median(without_values)) if without_values.size >= min_group else float("nan")
                    ),
                    "mann_whitney_p": p_value,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "lesion",
            "measure",
            "n_with",
            "n_without",
            "median_with",
            "median_without",
            "mann_whitney_p",
        ],
    )
