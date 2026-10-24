"""Tests for the explainability checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import timm
import torch

from drgrade.explain_checks import (
    area_share,
    blurred,
    brightest_region,
    cascade_stages,
    composite,
    edge_band,
    lesion_comparison,
    mass_share,
    masked_spearman,
    random_fraction_mask,
    randomised_copies,
    spread_50,
    ssim,
    top_fraction_mask,
)
from drgrade.io import LESION_COLS


def _fundus_mask(size: int) -> np.ndarray:
    """Boolean mask of the fundus circle, centre (size - 1) / 2, radius size / 2."""
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[:size, :size]
    return (yy - centre) ** 2 + (xx - centre) ** 2 <= (size / 2.0) ** 2


def _fundus(size: int, value: int = 60) -> np.ndarray:
    """A flat uint8 RGB image with the fundus circle filled and the corners black."""
    image = np.full((size, size, 3), value, dtype=np.uint8)
    image[~_fundus_mask(size)] = 0
    return image


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_ssim_identity_symmetry_and_noise() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((64, 64))
    noise = rng.normal(0.0, 0.2, a.shape)
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-9)
    assert ssim(a, a + noise) == pytest.approx(ssim(a + noise, a), abs=1e-12)
    assert ssim(a, a + noise) < ssim(a, a)
    assert ssim(a, a + noise) < ssim(a, a + 0.5 * noise)


def test_ssim_constant_images_reduce_to_luminance_term() -> None:
    m1, m2 = 0.3, 0.7
    c1 = (0.01 * 1.0) ** 2
    a = np.full((32, 32), m1)
    b = np.full((32, 32), m2)
    expected = (2 * m1 * m2 + c1) / (m1 * m1 + m2 * m2 + c1)
    assert ssim(a, b) == pytest.approx(expected, rel=1e-9)


def test_ssim_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ssim(np.zeros((8, 8)), np.zeros((8, 9)))


def test_masked_spearman_identity_reversed_constant_and_outside_mask() -> None:
    mask = np.ones((10, 10), dtype=bool)
    values = np.arange(100, dtype=float).reshape(10, 10)
    assert masked_spearman(values, values, mask) == pytest.approx(1.0)
    assert masked_spearman(values, -values, mask) == pytest.approx(-1.0)
    assert np.isnan(masked_spearman(np.ones((10, 10)), values, mask))

    # Pixels outside the mask must not influence the result.
    partial = np.zeros((10, 10), dtype=bool)
    partial[2:8, 2:8] = True
    other = values.copy()
    other[~partial] = np.random.default_rng(1).random(int((~partial).sum()))
    assert masked_spearman(values, other, partial) == pytest.approx(1.0)


def test_cascade_stages_efficientnet_names_and_order() -> None:
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    names = [name for name, _ in cascade_stages(model, "efficientnet_b0")]
    expected = ["classifier", "conv_head"] + [f"blocks.{i}" for i in range(6, -1, -1)] + ["stem"]
    assert names == expected


def test_cascade_stages_resnet_names_and_order() -> None:
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    names = [name for name, _ in cascade_stages(model, "resnet18")]
    assert names == ["fc", "layer4", "layer3", "layer2", "layer1", "stem"]


def test_cascade_stages_unknown_architecture_raises() -> None:
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    with pytest.raises(ValueError):
        cascade_stages(model, "vit_tiny")


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _changed_keys(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> set[str]:
    return {k for k in a if not torch.equal(a[k], b[k])}


def test_randomised_copies_cascade_on_resnet18() -> None:
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    original = _state(model)
    copies = list(randomised_copies(model, "resnet18", seed=123))

    assert len(copies) == 6
    first_name, first_model = copies[0]
    assert first_name == "fc"
    changed = _changed_keys(original, _state(first_model))
    assert changed and all(key.startswith("fc.") for key in changed)

    last_name, last_model = copies[-1]
    assert last_name == "stem"
    original_modules = dict(model.named_modules())
    for name, module in last_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            assert not torch.equal(module.weight.detach(), original_modules[name].weight.detach())

    # The input model is never modified.
    assert _changed_keys(original, _state(model)) == set()


def test_randomised_copies_same_seed_gives_same_weights_and_eval_mode() -> None:
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    run_a = list(randomised_copies(model, "resnet18", seed=7))
    run_b = list(randomised_copies(model, "resnet18", seed=7))
    for (name_a, copy_a), (name_b, copy_b) in zip(run_a, run_b):
        assert name_a == name_b
        assert _changed_keys(_state(copy_a), _state(copy_b)) == set()
        assert not any(m.training for m in copy_a.modules())


# ---------------------------------------------------------------------------
# Deletion test
# ---------------------------------------------------------------------------


def test_top_fraction_mask_counts_region_and_extremes() -> None:
    rng = np.random.default_rng(2)
    saliency = rng.random((20, 20))
    region = np.zeros((20, 20), dtype=bool)
    region[4:16, 3:17] = True
    n = int(region.sum())

    mask = top_fraction_mask(saliency, region, 0.25)
    assert mask.sum() == round(0.25 * n)
    assert not (mask & ~region).any()

    chosen_values = saliency[mask]
    unchosen_values = saliency[region & ~mask]
    assert chosen_values.min() >= unchosen_values.max()

    assert top_fraction_mask(saliency, region, 0.0).sum() == 0
    assert np.array_equal(top_fraction_mask(saliency, region, 1.0), region)


def test_top_fraction_mask_rejects_out_of_range_fraction() -> None:
    region = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        top_fraction_mask(np.zeros((4, 4)), region, 1.2)
    with pytest.raises(ValueError):
        top_fraction_mask(np.zeros((4, 4)), region, -0.1)


def test_top_fraction_mask_ties_are_deterministic() -> None:
    region = np.ones((4, 4), dtype=bool)
    saliency = np.ones((4, 4))
    first = top_fraction_mask(saliency, region, 0.25)
    second = top_fraction_mask(saliency, region, 0.25)
    assert np.array_equal(first, second)
    # Stable sort over row-major pixels keeps the first four pixels.
    assert np.flatnonzero(first.ravel()).tolist() == [0, 1, 2, 3]


def test_random_fraction_mask_count_region_and_reproducibility() -> None:
    region = np.zeros((30, 30), dtype=bool)
    region[5:25, 5:25] = True
    n = int(region.sum())
    mask_a = random_fraction_mask(region, 0.4, np.random.default_rng(9))
    mask_b = random_fraction_mask(region, 0.4, np.random.default_rng(9))
    assert mask_a.sum() == round(0.4 * n)
    assert not (mask_a & ~region).any()
    assert np.array_equal(mask_a, mask_b)


def test_blurred_keeps_outside_black_and_returns_uint8() -> None:
    image = _fundus(64, value=180)
    out = blurred(image, radius=2.0)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[-1, -1].tolist() == [0, 0, 0]
    assert out[32, 32].tolist() != [0, 0, 0]


def test_composite_replaces_exactly_the_masked_pixels() -> None:
    image = np.full((6, 6, 3), 10, dtype=np.uint8)
    replacement = np.full((6, 6, 3), 200, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:3, 2:5] = True
    out = composite(image, replacement, mask)
    assert out.dtype == np.uint8
    assert np.all(out[mask] == 200)
    assert np.all(out[~mask] == 10)
    assert np.all(image == 10)
    with pytest.raises(ValueError):
        composite(image, replacement[:5], mask)


# ---------------------------------------------------------------------------
# Shortcut checks
# ---------------------------------------------------------------------------


def test_edge_band_lies_inside_fundus_with_expected_area() -> None:
    size = 256
    frac = 0.1
    band = edge_band(size, frac)
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[:size, :size]
    inside = (yy - centre) ** 2 + (xx - centre) ** 2 <= (size / 2.0) ** 2
    assert not (band & ~inside).any()
    share = band.sum() / inside.sum()
    assert share == pytest.approx(1 - (1 - frac) ** 2, abs=0.02)


def test_brightest_region_finds_blob_and_ignores_edge_spot() -> None:
    size = 256
    image = _fundus(size, value=60)
    centre = (size - 1) / 2.0
    blob_row, blob_col = int(centre - 40), int(centre + 40)
    yy, xx = np.mgrid[:size, :size]
    # Gaussian-shaped blob, so the luminance has a single peak at the planted centre.
    squared = (yy - blob_row) ** 2 + (xx - blob_col) ** 2
    glow = 60 + 160 * np.exp(-squared / (2 * 8.0**2))
    grey = np.where(_fundus_mask(size), glow, 0.0).astype(np.uint8)
    image = np.repeat(grey[..., None], 3, axis=2)

    # Brighter spot planted inside the edge band (about 0.99 R from the centre).
    edge_row, edge_col = int(round(centre)), int(round(centre + 0.99 * size / 2.0))
    spot = (yy - edge_row) ** 2 + (xx - edge_col) ** 2 <= 3**2
    image[spot] = (255, 255, 255)

    row, col, mask = brightest_region(image, radius_frac=0.1, smooth_frac=0.01, edge_frac=0.05)
    assert abs(row - blob_row) <= 3 and abs(col - blob_col) <= 3
    assert mask[blob_row, blob_col]
    assert mask.sum() < _fundus_mask(size).sum()
    assert not mask[edge_row, edge_col]
    assert mask.dtype == bool


def test_mass_and_area_share_on_hand_example() -> None:
    saliency = np.array([[1.0, 1.0], [2.0, 4.0]])
    region = np.ones((2, 2), dtype=bool)
    within = np.array([[True, False], [False, True]])
    assert mass_share(saliency, region, within) == pytest.approx(5.0 / 8.0)
    assert area_share(region, within) == pytest.approx(0.5)
    assert np.isnan(mass_share(np.zeros((2, 2)), region, within))


def test_spread_50_uniform_and_single_hot_pixel() -> None:
    n_side = 10
    region = np.ones((n_side, n_side), dtype=bool)
    uniform = np.ones((n_side, n_side))
    assert spread_50(uniform, region) == pytest.approx(0.5, abs=0.02)

    hot = np.zeros((n_side, n_side))
    hot[3, 4] = 1.0
    assert spread_50(hot, region) == pytest.approx(1.0 / (n_side * n_side))
    assert np.isnan(spread_50(np.zeros((n_side, n_side)), region))


# ---------------------------------------------------------------------------
# Lesion comparison
# ---------------------------------------------------------------------------


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in LESION_COLS:
        data[col] = (rng.random(n) < 0.5).astype(int)
    data["peak"] = rng.normal(0.0, 1.0, n)
    data["spread"] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(data)


def test_lesion_comparison_shape_and_nan_below_min_group() -> None:
    frame = _frame(60)
    result = lesion_comparison(frame, LESION_COLS, ["peak", "spread"])
    assert len(result) == len(LESION_COLS) * 2
    assert list(result.columns) == [
        "lesion",
        "measure",
        "n_with",
        "n_without",
        "median_with",
        "median_without",
        "mann_whitney_p",
    ]

    tiny = frame.copy()
    tiny.loc[:, LESION_COLS[0]] = 0
    tiny.loc[:2, LESION_COLS[0]] = 1
    small = lesion_comparison(tiny, [LESION_COLS[0]], ["peak"], min_group=5)
    row = small.iloc[0]
    assert row["n_with"] == 3
    assert np.isnan(row["median_with"])
    assert np.isnan(row["mann_whitney_p"])


def test_lesion_comparison_detects_clear_difference() -> None:
    frame = _frame(80, seed=3)
    lesion = LESION_COLS[0]
    frame["peak"] = np.where(frame[lesion] == 1, 5.0, 0.0) + np.linspace(0, 0.1, len(frame))
    result = lesion_comparison(frame, [lesion], ["peak"])
    row = result.iloc[0]
    assert row["mann_whitney_p"] < 0.001
    assert row["median_with"] > row["median_without"]


def test_ssim_matches_skimage_when_available() -> None:
    """Our SSIM equals scikit-image's with the same settings (skipped when it is not installed)."""
    skimage_metrics = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(3)
    a = rng.random((64, 64))
    b = np.clip(a + rng.normal(0.0, 0.2, a.shape), 0.0, 1.0)
    reference = skimage_metrics.structural_similarity(
        a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0
    )
    assert ssim(a, b) == pytest.approx(reference, abs=1e-9)
