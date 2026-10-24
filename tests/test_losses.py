"""Tests for class weights and the loss factory (synthetic grades only)."""

from __future__ import annotations

import pytest
import torch

from drgrade.losses import build_loss, class_weights
from drgrade.run_config import (
    AugConfig,
    DataSection,
    ModelConfig,
    OptimiserConfig,
    RunConfig,
    ScheduleConfig,
)


def _run(loss: str, smoothing: float = 0.0) -> RunConfig:
    aug = AugConfig(180, 0.5, 0.5, 0.85, 1.0, 0.95, 1.05, 0.2, 0.2, 0.8, 1.2)
    return RunConfig(
        run_name="t",
        model=ModelConfig("M2", "efficientnet_b0", "efficientnet_b0.ra_in1k", 512, 0.2),
        loss=loss,
        label_smoothing=smoothing,
        optimiser=OptimiserConfig("adamw", 3e-4, 1e-4),
        schedule=ScheduleConfig("cosine", 1),
        max_epochs=20,
        early_stopping_patience=5,
        batch_size=16,
        accumulation_steps=1,
        grad_clip_norm=1.0,
        selection_metric="qwk",
        seed=42,
        num_workers=0,
        aug=aug,
        data=DataSection("splits/split_v1.csv"),
    )


def test_weights_follow_square_root_formula_and_have_mean_one() -> None:
    """Counts [4, 1, 1, 1, 1] give raw weights sqrt(N / n_c); normalised to mean 1."""
    grades = [0, 0, 0, 0, 1, 2, 3, 4]
    weights = class_weights(grades)
    raw = [(8 / 4) ** 0.5, (8 / 1) ** 0.5, (8 / 1) ** 0.5, (8 / 1) ** 0.5, (8 / 1) ** 0.5]
    mean = sum(raw) / 5
    expected = [r / mean for r in raw]
    assert weights.mean() == pytest.approx(1.0)
    assert list(weights) == pytest.approx(expected)
    assert weights[1] / weights[0] == pytest.approx(2.0)


def test_empty_class_raises() -> None:
    """A grade with no training image has no defined weight, so the call fails."""
    with pytest.raises(ValueError, match="no training images"):
        class_weights([0, 0, 1, 2, 3])


def test_out_of_range_grade_and_empty_input_raise() -> None:
    """Grades outside 0..4 and an empty list are rejected."""
    with pytest.raises(ValueError):
        class_weights([])
    with pytest.raises(ValueError):
        class_weights([0, 1, 2, 3, 5])


def test_ce_has_no_weights_and_wce_uses_them() -> None:
    """ce builds an unweighted loss; wce builds the weighted one, with label smoothing passed on."""
    grades = [0, 0, 0, 1, 2, 3, 4]
    plain = build_loss(_run("ce", smoothing=0.1), grades)
    assert plain.weight is None
    assert plain.label_smoothing == pytest.approx(0.1)
    weighted = build_loss(_run("wce"), grades)
    assert torch.allclose(weighted.weight, torch.as_tensor(class_weights(grades), dtype=torch.float32))
