"""Tests for the learning-rate schedule (warm-up then cosine, stepped per optimiser step)."""

from __future__ import annotations

import pytest
import torch

from drgrade.train_loop import lr_factor


def test_warmup_rises_linearly_to_one() -> None:
    """During warm-up the factor is (step + 1) / warmup_steps: equal increments that reach 1."""
    warmup = 10
    factors = [lr_factor(step, warmup, 100) for step in range(warmup)]
    steps = [f * warmup for f in factors]
    assert steps == pytest.approx([float(k + 1) for k in range(warmup)])
    diffs = [b - a for a, b in zip(factors, factors[1:])]
    assert all(d == pytest.approx(diffs[0]) for d in diffs)
    assert factors[-1] == pytest.approx(1.0)


def test_cosine_starts_at_one_and_ends_near_zero() -> None:
    """After warm-up the factor starts at 1 and falls monotonically to a value near 0."""
    warmup, total = 10, 1000
    assert lr_factor(warmup, warmup, total) == pytest.approx(1.0)
    tail = [lr_factor(step, warmup, total) for step in range(warmup, total)]
    assert all(b <= a + 1e-12 for a, b in zip(tail, tail[1:]))
    assert lr_factor(total - 1, warmup, total) < 1e-4


def test_no_warmup_starts_at_one() -> None:
    """With no warm-up the first step uses the full learning rate."""
    assert lr_factor(0, 0, 50) == pytest.approx(1.0)


def test_invalid_arguments_raise() -> None:
    """Warm-up must be shorter than the whole schedule."""
    with pytest.raises(ValueError):
        lr_factor(0, 10, 10)
    with pytest.raises(ValueError):
        lr_factor(0, 0, 0)


def test_scheduler_drives_adamw_learning_rate() -> None:
    """LambdaLR with the factor gives the expected AdamW learning rate at each step."""
    from functools import partial

    from torch.optim.lr_scheduler import LambdaLR

    param = torch.nn.Parameter(torch.zeros(1))
    optimiser = torch.optim.AdamW([param], lr=3e-4)
    scheduler = LambdaLR(optimiser, partial(lr_factor, warmup_steps=2, total_steps=6))
    seen = [optimiser.param_groups[0]["lr"]]
    for _ in range(6):
        optimiser.step()
        scheduler.step()
        seen.append(optimiser.param_groups[0]["lr"])
    assert seen[0] == pytest.approx(3e-4 * 0.5)
    assert seen[1] == pytest.approx(3e-4)
    assert seen[-1] == pytest.approx(0.0, abs=1e-12)
