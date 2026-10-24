"""Stage 4: loss functions (brief 8.2).

ce is plain cross-entropy. wce is cross-entropy with class weights
w_c = (N / n_c) ** 0.5, normalised to mean 1 over the classes, computed from the
training subset only (DECISIONS.md D15).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from drgrade.models import NUM_CLASSES
from drgrade.run_config import RunConfig


def class_weights(train_grades: Sequence[int], n_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return the square-root inverse-frequency weights, normalised to mean 1.

    Raises ValueError if the list is empty, holds a grade outside 0..n_classes-1,
    or leaves any class without an example (its weight would be infinite).
    """
    grades = np.asarray(list(train_grades), dtype=np.int64)
    if grades.size == 0:
        raise ValueError("no training grades given")
    if np.any((grades < 0) | (grades >= n_classes)):
        raise ValueError(f"grades must be in 0..{n_classes - 1}")
    counts = np.bincount(grades, minlength=n_classes)
    empty = [c for c in range(n_classes) if counts[c] == 0]
    if empty:
        raise ValueError(f"classes {empty} have no training images; class weights are undefined")
    raw = (grades.size / counts) ** 0.5
    return raw / raw.mean()


def build_loss(cfg: RunConfig, train_grades: Sequence[int]) -> nn.CrossEntropyLoss:
    """Return the cross-entropy loss named by cfg.loss, with label smoothing from the config."""
    if cfg.loss == "ce":
        weight = None
    elif cfg.loss == "wce":
        weight = torch.as_tensor(class_weights(train_grades), dtype=torch.float32)
    else:  # pragma: no cover - RunConfig already rejects other names
        raise ValueError(f"unknown loss {cfg.loss!r}")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
