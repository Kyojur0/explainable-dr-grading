"""Run configuration for training (brief sections 7, 8 and 13.1).

A run config (configs/e1_*.yaml, e2_*.yaml, e3_*.yaml) is read into frozen
dataclasses. Missing keys, unknown keys and out-of-range values raise
RunConfigError, so a typo cannot silently fall back to a default.

The data paths live in configs/data.yaml (processed_dir and target_size). The run
config only names the split file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

LOSS_NAMES: tuple[str, ...] = ("ce", "wce")
OPTIMISER_NAMES: tuple[str, ...] = ("adamw",)
SCHEDULE_NAMES: tuple[str, ...] = ("cosine",)
SELECTION_METRICS: tuple[str, ...] = ("qwk",)

_T = TypeVar("_T")


class RunConfigError(ValueError):
    """Raised when a run configuration is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and pretrained weights (brief 8.1)."""

    id: str
    timm_name: str
    timm_pretrained_tag: str
    input_size: int
    dropout: float

    def __post_init__(self) -> None:
        """Check that the architecture name is the prefix of the pretrained tag."""
        if self.timm_pretrained_tag.split(".")[0] != self.timm_name:
            raise RunConfigError(
                f"timm_pretrained_tag {self.timm_pretrained_tag!r} does not belong to "
                f"timm_name {self.timm_name!r}"
            )
        if self.input_size < 1:
            raise RunConfigError(f"model.input_size must be positive, got {self.input_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise RunConfigError(f"model.dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class OptimiserConfig:
    """Optimiser settings (brief 8.3)."""

    name: str
    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        """Check the optimiser name and the numeric ranges."""
        if self.name not in OPTIMISER_NAMES:
            raise RunConfigError(f"optimiser.name must be one of {OPTIMISER_NAMES}, got {self.name!r}")
        if self.lr <= 0:
            raise RunConfigError(f"optimiser.lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise RunConfigError(f"optimiser.weight_decay must be non-negative, got {self.weight_decay}")


@dataclass(frozen=True)
class ScheduleConfig:
    """Learning-rate schedule (brief 8.3)."""

    name: str
    warmup_epochs: int

    def __post_init__(self) -> None:
        """Check the schedule name and the warm-up length."""
        if self.name not in SCHEDULE_NAMES:
            raise RunConfigError(f"schedule.name must be one of {SCHEDULE_NAMES}, got {self.name!r}")
        if self.warmup_epochs < 0:
            raise RunConfigError(f"schedule.warmup_epochs must be non-negative, got {self.warmup_epochs}")


@dataclass(frozen=True)
class AugConfig:
    """Training augmentation (brief 7). Colour is not augmented beyond brightness, contrast and gamma."""

    rotation_degrees: float  # rotation drawn uniformly from [-value, value]; 180 gives the full 360
    hflip_p: float
    vflip_p: float
    scale_min: float
    scale_max: float
    ratio_min: float
    ratio_max: float
    brightness: float
    contrast: float
    gamma_min: float
    gamma_max: float

    def __post_init__(self) -> None:
        """Check the ranges of the augmentation parameters."""
        for name in ("hflip_p", "vflip_p"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise RunConfigError(f"aug.{name} must be in [0, 1]")
        if self.rotation_degrees < 0:
            raise RunConfigError("aug.rotation_degrees must be non-negative")
        if not 0.0 < self.scale_min <= self.scale_max <= 1.0:
            raise RunConfigError("aug scale must satisfy 0 < scale_min <= scale_max <= 1")
        if not 0.0 < self.ratio_min <= self.ratio_max:
            raise RunConfigError("aug ratio must satisfy 0 < ratio_min <= ratio_max")
        if self.brightness < 0 or self.contrast < 0:
            raise RunConfigError("aug brightness and contrast must be non-negative")
        if not 0.0 < self.gamma_min <= self.gamma_max:
            raise RunConfigError("aug gamma must satisfy 0 < gamma_min <= gamma_max")


@dataclass(frozen=True)
class DataSection:
    """Split file for the run. Image and manifest paths come from configs/data.yaml."""

    split_file: str


@dataclass(frozen=True)
class RunConfig:
    """A complete training run, as read from configs/e*.yaml (brief 8.3 to 8.5)."""

    run_name: str
    model: ModelConfig
    loss: str
    label_smoothing: float
    optimiser: OptimiserConfig
    schedule: ScheduleConfig
    max_epochs: int
    early_stopping_patience: int
    batch_size: int
    accumulation_steps: int
    grad_clip_norm: float
    selection_metric: str
    seed: int
    num_workers: int
    aug: AugConfig
    data: DataSection

    def __post_init__(self) -> None:
        """Validate cross-field constraints so that bad runs fail before training starts."""
        if self.loss not in LOSS_NAMES:
            raise RunConfigError(f"loss must be one of {LOSS_NAMES}, got {self.loss!r}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise RunConfigError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.selection_metric not in SELECTION_METRICS:
            raise RunConfigError(
                f"selection_metric must be one of {SELECTION_METRICS}, got {self.selection_metric!r}"
            )
        if self.max_epochs < 1:
            raise RunConfigError(f"max_epochs must be at least 1, got {self.max_epochs}")
        if self.early_stopping_patience < 1:
            raise RunConfigError("early_stopping_patience must be at least 1")
        if self.batch_size < 1 or self.accumulation_steps < 1:
            raise RunConfigError("batch_size and accumulation_steps must be at least 1")
        if self.grad_clip_norm <= 0:
            raise RunConfigError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")
        if self.num_workers < 0:
            raise RunConfigError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.schedule.warmup_epochs >= self.max_epochs and self.schedule.warmup_epochs > 0:
            raise RunConfigError(
                f"warmup_epochs ({self.schedule.warmup_epochs}) must be less than max_epochs ({self.max_epochs})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the run as plain nested dictionaries, for config.yaml and checkpoints."""
        return dataclasses.asdict(self)

    def with_overrides(self, **changes: Any) -> RunConfig:
        """Return a copy with some fields replaced (for example max_epochs from the command line)."""
        return dataclasses.replace(self, **changes)


def _section(raw: Any, cls: type[_T], name: str) -> _T:
    """Build one dataclass section from a mapping, checking its keys exactly."""
    if not isinstance(raw, Mapping):
        raise RunConfigError(f"{name} must be a mapping")
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise RunConfigError(f"{name}: unknown keys {unknown}, missing keys {missing}")
    try:
        return cls(**raw)  # type: ignore[call-arg]
    except TypeError as exc:
        raise RunConfigError(f"{name}: {exc}") from exc


def parse_run_config(raw: Any) -> RunConfig:
    """Build a RunConfig from a parsed YAML mapping, checking every key."""
    if not isinstance(raw, Mapping):
        raise RunConfigError("run config must be a YAML mapping")
    top_fields = {f.name for f in fields(RunConfig)}
    unknown = sorted(set(raw) - top_fields)
    missing = sorted(top_fields - set(raw))
    if unknown or missing:
        raise RunConfigError(f"run config: unknown keys {unknown}, missing keys {missing}")
    values: dict[str, Any] = dict(raw)
    values["model"] = _section(raw["model"], ModelConfig, "model")
    values["optimiser"] = _section(raw["optimiser"], OptimiserConfig, "optimiser")
    values["schedule"] = _section(raw["schedule"], ScheduleConfig, "schedule")
    values["aug"] = _section(raw["aug"], AugConfig, "aug")
    values["data"] = _section(raw["data"], DataSection, "data")
    return RunConfig(**values)


def load_run_config(path: Path | str) -> RunConfig:
    """Read and validate a run configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise RunConfigError(f"run config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        return parse_run_config(raw)
    except RunConfigError as exc:
        raise RunConfigError(f"{config_path}: {exc}") from exc
