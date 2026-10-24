"""Stage 8: explainability checks for one trained run.

Grad-CAM comes from src/drgrade/gradcam.py; the sanity checks, the deletion test, the shortcut
check and the lesion comparison use src/drgrade/explain_checks.py. Nothing here changes a model,
a threshold or a decision. Test images are read only to explain the locked test predictions:
each such read is preceded by the allowlist and predictions checks and an
"explain" entry in the access log (DECISIONS.md D38).

Order of commands to run on the real data:

    python scripts/explain.py maps --run $RUN --subset val --max-seconds 150      # repeat until exit 0
    python scripts/error_analysis.py --pred $RUN/predictions/val.csv                # if errors_val/ is missing
    python scripts/explain.py panels --run $RUN --subset val
    python scripts/explain.py maps --run $RUN --subset test --ids-from $RUN/predictions/errors_test/cases.csv
    python scripts/explain.py panels --run $RUN --subset test
    python scripts/explain.py randomisation --run $RUN --max-seconds 150            # repeat until exit 0
    python scripts/explain.py deletion --run $RUN --max-seconds 150                 # repeat until exit 0
    python scripts/explain.py stats --run $RUN

Every command takes --data-config, --explain-config, --device, --batch-size, --out (default
<run>/explain/) and --max-seconds. Exit status: 0 complete, 2 refused or bad input (including the
test-set gates), 3 stopped at --max-seconds (run the same command again to resume), 1 unexpected
error.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import rotate
from torch import nn

from drgrade.config import DataConfig, load_data_config
from drgrade.data import build_transforms
from drgrade.errors import CASE_COLUMNS
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
from drgrade.gradcam import GradCAM, GradCAMResult, fundus_mask, normalise, upsample
from drgrade.io import LESION_COLS, LESION_NAMES
from drgrade.models import gradcam_target_layer
from drgrade.predict import (
    LOGIT_COLUMNS,
    MANIFEST_NAME,
    PREDICTIONS_DIR,
    PROB_COLUMNS,
    LoadedModel,
    build_subset,
    check_data_settings,
    image_folder,
    load_model,
)
from drgrade.provenance import (
    ProvenanceError,
    append_access_event,
    load_allowlist,
    verify_for_use,
    verify_predictions,
)
from drgrade.utils import select_device, setup_logging, utc_now_iso, write_atomically
from drgrade.viz import FIGURE_DPI

import matplotlib.pyplot as plt  # noqa: E402  (viz selects the Agg backend on import)

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_UNEXPECTED = 1
STATUS_REFUSED = 2
STATUS_INCOMPLETE = 3

SUBSET_VAL = "val"
SUBSET_TEST = "test"
SUBSETS: tuple[str, ...] = (SUBSET_VAL, SUBSET_TEST)
DEFAULT_BATCH_SIZE = 2
PROB_TOLERANCE = 1e-4  # recomputed probabilities against maps_val.npz (deletion test)
LOGIT_TOLERANCE = 1e-3  # Grad-CAM logits against the predictions file (maps command)
ACCESS_EVENT = "explain"
EXPLAIN_DIRNAME = "explain"
CALIBRATED_FMT = "{subset}_calibrated.csv"
CALIBRATION_DIRNAME = "calibration"
REFERRAL_DIRNAME = "referral"
REFERRAL_JSON = "referral.json"
ERRORS_FMT = "errors_{subset}"
CASES_CSV = "cases.csv"
RANDOMISATION_DIRNAME = "randomisation"
DELETION_DIRNAME = "deletion"
STATS_DIRNAME = "stats"
PANELS_FMT = "panels_{subset}"
PANELS_CSV = "panels.csv"
EXAMPLES_COUNT = 3
DISC_CHECK_COUNT = 12
DISC_CHECK_COLUMNS = 4
DISC_CHECK_PANEL_INCH = 4.0
DISC_CHECK_MAX_SIDE = 1800  # longest side of disc_check.png in pixels
DISC_OUTLINE = (255, 255, 0)
REFERABLE_MIN_GRADE = 2
SHORTCUT_RATIO = 2.0
LESION_MEASURES: tuple[str, ...] = ("spread_50", "peak", "edge_mass", "disc_mass")
DELETION_CONDITIONS: tuple[str, ...] = ("salient", "random", "rotated")
ROTATION_SEED_OFFSET = 1_000_003  # separate seed space for the rotation angles, so random masks are unchanged
ROTATION_ANGLE_MIN = 90.0  # degrees; angles are drawn uniformly from [min, max)
ROTATION_ANGLE_MAX = 270.0
RANDOMISATION_COLUMNS: tuple[str, ...] = ("image_id", "stage", "stage_index", "spearman", "ssim")
REFERENCE_STAGE = "reference_other_image"
REFERENCE_STAGE_INDEX = -1
REFERENCE_LABEL = "unrelated images (reference)"
DELETION_COLUMNS: tuple[str, ...] = ("image_id", "condition", "fraction", "prob", "drop", "replaced_share")
PANEL_COLUMNS: tuple[str, ...] = (
    "reason",
    "image_id",
    "true_grade",
    "pred_grade",
    "calibrated_confidence",
    "referred",
    "lesions",
    "true_class_map",
    "spearman_pred_vs_true_map",
    "reviewer_note",
)
PANEL_INCH = 4.5
PANEL_GRID_COLUMNS = 3  # processed image, Grad-CAM for the predicted class, Grad-CAM for the true grade
TRUE_PANEL_TITLE = "Grad-CAM for the true grade ({grade})"
CAPTION_HEIGHT_RATIO = 0.22
CAPTION_WRAP = 96
FIGURE_FORMAT = "png"
MAPS_DTYPE_CAMS = np.float32


class ExplainRefused(RuntimeError):
    """Raised when a precondition, a gate or a consistency check forbids the requested step."""


@dataclass(frozen=True)
class ExplainConfig:
    """Settings of the explainability stage, read from configs/explain.yaml."""

    seed: int
    n_randomisation: int
    n_deletion: int
    deletion_fractions: tuple[float, ...]
    deletion_blur_radius: float
    compare_size: int
    overlay_alpha: float
    colormap: str
    edge_frac: float
    disc_radius_frac: float
    disc_smooth_frac: float
    min_group: int

    def __post_init__(self) -> None:
        """Check ranges so that a bad value fails at load time."""
        if self.n_randomisation < 1 or self.n_deletion < 1:
            raise ValueError("n_randomisation and n_deletion must be at least 1")
        if not self.deletion_fractions or any(not 0.0 <= f <= 1.0 for f in self.deletion_fractions):
            raise ValueError(f"deletion_fractions must be non-empty and in [0, 1], got {self.deletion_fractions}")
        if self.deletion_blur_radius <= 0:
            raise ValueError(f"deletion_blur_radius must be positive, got {self.deletion_blur_radius}")
        if self.compare_size < 2:
            raise ValueError(f"compare_size must be at least 2, got {self.compare_size}")
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError(f"overlay_alpha must be in [0, 1], got {self.overlay_alpha}")
        if self.colormap not in matplotlib.colormaps:
            raise ValueError(f"colormap {self.colormap!r} is not a matplotlib colormap")
        if not 0.0 < self.edge_frac <= 1.0:
            raise ValueError(f"edge_frac must be in (0, 1], got {self.edge_frac}")
        if not 0.0 < self.disc_radius_frac <= 1.0:
            raise ValueError(f"disc_radius_frac must be in (0, 1], got {self.disc_radius_frac}")
        if self.disc_smooth_frac <= 0.0:
            raise ValueError(f"disc_smooth_frac must be positive, got {self.disc_smooth_frac}")
        if self.min_group < 1:
            raise ValueError(f"min_group must be at least 1, got {self.min_group}")


def load_explain_config(path: Path | str) -> ExplainConfig:
    """Read configs/explain.yaml. Unknown or missing keys raise ValueError."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"explain config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    expected = {f.name for f in fields(ExplainConfig)}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(f"{config_path}: unknown keys {unknown}, missing keys {missing}")
    values = dict(raw)
    values["deletion_fractions"] = tuple(float(f) for f in raw["deletion_fractions"])
    return ExplainConfig(**values)


@dataclass(frozen=True)
class RunContext:
    """Everything one command needs: the run, its data settings, the model and the output folder."""

    run_dir: Path
    out_dir: Path
    data_cfg: DataConfig
    explain_cfg: ExplainConfig
    loaded: LoadedModel
    device: torch.device
    batch_size: int
    max_seconds: float | None
    started: float
    command: str

    @property
    def model(self) -> nn.Module:
        """The trained network in eval mode."""
        return self.loaded.model

    @property
    def arch(self) -> str:
        """The timm architecture name of the run."""
        return self.loaded.run.model.timm_name


@dataclass(frozen=True)
class SubsetData:
    """A validation or test subset: the sorted table, the evaluation dataset and the row of each id."""

    rows: pd.DataFrame
    dataset: Any
    positions: dict[str, int]

    @property
    def ids(self) -> list[str]:
        """Image ids in table order (sorted)."""
        return list(self.rows.index)


@dataclass(frozen=True)
class MapCache:
    """Cached raw Grad-CAM maps of one subset, read from maps_<subset>.npz."""

    ids: list[str]
    index: dict[str, int]
    cams: np.ndarray
    logits: np.ndarray
    target: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class PanelCase:
    """One representative case for a panel figure."""

    reason: str
    image_id: str
    true_grade: int
    pred_grade: int
    confidence: float
    referred: bool
    lesions: str
    cam: np.ndarray

    @property
    def wrong(self) -> bool:
        """True when the predicted grade differs from the true grade (a case with a true-grade panel)."""
        return self.pred_grade != self.true_grade


@dataclass(frozen=True)
class Render:
    """A case with its drawn panels, ready for a figure.

    true_overlay and spearman are set only for wrong cases; correct cases have neither.
    """

    case: PanelCase
    caption: str
    image: np.ndarray
    overlay: np.ndarray
    true_overlay: np.ndarray | None
    spearman: float | None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _context(args: argparse.Namespace, raw_args: Sequence[str]) -> RunContext:
    """Load the data settings, the explain settings and the trained model for one command."""
    if args.batch_size < 1:
        raise ExplainRefused("--batch-size must be at least 1")
    data_cfg = load_data_config(args.data_config)
    explain_cfg = load_explain_config(args.explain_config)
    device = select_device(args.device)
    loaded = load_model(args.run, device)
    check_data_settings(loaded, data_cfg)
    out_dir = args.out if args.out is not None else args.run / EXPLAIN_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_dir=args.run,
        out_dir=out_dir,
        data_cfg=data_cfg,
        explain_cfg=explain_cfg,
        loaded=loaded,
        device=device,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
        started=time.monotonic(),
        command=" ".join(["scripts/explain.py", *raw_args]),
    )


def _time_up(ctx: RunContext) -> bool:
    """True when --max-seconds has passed since this command started."""
    return ctx.max_seconds is not None and time.monotonic() - ctx.started >= ctx.max_seconds


def _subset_data(ctx: RunContext, subset: str) -> SubsetData:
    """Return the subset table with its dataset. For test, call _test_checks first."""
    table, dataset = build_subset(ctx.loaded, ctx.data_cfg, subset)
    rows = table.set_index("image_id", drop=False)
    ids = list(rows.index)
    return SubsetData(rows=rows, dataset=dataset, positions={image_id: n for n, image_id in enumerate(ids)})


def _test_checks(ctx: RunContext) -> None:
    """Refuse unless this run may be read on the test set (DECISIONS.md D38)."""
    name = ctx.loaded.run.run_name
    allowed = load_allowlist()
    if name not in allowed:
        raise ExplainRefused(f"run {name!r} is not in the test allowlist; only {sorted(allowed)} may be read on test")
    verify_predictions(ctx.run_dir / PREDICTIONS_DIR / f"{SUBSET_TEST}.csv", SUBSET_TEST, expected_rows=None)


def _log_test_access(ctx: RunContext, n_images: int) -> None:
    """Record in the access log that test images are about to be read for explanation only."""
    append_access_event(
        {
            "event": ACCESS_EVENT,
            "run_name": ctx.loaded.run.run_name,
            "folder": ctx.run_dir.as_posix(),
            "checkpoint_sha256": ctx.loaded.checkpoint_sha256,
            "n_images": n_images,
            "command": ctx.command,
        }
    )


def _predictions(ctx: RunContext, subset: str) -> pd.DataFrame:
    """Return the predictions file of a subset, indexed by image id, after its provenance check."""
    path = ctx.run_dir / PREDICTIONS_DIR / f"{subset}.csv"
    verify_for_use(path, subset, expected_rows=None, require_meta=False)
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    return frame.set_index("image_id")


def _logits_of(predictions: pd.DataFrame, ids: Sequence[str], path: Path) -> np.ndarray:
    """Return the reported logits for ids, refusing when an id is missing."""
    missing = [i for i in ids if i not in predictions.index]
    if missing:
        raise ExplainRefused(f"{len(missing)} ids are not in {path}, for example {missing[:5]}")
    return predictions.loc[list(ids), list(LOGIT_COLUMNS)].to_numpy(dtype=np.float64)


def _read_ids_file(path: Path) -> list[str]:
    """Return the sorted, unique image ids of a CSV with an image_id column."""
    if not path.is_file():
        raise ExplainRefused(f"ids file not found: {path}")
    frame = pd.read_csv(path, dtype={"image_id": str})
    if "image_id" not in frame.columns:
        raise ExplainRefused(f"{path} needs an image_id column")
    ids = sorted(set(frame["image_id"].astype(str)))
    if not ids:
        raise ExplainRefused(f"{path} lists no image ids")
    return ids


def _validation_sample(ctx: RunContext, val_ids: Sequence[str]) -> list[str]:
    """Return all validation ids in a seeded random order (the first n of it are used by each test)."""
    rng = np.random.default_rng(ctx.explain_cfg.seed)
    order = rng.permutation(len(val_ids))
    return [val_ids[int(i)] for i in order]


def _read_rgb(folder: Path, image_id: str) -> np.ndarray:
    """Return the processed image of an id as a uint8 (H, W, 3) array."""
    with Image.open(folder / f"{image_id}.jpg") as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def _to_input(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a batch to the device; on CPU use channels_last as load_model does (DECISIONS.md D36)."""
    images = images.to(device)
    if device.type == "cpu":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def _stack(dataset: Any, positions: Sequence[int]) -> torch.Tensor:
    """Stack the evaluation tensors of the given dataset positions."""
    return torch.stack([dataset[p][0] for p in positions])


def _gradcam(
    model: nn.Module,
    layer: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target_class: Sequence[int] | None = None,
) -> GradCAMResult:
    """Grad-CAM for a batch. Never called inside torch.inference_mode(): the backward pass needs autograd."""
    batch = _to_input(images, device)
    with GradCAM(model, layer) as cam:
        return cam(batch, target_class)


def _module_name(model: nn.Module, module: nn.Module) -> str:
    """Return the dotted name of a module inside the model (recorded in the maps sidecar)."""
    for name, candidate in model.named_modules():
        if candidate is module:
            return name
    raise ExplainRefused("the Grad-CAM layer is not a module of the model")


def _normalised(cams: np.ndarray, size: int, mask: np.ndarray) -> np.ndarray:
    """Upsample raw maps to size and normalise each one inside the fundus mask."""
    upsampled = upsample(np.asarray(cams), size)
    return np.stack([normalise(one, mask) for one in upsampled])


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write arrays to an .npz file atomically (the file handle stops numpy adding a suffix)."""

    def write(tmp: Path) -> None:
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)

    write_atomically(path, write)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read every array of an .npz file (no pickles)."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _write_csv(path: Path, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Write a table atomically with a fixed column order."""
    ordered = frame.loc[:, list(columns)]
    write_atomically(path, lambda p: ordered.to_csv(p, index=False))


def _read_csv_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Return the rows of a partial table, or an empty table with the columns when there is none."""
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    if list(frame.columns) != list(columns):
        raise ExplainRefused(f"{path} has other columns; remove it to start again")
    return frame


def _append_rows(frame: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Append rows (dicts in column order) to a table."""
    if not rows:
        return frame
    new = pd.DataFrame(rows)
    if frame.empty:
        return new
    return pd.concat([frame, new], ignore_index=True)


def _save_figure(fig: Any, path: Path, dpi: int = FIGURE_DPI) -> None:
    """Save a matplotlib figure atomically and close it.

    The figure is written to the temporary name from write_atomically and then moved into
    place, so a killed process never leaves a truncated PNG at path. The format is given
    explicitly because the temporary name ends in .tmp, which matplotlib cannot infer from.
    """
    try:
        write_atomically(path, lambda tmp: fig.savefig(tmp, dpi=dpi, format=FIGURE_FORMAT))
    finally:
        plt.close(fig)


def _maps_paths(ctx: RunContext, subset: str) -> tuple[Path, Path, Path]:
    """Return the cache file, its sidecar and the partial file for a subset."""
    base = ctx.out_dir
    return base / f"maps_{subset}.npz", base / f"maps_{subset}.json", base / f"maps_{subset}.partial.npz"


def _read_maps(ctx: RunContext, subset: str) -> MapCache:
    """Read maps_<subset>.npz, refusing when it is missing or was made from another checkpoint."""
    npz_path, meta_path, _ = _maps_paths(ctx, subset)
    if not npz_path.is_file() or not meta_path.is_file():
        raise ExplainRefused(
            f"{npz_path} is missing: run scripts/explain.py maps --run {ctx.run_dir} --subset {subset}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("checkpoint_sha256") != ctx.loaded.checkpoint_sha256:
        raise ExplainRefused(f"{npz_path} was made from another checkpoint; rerun the maps command")
    arrays = _load_npz(npz_path)
    ids = [str(i) for i in arrays["image_id"]]
    return MapCache(
        ids=ids,
        index={image_id: n for n, image_id in enumerate(ids)},
        cams=arrays["cams"],
        logits=arrays["logits"],
        target=arrays["target"],
        meta=meta,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax of one logit vector in float64."""
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


@torch.inference_mode()
def _probabilities(model: nn.Module, tensors: Sequence[torch.Tensor], device: torch.device, batch_size: int) -> np.ndarray:
    """Softmax probabilities (float64, N x 5) of evaluation tensors; plain forward passes only."""
    outputs: list[np.ndarray] = []
    for start in range(0, len(tensors), batch_size):
        batch = _to_input(torch.stack(list(tensors[start : start + batch_size])), device)
        logits = model(batch).to(dtype=torch.float64)
        outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs)


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------


def _merge_arrays(done: dict[str, np.ndarray] | None, new: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate two sets of map arrays (rows are images)."""
    if done is None:
        return new
    return {key: np.concatenate([done[key], new[key]]) for key in new}


def cmd_maps(ctx: RunContext, args: argparse.Namespace) -> int:
    """Grad-CAM maps for the predicted class, cached in maps_<subset>.npz (resumable).

    The test subset is explained only for the ids named in --ids-from, after the allowlist and
    predictions checks, and each run that reads test images writes an "explain" access event first.
    """
    subset = args.subset
    if subset == SUBSET_TEST:
        if args.ids_from is None:
            raise ExplainRefused("maps --subset test needs --ids-from CSV: only the representative cases are explained")
        _test_checks(ctx)
    data = _subset_data(ctx, subset)
    if not data.ids:
        raise ExplainRefused(f"the {subset} subset is empty")
    requested = _read_ids_file(args.ids_from) if args.ids_from is not None else list(data.ids)
    unknown = [i for i in requested if i not in data.positions]
    if unknown:
        raise ExplainRefused(f"{len(unknown)} ids are not in the {subset} subset, for example {unknown[:5]}")

    npz_path, meta_path, partial_path = _maps_paths(ctx, subset)
    if npz_path.is_file():
        existing = [str(i) for i in _load_npz(npz_path)["image_id"]]
        if existing == requested:
            logger.info("%s already holds maps for these %d ids", npz_path, len(requested))
            return STATUS_OK
        raise ExplainRefused(f"{npz_path} holds other ids; remove it to recompute")

    predictions = _predictions(ctx, subset)
    predicted_logits = _logits_of(predictions, requested, ctx.run_dir / PREDICTIONS_DIR / f"{subset}.csv")

    done: dict[str, np.ndarray] | None = None
    if partial_path.is_file():
        done = _load_npz(partial_path)
        stray = set(str(i) for i in done["image_id"]) - set(requested)
        if stray:
            raise ExplainRefused(f"{partial_path} holds ids that are not requested; remove it to start again")
    done_ids = set() if done is None else set(str(i) for i in done["image_id"])
    remaining = [i for i in requested if i not in done_ids]
    logger.info("%d of %d maps to compute for %s", len(remaining), len(requested), subset)

    if subset == SUBSET_TEST and remaining:
        _log_test_access(ctx, len(requested))

    layer = gradcam_target_layer(ctx.model, ctx.arch)
    layer_name = _module_name(ctx.model, layer)
    for start in range(0, len(remaining), ctx.batch_size):
        batch_ids = remaining[start : start + ctx.batch_size]
        images = _stack(data.dataset, [data.positions[i] for i in batch_ids])
        result = _gradcam(ctx.model, layer, images, ctx.device)
        new = {
            "image_id": np.array(batch_ids, dtype=str),
            "cams": result.cams.astype(MAPS_DTYPE_CAMS),
            "logits": result.logits,
            "target": result.target,
        }
        done = _merge_arrays(done, new)
        _save_npz(partial_path, done)
        logger.info("maps: %d of %d", len(done["image_id"]), len(requested))
        if _time_up(ctx) and start + ctx.batch_size < len(remaining):
            return STATUS_INCOMPLETE

    assert done is not None
    ids = [str(i) for i in done["image_id"]]
    order = {image_id: n for n, image_id in enumerate(requested)}
    perm = np.array(sorted(range(len(ids)), key=lambda n: order[ids[n]]), dtype=np.int64)
    arrays = {key: value[perm] for key, value in done.items()}
    final_ids = [str(i) for i in arrays["image_id"]]
    max_diff = float(np.max(np.abs(arrays["logits"] - predicted_logits)))
    if max_diff > LOGIT_TOLERANCE:
        raise ExplainRefused(
            f"Grad-CAM logits differ from the reported {subset} logits by {max_diff:.3g} "
            f"(limit {LOGIT_TOLERANCE}); the maps would not explain the reported predictions"
        )
    _save_npz(npz_path, arrays)
    meta = {
        "run_name": ctx.loaded.run.run_name,
        "checkpoint_sha256": ctx.loaded.checkpoint_sha256,
        "target_layer": layer_name,
        "subset": subset,
        "n": len(final_ids),
        "map_shape": [int(x) for x in arrays["cams"].shape[1:]],
        "max_abs_logit_difference": max_diff,
        "logit_tolerance": LOGIT_TOLERANCE,
        "predictions_file": (ctx.run_dir / PREDICTIONS_DIR / f"{subset}.csv").as_posix(),
        "command": ctx.command,
        "time_utc": utc_now_iso(),
    }
    write_atomically(meta_path, lambda p: p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    if partial_path.exists():
        partial_path.unlink()
    logger.info("wrote %s with %d maps (max logit difference %.2e)", npz_path, len(final_ids), max_diff)
    return STATUS_OK


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def _read_cases(path: Path) -> list[dict[str, Any]]:
    """Return the rows of a cases file, refusing when a column is missing."""
    if not path.is_file():
        raise ExplainRefused(f"cases file not found: {path} (run scripts/error_analysis.py first)")
    frame = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in CASE_COLUMNS if c not in frame.columns]
    if missing:
        raise ExplainRefused(f"{path} lacks the columns {missing}")
    if frame.empty:
        raise ExplainRefused(f"{path} has no cases")
    return frame.to_dict("records")


def _calibrated_confidence(ctx: RunContext, subset: str, ids: Sequence[str]) -> dict[str, float]:
    """Return the calibrated confidence (largest calibrated probability) of each id."""
    path = ctx.run_dir / CALIBRATION_DIRNAME / CALIBRATED_FMT.format(subset=subset)
    if not path.is_file():
        raise ExplainRefused(f"{path} is missing: run scripts/calibrate.py first")
    frame = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id")
    missing_cols = [c for c in PROB_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise ExplainRefused(f"{path} lacks the columns {missing_cols}")
    confidence = frame[list(PROB_COLUMNS)].max(axis=1)
    absent = [i for i in ids if i not in confidence.index]
    if absent:
        raise ExplainRefused(f"{path} lacks {len(absent)} case ids, for example {absent[:5]}")
    return {i: float(confidence.loc[i]) for i in ids}


def _referral_threshold(ctx: RunContext) -> float | None:
    """Return the primary referral threshold (referred when calibrated confidence is at or below it)."""
    path = ctx.run_dir / REFERRAL_DIRNAME / REFERRAL_JSON
    if not path.is_file():
        raise ExplainRefused(f"{path} is missing: run scripts/referral.py first")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["primary"]["threshold"]
    except (KeyError, TypeError) as exc:
        raise ExplainRefused(f"{path} has no primary.threshold") from exc
    return None if value is None else float(value)


def _lesion_text(ctx: RunContext, rows: pd.DataFrame, image_id: str) -> str:
    """Return the names of the annotated lesions of an image, or 'none annotated'."""
    row = rows.loc[image_id]
    names = [LESION_NAMES[column] for column in LESION_COLS if int(row[column]) == 1]
    return ", ".join(names) if names else "none annotated"


def _manifest_rows(ctx: RunContext, ids: Sequence[str]) -> pd.DataFrame:
    """Return the processed manifest rows of ids, indexed by image id."""
    path = image_folder(ctx.data_cfg) / MANIFEST_NAME
    frame = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id", drop=False)
    missing_cols = [c for c in LESION_COLS if c not in frame.columns]
    if missing_cols:
        raise ExplainRefused(f"{path} lacks the lesion columns {missing_cols}")
    absent = [i for i in ids if i not in frame.index]
    if absent:
        raise ExplainRefused(f"{path} lacks {len(absent)} case ids, for example {absent[:5]}")
    return frame


def _overlay(image: np.ndarray, cam: np.ndarray, alpha: float, colormap: str) -> np.ndarray:
    """Blend the normalised, upsampled map over the image, only inside the fundus mask."""
    size = image.shape[0]
    mask = fundus_mask(size)
    heat = normalise(upsample(np.asarray(cam)[None], size)[0], mask)
    rgb = matplotlib.colormaps[colormap](heat)[..., :3]
    weight = (alpha * mask)[..., None]
    base = image.astype(np.float64) / 255.0
    blended = base * (1.0 - weight) + rgb * weight
    return (np.clip(blended, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _caption(case: PanelCase) -> str:
    """Return the caption of one case: id, grades, calibrated confidence, referral and lesions."""
    referral = "referred" if case.referred else "not referred"
    text = (
        f"{case.image_id}   true grade {case.true_grade}   predicted grade {case.pred_grade}   "
        f"calibrated confidence {case.confidence:.3f}   {referral}   lesions: {case.lesions}"
    )
    return textwrap.fill(text, CAPTION_WRAP)


def _save_case_figure(renders: Sequence[Render], path: Path) -> None:
    """Save one figure with a row per case: a caption above three panels.

    The panels are the processed image, the Grad-CAM overlay for the predicted class and the
    Grad-CAM overlay for the true grade. Every row has the same three columns; the third panel
    is turned off for correct cases, which have no true-grade panel.
    """
    rows = len(renders)
    fig = plt.figure(figsize=(PANEL_GRID_COLUMNS * PANEL_INCH, rows * PANEL_INCH * (1.0 + CAPTION_HEIGHT_RATIO)))
    grid = fig.add_gridspec(
        2 * rows,
        PANEL_GRID_COLUMNS,
        height_ratios=[CAPTION_HEIGHT_RATIO, 1.0] * rows,
        hspace=0.15,
        wspace=0.04,
    )
    for index, render in enumerate(renders):
        caption_ax = fig.add_subplot(grid[2 * index, :])
        caption_ax.axis("off")
        caption_ax.text(0.0, 0.5, render.caption, ha="left", va="center", fontsize=8, transform=caption_ax.transAxes)
        panels = (
            (render.image, "processed image"),
            (render.overlay, "Grad-CAM overlay"),
            (render.true_overlay, TRUE_PANEL_TITLE.format(grade=render.case.true_grade)),
        )
        for column, (pixels, title) in enumerate(panels):
            ax = fig.add_subplot(grid[2 * index + 1, column])
            if pixels is None:
                ax.axis("off")
                continue
            ax.imshow(pixels)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    _save_figure(fig, path)


def _check_reason(reason: str) -> None:
    """Refuse reason names that cannot be folder names."""
    if not reason.strip() or reason in (".", "..") or "/" in reason or "\\" in reason:
        raise ExplainRefused(f"reason {reason!r} cannot be used as a folder name")


def _true_grade_maps(ctx: RunContext, data: SubsetData, cases: Sequence[PanelCase]) -> dict[str, np.ndarray]:
    """Grad-CAM for the true grade of each wrong case, computed live in batches.

    Never inside torch.inference_mode(): the backward pass needs autograd. The same model and
    target layer as the cached predicted-class maps are used.
    """
    if not cases:
        return {}
    layer = gradcam_target_layer(ctx.model, ctx.arch)
    maps: dict[str, np.ndarray] = {}
    for start in range(0, len(cases), ctx.batch_size):
        batch = cases[start : start + ctx.batch_size]
        images = _stack(data.dataset, [data.positions[case.image_id] for case in batch])
        result = _gradcam(ctx.model, layer, images, ctx.device, [case.true_grade for case in batch])
        for case, cam in zip(batch, result.cams):
            maps[case.image_id] = cam
    return maps


def _true_grade_comparison(
    case: PanelCase, true_cam: np.ndarray, image: np.ndarray, cfg: ExplainConfig
) -> tuple[np.ndarray, float]:
    """Overlay of the true-grade map, and its masked Spearman with the predicted-class map.

    Both maps are normalised and upsampled to compare_size inside fundus_mask. The Spearman is nan
    when either map is constant inside the mask.
    """
    overlay = _overlay(image, true_cam, cfg.overlay_alpha, cfg.colormap)
    mask = fundus_mask(cfg.compare_size)
    predicted = _normalised(case.cam[None], cfg.compare_size, mask)[0]
    true_norm = _normalised(true_cam[None], cfg.compare_size, mask)[0]
    return overlay, masked_spearman(predicted, true_norm, mask)


def cmd_panels(ctx: RunContext, args: argparse.Namespace) -> int:
    """Figures for the cases of one subset: per image, per reason grid, and panels.csv with a reviewer_note column.

    Each case has the processed image and the Grad-CAM overlay of the predicted class. A heatmap for
    a predicted "no DR" class shows evidence for normality, so it cannot show the lesions that were
    missed. A wrong case therefore also gets a third panel: Grad-CAM for the true grade, computed
    live. It shows whether the model had any evidence for the right answer and where that evidence
    lies. panels.csv records true_class_map ("yes" for wrong cases) and spearman_pred_vs_true_map
    (masked Spearman inside the fundus mask between the two normalised maps, empty for correct cases).
    """
    subset = args.subset
    if subset == SUBSET_TEST:
        _test_checks(ctx)
    cases_path = args.cases if args.cases is not None else (
        ctx.run_dir / PREDICTIONS_DIR / ERRORS_FMT.format(subset=subset) / CASES_CSV
    )
    records = _read_cases(cases_path)
    ids = [str(r["image_id"]) for r in records]
    maps = _read_maps(ctx, subset)
    absent = [i for i in ids if i not in maps.index]
    if absent:
        raise ExplainRefused(
            f"{len(absent)} case ids have no map; run scripts/explain.py maps --run {ctx.run_dir} "
            f"--subset {subset} --ids-from {cases_path}"
        )
    confidence = _calibrated_confidence(ctx, subset, ids)
    threshold = _referral_threshold(ctx)
    manifest = _manifest_rows(ctx, ids)
    if subset == SUBSET_TEST:
        _log_test_access(ctx, len(ids))

    folder = image_folder(ctx.data_cfg)
    cases: list[PanelCase] = []
    for record in records:
        image_id = str(record["image_id"])
        reason = str(record["reason"])
        _check_reason(reason)
        conf = confidence[image_id]
        cases.append(
            PanelCase(
                reason=reason,
                image_id=image_id,
                true_grade=int(record["true_grade"]),
                pred_grade=int(record["pred_grade"]),
                confidence=conf,
                referred=threshold is not None and conf <= threshold,
                lesions=_lesion_text(ctx, manifest, image_id),
                cam=maps.cams[maps.index[image_id]],
            )
        )

    wrong = [case for case in cases if case.wrong]
    true_maps: dict[str, np.ndarray] = {}
    if wrong:
        true_maps = _true_grade_maps(ctx, _subset_data(ctx, subset), wrong)

    out_dir = ctx.out_dir / PANELS_FMT.format(subset=subset)
    renders: list[Render] = []
    for case in cases:
        image = _read_rgb(folder, case.image_id)
        overlay = _overlay(image, case.cam, ctx.explain_cfg.overlay_alpha, ctx.explain_cfg.colormap)
        true_overlay: np.ndarray | None = None
        spearman: float | None = None
        if case.wrong:
            true_overlay, spearman = _true_grade_comparison(case, true_maps[case.image_id], image, ctx.explain_cfg)
        renders.append(
            Render(
                case=case,
                caption=_caption(case),
                image=image,
                overlay=overlay,
                true_overlay=true_overlay,
                spearman=spearman,
            )
        )

    for render in renders:
        target = out_dir / render.case.reason / f"{render.case.image_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        _save_case_figure([render], target)
    grouped: dict[str, list[Render]] = {}
    for render in renders:
        grouped.setdefault(render.case.reason, []).append(render)
    for reason, group in grouped.items():
        _save_case_figure(group, out_dir / f"{reason}.png")

    table = pd.DataFrame(
        [
            {
                "reason": r.case.reason,
                "image_id": r.case.image_id,
                "true_grade": r.case.true_grade,
                "pred_grade": r.case.pred_grade,
                "calibrated_confidence": f"{r.case.confidence:.3f}",
                "referred": r.case.referred,
                "lesions": r.case.lesions,
                "true_class_map": "yes" if r.case.wrong else "no",
                "spearman_pred_vs_true_map": (
                    f"{r.spearman:.4f}" if r.spearman is not None and np.isfinite(r.spearman) else ""
                ),
                "reviewer_note": "",
            }
            for r in renders
        ]
    )
    _write_csv(out_dir / PANELS_CSV, table, PANEL_COLUMNS)
    logger.info("wrote %d case figures and %s", len(renders), out_dir / PANELS_CSV)
    return STATUS_OK


# ---------------------------------------------------------------------------
# randomisation
# ---------------------------------------------------------------------------


def _trained_maps(ctx: RunContext, data: SubsetData, ids: Sequence[str], path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw Grad-CAM maps and predicted targets of the trained model for ids, cached in trained.npz."""
    if path.is_file():
        arrays = _load_npz(path)
        if [str(i) for i in arrays["image_id"]] != list(ids):
            raise ExplainRefused(f"{path} was made for other images; remove it to start again")
        return arrays["cams"], arrays["target"]
    layer = gradcam_target_layer(ctx.model, ctx.arch)
    cams: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, len(ids), ctx.batch_size):
        images = _stack(data.dataset, [data.positions[i] for i in ids[start : start + ctx.batch_size]])
        result = _gradcam(ctx.model, layer, images, ctx.device)
        cams.append(result.cams.astype(MAPS_DTYPE_CAMS))
        targets.append(result.target)
    stacked_cams = np.concatenate(cams)
    stacked_targets = np.concatenate(targets)
    _save_npz(path, {"image_id": np.array(ids, dtype=str), "cams": stacked_cams, "target": stacked_targets})
    return stacked_cams, stacked_targets


def _randomisation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per stage: n, mean and sd of Spearman and SSIM, and the number of undefined Spearman values."""
    grouped = frame.groupby(["stage_index", "stage"], sort=True)
    return grouped.agg(
        n=("image_id", "size"),
        mean_spearman=("spearman", "mean"),
        sd_spearman=("spearman", "std"),
        mean_ssim=("ssim", "mean"),
        sd_ssim=("ssim", "std"),
        n_nan=("spearman", lambda values: int(values.isna().sum())),
    ).reset_index()


def _reference_rows(trained_norm: np.ndarray, ids: Sequence[str], mask: np.ndarray) -> list[dict[str, Any]]:
    """Reference similarity of unrelated images, from the trained maps only.

    Image i is paired with image (i + 1) mod n in sample order, so each pair holds two different
    images. The values show what similarity two unrelated maps reach under the same normalisation.
    Fewer than two images give no reference rows.
    """
    n = len(ids)
    if n < 2:
        logger.warning("randomisation: fewer than two images, so there is no reference row for unrelated images")
        return []
    rows: list[dict[str, Any]] = []
    for i in range(n):
        j = (i + 1) % n
        rows.append(
            {
                "image_id": ids[i],
                "stage": REFERENCE_STAGE,
                "stage_index": REFERENCE_STAGE_INDEX,
                "spearman": masked_spearman(trained_norm[i], trained_norm[j], mask),
                "ssim": ssim(trained_norm[i], trained_norm[j]),
            }
        )
    return rows


def _complete_stage_frame(path: Path, ids: Sequence[str], n_stages: int) -> pd.DataFrame | None:
    """Return randomisation.csv when it holds every stage for every sampled id, otherwise None."""
    if not path.is_file():
        return None
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    if list(frame.columns) != list(RANDOMISATION_COLUMNS):
        return None
    for stage_index in range(n_stages):
        part = frame.loc[frame["stage_index"] == stage_index, "image_id"].astype(str)
        if sorted(part) != sorted(ids):
            return None
    return frame


def _write_randomisation_outputs(frame: pd.DataFrame, rdir: Path) -> None:
    """Write summary.csv and randomisation.png from the stage rows and the reference rows."""
    summary = _randomisation_summary(frame)
    _write_csv(rdir / "summary.csv", summary, list(summary.columns))
    _plot_randomisation(summary, rdir / "randomisation.png")


def _plot_randomisation(summary: pd.DataFrame, path: Path) -> None:
    """Mean Spearman and mean SSIM against stage, with sd bars, in cascade order.

    Two dashed horizontal lines show the mean of the unrelated-images reference, one per measure.
    """
    stages = summary[summary["stage_index"] >= 0]
    reference = summary[summary["stage_index"] == REFERENCE_STAGE_INDEX]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(stages) + 2.0), 4.5))
    spearman_line = ax.errorbar(
        x, stages["mean_spearman"], yerr=stages["sd_spearman"], marker="o", capsize=3, label="Spearman (fundus mask)"
    )
    ssim_line = ax.errorbar(x, stages["mean_ssim"], yerr=stages["sd_ssim"], marker="s", capsize=3, label="SSIM")
    if not reference.empty:
        ref = reference.iloc[0]
        ax.axhline(ref["mean_spearman"], linestyle="--", color=spearman_line[0].get_color(), label=REFERENCE_LABEL)
        ax.axhline(ref["mean_ssim"], linestyle="--", color=ssim_line[0].get_color(), label=REFERENCE_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(stages["stage"], rotation=45, ha="right")
    ax.set_xlabel("re-initialised stage (top layers first)")
    ax.set_ylabel("mean similarity with the trained map")
    ax.set_title("Model randomisation test (validation sample)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, path)


def _randomisation_examples(
    ctx: RunContext,
    data: SubsetData,
    ids: Sequence[str],
    trained_cams: np.ndarray,
    targets: np.ndarray,
    path: Path,
) -> None:
    """Trained map and the map after each stage for the first few images, one row per image."""
    compare = ctx.explain_cfg.compare_size
    mask = fundus_mask(compare)
    positions = [data.positions[i] for i in ids]
    columns: list[tuple[str, np.ndarray]] = [("trained", _normalised(trained_cams, compare, mask))]
    for name, copy_model in randomised_copies(ctx.model, ctx.arch, ctx.explain_cfg.seed):
        layer = gradcam_target_layer(copy_model, ctx.arch)
        result = _gradcam(copy_model, layer, _stack(data.dataset, positions), ctx.device, targets.tolist())
        columns.append((name, _normalised(result.cams, compare, mask)))
    fig, axes = plt.subplots(len(ids), len(columns), figsize=(1.7 * len(columns), 1.9 * len(ids)), squeeze=False)
    for row in range(len(ids)):
        for column, (title, maps) in enumerate(columns):
            ax = axes[row][column]
            ax.imshow(maps[row], cmap=ctx.explain_cfg.colormap, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=7)
            if column == 0:
                ax.set_ylabel(ids[row], fontsize=7)
    fig.suptitle("Randomisation test: trained map and maps after each stage (normalised)", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, path)


def cmd_randomisation(ctx: RunContext, args: argparse.Namespace) -> int:
    """Model randomisation test: similarity of Grad-CAM maps after re-initialising stages top-down.

    Resumable between stages: each finished stage is saved to randomisation.partial.csv; the
    time limit is checked between stages, so every invocation finishes at least one stage.

    Each run also adds the reference rows (stage "reference_other_image", stage_index -1): the
    similarity of trained maps of different images, the level that unrelated maps reach.

    When randomisation.csv already holds every stage for the sampled images, no stage is
    recomputed. summary.csv and randomisation.png are rebuilt from it, and examples.png is
    rebuilt only when it is missing (it needs model passes).
    """
    cfg = ctx.explain_cfg
    data = _subset_data(ctx, SUBSET_VAL)
    sample = _validation_sample(ctx, data.ids)
    ids = sample[: min(cfg.n_randomisation, len(sample))]
    rdir = ctx.out_dir / RANDOMISATION_DIRNAME
    rdir.mkdir(parents=True, exist_ok=True)
    trained_cams, targets = _trained_maps(ctx, data, ids, rdir / "trained.npz")
    compare = cfg.compare_size
    mask = fundus_mask(compare)
    trained_norm = _normalised(trained_cams, compare, mask)
    n_stages = len(cascade_stages(ctx.model, ctx.arch))
    frame_path = rdir / "randomisation.csv"
    examples_path = rdir / "examples.png"

    complete = _complete_stage_frame(frame_path, ids, n_stages)
    if complete is not None:
        logger.info(
            "randomisation: %s holds all %d stages for %d images; rebuilding outputs without recomputing stages",
            frame_path,
            n_stages,
            len(ids),
        )
        frame = complete
        if not bool((frame["stage_index"] == REFERENCE_STAGE_INDEX).any()):
            reference = _reference_rows(trained_norm, ids, mask)
            if reference:
                frame = _append_rows(frame, reference).sort_values("stage_index", kind="stable").reset_index(drop=True)
                _write_csv(frame_path, frame, RANDOMISATION_COLUMNS)
        _write_randomisation_outputs(frame, rdir)
        if not examples_path.is_file():
            _randomisation_examples(ctx, data, ids[:EXAMPLES_COUNT], trained_cams[:EXAMPLES_COUNT], targets[:EXAMPLES_COUNT], examples_path)
        logger.info("rebuilt randomisation summary and figures for %d images", len(ids))
        return STATUS_OK

    partial = rdir / "randomisation.partial.csv"
    done = _read_csv_or_empty(partial, RANDOMISATION_COLUMNS)
    done_stages = set(int(s) for s in done["stage_index"]) if not done.empty else set()
    positions = [data.positions[i] for i in ids]

    for stage_index, (name, copy_model) in enumerate(randomised_copies(ctx.model, ctx.arch, cfg.seed)):
        if stage_index in done_stages:
            continue
        layer = gradcam_target_layer(copy_model, ctx.arch)
        rows: list[dict[str, Any]] = []
        for start in range(0, len(ids), ctx.batch_size):
            stop = start + ctx.batch_size
            images = _stack(data.dataset, positions[start:stop])
            result = _gradcam(copy_model, layer, images, ctx.device, targets[start:stop].tolist())
            stage_norm = _normalised(result.cams, compare, mask)
            for offset, image_id in enumerate(ids[start:stop]):
                reference = trained_norm[start + offset]
                rows.append(
                    {
                        "image_id": image_id,
                        "stage": name,
                        "stage_index": stage_index,
                        "spearman": masked_spearman(reference, stage_norm[offset], mask),
                        "ssim": ssim(reference, stage_norm[offset]),
                    }
                )
        done = _append_rows(done, rows)
        _write_csv(partial, done, RANDOMISATION_COLUMNS)
        logger.info("randomisation: stage %d (%s) done", stage_index, name)
        if stage_index < n_stages - 1 and _time_up(ctx):
            return STATUS_INCOMPLETE

    frame = done.sort_values("stage_index", kind="stable").reset_index(drop=True)
    reference = _reference_rows(trained_norm, ids, mask)
    if reference:
        frame = _append_rows(frame, reference).sort_values("stage_index", kind="stable").reset_index(drop=True)
    _write_csv(frame_path, frame, RANDOMISATION_COLUMNS)
    _write_randomisation_outputs(frame, rdir)
    _randomisation_examples(ctx, data, ids[:EXAMPLES_COUNT], trained_cams[:EXAMPLES_COUNT], targets[:EXAMPLES_COUNT], examples_path)
    if partial.exists():
        partial.unlink()
    logger.info("wrote randomisation outputs for %d images and %d stages", len(ids), n_stages)
    return STATUS_OK


# ---------------------------------------------------------------------------
# deletion
# ---------------------------------------------------------------------------


def _deletion_rows(
    ctx: RunContext,
    data: SubsetData,
    maps: MapCache,
    image_id: str,
    position: int,
    transform: Callable[[Image.Image], torch.Tensor],
) -> list[dict[str, Any]]:
    """Deletion rows of one image: salient, random and rotated conditions at each fraction, plus fraction 0.

    The rotated condition takes the salient mask, rotates it about the image centre by an angle
    drawn uniformly from [ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX), and keeps the part inside the
    fundus. It has the salient mask's shape and nearly its area, in a different place, so it
    separates location from contiguity: random pixels are scattered, a blob is contiguous. The
    angles come from their own generator, so the random masks draw exactly as before.
    replaced_share is the replaced pixel count divided by the fundus pixel count.
    """
    cfg = ctx.explain_cfg
    row = maps.index[image_id]
    target = int(maps.target[row])
    expected = float(_softmax(maps.logits[row])[target])
    image = _read_rgb(image_folder(ctx.data_cfg), image_id)
    size = image.shape[0]
    saliency = upsample(np.asarray(maps.cams[row])[None], size)[0]
    region = fundus_mask(size)
    fundus_pixels = int(region.sum())
    replacement = blurred(image, cfg.deletion_blur_radius)
    rng = np.random.default_rng(cfg.seed + position)
    angle_rng = np.random.default_rng(cfg.seed + ROTATION_SEED_OFFSET + position)
    variants: list[np.ndarray] = [image]
    shares: list[float] = [0.0]
    for fraction in cfg.deletion_fractions:
        salient = top_fraction_mask(saliency, region, fraction)
        random_mask = random_fraction_mask(region, fraction, rng)
        angle = float(angle_rng.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX))
        rotated = (rotate(salient.astype(np.uint8), angle, reshape=False, order=0) > 0) & region
        masks = {"salient": salient, "random": random_mask, "rotated": rotated}
        for condition in DELETION_CONDITIONS:
            variants.append(composite(image, replacement, masks[condition]))
            shares.append(float(masks[condition].sum()) / fundus_pixels)
    tensors = [transform(Image.fromarray(v)) for v in variants]
    probs = _probabilities(ctx.model, tensors, ctx.device, ctx.batch_size)
    original = float(probs[0, target])
    if abs(original - expected) > PROB_TOLERANCE:
        raise ExplainRefused(
            f"recomputed probability for {image_id} differs from maps_val.npz by {abs(original - expected):.2e} "
            f"(limit {PROB_TOLERANCE}); the maps and the model disagree"
        )
    rows: list[dict[str, Any]] = [
        {
            "image_id": image_id,
            "condition": condition,
            "fraction": 0.0,
            "prob": original,
            "drop": 0.0,
            "replaced_share": 0.0,
        }
        for condition in DELETION_CONDITIONS
    ]
    per_fraction = len(DELETION_CONDITIONS)
    for k, fraction in enumerate(cfg.deletion_fractions):
        for offset, condition in enumerate(DELETION_CONDITIONS):
            variant = 1 + per_fraction * k + offset
            prob = float(probs[variant, target])
            rows.append(
                {
                    "image_id": image_id,
                    "condition": condition,
                    "fraction": fraction,
                    "prob": prob,
                    "drop": original - prob,
                    "replaced_share": shares[variant],
                }
            )
    return rows


def _read_deletion_partial(path: Path) -> pd.DataFrame:
    """Return the rows of a partial deletion file, refusing one written without rotated or replaced_share."""
    if not path.is_file():
        return pd.DataFrame(columns=list(DELETION_COLUMNS))
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    stale_columns = list(frame.columns) != list(DELETION_COLUMNS)
    missing_rotated = not frame.empty and "rotated" not in set(frame["condition"])
    if stale_columns or missing_rotated:
        raise ExplainRefused(
            f"{path} was written without the rotated condition or replaced_share; remove it to start again"
        )
    return frame


def _mean_sd_se(values: pd.Series) -> tuple[float, float, float]:
    """Mean, sample sd and standard error of the mean (sd and se are nan with fewer than two values)."""
    n = int(len(values))
    mean = float(values.mean()) if n > 0 else float("nan")
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 0 else float("nan")
    return mean, sd, se


def _deletion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per condition and fraction: drop statistics, mean replaced share, and the paired salient minus random and salient minus rotated differences with their se.

    The paired differences are matched by image and fraction, and are repeated on every condition's row for that fraction.
    """
    drops = {c: frame[frame["condition"] == c].set_index(["image_id", "fraction"])["drop"] for c in DELETION_CONDITIONS}
    diff_random = drops["salient"] - drops["random"]
    diff_rotated = drops["salient"] - drops["rotated"]
    rows: list[dict[str, Any]] = []
    fractions = sorted(float(f) for f in frame["fraction"].unique())
    for condition in DELETION_CONDITIONS:
        for fraction in fractions:
            part = frame[(frame["condition"] == condition) & (frame["fraction"] == fraction)]
            mean_drop, sd_drop, se_drop = _mean_sd_se(part["drop"])
            mean_random, _, se_random = _mean_sd_se(diff_random.xs(fraction, level="fraction"))
            mean_rotated, _, se_rotated = _mean_sd_se(diff_rotated.xs(fraction, level="fraction"))
            rows.append(
                {
                    "condition": condition,
                    "fraction": fraction,
                    "n": int(len(part)),
                    "mean_drop": mean_drop,
                    "sd_drop": sd_drop,
                    "se_drop": se_drop,
                    "mean_replaced_share": float(part["replaced_share"].mean()),
                    "mean_diff_salient_minus_random": mean_random,
                    "se_diff_salient_minus_random": se_random,
                    "mean_diff_salient_minus_rotated": mean_rotated,
                    "se_diff_salient_minus_rotated": se_rotated,
                }
            )
    return pd.DataFrame(rows)


def _plot_deletion(summary: pd.DataFrame, path: Path) -> None:
    """Mean drop in predicted-class probability against the nominal share of pixels replaced, with se bars.

    The three curves are the salient, random and rotated conditions (rotated keeps the salient shape).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for condition, marker in (("salient", "o"), ("random", "s"), ("rotated", "^")):
        part = summary[summary["condition"] == condition]
        ax.errorbar(part["fraction"], part["mean_drop"], yerr=part["se_drop"], marker=marker, capsize=3, label=condition)
    ax.set_xlabel("share of fundus pixels replaced")
    ax.set_ylabel("mean drop in predicted-class probability")
    ax.set_title("Deletion test (validation sample)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, path)


def cmd_deletion(ctx: RunContext, args: argparse.Namespace) -> int:
    """Deletion test on the validation sample, resumable per image (maps_val.npz must cover the sample)."""
    cfg = ctx.explain_cfg
    data = _subset_data(ctx, SUBSET_VAL)
    maps = _read_maps(ctx, SUBSET_VAL)
    sample = _validation_sample(ctx, data.ids)
    ids = sample[: min(cfg.n_deletion, len(sample))]
    absent = [i for i in ids if i not in maps.index]
    if absent:
        raise ExplainRefused(
            f"maps_val.npz lacks {len(absent)} of the sampled images; run scripts/explain.py maps "
            f"--run {ctx.run_dir} --subset val (all validation ids)"
        )
    ddir = ctx.out_dir / DELETION_DIRNAME
    ddir.mkdir(parents=True, exist_ok=True)
    partial = ddir / "deletion.partial.csv"
    done = _read_deletion_partial(partial)
    transform = build_transforms(ctx.loaded.run.model.input_size, False, ctx.loaded.run.aug, ctx.data_cfg.target_size)

    for position, image_id in enumerate(ids):
        if not done.empty and image_id in set(done["image_id"]):
            continue
        rows = _deletion_rows(ctx, data, maps, image_id, position, transform)
        done = _append_rows(done, rows)
        _write_csv(partial, done, DELETION_COLUMNS)
        logger.info("deletion: %d of %d images", len(set(done["image_id"])), len(ids))
        pending = [i for i in ids if i not in set(done["image_id"])]
        if pending and _time_up(ctx):
            return STATUS_INCOMPLETE

    frame = done.reset_index(drop=True)
    _write_csv(ddir / "deletion.csv", frame, DELETION_COLUMNS)
    summary = _deletion_summary(frame)
    _write_csv(ddir / "summary.csv", summary, list(summary.columns))
    _plot_deletion(summary, ddir / "deletion.png")
    if partial.exists():
        partial.unlink()
    logger.info("wrote deletion outputs for %d images", len(ids))
    return STATUS_OK


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def _finite_or_none(value: float) -> float | None:
    """Return a float, or None for NaN (JSON has no NaN)."""
    return float(value) if np.isfinite(value) else None


def _mean_or_none(values: pd.Series, median: bool = False) -> float | None:
    """Mean or median of the finite values, or None when there are none."""
    array = values.to_numpy(dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.median(finite)) if median else float(finite.mean())


def _share_above(mass: pd.Series, area: pd.Series) -> dict[str, Any]:
    """Share of images whose mass share exceeds SHORTCUT_RATIO times their area share (finite pairs only)."""
    a = mass.to_numpy(dtype=np.float64)
    b = area.to_numpy(dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    share = float(np.mean(a[ok] > SHORTCUT_RATIO * b[ok])) if ok.any() else None
    return {"share": share, "n_valid": int(ok.sum())}


def _shortcut_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """The shortcut summary written to shortcut_summary.json."""
    summary: dict[str, Any] = {"n": int(len(frame))}
    for measure in ("edge_mass", "edge_area", "disc_mass", "disc_area"):
        summary[f"mean_{measure}"] = _mean_or_none(frame[measure])
        summary[f"median_{measure}"] = _mean_or_none(frame[measure], median=True)
    summary["share_edge_mass_above_2x_edge_area"] = _share_above(frame["edge_mass"], frame["edge_area"])
    summary["share_disc_mass_above_2x_disc_area"] = _share_above(frame["disc_mass"], frame["disc_area"])
    by_class: dict[str, Any] = {}
    for grade, part in frame.groupby("pred_grade"):
        by_class[str(int(grade))] = {
            "n": int(len(part)),
            "mean_edge_mass": _mean_or_none(part["edge_mass"]),
            "mean_edge_area": _mean_or_none(part["edge_area"]),
            "mean_disc_mass": _mean_or_none(part["disc_mass"]),
            "mean_disc_area": _mean_or_none(part["disc_area"]),
        }
    summary["by_predicted_class"] = by_class
    return summary


def _disc_check_figure(ctx: RunContext, ids: Sequence[str], path: Path) -> None:
    """Grid of images with the optic-disc proxy circle drawn, to check the proxy by eye."""
    cfg = ctx.explain_cfg
    folder = image_folder(ctx.data_cfg)
    panels: list[tuple[Image.Image, str]] = []
    for image_id in ids:
        image = _read_rgb(folder, image_id)
        size = image.shape[0]
        row, col, _ = brightest_region(image, cfg.disc_radius_frac, cfg.disc_smooth_frac, cfg.edge_frac)
        radius = cfg.disc_radius_frac * (size / 2.0)
        canvas = Image.fromarray(image.copy())
        ImageDraw.Draw(canvas).ellipse(
            [int(round(col - radius)), int(round(row - radius)), int(round(col + radius)), int(round(row + radius))],
            outline=DISC_OUTLINE,
            width=max(2, size // 128),
        )
        panels.append((canvas, f"{image_id}  proxy centre row {row}, col {col}"))
    if not panels:
        return
    rows = math.ceil(len(panels) / DISC_CHECK_COLUMNS)
    width = DISC_CHECK_COLUMNS * DISC_CHECK_PANEL_INCH
    height = rows * DISC_CHECK_PANEL_INCH
    # Only this figure changes its dpi: the longest side stays within DISC_CHECK_MAX_SIDE pixels.
    dpi = min(FIGURE_DPI, int(DISC_CHECK_MAX_SIDE // max(width, height)))
    fig, axes = plt.subplots(rows, DISC_CHECK_COLUMNS, figsize=(width, height), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes:
        ax.axis("off")
    for ax, (canvas, caption) in zip(flat_axes, panels):
        ax.imshow(np.asarray(canvas.convert("RGB")))
        ax.set_title(caption, fontsize=7)
    fig.suptitle("Optic-disc proxy (brightest region away from the edge)", fontsize=10)
    fig.tight_layout()
    _save_figure(fig, path, dpi=dpi)


def cmd_stats(ctx: RunContext, args: argparse.Namespace) -> int:
    """Shortcut check, per-image heatmap statistics and the lesion comparison (all validation images)."""
    cfg = ctx.explain_cfg
    data = _subset_data(ctx, SUBSET_VAL)
    maps = _read_maps(ctx, SUBSET_VAL)
    absent = [i for i in data.ids if i not in maps.index]
    if absent:
        raise ExplainRefused(
            f"maps_val.npz does not cover all validation images ({len(absent)} missing); run "
            f"scripts/explain.py maps --run {ctx.run_dir} --subset val without --ids-from"
        )
    folder = image_folder(ctx.data_cfg)
    records: list[dict[str, Any]] = []
    for image_id in data.ids:
        row = maps.index[image_id]
        meta = data.rows.loc[image_id]
        image = _read_rgb(folder, image_id)
        size = image.shape[0]
        saliency = upsample(np.asarray(maps.cams[row])[None], size)[0]
        region = fundus_mask(size)
        edge = edge_band(size, cfg.edge_frac)
        disc_row, disc_col, disc = brightest_region(image, cfg.disc_radius_frac, cfg.disc_smooth_frac, cfg.edge_frac)
        record: dict[str, Any] = {
            "image_id": image_id,
            "true_grade": int(meta["grade"]),
            "pred_grade": int(maps.target[row]),
            "spread_50": spread_50(saliency, region),
            "peak": float(np.max(maps.cams[row])),
            "edge_mass": mass_share(saliency, region, edge),
            "edge_area": area_share(region, edge),
            "disc_mass": mass_share(saliency, region, disc),
            "disc_area": area_share(region, disc),
            "disc_row": disc_row,
            "disc_col": disc_col,
            "border_level": float(meta["border_level"]),
        }
        for column in LESION_COLS:
            record[column] = int(meta[column])
        records.append(record)
    frame = pd.DataFrame(records)

    sdir = ctx.out_dir / STATS_DIRNAME
    sdir.mkdir(parents=True, exist_ok=True)
    _write_csv(sdir / "heatmap_stats.csv", frame, list(frame.columns))
    summary = _shortcut_summary(frame)
    write_atomically(
        sdir / "shortcut_summary.json",
        lambda p: p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"),
    )
    all_rows = lesion_comparison(frame, LESION_COLS, LESION_MEASURES, cfg.min_group)
    all_rows.insert(0, "scope", "all")
    referable = frame[frame["true_grade"] >= REFERABLE_MIN_GRADE]
    referable_rows = lesion_comparison(referable, LESION_COLS, LESION_MEASURES, cfg.min_group)
    referable_rows.insert(0, "scope", "referable")
    comparison = pd.concat([all_rows, referable_rows], ignore_index=True)
    _write_csv(sdir / "lesion_comparison.csv", comparison, list(comparison.columns))

    _disc_check_figure(ctx, _validation_sample(ctx, data.ids)[:DISC_CHECK_COUNT], sdir / "disc_check.png")
    logger.info("wrote shortcut, heatmap and lesion outputs for %d images", len(frame))
    return STATUS_OK


HANDLERS: dict[str, Callable[[RunContext, argparse.Namespace], int]] = {
    "maps": cmd_maps,
    "panels": cmd_panels,
    "randomisation": cmd_randomisation,
    "deletion": cmd_deletion,
    "stats": cmd_stats,
}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser with one subcommand per step."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True, type=Path, help="run folder with best.pt, config.yaml and predictions/")
    common.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    common.add_argument("--explain-config", type=Path, default=Path("configs/explain.yaml"))
    common.add_argument("--device", default="cpu", help="cpu (default), cuda or mps")
    common.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    common.add_argument("--max-seconds", type=float, default=None, help="stop after the first unit past this; exit 3")
    common.add_argument("--out", type=Path, default=None, help="output folder (default <run>/explain/)")

    parser = argparse.ArgumentParser(description="Explainability checks for one trained run.")
    steps = parser.add_subparsers(dest="step", required=True)
    maps = steps.add_parser("maps", parents=[common], help="Grad-CAM maps for the predicted class")
    maps.add_argument("--subset", choices=SUBSETS, default=SUBSET_VAL)
    maps.add_argument("--ids-from", type=Path, default=None, help="CSV with image_id; required for test")
    panels = steps.add_parser("panels", parents=[common], help="figures for the representative cases")
    panels.add_argument("--subset", choices=SUBSETS, default=SUBSET_VAL)
    panels.add_argument("--cases", type=Path, default=None, help="cases CSV (default predictions/errors_<subset>/cases.csv)")
    steps.add_parser("randomisation", parents=[common], help="model randomisation test (12.3.1)")
    steps.add_parser("deletion", parents=[common], help="deletion test (12.3.2)")
    steps.add_parser("stats", parents=[common], help="shortcut check and lesion comparison (12.3.3, 12.4)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explainability step and return the exit status."""
    setup_logging()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        ctx = _context(args, raw_args)
        return HANDLERS[args.step](ctx, args)
    except ExplainRefused as exc:
        logger.error("%s", exc)
        return STATUS_REFUSED
    except (ValueError, FileNotFoundError, ProvenanceError) as exc:
        logger.error("%s", exc)
        return STATUS_REFUSED
    except Exception:
        logger.exception("explainability step failed")
        return STATUS_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
