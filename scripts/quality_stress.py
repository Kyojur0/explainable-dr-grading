"""Stage 7: synthetic image-quality stress test on the validation subset (DECISIONS.md D33).

Each level perturbs the processed validation images (Gaussian blur, or darkening), then runs the
run's best.pt with the evaluation transform and no TTA. The unperturbed baseline ("none") goes
through the same code path. Test images are never used.

Usage (repeat until the command exits with status 0):
    PYTHONPATH=src python scripts/quality_stress.py --run runs/e2_effb0_512_ce --max-seconds 150

Each level is saved as <run>/quality_stress/<level>.csv with the prediction columns, and an
interrupted level resumes from <level>.csv.partial.csv (predict_subset, DECISIONS.md D12).
When every level is complete the script writes summary.csv and quality_stress.png.
The temperature comes from <run>/calibration/calibration.json when it exists, otherwise the
confidence is uncalibrated and the summary says so. The referral threshold comes from
<run>/referral/referral.json when it exists.

Exit status: 0 done, 2 bad input, 3 stopped at --max-seconds with work left (run again), 1 unexpected error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from drgrade.calibration import CALIBRATION_JSON, apply_temperature
from drgrade.config import ConfigError, load_data_config
from drgrade.data import IMAGE_SUFFIX, FundusDataset, build_transforms
from drgrade.metrics import full_metrics
from drgrade.predict import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    LOGIT_COLUMNS,
    SUBSET_VAL,
    build_subset,
    check_data_settings,
    image_folder,
    load_model,
    predict_subset,
)
from drgrade.preprocess import apply_circle_mask
from drgrade.reliability_figures import save_stress
from drgrade.run_config import RunConfigError
from drgrade.utils import select_device, setup_logging, write_atomically

logger = logging.getLogger(__name__)

STATUS_OK = 0
STATUS_UNEXPECTED = 1
STATUS_BAD_INPUT = 2
STATUS_INCOMPLETE = 3
STRESS_DIR = "quality_stress"
SUMMARY_CSV = "summary.csv"
STRESS_PNG = "quality_stress.png"
REFERRAL_JSON = "referral.json"
KIND_NONE = "none"
KIND_BLUR = "blur"
KIND_DARKEN = "darken"
BLUR_RADII_PX: tuple[float, ...] = (1.0, 2.0, 4.0)  # Gaussian radius in pixels at the processed size (DECISIONS.md D33)
DARKEN_FACTORS: tuple[float, ...] = (0.8, 0.6, 0.4)  # RGB multiplied by these factors
UINT8_MAX = 255


@dataclass(frozen=True)
class Perturbation:
    """One stress level: kind 'none', 'blur' (value = radius in px) or 'darken' (value = factor)."""

    kind: str
    value: float

    def __post_init__(self) -> None:
        """Reject unknown kinds and out-of-range values."""
        if self.kind == KIND_NONE:
            return
        if self.kind == KIND_BLUR and self.value > 0:
            return
        if self.kind == KIND_DARKEN and 0.0 < self.value <= 1.0:
            return
        raise ValueError(f"invalid perturbation {self.kind!r} with value {self.value}")

    @property
    def name(self) -> str:
        """Return the level name used for files and rows, for example 'none', 'blur_r2' or 'darken_f0.6'."""
        if self.kind == KIND_NONE:
            return KIND_NONE
        if self.kind == KIND_BLUR:
            return f"{KIND_BLUR}_r{self.value:g}"
        return f"{KIND_DARKEN}_f{self.value:g}"

    @property
    def severity(self) -> float:
        """Return 0 for the baseline, the blur radius in px, or 1 minus the darkening factor."""
        if self.kind == KIND_NONE:
            return 0.0
        if self.kind == KIND_BLUR:
            return self.value
        return 1.0 - self.value

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return the perturbed RGB image. Pixels outside the inscribed circle stay black for blur."""
        if self.kind == KIND_NONE:
            return image
        if self.kind == KIND_BLUR:
            width, height = image.size
            if width != height:
                raise ValueError(f"blur needs a square image, got {image.size}")
            blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=self.value))).copy()
            blurred[~_inside_circle(width)] = 0
            return Image.fromarray(blurred)
        scaled = np.asarray(image).astype(np.float64) * self.value
        return Image.fromarray(np.clip(np.rint(scaled), 0, UINT8_MAX).astype(np.uint8))


@lru_cache(maxsize=8)
def _inside_circle(size: int) -> np.ndarray:
    """Return a boolean mask of the circle inscribed in a size by size square, using the preprocessing geometry."""
    white = Image.new("RGB", (size, size), (UINT8_MAX, UINT8_MAX, UINT8_MAX))
    return np.asarray(apply_circle_mask(white))[..., 0] > 0


class PerturbedDataset(FundusDataset):
    """FundusDataset that applies a perturbation to the RGB image before the evaluation transform."""

    def __init__(
        self,
        table: pd.DataFrame,
        image_dir: Path | str,
        transform: Callable[[Image.Image], Any],
        perturbation: Perturbation,
    ) -> None:
        """Store the table, folder, transform and the perturbation to apply."""
        super().__init__(table, image_dir, transform)
        self._perturbation = perturbation

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        """Load one JPEG as RGB, perturb it, transform it, and return it with its grade and id."""
        image_id = self._image_ids[index]
        with Image.open(self._image_dir / f"{image_id}{IMAGE_SUFFIX}") as handle:
            rgb = handle.convert("RGB")
        return self._transform(self._perturbation(rgb)), self._grades[index], image_id


def levels() -> list[Perturbation]:
    """Return the baseline followed by the blur and darkening levels, in a fixed order."""
    out = [Perturbation(KIND_NONE, 0.0)]
    out += [Perturbation(KIND_BLUR, radius) for radius in BLUR_RADII_PX]
    out += [Perturbation(KIND_DARKEN, factor) for factor in DARKEN_FACTORS]
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Stage 7 image-quality stress test.")
    parser.add_argument("--run", required=True, type=Path, help="run folder with best.pt and config.yaml")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=None, help="overrides processed_dir from the data config")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda, then mps, then cpu)")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop after the first batch past this; exit 3")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default <run>/quality_stress/)")
    parser.add_argument("--calibration", type=Path, default=None, help="calibration.json (default <run>/calibration/)")
    parser.add_argument("--referral", type=Path, default=None, help="referral.json (default <run>/referral/)")
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON file, or None when it does not exist."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration_and_threshold(
    run_dir: Path, calibration_path: Path | None, referral_path: Path | None
) -> tuple[float | None, float | None]:
    """Return the temperature (None when not calibrated) and the primary referral threshold (None when unavailable)."""
    cal_file = calibration_path if calibration_path is not None else run_dir / "calibration" / CALIBRATION_JSON
    ref_file = referral_path if referral_path is not None else run_dir / "referral" / REFERRAL_JSON
    cal = _read_json(cal_file)
    ref = _read_json(ref_file)
    if cal is None:
        logger.warning("no %s: confidence is uncalibrated and no referral fraction is reported", cal_file)
        if ref is not None:
            logger.warning("%s is ignored because it is defined on calibrated confidence", ref_file)
        return None, None
    temperature = float(cal["temperature"])
    if ref is None:
        logger.warning("no %s: the referral fraction is not reported", ref_file)
        return temperature, None
    return temperature, float(ref["primary"]["threshold"])


def _level_row(
    level: Perturbation,
    path: Path,
    temperature: float | None,
    threshold: float | None,
) -> dict[str, Any]:
    """Return one summary row for a finished level: metrics, mean confidence and referral fraction."""
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    metrics = full_metrics(frame, n_boot=0)
    logits = frame[list(LOGIT_COLUMNS)].to_numpy().astype(np.float64)
    confidence = apply_temperature(logits, temperature if temperature is not None else 1.0).max(axis=1)
    return {
        "level": level.name,
        "kind": level.kind,
        "value": level.value,
        "severity": level.severity,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "qwk": metrics["qwk"],
        "referable_miss_rate": metrics["errors"]["referable_miss_rate"],
        "mean_confidence": float(confidence.mean()),
        "referral_fraction": float((confidence <= threshold).mean()) if threshold is not None else None,
        "confidence_scale": "calibrated" if temperature is not None else "uncalibrated",
        "temperature": temperature,
        "referral_threshold": threshold,
        "predictions": path.name,
    }


def main(argv: list[str] | None = None) -> int:
    """Run or resume the stress levels; write the summary when every level is complete. Returns the exit status."""
    setup_logging()
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run
    out_dir = args.out if args.out is not None else run_dir / STRESS_DIR
    clock = time.monotonic
    started = clock()
    try:
        data_cfg = load_data_config(args.data_config)
        if args.processed_dir is not None:
            data_cfg = replace(data_cfg, processed_dir=str(args.processed_dir))
        if args.batch_size < 1 or args.num_workers < 0:
            raise ValueError("--batch-size must be at least 1 and --num-workers non-negative")
        device = select_device(args.device)
        loaded = load_model(run_dir, device)
        check_data_settings(loaded, data_cfg)
        table, _ = build_subset(loaded, data_cfg, SUBSET_VAL)
        folder = image_folder(data_cfg)
        transform = build_transforms(loaded.run.model.input_size, False, loaded.run.aug, data_cfg.target_size)
        temperature, threshold = _calibration_and_threshold(run_dir, args.calibration, args.referral)
        out_dir.mkdir(parents=True, exist_ok=True)
        for level in levels():
            out_path = out_dir / f"{level.name}.csv"
            if out_path.is_file():
                continue
            budget = None if args.max_seconds is None else max(0.0, args.max_seconds - (clock() - started))
            dataset = PerturbedDataset(table, folder, transform, level)
            logger.info("level %s (budget %s s)", level.name, "unlimited" if budget is None else f"{budget:.0f}")
            complete = predict_subset(
                loaded,
                table,
                dataset,
                out_path,
                device,
                tta=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_seconds=budget,
                clock=clock,
            )
            if not complete:
                logger.info("stopped in level %s; run the same command again", level.name)
                return STATUS_INCOMPLETE
        rows = [_level_row(level, out_dir / f"{level.name}.csv", temperature, threshold) for level in levels()]
    except (ConfigError, RunConfigError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return STATUS_BAD_INPUT
    except Exception:
        logger.exception("quality stress failed")
        return STATUS_UNEXPECTED

    summary = pd.DataFrame(rows)
    write_atomically(out_dir / SUMMARY_CSV, lambda p: summary.to_csv(p, index=False))
    save_stress(summary, out_dir / STRESS_PNG, f"{run_dir.name}: validation quality stress")
    logger.info("summary written to %s", out_dir / SUMMARY_CSV)
    return STATUS_OK


if __name__ == "__main__":
    sys.exit(main())
