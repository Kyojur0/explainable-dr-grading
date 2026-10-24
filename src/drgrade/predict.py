"""Stage 5: predictions from a trained run, resumable in chunks, with the test-set lock (brief 9.1).

Entry points:
- load_model(run_dir, device): rebuilds the architecture with pretrained=False and loads
  best.pt (keys model_state, epoch, metrics, config; see train_loop._save_best).
- predict_subset(...): predicts one subset in batches and rewrites <out>.partial.csv
  after each batch. It returns False when --max-seconds stops the run before the end; the
  final CSV is written atomically, sorted by image_id, once every image is done.
- decide_test_access(...), record_test_output(...), close_test_lock(...): the test-set
  lock in <run>/TEST_LOCK.json (DECISIONS.md D23).

Test-time augmentation (DECISIONS.md D24) averages the softmax probabilities over four
views of the normalised batch: identity, horizontal flip, vertical flip, and both flips
(a 180 degree rotation). The logit columns of a TTA file are the log of the averaged
probabilities. A plain file holds the raw logits of the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import Subset

from drgrade.config import DataConfig
from drgrade.data import FundusDataset, build_transforms, load_run_table, make_loader
from drgrade.io import GRADE_VALUES
from drgrade.models import build_model
from drgrade.run_config import RunConfig, parse_run_config
from drgrade.train_loop import RESUME_FREE_KEYS
from drgrade.utils import get_git_info, utc_now_iso, write_atomically

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "best.pt"
RUN_CONFIG_NAME = "config.yaml"
LOCK_NAME = "TEST_LOCK.json"
PREDICTIONS_DIR = "predictions"
MANIFEST_NAME = "manifest.csv"
PARTIAL_SUFFIX = ".partial.csv"
SUBSET_VAL = "val"
SUBSET_TEST = "test"
SUBSETS: tuple[str, ...] = (SUBSET_VAL, SUBSET_TEST)
LOCK_OPEN = "open"
LOCK_CLOSED = "closed"
PROB_COLUMNS: tuple[str, ...] = tuple(f"prob_{g}" for g in GRADE_VALUES)
LOGIT_COLUMNS: tuple[str, ...] = tuple(f"logit_{g}" for g in GRADE_VALUES)
PREDICTION_COLUMNS: tuple[str, ...] = ("image_id", "true_grade", "pred_grade", *PROB_COLUMNS, *LOGIT_COLUMNS)
HASH_CHUNK_BYTES = 1 << 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
TTA_SUFFIX = "_tta"
TTA_FLIPS_LAST_AXIS = -1
TTA_FLIPS_HEIGHT_AXIS = -2
PLAIN_VARIANT = "plain"
TTA_VARIANT = "tta"


class PredictionRefused(RuntimeError):
    """Raised when the test-set lock or a precondition forbids the requested prediction."""


@dataclass(frozen=True)
class LoadedModel:
    """A trained model in eval mode, with the run settings it was trained with."""

    model: nn.Module
    run: RunConfig
    checkpoint_sha256: str
    saved_target_size: int | None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(run_dir: Path | str, device: torch.device) -> LoadedModel:
    """Rebuild the run's network with random weights, load best.pt into it, and return it in eval mode.

    The run settings come from the checkpoint. config.yaml must agree with them, except for
    the keys that a resume may change (max_epochs and num_workers). A disagreement raises
    ValueError, because the two files would then describe different models.
    """
    run_path = Path(run_dir)
    checkpoint_path = run_path / CHECKPOINT_NAME
    config_path = run_path / RUN_CONFIG_NAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{checkpoint_path} is missing")
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run = parse_run_config(checkpoint["config"])
    saved_run, saved_target = _read_run_config_yaml(config_path)
    _check_same_run(saved_run, run.to_dict())
    model = build_model(run, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    if device.type == "cpu":
        # channels_last is about 1.7x faster for this network on CPU and gave
        # bit-identical logits in a check on the candidate's machine (DECISIONS.md D36).
        model.to(memory_format=torch.channels_last)
    return LoadedModel(
        model=model,
        run=run,
        checkpoint_sha256=sha256_file(checkpoint_path),
        saved_target_size=saved_target,
    )


def _read_run_config_yaml(path: Path) -> tuple[dict[str, Any], int | None]:
    """Return the run section and the recorded target size from config.yaml."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "run" not in raw:
        raise ValueError(f"{path} has no run section")
    data = raw.get("data") or {}
    target = data.get("target_size")
    return dict(raw["run"]), (int(target) if target is not None else None)


def _check_same_run(saved: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    """Raise ValueError when config.yaml and the checkpoint disagree on a setting that matters."""
    a = {k: v for k, v in saved.items() if k not in RESUME_FREE_KEYS}
    b = {k: v for k, v in checkpoint.items() if k not in RESUME_FREE_KEYS}
    if a != b:
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise ValueError(f"config.yaml and best.pt disagree on {changed}")


def image_folder(data_cfg: DataConfig) -> Path:
    """Return processed_dir/s<target_size>, the folder that holds the JPEGs and manifest.csv."""
    return Path(data_cfg.processed_dir) / f"s{data_cfg.target_size}"


def check_data_settings(loaded: LoadedModel, data_cfg: DataConfig) -> None:
    """Raise ValueError when the processed size differs from the size the run was trained on."""
    if loaded.saved_target_size is not None and loaded.saved_target_size != data_cfg.target_size:
        raise ValueError(
            f"the run was trained on target_size {loaded.saved_target_size}; "
            f"the data config says {data_cfg.target_size}"
        )


def build_subset(loaded: LoadedModel, data_cfg: DataConfig, subset: str) -> tuple[pd.DataFrame, FundusDataset]:
    """Return the subset rows (sorted by image id) and a dataset with the evaluation transform.

    The test subset is allowed here only. The caller has already passed the lock check, and
    this is the one place that reads the test subset.
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    folder = image_folder(data_cfg)
    table = load_run_table(
        folder / MANIFEST_NAME,
        Path(loaded.run.data.split_file),
        (subset,),
        folder,
        allow_test=subset == SUBSET_TEST,
    )
    transform = build_transforms(loaded.run.model.input_size, False, loaded.run.aug, data_cfg.target_size)
    return table, FundusDataset(table, folder, transform)


def _softmax64(logits: np.ndarray) -> np.ndarray:
    """Return softmax probabilities computed in float64 from float64 logits (as in train_loop)."""
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1).numpy()


@torch.inference_mode()
def batch_outputs(model: nn.Module, images: torch.Tensor, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, logit columns) for a batch of normalised images, both float64 with shape N by 5.

    Plain: the raw logits and their softmax. TTA: the softmax averaged over the four views;
    the logit columns are the log of those averaged probabilities.
    """
    if images.device.type == "cpu":
        images = images.contiguous(memory_format=torch.channels_last)
    if not tta:
        logits = model(images).float().cpu().numpy().astype(np.float64)
        return _softmax64(logits), logits
    views = [
        images,
        images.flip(TTA_FLIPS_LAST_AXIS),
        images.flip(TTA_FLIPS_HEIGHT_AXIS),
        images.flip(TTA_FLIPS_LAST_AXIS).flip(TTA_FLIPS_HEIGHT_AXIS),
    ]
    total = np.zeros((images.shape[0], len(GRADE_VALUES)), dtype=np.float64)
    for view in views:
        if view.device.type == "cpu":
            view = view.contiguous(memory_format=torch.channels_last)
        total += _softmax64(model(view).float().cpu().numpy().astype(np.float64))
    probs = total / len(views)
    return probs, np.log(probs)


def _frame(ids: list[str], labels: np.ndarray, probs: np.ndarray, logits: np.ndarray) -> pd.DataFrame:
    """Return the output rows for a set of images, in the column order of PREDICTION_COLUMNS."""
    frame = pd.DataFrame(
        {
            "image_id": ids,
            "true_grade": labels.astype(int),
            "pred_grade": logits.argmax(axis=1).astype(int),
        }
    )
    for index, grade in enumerate(GRADE_VALUES):
        frame[f"prob_{grade}"] = probs[:, index]
    for index, grade in enumerate(GRADE_VALUES):
        frame[f"logit_{grade}"] = logits[:, index]
    return frame


def _read_partial(path: Path) -> pd.DataFrame:
    """Return the rows already saved in the partial file, or an empty table when there is none.

    Floats are read with round-trip precision, so a resumed file equals an uninterrupted one.
    """
    if not path.is_file():
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    frame = pd.read_csv(path, dtype={"image_id": str}, float_precision="round_trip")
    if list(frame.columns) != list(PREDICTION_COLUMNS):
        raise ValueError(f"{path} does not have the prediction columns; remove it to start again")
    return frame


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write a prediction table atomically with the fixed column order."""
    ordered = frame.loc[:, list(PREDICTION_COLUMNS)]
    write_atomically(path, lambda p: ordered.to_csv(p, index=False))


def predict_subset(
    loaded: LoadedModel,
    table: pd.DataFrame,
    dataset: FundusDataset,
    out_path: Path,
    device: torch.device,
    tta: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Predict every row of table, resuming from <out>.partial.csv. Returns True when the final file is written.

    Each batch is saved to the partial file before the time limit is checked, so a stop loses
    no finished work. When max_seconds has passed with images left, the function returns
    False and the caller exits with status 3; running the same command again continues.
    The partial file is rewritten atomically after each batch, so a killed process cannot
    leave a truncated line behind.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_seconds is not None and max_seconds < 0:
        raise ValueError("max_seconds must be non-negative")
    partial = out_path.with_name(out_path.name + PARTIAL_SUFFIX)
    started = clock()
    done = _read_partial(partial)
    total = len(table)
    table_ids = set(table["image_id"])
    if len(done) and not set(done["image_id"]) <= table_ids:
        raise ValueError(f"{partial} holds ids that are not in this subset; remove it to start again")
    done_ids = set(done["image_id"])
    remaining = [position for position, image_id in enumerate(table["image_id"]) if image_id not in done_ids]
    logger.info("%d of %d images to predict (tta=%s, device=%s)", len(remaining), total, tta, device)

    if remaining:
        loader = make_loader(
            Subset(dataset, remaining),
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=loaded.run.seed,
            device_type=device.type,
        )
        for images, labels, ids in loader:
            probs, logits = batch_outputs(loaded.model, images.to(device), tta)
            batch = _frame(list(ids), labels.numpy(), probs, logits)
            done = pd.concat([done, batch], ignore_index=True) if len(done) else batch
            _write_frame(partial, done)
            logger.info("predicted %d of %d", len(done), total)
            if max_seconds is not None and clock() - started >= max_seconds and len(done) < total:
                logger.info("stopped at --max-seconds with %d images left; run the same command again", total - len(done))
                return False

    if len(done) != total or set(done["image_id"]) != table_ids or done["image_id"].duplicated().any():
        raise RuntimeError("predictions do not cover the subset exactly once")
    final = done.sort_values("image_id").reset_index(drop=True)
    _write_frame(out_path, final)
    if partial.exists():
        partial.unlink()
    logger.info("wrote %s with %d rows", out_path, len(final))
    return True


def default_output(run_dir: Path, subset: str, tta: bool) -> Path:
    """Return <run>/predictions/<subset>[_tta].csv."""
    suffix = TTA_SUFFIX if tta else ""
    return run_dir / PREDICTIONS_DIR / f"{subset}{suffix}.csv"


# ---------------------------------------------------------------------------
# Test-set lock (DECISIONS.md D23)
# ---------------------------------------------------------------------------


def _variant(tta: bool) -> str:
    """Return the name of a prediction variant in the lock: 'plain' or 'tta'."""
    return TTA_VARIANT if tta else PLAIN_VARIANT


def read_lock(run_dir: Path) -> dict[str, Any] | None:
    """Return the parsed TEST_LOCK.json of a run, or None when it does not exist."""
    path = run_dir / LOCK_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_lock(run_dir: Path, lock: dict[str, Any]) -> None:
    """Write TEST_LOCK.json atomically as sorted, indented JSON."""
    text = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    write_atomically(run_dir / LOCK_NAME, lambda p: p.write_text(text, encoding="utf-8"))


def decide_test_access(
    lock: dict[str, Any] | None,
    checkpoint_sha256: str,
    tta: bool,
    final: bool,
    force: bool,
    reason: str | None,
) -> bool:
    """Return True when the override (--force with --reason) is in use, or raise PredictionRefused.

    Rules (DECISIONS.md D23):
    - a test prediction needs --final;
    - --force and --reason are given together;
    - with no lock, the prediction is allowed (the lock is created when the output is complete);
    - the checkpoint must match the locked hash, unless overridden;
    - while the lock is open, each variant (plain or TTA) may be produced once without an
      override; producing a variant again needs an override;
    - once the lock is closed, every test prediction needs an override.
    """
    if not final:
        raise PredictionRefused("test predictions need --final")
    if force != bool(reason):
        raise PredictionRefused("--force and --reason must be given together")
    override = force and bool(reason)
    if lock is None:
        # No lock in this folder, but the central access log may still record an earlier
        # completed run (for example a reproduction from a fresh clone); the override is
        # passed on so that provenance.check_test_access can allow it and log it.
        return override
    locked_sha = lock.get("checkpoint", {}).get("sha256")
    if locked_sha != checkpoint_sha256 and not override:
        raise PredictionRefused("best.pt differs from the hash in TEST_LOCK.json; pass --force and --reason to rerun")
    status = lock.get("status")
    if status == LOCK_CLOSED and not override:
        raise PredictionRefused("TEST_LOCK.json is closed; pass --force and --reason to rerun on test")
    if status not in (LOCK_OPEN, LOCK_CLOSED):
        raise PredictionRefused(f"TEST_LOCK.json has an unknown status {status!r}; refusing")
    if status == LOCK_OPEN and not override:
        recorded = {entry["variant"] for entry in lock.get("outputs", [])}
        if _variant(tta) in recorded:
            raise PredictionRefused(
                f"the {_variant(tta)} test output is already in TEST_LOCK.json; pass --force and --reason to rerun"
            )
    return override


def record_test_output(
    run_dir: Path,
    run_name: str,
    checkpoint_sha256: str,
    tta: bool,
    out_file: Path,
    rows: int,
    command: str,
    forced: bool,
    reason: str | None,
    csv_sha256: str | None = None,
) -> dict[str, Any]:
    """Create or update TEST_LOCK.json after a complete test prediction has been written.

    `csv_sha256` is the hash of the written CSV; downstream scripts refuse a test file
    whose hash is not in the lock (src/drgrade/provenance.py, DECISIONS.md D38).
    """
    now = utc_now_iso()
    git = get_git_info()
    entry = {
        "variant": _variant(tta),
        "file": out_file.resolve().relative_to(run_dir.resolve()).as_posix(),
        "rows": rows,
        "written_utc": now,
        "command": command,
        "csv_sha256": csv_sha256,
    }
    lock = read_lock(run_dir)
    if lock is None:
        lock = {
            "run_name": run_name,
            "status": LOCK_OPEN,
            "created_utc": now,
            "checkpoint": {"file": CHECKPOINT_NAME, "sha256": checkpoint_sha256},
            "git": git,
            "command": command,
            "tta": tta,
            "outputs": [entry],
            "closed_utc": None,
            "reruns": [],
        }
    else:
        recorded = {output["variant"] for output in lock.get("outputs", [])}
        if _variant(tta) not in recorded:
            lock.setdefault("outputs", []).append(entry)
    if forced:
        lock.setdefault("reruns", []).append(
            {
                "time_utc": now,
                "reason": reason,
                "command": command,
                "variant": _variant(tta),
                "file": entry["file"],
                "rows": rows,
                "checkpoint_sha256": checkpoint_sha256,
                "csv_sha256": csv_sha256,
                "git": git,
            }
        )
    _write_lock(run_dir, lock)
    return lock


def close_test_lock(run_dir: Path) -> dict[str, Any]:
    """Set the lock status to closed with a timestamp. Raises PredictionRefused when there is no lock."""
    lock = read_lock(run_dir)
    if lock is None:
        raise PredictionRefused(f"no {LOCK_NAME} in {run_dir}: nothing to close")
    if lock.get("status") == LOCK_CLOSED:
        return lock
    lock["status"] = LOCK_CLOSED
    lock["closed_utc"] = utc_now_iso()
    _write_lock(run_dir, lock)
    return lock


def format_command(argv: Sequence[str]) -> str:
    """Return the command line as one string for the lock file."""
    return " ".join(["scripts/predict.py", *argv])
