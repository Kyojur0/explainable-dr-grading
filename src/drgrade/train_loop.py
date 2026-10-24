"""Stage 4: training loop, checkpoints, resume and validation selection (brief 8.3 to 8.5).

Entry point: train(). It never loads the test subset (load_run_table is called with
the train and val subsets only).

Per epoch the loop trains, evaluates on the validation subset, appends one row to
history.csv, updates the best model when validation QWK improves (ties broken by
macro-F1), writes last.pt (for resume) and best.pt, and stops early after
early_stopping_patience epochs without improvement or at max_epochs.

After the last epoch the best model predicts the validation subset into
val_predictions.csv and summary.json is written. Every file is written through a
temporary name and os.replace, because the output folder may be a synced Google Drive
folder.

Resume (DECISIONS.md D20): last.pt holds the model, optimiser, scheduler, AMP scaler,
history, best metrics and the random number generator states at the end of the last
finished epoch. Resuming restores all of them, so with num_workers 0 on CPU the
resumed weights equal those of an uninterrupted run (tested in
tests/test_train_smoke.py). With DataLoader workers the augmentation streams after a
resume are not guaranteed to match, and the README says so.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from drgrade.config import DataConfig
from drgrade.data import (
    FundusDataset,
    build_transforms,
    load_run_table,
    make_loader,
    steps_per_epoch,
)
from drgrade.io import GRADE_VALUES
from drgrade.losses import build_loss
from drgrade.metrics import validation_metrics
from drgrade.models import build_model
from drgrade.run_config import RunConfig
from drgrade.utils import (
    describe_device,
    get_git_info,
    package_versions,
    seed_everything,
    utc_now_iso,
    write_atomically,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.csv"
FILE_LAST = "last.pt"
FILE_BEST = "best.pt"
FILE_HISTORY = "history.csv"
FILE_PREDICTIONS = "val_predictions.csv"
FILE_SUMMARY = "summary.json"
FILE_ENV = "env.json"
FILE_CONFIG = "config.yaml"
RUN_FILES: tuple[str, ...] = (
    FILE_LAST,
    FILE_BEST,
    FILE_HISTORY,
    FILE_PREDICTIONS,
    FILE_SUMMARY,
    FILE_ENV,
    FILE_CONFIG,
)
HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_qwk",
    "lr",
    "epoch_seconds",
)
# Run config keys that may change between a run and its resume. Everything else must match.
RESUME_FREE_KEYS: frozenset[str] = frozenset({"max_epochs", "num_workers"})

# Smoke test settings (brief 14.3): 64 train and 32 validation images, 2 epochs, with a
# checkpoint after epoch 1 and a resume for epoch 2 in the same process.
SMOKE_TRAIN_IMAGES = 64
SMOKE_VAL_IMAGES = 32
SMOKE_BATCH_SIZE = 8
SMOKE_EPOCHS = 2
SMOKE_STOP_AFTER_EPOCH = 1
SMOKE_OUTPUT_PREFIX = "smoke_"

SELECTION_RULE = "highest validation QWK; ties broken by higher validation macro-F1; early stopping on QWK"


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Return the learning-rate multiplier for optimiser step `step` (0-based).

    Linear warm-up: (step + 1) / warmup_steps for steps before warmup_steps.
    Then cosine decay from 1 to 0 over the remaining steps, reaching 0 at total_steps.
    With warmup_steps 0 the cosine starts at 1.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(f"warmup_steps must be in [0, total_steps), got {warmup_steps} of {total_steps}")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def is_improvement(
    qwk: float,
    macro_f1: float,
    best_qwk: float | None,
    best_macro_f1: float | None,
) -> bool:
    """Return True when this epoch beats the best so far (QWK first, then macro-F1).

    A None best means nothing has been recorded yet. NaN counts as the worst value.
    """
    if best_qwk is None or best_macro_f1 is None:
        return True
    qwk_now, qwk_best = _nan_low(qwk), _nan_low(best_qwk)
    if qwk_now != qwk_best:
        return qwk_now > qwk_best
    return _nan_low(macro_f1) > _nan_low(best_macro_f1)


def _nan_low(value: float) -> float:
    """Map NaN to negative infinity so that comparisons never select it."""
    return -math.inf if math.isnan(value) else float(value)


@dataclass
class _Job:
    """Everything fixed for one call of train(): configs, paths, device and datasets."""

    run: RunConfig
    data: DataConfig
    out_dir: Path
    device: torch.device
    pretrained: bool
    smoke: bool
    image_dir: Path
    train_dataset: FundusDataset
    val_dataset: FundusDataset
    train_grades: list[int]
    val_ids: frozenset[str]


@dataclass
class _State:
    """Mutable training state: model, optimiser and everything saved in last.pt."""

    model: nn.Module
    optimiser: torch.optim.Optimizer
    scheduler: LambdaLR
    scaler: torch.amp.GradScaler
    criterion: nn.Module
    generator: torch.Generator
    train_loader: DataLoader
    val_loader: DataLoader
    epochs_done: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    epochs_without_improvement: int = 0
    stopped_early: bool = False
    finished: bool = False
    train_seconds: float = 0.0
    resumed_from: int = 0


@dataclass
class _Predictions:
    """Loss, ids, true grades and logits for one pass over a loader."""

    loss: float
    ids: list[str]
    labels: np.ndarray
    logits: np.ndarray


def train(
    run_cfg: RunConfig,
    data_cfg: DataConfig,
    out_dir: Path | str,
    device: torch.device,
    pretrained: bool,
    smoke: bool = False,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Train one run and write its outputs to out_dir. Returns a summary dictionary.

    smoke: use 64 train and 32 validation images, batch size at most 8, 2 epochs.
        The run stops after epoch 1 and resumes for epoch 2 in the same process,
        and the call fails unless history.csv has 2 rows and the resume really
        loaded the epoch-1 checkpoint. Any existing run files in out_dir are removed first.
    resume: continue from out_dir/last.pt if it exists. Without resume, an out_dir
        that already holds last.pt raises FileExistsError, so a finished run is never
        overwritten by accident.
    stop_after_epoch: stop once this epoch (1-based) is saved, without marking the
        run finished. Used by the smoke test and the resume test to simulate an
        interruption.

    Only the train and val subsets are loaded. Missing processed data raises
    FileNotFoundError telling the user to run scripts/preprocess.py first.
    """
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be at least 1")
    out = Path(out_dir)
    run = _smoke_settings(run_cfg) if smoke else run_cfg
    manifest_path, image_dir = _processed_paths(data_cfg)
    split_path = Path(run.data.split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"split file not found: {split_path}")

    if smoke:
        _clear_run_files(out)
    elif not resume and (out / FILE_LAST).exists():
        raise FileExistsError(f"{out} already holds a run (last.pt); pass resume or choose another --out")
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(run.seed)
    _env_start(out, run, device, pretrained, smoke)
    try:
        job = _make_job(run, data_cfg, out, device, pretrained, smoke, manifest_path, split_path, image_dir)
        _write_config(job)
        if smoke:
            # First process-like phase: train epoch 1 and save last.pt, then stop.
            state = _build_state(job, resume=False)
            _fit(job, state, stop_after_epoch=SMOKE_STOP_AFTER_EPOCH)
            resume = True
        state = _build_state(job, resume=resume)
        epochs_before = state.epochs_done
        _fit(job, state, stop_after_epoch=stop_after_epoch)
        if state.finished:
            ran = state.epochs_done > epochs_before
            if ran or not (out / FILE_PREDICTIONS).exists():
                _write_predictions(job, state)
            if ran or not (out / FILE_SUMMARY).exists():
                _write_summary(job, state)
        if smoke:
            _check_smoke(job, state)
    finally:
        _env_end(out, utc_now_iso())

    best = state.best or {}
    return {
        "out_dir": str(out),
        "run_name": run.run_name,
        "finished": state.finished,
        "stopped_early": state.stopped_early,
        "epochs_run": state.epochs_done,
        "best_epoch": best.get("epoch"),
        "best_val": best or None,
        "total_seconds": state.train_seconds,
    }


def _smoke_settings(run: RunConfig) -> RunConfig:
    """Return the run with the smoke-test batch size and epoch count."""
    return run.with_overrides(
        batch_size=min(run.batch_size, SMOKE_BATCH_SIZE),
        max_epochs=SMOKE_EPOCHS,
    )


def _processed_paths(data_cfg: DataConfig) -> tuple[Path, Path]:
    """Return the manifest path and the image folder (processed_dir/s<target_size>)."""
    image_dir = Path(data_cfg.processed_dir) / f"s{data_cfg.target_size}"
    manifest = image_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(
            f"processed data not found: {manifest} is missing. Run scripts/preprocess.py first "
            "(or point --processed-dir at the folder that contains s<target_size>/)."
        )
    return manifest, image_dir


def _make_job(
    run: RunConfig,
    data_cfg: DataConfig,
    out: Path,
    device: torch.device,
    pretrained: bool,
    smoke: bool,
    manifest_path: Path,
    split_path: Path,
    image_dir: Path,
) -> _Job:
    """Load the train and validation tables and build the datasets."""
    table = load_run_table(manifest_path, split_path, ("train", "val"), image_dir)
    train_table = table.loc[table["subset"] == "train"]
    val_table = table.loc[table["subset"] == "val"]
    if smoke:
        train_table = train_table.head(SMOKE_TRAIN_IMAGES)
        val_table = val_table.head(SMOKE_VAL_IMAGES)
    size = run.model.input_size
    source = data_cfg.target_size
    train_dataset = FundusDataset(train_table, image_dir, build_transforms(size, True, run.aug, source))
    val_dataset = FundusDataset(val_table, image_dir, build_transforms(size, False, run.aug, source))
    return _Job(
        run=run,
        data=data_cfg,
        out_dir=out,
        device=device,
        pretrained=pretrained,
        smoke=smoke,
        image_dir=image_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_grades=train_table["grade"].astype(int).tolist(),
        val_ids=frozenset(val_table["image_id"].astype(str)),
    )


def _use_amp(device: torch.device) -> bool:
    """Mixed precision is used on CUDA only (brief 4)."""
    return device.type == "cuda"


def _build_state(job: _Job, resume: bool) -> _State:
    """Create the model, optimiser, scheduler, loaders and AMP scaler. Restore last.pt if resuming."""
    run = job.run
    model = build_model(run, pretrained=job.pretrained).to(job.device)
    criterion = build_loss(run, job.train_grades).to(job.device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=run.optimiser.lr,
        weight_decay=run.optimiser.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(run.seed)
    train_loader = make_loader(
        job.train_dataset,
        run.batch_size,
        shuffle=True,
        num_workers=run.num_workers,
        seed=run.seed,
        device_type=job.device.type,
        generator=generator,
    )
    val_loader = make_loader(
        job.val_dataset,
        run.batch_size,
        shuffle=False,
        num_workers=run.num_workers,
        seed=run.seed,
        device_type=job.device.type,
    )
    per_epoch = steps_per_epoch(len(job.train_dataset), run.batch_size, run.accumulation_steps)
    scheduler = LambdaLR(
        optimiser,
        lr_lambda=functools.partial(
            lr_factor,
            warmup_steps=run.schedule.warmup_epochs * per_epoch,
            total_steps=run.max_epochs * per_epoch,
        ),
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=_use_amp(job.device))
    state = _State(
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        scaler=scaler,
        criterion=criterion,
        generator=generator,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    checkpoint_path = job.out_dir / FILE_LAST
    if resume and checkpoint_path.is_file():
        _restore(job, state, torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        logger.info("resumed from %s after epoch %d", checkpoint_path, state.epochs_done)
    elif resume:
        logger.info("no %s found; starting a fresh run", FILE_LAST)
    return state


def _restore(job: _Job, state: _State, ckpt: dict[str, Any]) -> None:
    """Load a checkpoint into the state, after checking that the run config matches."""
    _check_resume_config(ckpt["config"], job.run.to_dict())
    state.model.load_state_dict(ckpt["model_state"])
    state.optimiser.load_state_dict(ckpt["optimiser_state"])
    state.scheduler.load_state_dict(ckpt["scheduler_state"])
    state.scaler.load_state_dict(ckpt["scaler_state"])
    _restore_rng(ckpt["rng"], state.generator)
    state.epochs_done = int(ckpt["epochs_done"])
    state.history = list(ckpt["history"])
    state.best = ckpt["best"]
    state.epochs_without_improvement = int(ckpt["epochs_without_improvement"])
    state.stopped_early = bool(ckpt["stopped_early"])
    state.finished = bool(ckpt["finished"])
    state.train_seconds = float(ckpt["train_seconds"])
    state.resumed_from = state.epochs_done


def _check_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse to resume when anything except the epoch limit and worker count has changed."""

    saved_part = {k: v for k, v in saved.items() if k not in RESUME_FREE_KEYS}
    current_part = {k: v for k, v in current.items() if k not in RESUME_FREE_KEYS}
    if saved_part != current_part:
        changed = sorted(k for k in set(saved_part) | set(current_part) if saved_part.get(k) != current_part.get(k))
        raise ValueError(f"resume refused: the run config differs from last.pt in {changed}")


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    """Return every random number generator state that influences training."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    """Put back the generator states saved by _capture_rng."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _fit(job: _Job, state: _State, stop_after_epoch: int | None) -> None:
    """Run epochs until the run finishes or stop_after_epoch is reached."""
    run = job.run
    while not state.finished and state.epochs_done < run.max_epochs:
        epoch = state.epochs_done + 1
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            logger.info("stopping before epoch %d as requested", epoch)
            return
        started = time.monotonic()
        train_loss, lr = _train_one_epoch(job, state)
        val = _evaluate(job, state.model, state.val_loader, state.criterion)
        metrics = validation_metrics(val.labels, val.logits.argmax(axis=1))
        seconds = time.monotonic() - started
        state.train_seconds += seconds
        state.history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val.loss),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_qwk": metrics["qwk"],
                "lr": lr,
                "epoch_seconds": seconds,
            }
        )
        state.epochs_done = epoch
        best_qwk = None if state.best is None else state.best["qwk"]
        best_f1 = None if state.best is None else state.best["macro_f1"]
        if is_improvement(metrics["qwk"], metrics["macro_f1"], best_qwk, best_f1):
            state.best = {
                "epoch": epoch,
                "val_loss": float(val.loss),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "qwk": metrics["qwk"],
            }
            state.epochs_without_improvement = 0
            _save_best(job, state)
        else:
            state.epochs_without_improvement += 1
        state.stopped_early = state.epochs_without_improvement >= run.early_stopping_patience
        state.finished = state.stopped_early or state.epochs_done >= run.max_epochs
        _save_last(job, state)
        _write_history(job, state)
        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_qwk=%.4f val_macro_f1=%.4f lr=%.2e %.1fs",
            epoch,
            run.max_epochs,
            train_loss,
            val.loss,
            metrics["qwk"],
            metrics["macro_f1"],
            lr,
            seconds,
        )
        if state.stopped_early:
            logger.info("early stopping after %d epochs without improvement", state.epochs_without_improvement)


def _train_one_epoch(job: _Job, state: _State) -> tuple[float, float]:
    """Run one training epoch with gradient accumulation and clipping.

    Returns the mean sample loss and the learning rate used by the epoch's last optimiser step.
    """
    run = job.run
    device = job.device
    use_amp = _use_amp(device)
    state.model.train()
    accumulation = run.accumulation_steps
    n_batches = len(state.train_loader)
    total_loss = 0.0
    n_seen = 0
    last_lr = float(state.optimiser.param_groups[0]["lr"])
    state.optimiser.zero_grad(set_to_none=True)
    for batch_index, (images, labels, _ids) in enumerate(state.train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = state.model(images)
        loss = state.criterion(logits.float(), labels)
        state.scaler.scale(loss / accumulation).backward()
        total_loss += float(loss.detach()) * labels.size(0)
        n_seen += labels.size(0)
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == n_batches:
            state.scaler.unscale_(state.optimiser)
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), run.grad_clip_norm)
            state.scaler.step(state.optimiser)
            state.scaler.update()
            last_lr = float(state.optimiser.param_groups[0]["lr"])
            state.scheduler.step()
            state.optimiser.zero_grad(set_to_none=True)
    return total_loss / n_seen, last_lr


def _autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """Return an autocast context for CUDA, or a no-op context elsewhere."""
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def _evaluate(job: _Job, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> _Predictions:
    """Run the model over a loader without gradients. Returns loss, ids, labels and logits."""
    model.eval()
    device = job.device
    use_amp = _use_amp(device)
    total_loss = 0.0
    n_seen = 0
    ids: list[str] = []
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    for images, labels, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images)
        logits = logits.float()
        loss = criterion(logits, labels_dev)
        total_loss += float(loss) * labels.size(0)
        n_seen += labels.size(0)
        ids.extend(batch_ids)
        labels_all.append(labels.numpy())
        logits_all.append(logits.cpu().numpy())
    return _Predictions(
        loss=total_loss / n_seen,
        ids=ids,
        labels=np.concatenate(labels_all),
        logits=np.concatenate(logits_all),
    )


def _write_predictions(job: _Job, state: _State) -> None:
    """Predict the validation subset with best.pt and write val_predictions.csv, sorted by id."""
    best_path = job.out_dir / FILE_BEST
    if not best_path.is_file():
        raise FileNotFoundError(f"{best_path} is missing; cannot write validation predictions")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model = build_model(job.run, pretrained=False).to(job.device)
    model.load_state_dict(checkpoint["model_state"])
    pred = _evaluate(job, model, state.val_loader, state.criterion)
    if not set(pred.ids) <= job.val_ids:
        raise RuntimeError("validation predictions contain ids outside the validation subset")

    probs = torch.softmax(torch.as_tensor(pred.logits, dtype=torch.float64), dim=1).numpy()
    frame = pd.DataFrame(
        {
            "image_id": pred.ids,
            "true_grade": pred.labels.astype(int),
            "pred_grade": pred.logits.argmax(axis=1).astype(int),
        }
    )
    for index, grade in enumerate(GRADE_VALUES):
        frame[f"prob_{grade}"] = probs[:, index]
    for index, grade in enumerate(GRADE_VALUES):
        frame[f"logit_{grade}"] = pred.logits[:, index]
    frame = frame.sort_values("image_id").reset_index(drop=True)
    write_atomically(job.out_dir / FILE_PREDICTIONS, lambda p: frame.to_csv(p, index=False))
    logger.info("wrote %s with %d rows", FILE_PREDICTIONS, len(frame))


def _save_last(job: _Job, state: _State) -> None:
    """Save the complete training state at the end of an epoch (for --resume)."""
    payload = {
        "model_state": state.model.state_dict(),
        "optimiser_state": state.optimiser.state_dict(),
        "scheduler_state": state.scheduler.state_dict(),
        "scaler_state": state.scaler.state_dict(),
        "epochs_done": state.epochs_done,
        "history": state.history,
        "best": state.best,
        "epochs_without_improvement": state.epochs_without_improvement,
        "stopped_early": state.stopped_early,
        "finished": state.finished,
        "train_seconds": state.train_seconds,
        "rng": _capture_rng(state.generator),
        "config": job.run.to_dict(),
    }
    write_atomically(job.out_dir / FILE_LAST, lambda p: torch.save(payload, p))


def _save_best(job: _Job, state: _State) -> None:
    """Save the model that has the best validation metrics so far (best.pt)."""
    payload = {
        "model_state": state.model.state_dict(),
        "epoch": state.epochs_done,
        "metrics": state.best,
        "config": job.run.to_dict(),
    }
    write_atomically(job.out_dir / FILE_BEST, lambda p: torch.save(payload, p))


def _write_history(job: _Job, state: _State) -> None:
    """Write history.csv from the in-memory history (atomically)."""
    frame = pd.DataFrame(state.history, columns=list(HISTORY_COLUMNS))
    write_atomically(job.out_dir / FILE_HISTORY, lambda p: frame.to_csv(p, index=False))


def _write_summary(job: _Job, state: _State) -> None:
    """Write summary.json: the selected epoch, its validation metrics and the run totals."""
    best = state.best or {}
    summary = {
        "run_name": job.run.run_name,
        "selection": {"split": "val", "rule": SELECTION_RULE},
        "best_epoch": best.get("epoch"),
        "best_val": best or None,
        "epochs_run": state.epochs_done,
        "max_epochs": job.run.max_epochs,
        "stopped_early": state.stopped_early,
        "total_seconds": round(state.train_seconds, 3),
        "smoke": job.smoke,
        "pretrained": job.pretrained,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    write_atomically(job.out_dir / FILE_SUMMARY, lambda p: p.write_text(text, encoding="utf-8"))


def _write_config(job: _Job) -> None:
    """Write config.yaml: the fully resolved run and the data settings it used."""
    resolved = {
        "run": job.run.to_dict(),
        "data": {
            "processed_dir": str(job.data.processed_dir),
            "image_dir": str(job.image_dir),
            "target_size": job.data.target_size,
            "split_file": job.run.data.split_file,
        },
    }
    text = yaml.safe_dump(resolved, sort_keys=False)
    write_atomically(job.out_dir / FILE_CONFIG, lambda p: p.write_text(text, encoding="utf-8"))


def _env_start(out: Path, run: RunConfig, device: torch.device, pretrained: bool, smoke: bool) -> None:
    """Record a new session in env.json (versions, device, git commit, argv, start time)."""
    path = out / FILE_ENV
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    sessions = list(previous.get("sessions", []))
    started = utc_now_iso()
    sessions.append(
        {
            "start_utc": started,
            "end_utc": None,
            "argv": list(sys.argv),
            "device": describe_device(device),
        }
    )
    env = {
        "run_name": run.run_name,
        "seed": run.seed,
        "pretrained": pretrained,
        "smoke": smoke,
        "versions": package_versions(),
        "device": describe_device(device),
        "git": get_git_info(),
        "start_utc": sessions[0]["start_utc"],
        "end_utc": None,
        "sessions": sessions,
    }
    _write_env(path, env)


def _env_end(out: Path, ended: str) -> None:
    """Record the end time of the current session in env.json."""
    path = out / FILE_ENV
    if not path.is_file():
        return
    env = json.loads(path.read_text(encoding="utf-8"))
    if env.get("sessions"):
        env["sessions"][-1]["end_utc"] = ended
    env["end_utc"] = ended
    _write_env(path, env)


def _write_env(path: Path, env: dict[str, Any]) -> None:
    """Write env.json atomically as sorted, indented JSON."""
    text = json.dumps(env, indent=2, sort_keys=True) + "\n"
    write_atomically(path, lambda p: p.write_text(text, encoding="utf-8"))


def _clear_run_files(out: Path) -> None:
    """Remove the files of an earlier run from out, so that a smoke test starts clean."""
    for name in RUN_FILES:
        target = out / name
        if target.is_file():
            target.unlink()


def _check_smoke(job: _Job, state: _State) -> None:
    """Fail unless the smoke run wrote its files, two history rows, and resumed from epoch 1."""
    out = job.out_dir
    history = pd.read_csv(out / FILE_HISTORY)
    if len(history) != SMOKE_EPOCHS:
        raise RuntimeError(f"smoke check failed: history.csv has {len(history)} rows, expected {SMOKE_EPOCHS}")
    if state.resumed_from != SMOKE_STOP_AFTER_EPOCH:
        raise RuntimeError(
            f"smoke check failed: the resume started from epoch {state.resumed_from}, "
            f"expected {SMOKE_STOP_AFTER_EPOCH}"
        )
    missing = [name for name in RUN_FILES if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"smoke check failed: missing outputs {missing}")
    logger.info("smoke check passed: %d epochs, resume from epoch %d", SMOKE_EPOCHS, state.resumed_from)
