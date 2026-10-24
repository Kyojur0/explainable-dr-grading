"""Shared helpers: logging, seeding, device selection, atomic writes and environment capture.

Torch is imported only inside the functions that need it, so stage 1 code keeps
working on machines without torch.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    import torch

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# cuBLAS needs this workspace setting for deterministic kernels on CUDA 10.2 and later
# (PyTorch reproducibility notes). It must be set before the first CUDA call.
CUBLAS_WORKSPACE_VALUE = ":4096:8"

# Distributions whose versions go into env.json (brief 8.5). scikit-learn is the
# distribution name of the sklearn package.
ENV_DISTRIBUTIONS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "scikit-learn",
    "pandas",
    "PyYAML",
)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for command-line scripts."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch, and switch torch to deterministic settings.

    With torch present this also sets cuDNN to non-benchmark mode, asks PyTorch to use
    deterministic algorithms (warn only, so ops without a deterministic version warn
    rather than crash) and sets CUBLAS_WORKSPACE_CONFIG. Stage 1 code never reaches the
    torch branch.
    PYTHONHASHSEED only affects interpreters started after this call; set it in the
    environment for full effect. DataLoader worker seeding is in drgrade.data.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_VALUE)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _mps_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _mps_available() -> bool:
    """Return True when the Apple MPS backend is usable. Requires torch to be installed."""
    import torch

    return bool(hasattr(torch, "mps") and torch.backends.mps.is_available())


def select_device(requested: str | None = None) -> torch.device:
    """Return the torch device to train on.

    Without a request the order is CUDA, then MPS, then CPU. A requested device
    is used as given, but a CUDA or MPS request that the machine cannot satisfy
    raises RuntimeError rather than silently falling back.
    """
    import torch

    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not _mps_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict[str, str]:
    """Return the device type and a human-readable name (for example the GPU model)."""
    import torch

    if device.type == "cuda":
        return {"type": "cuda", "name": torch.cuda.get_device_name(device)}
    return {"type": device.type, "name": platform.processor() or platform.machine()}


def write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Write a file through a temporary name and move it into place with os.replace.

    The output folder may be a synced Google Drive folder, so a reader never sees a
    half-written file. The temporary file is removed if writing fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def package_versions() -> dict[str, str | None]:
    """Return the Python version and the installed versions of the packages in env.json."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ENV_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with whole seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_git_info(repo_dir: Path | None = None) -> dict[str, str | bool | None]:
    """Return the current commit hash and whether the working tree is dirty.

    Returns {"commit": None, "dirty": None} when git is unavailable or the
    directory is not a repository. Never raises.
    """
    cwd = repo_dir if repo_dir is not None else Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}
