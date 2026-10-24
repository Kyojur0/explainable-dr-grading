"""Provenance and test-set access controls (DECISIONS.md D38).

Three safeguards added after an independent pre-test audit:

1. An allowlist of the runs that may ever be predicted on the test set
   (`configs/test_allowlist.yaml`, committed). Copies of a run folder share the run
   name, so a copy cannot obtain a fresh lock.
2. A central, append-only access log (`results/test_access_log.jsonl`, committed).
   Every test prediction writes an "attempt" event before the test images are
   loaded and a "complete" event, with the SHA-256 of the output CSV, when it ends.
   An interrupted test run therefore always leaves a trace.
3. A sidecar `<predictions>.meta.json` written next to every predictions CSV
   (subset, run name, checkpoint hash, TTA flag, row count, CSV hash), and
   `verify_predictions`, used by every script that reads predictions, which checks
   the sidecar and, for test files, that the CSV hash matches the log and the lock.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from drgrade.utils import get_git_info, utc_now_iso

ALLOWLIST_PATH = Path("configs/test_allowlist.yaml")
ACCESS_LOG_PATH = Path("results/test_access_log.jsonl")
META_SUFFIX = ".meta.json"
EVENT_ATTEMPT = "attempt"
EVENT_COMPLETE = "complete"
# Environment overrides, used by the test suite so that tests never touch the real,
# committed allowlist or access log (tests/conftest.py).
ENV_ALLOWLIST = "DRGRADE_TEST_ALLOWLIST"
ENV_ACCESS_LOG = "DRGRADE_ACCESS_LOG"


def allowlist_path() -> Path:
    """Return the allowlist path in force (environment override, else the committed file)."""
    return Path(os.environ.get(ENV_ALLOWLIST, ALLOWLIST_PATH))


def access_log_path() -> Path:
    """Return the access log path in force (environment override, else the committed file)."""
    return Path(os.environ.get(ENV_ACCESS_LOG, ACCESS_LOG_PATH))


class ProvenanceError(ValueError):
    """Raised when a predictions file or a test access fails a provenance check."""


def sha256_of(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_allowlist(path: Path | None = None) -> set[str]:
    """Return the run names allowed on the test set. Raises ProvenanceError if the file is missing."""
    path = allowlist_path() if path is None else path
    if not path.is_file():
        raise ProvenanceError(f"{path} is missing: no run may be predicted on test without it")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    runs = raw.get("allowed_runs")
    if not isinstance(runs, list) or not all(isinstance(r, str) for r in runs):
        raise ProvenanceError(f"{path} must contain a list 'allowed_runs'")
    return set(runs)


def read_access_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Return every event in the access log (empty when the log does not exist yet)."""
    path = access_log_path() if path is None else path
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def append_access_event(event: dict[str, Any], path: Path | None = None) -> None:
    """Append one event (one JSON line) to the access log, creating it if needed.

    Append-only by design: earlier lines are never rewritten.
    """
    path = access_log_path() if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": utc_now_iso(), "git": get_git_info(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def check_test_access(
    run_name: str,
    variant: str,
    forced: bool,
    allowlist: Path | None = None,
    log_path: Path | None = None,
) -> None:
    """Refuse a test prediction for a run outside the allowlist, or a repeat of a completed output.

    A repeat is any earlier "complete" event for the same run name and variant, in any
    folder; it is allowed only with the --force/--reason override (`forced`).
    """
    allowlist = allowlist_path() if allowlist is None else allowlist
    allowed = load_allowlist(allowlist)
    if run_name not in allowed:
        raise ProvenanceError(
            f"run {run_name!r} is not in {allowlist}; only {sorted(allowed)} may be predicted on test"
        )
    if forced:
        return
    for event in read_access_log(log_path):
        if event.get("event") == EVENT_COMPLETE and event.get("run_name") == run_name and event.get("variant") == variant:
            raise ProvenanceError(
                f"the {variant} test output of {run_name} was already completed at {event.get('time_utc')} "
                f"({event.get('folder')}); pass --force and --reason to repeat it"
            )


def meta_path(csv_path: Path | str) -> Path:
    """Return the sidecar metadata path of a predictions CSV."""
    csv = Path(csv_path)
    return csv.with_name(csv.name + META_SUFFIX)


def write_meta(csv_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Write the sidecar for a complete predictions CSV, adding its hash; return what was written."""
    record = {**meta, "csv_sha256": sha256_of(csv_path), "written_utc": utc_now_iso()}
    meta_path(csv_path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def read_meta(csv_path: Path | str) -> dict[str, Any] | None:
    """Return the sidecar of a predictions CSV, or None when it has none (for example training output)."""
    path = meta_path(csv_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_predictions(
    csv_path: Path | str,
    expected_subset: str,
    expected_rows: int | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Check a predictions CSV before it is used, and return its sidecar.

    - the sidecar must exist, name the expected subset, and hold the CSV's current hash;
    - when `expected_rows` is given, the sidecar row count must match it (completeness);
    - for the test subset, the CSV hash must appear in a "complete" event of the access
      log and in the run's TEST_LOCK.json, so a replaced or hand-made test file is refused.
    Raises ProvenanceError with the reason.
    """
    csv = Path(csv_path)
    meta = read_meta(csv)
    if meta is None:
        raise ProvenanceError(f"{csv} has no {META_SUFFIX} sidecar; produce it with scripts/predict.py")
    if meta.get("subset") != expected_subset:
        raise ProvenanceError(f"{csv} holds subset {meta.get('subset')!r}, expected {expected_subset!r}")
    current = sha256_of(csv)
    if meta.get("csv_sha256") != current:
        raise ProvenanceError(f"{csv} has changed since prediction (hash differs from its sidecar)")
    if expected_rows is not None and meta.get("rows") != expected_rows:
        raise ProvenanceError(f"{csv} has {meta.get('rows')} rows, expected {expected_rows} (incomplete)")
    if expected_subset == "test":
        logged = {
            event.get("csv_sha256")
            for event in read_access_log(log_path)
            if event.get("event") == EVENT_COMPLETE
        }
        if current not in logged:
            shown = access_log_path() if log_path is None else log_path
            raise ProvenanceError(f"{csv} is not recorded as a completed test prediction in {shown}")
        lock_file = csv.parent.parent / "TEST_LOCK.json"
        if not lock_file.is_file():
            raise ProvenanceError(f"{lock_file} is missing for test predictions {csv}")
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        hashes = {o.get("csv_sha256") for o in lock.get("outputs", [])} | {
            r.get("csv_sha256") for r in lock.get("reruns", [])
        }
        if current not in hashes:
            raise ProvenanceError(f"{csv} hash is not in {lock_file}")
    return meta


def check_same_checkpoint(*metas: dict[str, Any]) -> None:
    """Raise ProvenanceError unless every sidecar names the same run and checkpoint hash."""
    keys = {(m.get("run_name"), m.get("checkpoint_sha256")) for m in metas}
    if len(keys) != 1:
        raise ProvenanceError(f"predictions come from different runs or checkpoints: {sorted(keys)}")


def subset_size(split_path: Path | str, subset: str) -> int:
    """Return the number of images the split assigns to `subset` (for completeness checks)."""
    import pandas as pd  # local import keeps this module light for the predict path

    split = pd.read_csv(split_path, dtype={"image_id": str, "subset": str})
    return int((split["subset"] == subset).sum())


def verify_for_use(
    csv_path: Path | str,
    subset: str,
    expected_rows: int,
    require_meta: bool,
    log_path: Path | None = None,
) -> dict[str, Any] | None:
    """Verify a predictions file before an analysis script uses it.

    Test files are always verified in full (sidecar, hash, log, lock, row count).
    Validation files are verified in full when `require_meta` is set or when a sidecar
    exists; a validation file without a sidecar (training-time output, Colab) is
    accepted only when `require_meta` is False, and its completeness is then checked
    by the caller against `expected_rows`. Returns the sidecar, or None.
    """
    if subset == "test" or require_meta or read_meta(csv_path) is not None:
        return verify_predictions(csv_path, subset, expected_rows=expected_rows, log_path=log_path)
    return None
