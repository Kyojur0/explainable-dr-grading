"""Synthetic predictions tables for the stage 5 and 6 tests (no real data, no model)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def make_predictions(
    ids: Sequence[str],
    true: Sequence[int],
    pred: Sequence[int],
    confidence: Sequence[float],
) -> pd.DataFrame:
    """Return a predictions table: the predicted grade gets the given confidence, the rest share the remainder."""
    rows = []
    for image_id, t, p, c in zip(ids, true, pred, confidence):
        probs = np.full(5, (1.0 - c) / 4.0)
        probs[p] = c
        logits = np.log(probs)
        row = {"image_id": image_id, "true_grade": int(t), "pred_grade": int(p)}
        row.update({f"prob_{g}": float(probs[g]) for g in range(5)})
        row.update({f"logit_{g}": float(logits[g]) for g in range(5)})
        rows.append(row)
    return pd.DataFrame(rows)


def ids_of_subset(split: pd.DataFrame, subset: str) -> list[str]:
    """Return the image ids of one subset from a split frame, sorted."""
    return sorted(split.loc[split["subset"] == subset, "image_id"].astype(str).tolist())


def register_val(csv_path, run_name: str = "synthetic", checkpoint: str = "c0") -> dict:  # type: ignore[no-untyped-def]
    """Write the provenance sidecar for a synthetic validation predictions file (DECISIONS.md D38)."""
    from drgrade.provenance import write_meta

    rows = len(pd.read_csv(csv_path))
    return write_meta(csv_path, {"subset": "val", "run_name": run_name, "checkpoint_sha256": checkpoint, "tta": False, "rows": rows})


def register_test(csv_path, run_name: str = "synthetic", checkpoint: str = "c0") -> dict:  # type: ignore[no-untyped-def]
    """Register a synthetic test predictions file as scripts/predict.py would: sidecar, log event and lock.

    The access log is the per-test temporary one set by tests/conftest.py, never the real log.
    The file must sit at <run>/predictions/<name>.csv so that the lock is <run>/TEST_LOCK.json.
    """
    import json
    from pathlib import Path

    from drgrade.provenance import EVENT_COMPLETE, append_access_event, write_meta

    csv = Path(csv_path)
    rows = len(pd.read_csv(csv))
    meta = write_meta(csv, {"subset": "test", "run_name": run_name, "checkpoint_sha256": checkpoint, "tta": False, "rows": rows})
    append_access_event({"event": EVENT_COMPLETE, "run_name": run_name, "variant": "plain", "csv_sha256": meta["csv_sha256"], "rows": rows})
    lock = {"run_name": run_name, "status": "open", "outputs": [{"variant": "plain", "csv_sha256": meta["csv_sha256"]}], "reruns": []}
    (csv.parent.parent / "TEST_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    return meta


def write_split_for(path, val_csv=None, test_csv=None) -> None:  # type: ignore[no-untyped-def]
    """Write a split CSV whose val and test subsets are exactly the ids of the given predictions files."""
    rows = []
    for subset, csv in (("val", val_csv), ("test", test_csv)):
        if csv is not None:
            ids = pd.read_csv(csv, dtype={"image_id": str})["image_id"].astype(str)
            rows.extend({"image_id": i, "subset": subset} for i in ids)
    pd.DataFrame(rows).to_csv(path, index=False)
