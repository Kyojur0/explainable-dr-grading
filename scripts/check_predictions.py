"""Check that predictions made from the committed models match the committed predictions.

Use after re-running scripts/predict.py with a model in models/ (README, "Reproduce from the
trained models"). Same image ids, same predicted grades and logits within --tolerance means
every number in the report follows, because scripts/reproduce_report.py derives them all from
these files.

Usage:
    python scripts/check_predictions.py --run models/e2_effb0_512_ce
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOGITS = [f"logit_{k}" for k in range(5)]


def compare(fresh: Path, saved: Path) -> tuple[int, int, float]:
    """Return (rows, grade disagreements, max absolute logit difference)."""
    a = pd.read_csv(fresh, dtype={"image_id": str}).set_index("image_id").sort_index()
    b = pd.read_csv(saved, dtype={"image_id": str}).set_index("image_id").sort_index()
    if list(a.index) != list(b.index):
        raise ValueError(f"{fresh} and {saved} hold different image ids")
    grades = int((a["pred_grade"] != b["pred_grade"]).sum())
    diff = float(np.abs(a[LOGITS].to_numpy() - b[LOGITS].to_numpy()).max())
    return len(a), grades, diff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="model folder, for example models/e2_effb0_512_ce")
    parser.add_argument("--results", type=Path, default=Path("results/predictions"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)
    ok = True
    for subset in ("val", "test"):
        fresh = args.run / "predictions" / f"{subset}.csv"
        saved = args.results / args.run.name / f"{subset}.csv"
        if not fresh.is_file():
            print(f"{subset}: {fresh} not found (run scripts/predict.py first)")
            ok = False
            continue
        rows, grades, diff = compare(fresh, saved)
        match = grades == 0 and diff <= args.tolerance
        ok &= match
        print(f"{subset}: {rows} images, {grades} grades differ, largest logit difference {diff:.2e} -> {'match' if match else 'DIFFERENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
