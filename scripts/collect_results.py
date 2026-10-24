"""Copy the outputs the report relies on from runs/ and outputs/ into results/ (committed).

After this, every number in the report can be recomputed from results/ alone with
scripts/reproduce_report.py: no images, no GPU, no model. MMRDR is CC BY 4.0, so image ids,
grades and per-image metadata may be redistributed with attribution (DECISIONS.md D42).

Writes results/MANIFEST.json with the SHA-256, size and source of every copied file and the
git commit it was collected at.

Usage:
    python scripts/collect_results.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from drgrade.utils import get_git_info, setup_logging, utc_now_iso

logger = logging.getLogger(__name__)

RUNS = Path("runs")
OUT = Path("results")
SELECTED = "e2_effb0_512_ce"
SECONDARY = "e3_effb0_512_wce"
BASELINE = "e1_resnet18_384_wce"
TESTED = (SELECTED, SECONDARY)

# (source, destination inside results/)
FILES: list[tuple[str, str]] = [
    ("outputs/audit/summary.json", "audit/summary.json"),
    ("outputs/audit/counts_split_grade.csv", "audit/counts_split_grade.csv"),
    ("outputs/compare/val_comparison.csv", "compare/val_comparison.csv"),
    ("outputs/compare_extras/val_comparison.csv", "compare/val_comparison_extras.csv"),
]
for run in (BASELINE, SELECTED, SECONDARY):
    FILES += [
        (f"runs/{run}/val_predictions.csv", f"predictions/{run}/val_training.csv"),
        (f"runs/{run}/summary.json", f"runs/{run}/summary.json"),
        (f"runs/{run}/env.json", f"runs/{run}/env.json"),
        (f"runs/{run}/config.yaml", f"runs/{run}/config.yaml"),
    ]
for run in TESTED:
    for subset in ("val", "test"):
        FILES += [
            (f"runs/{run}/predictions/{subset}.csv", f"predictions/{run}/{subset}.csv"),
            (f"runs/{run}/predictions/{subset}.csv.meta.json", f"predictions/{run}/{subset}.csv.meta.json"),
        ]
        for name in ("metrics.json", "bootstrap_ci.json", "per_class.csv", "confusion_counts.csv"):
            FILES.append((f"runs/{run}/predictions/eval_{subset}/{name}", f"runs/{run}/eval_{subset}/{name}"))
    FILES += [
        (f"runs/{run}/calibration/calibration.json", f"runs/{run}/calibration.json"),
        (f"runs/{run}/referral/referral.json", f"runs/{run}/referral.json"),
        (f"runs/{run}/referral/referable_threshold.json", f"runs/{run}/referable_threshold.json"),
    ]
FILES += [
    (f"runs/{SELECTED}/predictions/errors_test/dangerous_misses.csv", f"runs/{SELECTED}/errors_test/dangerous_misses.csv"),
    (f"runs/{SELECTED}/predictions/errors_test/cases.csv", f"runs/{SELECTED}/errors_test/cases.csv"),
    (f"runs/{SELECTED}/explain/randomisation/summary.csv", "explain/randomisation_summary.csv"),
    (f"runs/{SELECTED}/explain/deletion/summary.csv", "explain/deletion_summary.csv"),
    (f"runs/{SELECTED}/explain/stats/shortcut_summary.json", "explain/shortcut_summary.json"),
    (f"runs/{SELECTED}/explain/stats/lesion_comparison.csv", "explain/lesion_comparison.csv"),
    (f"runs/{SELECTED}/explain/maps_val.json", "explain/maps_val.json"),
    (f"runs/{SELECTED}/explain/maps_test.json", "explain/maps_test.json"),
    (f"runs/{SELECTED}/explain/panels_test/panels.csv", "explain/panels_test.csv"),
    (f"runs/{SELECTED}/explain/panels_val/panels.csv", "explain/panels_val.csv"),
]

MANIFEST_COLUMNS = (
    "image_id", "official_split", "grade", "lr",
    "les_ma", "les_he", "les_irh", "les_vb_irma", "les_nv", "les_vh", "les_rd",
    "orig_width", "orig_height", "crop_failed", "border_level", "brightness", "contrast", "sharpness", "file",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/s512/manifest.csv"))
    args = parser.parse_args(argv)

    entries = []
    missing = [src for src, _ in FILES if not Path(src).is_file()]
    if missing or not args.manifest.is_file():
        for src in missing + ([] if args.manifest.is_file() else [str(args.manifest)]):
            logger.error("missing: %s", src)
        return 2
    for src, dst in FILES:
        target = OUT / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        entries.append({"file": target.as_posix(), "source": src})

    # Per-image metadata (no pixels): grades, lesion flags, crop box, border level, quality measures.
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    meta_path = OUT / "manifest_meta.csv"
    manifest.loc[:, list(MANIFEST_COLUMNS)].to_csv(meta_path, index=False)
    entries.append({"file": meta_path.as_posix(), "source": args.manifest.as_posix() + " (selected columns)"})

    for entry in entries:
        path = Path(entry["file"])
        entry["sha256"] = sha256(path)
        entry["bytes"] = path.stat().st_size
    manifest_out = {
        "collected_utc": utc_now_iso(),
        "git": get_git_info(),
        "files": sorted(entries, key=lambda e: e["file"]),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8")
    logger.info("copied %d files into %s", len(entries), OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
