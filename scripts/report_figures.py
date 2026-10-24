"""Draw the report figures in one consistent style (greyscale plus one blue accent).

Reads only existing outputs (no model passes, no test images beyond the case panels that
scripts/explain.py already wrote) and writes PNG files to --out.

Figures:
- overview.png: the pipeline at a glance, with the boundary between development and test.
- confusion.png: E2 test confusion matrix, normalised by true grade, with counts, and
  per-class precision, recall and F1 beside it.
- reliability.png: E2 test reliability before and after temperature scaling, and the
  referable miss rate among retained cases as uncertain cases are referred.
- deletion.png: deletion test (validation), mean drop in predicted-class probability.
- cam_cases.png: three test cases, cropped from the explain.py case panels.

Usage:
    python scripts/report_figures.py --run runs/e2_effb0_512_ce --out report/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

BLUE = "#2B6CB0"
DARK = "#2D2D2D"
MID = "#8A8A8A"
LIGHT = "#D0D0D0"
DPI = 300
GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]
CASES = (
    ("correct_grade_4", "ts001331", "a  True grade 4, predicted 4", ["Image", "Grad-CAM, predicted grade 4"]),
    ("referable_miss", "ts001947", "b  True grade 2, predicted 0, confidence 0.97",
     ["Image", "Grad-CAM, predicted grade 0", "Grad-CAM, true grade 2"]),
    ("referable_miss", "ts001606", "c  True grade 2, predicted 1, confidence 0.91",
     ["Image", "Grad-CAM, predicted grade 1", "Grad-CAM, true grade 2"]),
)


def style() -> None:
    """Shared matplotlib style: small sans text, no top or right spines, light grid."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#EDEDED",
            "grid.linewidth": 0.6,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def confusion(run: Path, out: Path) -> None:
    counts = pd.read_csv(run / "predictions/eval_test/confusion_counts.csv", index_col=0).to_numpy()
    norm = counts / counts.sum(axis=1, keepdims=True)
    per_class = pd.read_csv(run / "predictions/eval_test/per_class.csv")
    cmap = LinearSegmentedColormap.from_list("blue", ["#FFFFFF", BLUE])
    fig, (ax, side) = plt.subplots(1, 2, figsize=(5.2, 2.9), gridspec_kw={"width_ratios": [5, 2.3], "wspace": 0.05})
    ax.grid(False)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            colour = "white" if norm[i, j] > 0.55 else DARK
            ax.text(j, i, f"{norm[i, j]:.2f}\n({counts[i, j]})", ha="center", va="center", fontsize=6.3, color=colour)
    ax.set_xticks(range(5), [g.split(" ", 1)[0] for g in GRADES])
    ax.set_yticks(range(5), GRADES)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    side.axis("off")
    side.set_xlim(0, 3)
    side.set_ylim(4.5, -0.5)
    for k, name in enumerate(("Precision", "Recall", "F1")):
        side.text(k + 0.5, -0.75, name, ha="center", va="bottom", fontsize=6.5, color=MID)
    for i in range(5):
        row = per_class.iloc[i]
        for k, col in enumerate(("precision", "recall", "f1")):
            side.text(k + 0.5, i, f"{row[col]:.2f}", ha="center", va="center", fontsize=6.8, color=DARK)
    fig.savefig(out / "confusion.png")
    plt.close(fig)


def reliability(run: Path, out: Path) -> None:
    bins = pd.read_csv(run / "calibration/reliability_bins.csv")
    cal = json.loads((run / "calibration/calibration.json").read_text())
    test = bins.loc[bins["split"] == "test"]
    rc = pd.read_csv(run / "referral/risk_coverage_test.csv")
    ref = json.loads((run / "referral/referral.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a.plot([0, 1], [0, 1], color=LIGHT, lw=1, ls="--", label="Perfect calibration")
    for stage, colour, marker, label in (
        ("before", MID, "o", f"Before (ECE {cal['test']['before']['ece']:.3f})"),
        ("after", BLUE, "s", f"After, T = {cal['temperature']:.2f} (ECE {cal['test']['after']['ece']:.3f})"),
    ):
        rows = test.loc[(test["stage"] == stage) & (test["count"] >= 10)]
        a.plot(rows["mean_confidence"], rows["accuracy"], color=colour, marker=marker, ms=3, lw=1.2, label=label)
    a.set_xlim(0.2, 1.0)
    a.set_ylim(0.2, 1.0)
    a.set_xlabel("Mean confidence (bins with at least 10 images)")
    a.set_ylabel("Accuracy")
    a.set_title("Calibration on test", loc="left")
    a.legend(loc="upper left")
    b.plot(rc["referral_fraction"] * 100, rc["referable_miss_rate_retained"] * 100, color=BLUE, lw=1.4)
    primary = ref["primary"]["test"]
    b.axvline(primary["referral_fraction"] * 100, color=MID, lw=0.8, ls="--")
    b.text(primary["referral_fraction"] * 100 + 1, b.get_ylim()[1] * 0.92 if b.get_ylim()[1] > 0 else 20,
           "Rule fixed on validation:\nrefer least confident 20%", fontsize=7, color=DARK, va="top")
    b.set_xlabel("Cases referred for review (%)")
    b.set_ylabel("Referable miss rate, retained (%)")
    b.set_title("Referral of uncertain cases on test", loc="left")
    b.set_xlim(0, 50)
    b.set_ylim(bottom=0)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "reliability.png")
    plt.close(fig)


def deletion(run: Path, out: Path) -> None:
    s = pd.read_csv(run / "explain/deletion/summary.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for condition, colour, marker, label in (
        ("salient", BLUE, "o", "Most salient pixels"),
        ("rotated", DARK, "^", "Same mask, rotated"),
        ("random", MID, "s", "Random pixels"),
    ):
        rows = s.loc[s["condition"] == condition].sort_values("fraction")
        ax.errorbar(rows["fraction"] * 100, rows["mean_drop"], yerr=rows["se_drop"], color=colour, marker=marker,
                    ms=3, lw=1.2, capsize=1.5, label=label)
    ax.set_xlabel("Fundus pixels blurred (%)")
    ax.set_ylabel("Drop in predicted-class probability")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.savefig(out / "deletion.png")
    plt.close(fig)


def _panels(png: Path) -> list[Image.Image]:
    """Crop the image panels (large dark squares) from an explain.py case figure, left to right."""
    rgb = np.asarray(Image.open(png).convert("RGB"))
    dark = rgb.mean(axis=2) < 200
    labels, _ = ndimage.label(ndimage.binary_closing(dark, iterations=3))
    boxes = []
    for sl in ndimage.find_objects(labels):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > 0.25 * rgb.shape[0] and abs(h - w) < 0.1 * max(h, w):
            boxes.append(sl)
    boxes.sort(key=lambda sl: sl[1].start)
    return [Image.fromarray(rgb[sl]) for sl in boxes]


def cam_cases(run: Path, out: Path) -> None:
    rows = []
    for reason, image_id, label, titles in CASES:
        crops = _panels(run / "explain/panels_test" / reason / f"{image_id}.png")
        if len(crops) != len(titles):
            raise ValueError(f"{image_id}: expected {len(titles)} panels, found {len(crops)}")
        rows.append((label, crops, titles))
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.6, 2.15 * len(rows)))
    for r, (label, crops, titles) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(crops):
                ax.imshow(crops[c])
                ax.set_title(titles[c], fontsize=7, color=BLUE if "true" in titles[c] else DARK, pad=3)
        axes[r, 0].text(-0.02, 1.2, label, transform=axes[r, 0].transAxes, fontsize=7.8, fontweight="bold",
                        color=DARK, ha="left", va="bottom")
    fig.subplots_adjust(hspace=0.42, wspace=0.04)
    fig.savefig(out / "cam_cases.png")
    plt.close(fig)


def overview(out: Path) -> None:
    """Pipeline at a glance: development on validation only, then one locked pass on test."""
    from matplotlib.patches import FancyBboxPatch

    steps = [
        ("Audit", "11,118 images\nrepeat captures\nlabel checks"),
        ("Split", "validation from\ntrain, grouped by\nrepeat capture"),
        ("Train", "E1, E2, E3\nfixed in\nadvance"),
        ("Select", "model by\nval QWK; T and\nthresholds on val"),
        ("Test", "once, locked\nand logged\nE2 and E3"),
        ("Analyse", "errors\nreliability\nGrad-CAM"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 1.55))
    ax.axis("off")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.35)
    w, h, y = 0.86, 0.78, 0.18
    for k, (title, body) in enumerate(steps):
        x = k + 0.07
        test = title.startswith("Test")
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                             linewidth=1.2 if test else 0.7, edgecolor=BLUE if test else MID,
                             facecolor="#EAF1FA" if test else "white")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="center", fontsize=7.2, fontweight="bold",
                color=BLUE if test else DARK)
        ax.text(x + w / 2, y + 0.3, body, ha="center", va="center", fontsize=5.8, color=DARK, linespacing=1.25)
        if k < len(steps) - 1:
            ax.annotate("", xy=(k + 1.07, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops={"arrowstyle": "-|>", "color": MID, "lw": 0.8, "shrinkA": 1, "shrinkB": 1})
    ax.plot([3.965, 3.965], [0.05, 1.3], color=BLUE, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.95, 1.2, "Development: no test predictions are used for any choice", ha="center", fontsize=6.8, color=MID)
    ax.text(4.98, 1.2, "Held-out test images", ha="center", fontsize=6.8, color=BLUE)
    fig.savefig(out / "overview.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    overview(args.out)
    confusion(args.run, args.out)
    reliability(args.run, args.out)
    deletion(args.run, args.out)
    cam_cases(args.run, args.out)
    print("wrote", sorted(p.name for p in args.out.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
