"""Figures for stage 7: reliability diagrams, risk-coverage curves and the quality stress summary.

Figures use matplotlib's Figure directly (no pyplot state), written atomically at FIGURE_DPI
as PNG, the same pattern as drgrade.reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from drgrade.reporting import FIGURE_DPI
from drgrade.utils import write_atomically

DIAGONAL_STYLE = "--"
BAR_WIDTH_FRACTION = 0.9  # bars fill this share of each bin's width


def _save(fig: Figure, path: Path) -> None:
    """Write the figure atomically as PNG."""
    write_atomically(path, lambda p: fig.savefig(p, dpi=FIGURE_DPI, format="png"))


def save_reliability(panels: Sequence[tuple[str, list[dict[str, Any]]]], path: Path, title: str) -> None:
    """Draw one column per panel: accuracy per confidence bin against the diagonal, and the count per bin below.

    Each panel is (label, bins) where bins comes from calibration.reliability_bins. Empty bins draw
    no accuracy bar.
    """
    if not panels:
        raise ValueError("at least one panel is needed")
    fig = Figure(figsize=(4.6 * len(panels), 5.2))
    axes = fig.subplots(2, len(panels), squeeze=False, gridspec_kw={"height_ratios": [3, 1]})
    for col, (label, bins) in enumerate(panels):
        n_bins = len(bins)
        width = (bins[0]["upper"] - bins[0]["lower"]) * BAR_WIDTH_FRACTION
        centres = [(row["lower"] + row["upper"]) / 2.0 for row in bins]
        accuracy = [row["accuracy"] if row["count"] else 0.0 for row in bins]
        counts = [row["count"] for row in bins]
        top = axes[0][col]
        top.bar(centres, accuracy, width=width, color="#4C72B0", edgecolor="#2d4a73", label="accuracy")
        top.plot([0.0, 1.0], [0.0, 1.0], DIAGONAL_STYLE, color="#444444", linewidth=1.0, label="perfect calibration")
        top.set_xlim(0.0, 1.0)
        top.set_ylim(0.0, 1.0)
        top.set_ylabel("accuracy")
        top.set_title(f"{label} (bins: {n_bins})", fontsize=10)
        top.legend(loc="upper left", fontsize=8)
        bottom = axes[1][col]
        bottom.bar(centres, counts, width=width, color="#8c8c8c")
        bottom.set_xlim(0.0, 1.0)
        bottom.set_xlabel("confidence (largest calibrated probability)")
        bottom.set_ylabel("images")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def save_risk_coverage(table: pd.DataFrame, path: Path, title: str) -> None:
    """Draw the risk-coverage curves against the referral fraction.

    Left: accuracy and QWK on the retained cases. Right: the referable miss rate among retained
    cases, and the share of all referable misses that were referred. Undefined values (None)
    are left out of the lines.
    """
    fig = Figure(figsize=(10.0, 4.2))
    left, right = fig.subplots(1, 2)
    x = table["referral_fraction"].astype(float)
    left.plot(x, table["accuracy"].astype(float), label="accuracy")
    left.plot(x, table["qwk"].astype(float), label="QWK")
    left.set_xlabel("referred fraction")
    left.set_ylabel("metric on retained cases")
    left.set_ylim(0.0, 1.0)
    left.legend(fontsize=8)
    right.plot(x, table["referable_miss_rate_retained"].astype(float), label="referable miss rate (retained)")
    right.plot(x, table["share_of_misses_referred"].astype(float), label="share of referable misses referred")
    right.set_xlabel("referred fraction")
    right.set_ylabel("rate")
    right.set_ylim(0.0, 1.0)
    right.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def save_stress(rows: pd.DataFrame, path: Path, title: str) -> None:
    """Draw QWK and mean calibrated confidence against severity, one line per perturbation kind.

    rows needs the columns kind, severity, qwk and mean_confidence. The unperturbed baseline
    has kind 'none' and severity 0, and is drawn as a point at the start of each line.
    """
    fig = Figure(figsize=(10.0, 4.2))
    left, right = fig.subplots(1, 2)
    baseline = rows.loc[rows["kind"] == "none"]
    for kind, sub in rows.loc[rows["kind"] != "none"].groupby("kind", sort=True):
        line = pd.concat([baseline, sub]).sort_values("severity")
        left.plot(line["severity"], line["qwk"].astype(float), marker="o", label=str(kind))
        right.plot(line["severity"], line["mean_confidence"].astype(float), marker="o", label=str(kind))
    for axis, ylabel in ((left, "QWK"), (right, "mean calibrated confidence")):
        axis.set_xlabel("severity (blur: radius in px; darkening: 1 - factor)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)
