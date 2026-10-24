"""Plotting helpers for the audit figures (matplotlib, Agg backend, 150 dpi)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

FIGURE_DPI = 150
PANEL_INCH = 4.0


def save_grouped_bar(table: pd.DataFrame, path: Path, title: str, ylabel: str) -> None:
    """Save a grouped bar chart. Index holds the x categories, columns the series."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        table.plot.bar(ax=ax, rot=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(title=table.columns.name)
        fig.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: Path,
) -> None:
    """Save a heatmap of values in [0, 1] with each cell annotated. NaN cells show n/a."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(col_labels)), 4.0))
    try:
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for (row, col), value in np.ndenumerate(matrix):
            text = "n/a" if np.isnan(value) else f"{value:.2f}"
            colour = "white" if not np.isnan(value) and value < 0.5 else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=colour)
        fig.colorbar(image, ax=ax, label="prevalence")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)


def save_contact_sheet(panels: list[tuple[Image.Image, str]], path: Path, title: str, ncols: int = 3) -> bool:
    """Save a grid of images, each with a caption. Returns False and writes nothing if empty."""
    if not panels:
        return False
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * PANEL_INCH, nrows * PANEL_INCH))
    flat_axes = np.atleast_1d(axes).ravel()
    try:
        for ax in flat_axes:
            ax.axis("off")
        for ax, (image, caption) in zip(flat_axes, panels):
            ax.imshow(np.asarray(image.convert("RGB")))
            ax.set_title(caption, fontsize=7)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return True
