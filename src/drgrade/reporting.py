"""Stage 5 reporting: headline summaries, the validation comparison and the confusion figure.

Figures use matplotlib's Figure with the Agg canvas directly, so no pyplot state is kept
and the code runs headless. Every file is written through a temporary name and os.replace.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from drgrade.io import GRADE_VALUES
from drgrade.utils import write_atomically

FIGURE_DPI = 150
GRADE_NAMES: dict[int, str] = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"}
COMPARE_COLUMNS: tuple[str, ...] = (
    "run",
    "n",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "qwk",
    "referable_sensitivity",
    "referable_miss_rate",
)
RULE_TEXT = "highest validation QWK; ties broken by higher validation macro-F1"


def grade_label(grade: int) -> str:
    """Return the axis label for a grade, for example '4 PDR'."""
    return f"{grade} {GRADE_NAMES[grade]}"


def nan_low(value: float | None) -> float:
    """Map None and NaN to negative infinity, so that they never win a comparison."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return -math.inf
    return float(value)


def select_run(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the row the pre-declared rule selects: highest QWK, then macro-F1, then run name."""
    if not rows:
        raise ValueError("no runs to compare")
    return sorted(rows, key=lambda r: (-nan_low(r["qwk"]), -nan_low(r["macro_f1"]), str(r["run"])))[0]


def comparison_markdown(rows: Sequence[dict[str, Any]], selected: str | None, note: str | None = None) -> str:
    """Return the validation comparison as a markdown table.

    With `selected`, the pre-declared selection is stated underneath. Without it (tables of
    stability runs or ablations, which are not candidates for selection), `note` is written
    instead, so the table cannot be read as a model selection.
    """
    header = "| " + " | ".join(COMPARE_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(COMPARE_COLUMNS)) + "|"
    body = []
    for row in rows:
        cells = []
        for column in COMPARE_COLUMNS:
            value = row[column]
            cells.append("" if value is None or (isinstance(value, float) and math.isnan(value)) else _fmt(value))
        body.append("| " + " | ".join(cells) + " |")
    footer = (
        f"Pre-declared rule ({RULE_TEXT}) selects: **{selected}**."
        if selected is not None
        else (note or "No selection: these runs are not candidates for model selection.")
    )
    lines = [header, divider, *body, "", footer]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    """Format a number for markdown; integers are written without decimals."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def write_comparison(
    rows: Sequence[dict[str, Any]], csv_path: Path, md_path: Path, select: bool = True, note: str | None = None
) -> str | None:
    """Write the comparison CSV and markdown; return the selected run name, or None when select is False."""
    selected = str(select_run(rows)["run"]) if select else None
    frame = pd.DataFrame(list(rows), columns=list(COMPARE_COLUMNS))
    write_atomically(csv_path, lambda p: frame.to_csv(p, index=False))
    text = comparison_markdown(rows, selected, note)
    write_atomically(md_path, lambda p: p.write_text(text, encoding="utf-8"))
    return selected


def write_confusion_figure(counts: np.ndarray, norm: list[list[float | None]], title: str, path: Path) -> None:
    """Write an annotated heatmap of the row-normalised confusion matrix (PNG, 150 dpi).

    Each cell shows the row share and the count. Rows with no true images are blank.
    """
    grades = list(GRADE_VALUES)
    values = np.array([[np.nan if v is None else v for v in row] for row in norm], dtype=float)
    fig = Figure(figsize=(7.0, 6.0), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    image = ax.imshow(np.nan_to_num(values, nan=0.0), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(grades, [grade_label(g) for g in grades], rotation=35, ha="right")
    ax.set_yticks(grades, [grade_label(g) for g in grades])
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    for i in grades:
        for j in grades:
            if math.isnan(values[i, j]):
                label = f"n/a\n{counts[i, j]}"
                colour = "black"
            else:
                label = f"{values[i, j]:.2f}\n{counts[i, j]}"
                colour = "white" if values[i, j] > 0.5 else "black"
            ax.text(j, i, label, ha="center", va="center", color=colour, fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8, label="share of true grade")
    write_atomically(path, lambda p: fig.savefig(p, dpi=FIGURE_DPI, format="png"))


def metric_or_blank(value: Any) -> float | None:
    """Return a finite float, or None for undefined values, for CSV cells."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def headline_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Return one comparison-style row from full_metrics output."""
    referable = metrics["referable"]
    return {
        "run": name,
        "n": metrics["n"],
        "accuracy": metric_or_blank(metrics["accuracy"]),
        "macro_f1": metric_or_blank(metrics["macro_f1"]),
        "weighted_f1": metric_or_blank(metrics["weighted_f1"]),
        "qwk": metric_or_blank(metrics["qwk"]),
        "referable_sensitivity": metric_or_blank(referable["sensitivity"]),
        "referable_miss_rate": metric_or_blank(metrics["errors"]["referable_miss_rate"]),
    }


def subset_of_ids(ids: Sequence[str], split: pd.DataFrame) -> str:
    """Return the single split subset (train, val or test) that all the ids belong to.

    Raises ValueError for an id missing from the split file, or ids from more than one subset.
    """
    subset_of = dict(zip(split["image_id"].astype(str), split["subset"].astype(str)))
    missing = [i for i in ids if i not in subset_of]
    if missing:
        raise ValueError(f"{len(missing)} predicted ids are not in the split file, first: {missing[:5]}")
    names = {subset_of[i] for i in ids}
    if len(names) != 1:
        raise ValueError(f"predictions mix subsets {sorted(names)}")
    return names.pop()


def json_safe(value: Any) -> Any:
    """Return the value with NaN and infinite floats replaced by None, recursively (valid JSON)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    return value


def contact_sheet(
    rows: pd.DataFrame,
    image_dir: Path,
    path: Path,
    title: str,
    columns: int = 3,
) -> None:
    """Write a captioned grid of the case images (id, true, predicted, confidence, lesions present).

    A missing JPEG is shown as a labelled blank tile, and the sheet is still written.
    """
    count = len(rows)
    if count == 0:
        raise ValueError("a contact sheet needs at least one case")
    n_cols = min(columns, count)
    n_rows = math.ceil(count / n_cols)
    fig = Figure(figsize=(n_cols * 3.4, n_rows * 3.9), constrained_layout=True)
    FigureCanvasAgg(fig)
    for index, row in enumerate(rows.itertuples(index=False)):
        ax = fig.add_subplot(n_rows, n_cols, index + 1)
        jpg = Path(image_dir) / f"{row.image_id}.jpg"
        if jpg.is_file():
            with Image.open(jpg) as handle:
                ax.imshow(handle.convert("RGB"))
        else:
            ax.text(0.5, 0.5, "image missing", ha="center", va="center")
        ax.axis("off")
        lesions = row.lesions_present if isinstance(row.lesions_present, str) and row.lesions_present else "none"
        ax.set_title(
            f"{row.image_id}\ntrue {int(row.true_grade)}  pred {int(row.pred_grade)}  conf {float(row.confidence):.2f}\n"
            f"lesions: {lesions}",
            fontsize=8,
        )
    fig.suptitle(title)
    write_atomically(path, lambda p: fig.savefig(p, dpi=FIGURE_DPI, format="png"))
