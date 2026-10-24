"""CSV loading and lesion parsing for the MMRDR CFP annotations (brief 3.3).

Confirmed format (see DECISIONS.md, D01):
    header:  type,image,grade,lesion,lr
    example: FP,img/tr000002.jpg,3,"[1, 1, 1, 0, 0, 0, 0]",1
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

CFP_DIRNAME = "MMRDR-CFP"
FP_CSV_NAME = "FP.csv"
EXPECTED_COLUMNS: tuple[str, ...] = ("type", "image", "grade", "lesion", "lr")
RECORD_TYPE = "FP"

# Seven binary lesion columns, in the order used by the dataset README.
LESION_COLS: tuple[str, ...] = (
    "les_ma",
    "les_he",
    "les_irh",
    "les_vb_irma",
    "les_nv",
    "les_vh",
    "les_rd",
)
LESION_NAMES: dict[str, str] = {
    "les_ma": "microaneurysm",
    "les_he": "hard exudate",
    "les_irh": "intraretinal haemorrhage",
    "les_vb_irma": "venous beading / IRMA",
    "les_nv": "neovascularisation",
    "les_vh": "vitreous haemorrhage",
    "les_rd": "retinal detachment",
}

GRADE_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4)
LR_VALUES: tuple[int, ...] = (0, 1)  # 0 = left eye, 1 = right eye
SPLIT_PREFIXES: dict[str, str] = {"tr": "train", "ts": "test"}
SPLIT_NAMES: tuple[str, ...] = ("train", "test")

OUTPUT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "rel_path",
    "official_split",
    "grade",
    "lr",
    *LESION_COLS,
)


def official_split(image_id: str) -> str:
    """Map an image id prefix (tr or ts) to its official split name.

    Raises ValueError for any other prefix so that unknown files are never
    silently assigned to a split.
    """
    prefix = image_id[:2]
    if prefix not in SPLIT_PREFIXES:
        raise ValueError(f"{image_id}: unknown split prefix {prefix!r} (expected tr or ts)")
    return SPLIT_PREFIXES[prefix]


def parse_lesion(raw: str, image_id: str) -> tuple[bool, ...]:
    """Parse a lesion field such as "[1, 0, 0, 0, 0, 0, 0]" into seven booleans.

    Raises ValueError (naming the image id) for malformed strings, a wrong
    number of values, or any value outside {0, 1}.
    """
    try:
        values: Any = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"{image_id}: lesion field is not a parsable list: {raw!r}") from exc
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{image_id}: lesion field is not a list: {raw!r}")
    if len(values) != len(LESION_COLS):
        raise ValueError(
            f"{image_id}: lesion field has {len(values)} values, expected {len(LESION_COLS)}: {raw!r}"
        )
    for value in values:
        if not isinstance(value, int) or value not in (0, 1):
            raise ValueError(f"{image_id}: lesion values must be 0 or 1, got {raw!r}")
    return tuple(bool(value) for value in values)


def image_path(root: Path, rel_path: str) -> Path:
    """Return the on-disk path of an image given its CSV 'image' field."""
    return Path(root) / CFP_DIRNAME / rel_path


def _parse_int(raw: str, image_id: str, field: str) -> int:
    """Parse an integer field, naming the image id on failure."""
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{image_id}: {field} is not an integer: {raw!r}") from exc


def load_fp_csv(root: Path) -> pd.DataFrame:
    """Load and validate FP.csv under the dataset root.

    Returns one row per image with columns: image_id, rel_path, official_split,
    grade (int 0..4), lr (int 0 or 1), and the seven boolean lesion columns.
    Raises ValueError on any malformed row. Labels are never modified.
    """
    csv_path = Path(root) / CFP_DIRNAME / FP_CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"{FP_CSV_NAME} not found at {csv_path}")
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if list(raw.columns) != list(EXPECTED_COLUMNS):
        raise ValueError(f"{FP_CSV_NAME} header is {list(raw.columns)}, expected {list(EXPECTED_COLUMNS)}")
    if raw.empty:
        raise ValueError(f"{FP_CSV_NAME} has no rows")

    records: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        image_id = Path(row["image"]).stem
        if row["type"] != RECORD_TYPE:
            raise ValueError(f"{image_id}: type is {row['type']!r}, expected {RECORD_TYPE!r}")
        grade = _parse_int(row["grade"], image_id, "grade")
        if grade not in GRADE_VALUES:
            raise ValueError(f"{image_id}: grade {grade} outside 0..4")
        lr = _parse_int(row["lr"], image_id, "lr")
        if lr not in LR_VALUES:
            raise ValueError(f"{image_id}: lr {lr} is not 0 or 1")
        lesions = parse_lesion(row["lesion"], image_id)
        record: dict[str, Any] = {
            "image_id": image_id,
            "rel_path": row["image"],
            "official_split": official_split(image_id),
            "grade": grade,
            "lr": lr,
        }
        record.update(zip(LESION_COLS, lesions))
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=list(OUTPUT_COLUMNS))
    duplicated = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
    if duplicated:
        raise ValueError(f"duplicate image ids in {FP_CSV_NAME}: {duplicated[:10]}")
    df["grade"] = df["grade"].astype("int64")
    df["lr"] = df["lr"].astype("int64")
    for col in LESION_COLS:
        df[col] = df[col].astype(bool)
    return df
