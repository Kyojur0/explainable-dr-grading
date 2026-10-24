"""Hand-made cases for each consistency flag F1 to F5 (brief 5.3)."""

from __future__ import annotations

import pandas as pd

from drgrade.audit_rules import FLAG_NAMES, compute_flags
from drgrade.io import LESION_COLS


def _row(grade: int, **lesions: int) -> dict[str, object]:
    """One row with the given grade and lesions switched on (others off)."""
    row: dict[str, object] = {"grade": grade}
    for col in LESION_COLS:
        row[col] = bool(lesions.get(col, 0))
    return row


def _flags_for(rows: list[dict[str, object]]) -> pd.DataFrame:
    return compute_flags(pd.DataFrame(rows))


def test_f1_grade0_with_any_lesion() -> None:
    flags = _flags_for([_row(0), _row(0, les_ma=1), _row(0, les_rd=1)])
    assert list(flags[FLAG_NAMES[0]]) == [False, True, True]


def test_f2_grade1_with_non_ma_lesion() -> None:
    flags = _flags_for([_row(1, les_ma=1), _row(1, les_ma=1, les_he=1), _row(1, les_vh=1)])
    assert list(flags[FLAG_NAMES[1]]) == [False, True, True]


def test_f3_pdr_signs_below_grade_4() -> None:
    flags = _flags_for([_row(3, les_nv=1), _row(4, les_nv=1), _row(2, les_vh=1), _row(2, les_ma=1)])
    assert list(flags[FLAG_NAMES[2]]) == [True, False, True, False]


def test_f4_grade4_without_pdr_signs() -> None:
    flags = _flags_for([_row(4, les_nv=1), _row(4, les_vh=1), _row(4, les_he=1), _row(3)])
    assert list(flags[FLAG_NAMES[3]]) == [False, False, True, False]


def test_f5_grade_ge1_without_lesion() -> None:
    flags = _flags_for([_row(0), _row(1), _row(2, les_ma=1), _row(4)])
    assert list(flags[FLAG_NAMES[4]]) == [False, True, False, True]


def test_flags_keep_the_input_index() -> None:
    df = pd.DataFrame([_row(1)], index=[42])
    flags = compute_flags(df)
    assert list(flags.index) == [42]
    assert list(flags.columns) == list(FLAG_NAMES)
