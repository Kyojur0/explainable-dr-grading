"""Label-consistency rules for the data audit (brief 5.3).

Flags are informational only. A flag may reflect incomplete lesion annotation
rather than a wrong grade. Labels are never changed based on these rules.
"""

from __future__ import annotations

import pandas as pd

from drgrade.io import LESION_COLS

FLAG_NAMES: tuple[str, ...] = (
    "F1_grade0_with_lesion",
    "F2_grade1_beyond_ma",
    "F3_pdr_signs_below_4",
    "F4_grade4_without_pdr_signs",
    "F5_grade_ge1_no_lesion",
)

MICROANEURYSM_COL = "les_ma"
PDR_SIGN_COLS: tuple[str, ...] = ("les_nv", "les_vh")


def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return one boolean column per flag in FLAG_NAMES, indexed like `df`.

    `df` needs an integer `grade` column and the columns in LESION_COLS (boolean).
    """
    grade = df["grade"]
    lesions = df[list(LESION_COLS)].astype(bool)
    any_lesion = lesions.any(axis=1)
    any_non_ma = lesions.drop(columns=[MICROANEURYSM_COL]).any(axis=1)
    pdr_signs = lesions[list(PDR_SIGN_COLS)].any(axis=1)

    return pd.DataFrame(
        {
            FLAG_NAMES[0]: (grade == 0) & any_lesion,
            FLAG_NAMES[1]: (grade == 1) & any_non_ma,
            FLAG_NAMES[2]: pdr_signs & (grade < 4),
            FLAG_NAMES[3]: (grade == 4) & ~pdr_signs,
            FLAG_NAMES[4]: (grade >= 1) & ~any_lesion,
        },
        index=df.index,
    )
