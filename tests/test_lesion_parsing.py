"""Tests for FP.csv loading, lesion parsing and split mapping (brief 3.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from drgrade.io import LESION_COLS, load_fp_csv, official_split, parse_lesion

HEADER = "type,image,grade,lesion,lr"


def _write_csv(root: Path, lines: list[str]) -> Path:
    """Write a synthetic MMRDR root with the given FP.csv data lines."""
    cfp = root / "MMRDR-CFP"
    cfp.mkdir(parents=True)
    (cfp / "FP.csv").write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return root


def test_good_rows_expand_to_named_columns(tmp_path: Path) -> None:
    root = _write_csv(
        tmp_path,
        [
            'FP,img/tr000001.jpg,0,"[0, 0, 0, 0, 0, 0, 0]",0',
            'FP,img/tr000002.jpg,3,"[1, 1, 1, 0, 0, 0, 0]",1',
            'FP,img/ts000001.jpg,4,"[0, 0, 0, 0, 1, 1, 0]",1',
        ],
    )
    df = load_fp_csv(root)
    assert len(df) == 3
    assert list(df["official_split"]) == ["train", "train", "test"]
    assert list(df["grade"]) == [0, 3, 4]
    assert list(df["lr"]) == [0, 1, 1]
    assert df.loc[1, "les_ma"] and df.loc[1, "les_he"] and df.loc[1, "les_irh"]
    assert not df.loc[1, "les_nv"]
    assert df.loc[2, "les_nv"] and df.loc[2, "les_vh"]
    for col in LESION_COLS:
        assert df[col].dtype == bool


def test_bad_length_raises_with_image_id(tmp_path: Path) -> None:
    root = _write_csv(tmp_path, ['FP,img/tr000007.jpg,1,"[1, 0, 0, 0, 0, 0]",0'])
    with pytest.raises(ValueError, match="tr000007"):
        load_fp_csv(root)


def test_bad_value_raises(tmp_path: Path) -> None:
    root = _write_csv(tmp_path, ['FP,img/tr000008.jpg,1,"[2, 0, 0, 0, 0, 0, 0]",0'])
    with pytest.raises(ValueError, match="tr000008"):
        load_fp_csv(root)


def test_malformed_string_raises() -> None:
    with pytest.raises(ValueError, match="tr000009"):
        parse_lesion("[0, 1, 0", "tr000009")


def test_non_list_raises() -> None:
    with pytest.raises(ValueError, match="tr000010"):
        parse_lesion("(0, 0, 0, 0, 0, 0, 0)x", "tr000010")


def test_float_values_rejected() -> None:
    with pytest.raises(ValueError):
        parse_lesion("[0.0, 0, 0, 0, 0, 0, 0]", "tr000011")


def test_parse_lesion_returns_booleans() -> None:
    assert parse_lesion("[1, 0, 0, 0, 0, 0, 1]", "tr000012") == (
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    )


def test_split_prefix_mapping() -> None:
    assert official_split("tr000001") == "train"
    assert official_split("ts000001") == "test"


def test_unknown_prefix_raises() -> None:
    with pytest.raises(ValueError, match="va000001"):
        official_split("va000001")


def test_bad_prefix_in_csv_raises(tmp_path: Path) -> None:
    root = _write_csv(tmp_path, ['FP,img/va000001.jpg,0,"[0, 0, 0, 0, 0, 0, 0]",0'])
    with pytest.raises(ValueError, match="va000001"):
        load_fp_csv(root)


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    root = _write_csv(
        tmp_path,
        [
            'FP,img/tr000001.jpg,0,"[0, 0, 0, 0, 0, 0, 0]",0',
            'FP,img/tr000001.jpg,0,"[0, 0, 0, 0, 0, 0, 0]",0',
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_fp_csv(root)
