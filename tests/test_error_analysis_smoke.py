"""Tests for stage 6 (error analysis): case quotas, tie-breaks and an end-to-end smoke run on synthetic data."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from _prediction_data import ids_of_subset, make_predictions, register_test
from _training_data import write_tiny_dataset
from drgrade.errors import (
    CASE_ADJACENT,
    CASE_DISTANT,
    CASE_MOST_CONFIDENT,
    CASE_REFERABLE_MISS,
    add_scores,
    audit_flags_by_image,
    dangerous_misses,
    error_table,
    join_predictions,
    lesion_stratified,
    quality_stratified,
    select_cases,
)

REPO = Path(__file__).resolve().parents[1]
ERROR_SCRIPT = REPO / "scripts" / "error_analysis.py"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _scored(true: list[int], pred: list[int], conf: list[float]) -> pd.DataFrame:
    ids = [f"tr{k:06d}" for k in range(len(true))]
    frame = make_predictions(ids, true, pred, conf)
    manifest = pd.DataFrame(
        {
            "image_id": ids,
            "border_level": [0] * len(ids),
            "brightness": [100.0 + k for k in range(len(ids))],
            "contrast": [20.0] * len(ids),
            "sharpness": [5.0 + k for k in range(len(ids))],
            **{c: [0] * len(ids) for c in ("les_ma", "les_he", "les_irh", "les_vb_irma", "les_nv", "les_vh", "les_rd")},
            "file": [f"{i}.jpg" for i in ids],
        }
    )
    return add_scores(join_predictions(frame, manifest))


def test_case_quotas_and_reasons() -> None:
    """Quotas: 6 most confident errors, referable misses up to 6, 4 adjacent and 4 two-or-more apart, 2 correct per grade."""
    true: list[int] = []
    pred: list[int] = []
    conf: list[float] = []
    # 10 referable misses (true 2 to 4, predicted 0 or 1), with descending confidence.
    for k in range(10):
        true.append(2 + k % 3)
        pred.append(k % 2)
        conf.append(0.95 - 0.02 * k)
    # 6 adjacent errors (one grade apart, predicted higher) with mid confidence.
    for k in range(6):
        true.append(0)
        pred.append(1)
        conf.append(0.6 - 0.01 * k)
    # 6 errors two or more grades apart with low confidence.
    for k in range(6):
        true.append(0)
        pred.append(3)
        conf.append(0.3 - 0.01 * k)
    # Correct predictions: 4 per grade, descending confidence.
    for grade in range(5):
        for k in range(4):
            true.append(grade)
            pred.append(grade)
            conf.append(0.9 - 0.05 * k - 0.01 * grade)
    frame = _scored(true, pred, conf)
    cases = select_cases(frame)

    counts = cases["reason"].value_counts().to_dict()
    assert counts[CASE_MOST_CONFIDENT] == 6
    assert counts[CASE_REFERABLE_MISS] == 6
    assert counts[CASE_ADJACENT] == 4
    assert counts[CASE_DISTANT] == 4
    for grade in range(5):
        assert counts[f"correct_grade_{grade}"] == 2
    most_confident = cases.loc[cases["reason"] == CASE_MOST_CONFIDENT, "confidence"].tolist()
    assert most_confident == sorted(most_confident, reverse=True)
    # The referable-miss selection is the top six by confidence among the referable misses.
    misses = cases.loc[cases["reason"] == CASE_REFERABLE_MISS]
    assert (misses["true_grade"] >= 2).all() and (misses["pred_grade"] <= 1).all()


def test_selection_is_deterministic_and_breaks_ties_by_id() -> None:
    """Equal confidence is ordered by image_id, and the same input gives the same cases."""
    frame = _scored([0, 0, 0], [1, 1, 1], [0.7, 0.7, 0.7])
    first = select_cases(frame)
    second = select_cases(frame)
    pd.testing.assert_frame_equal(first, second)
    adjacent = first.loc[first["reason"] == CASE_ADJACENT, "image_id"].tolist()
    assert adjacent == sorted(adjacent)


def test_error_table_lists_only_misclassified_images_with_flags() -> None:
    """The error table has one row per error, with audit flags and the leakage flag."""
    # tr000000 and tr000002 are correct; tr000001 is one grade low; tr000003 is a referable miss.
    frame = _scored([0, 1, 2, 4], [0, 0, 2, 1], [0.9, 0.6, 0.8, 0.5])
    flags = pd.DataFrame({"image_id": ["tr000001", "tr000001"], "split": ["train"] * 2, "grade": [1, 1], "flag": ["F1", "F5"]})
    errors = error_table(frame, audit_flags_by_image(flags), leakage_ids=["tr000003"])
    assert set(errors["image_id"]) == {"tr000001", "tr000003"}
    rows = errors.set_index("image_id")
    assert rows.loc["tr000001", "audit_flags"] == "F1;F5"
    assert rows.loc["tr000003", "audit_flags"] == ""
    assert bool(rows.loc["tr000003", "in_leakage_list"]) is True
    assert bool(rows.loc["tr000001", "in_leakage_list"]) is False
    assert bool(rows.loc["tr000003", "referable_miss"]) is True
    assert bool(rows.loc["tr000001", "referable_miss"]) is False


def test_lesion_and_quality_strata_have_expected_shape() -> None:
    """Seven lesion rows with present and absent counts, and quartile rows for each quality measure."""
    frame = _scored([0, 2, 3, 4, 2, 0, 1, 4], [0, 1, 3, 4, 2, 0, 1, 2], [0.9] * 8)
    lesions = lesion_stratified(frame)
    assert len(lesions) == 7
    assert (lesions["n_present"] + lesions["n_absent"] == len(frame)).all()
    quality = quality_stratified(frame)
    assert set(quality["measure"]) == {"brightness", "contrast", "sharpness"}
    assert quality.groupby("measure")["n"].sum().tolist() == [len(frame)] * 3


def test_dangerous_misses_use_nv_or_vh_below_grade_4() -> None:
    """An image with neovascularisation predicted at grade 3 is listed; the same image at grade 4 is not."""
    frame = _scored([4, 4], [3, 4], [0.7, 0.8])
    frame.loc[:, "les_nv"] = [1, 1]
    listed = dangerous_misses(frame)
    assert list(listed["image_id"]) == ["tr000000"]


def _write_case(tmp_path: Path) -> dict[str, Path]:
    dataset = write_tiny_dataset(tmp_path / "data", target_size=64, seed=4)
    split = pd.read_csv(dataset.split, dtype={"image_id": str, "subset": str})
    manifest = pd.read_csv(dataset.manifest, dtype={"image_id": str})
    manifest.loc[manifest.index[:1], "les_nv"] = 1
    manifest.to_csv(dataset.manifest, index=False)
    truth = dict(zip(manifest["image_id"], manifest["grade"].astype(int)))
    val_ids = ids_of_subset(split, "val")
    pred = [(truth[i] + (1 if k % 2 else -1)) % 5 for k, i in enumerate(val_ids)]
    conf = [0.4 + 0.05 * k for k in range(len(val_ids))]
    pred_path = tmp_path / "runs" / "x" / "predictions" / "val.csv"
    pred_path.parent.mkdir(parents=True)
    make_predictions(val_ids, [truth[i] for i in val_ids], pred, conf).to_csv(pred_path, index=False)
    flags = tmp_path / "flagged.csv"
    pd.DataFrame({"image_id": [val_ids[0]], "split": ["train"], "grade": [truth[val_ids[0]]], "flag": ["F5_grade_ge1_no_lesion"]}).to_csv(flags, index=False)
    leak = tmp_path / "leak.csv"
    pd.DataFrame({"image_id": ["ts000001"], "linked_train_ids": [val_ids[0]], "max_similarity": [0.99], "exact": [True]}).to_csv(leak, index=False)
    return {"dataset": Path(dataset.split), "manifest": Path(dataset.manifest), "pred": pred_path, "flags": flags, "leak": leak, "split": Path(dataset.split)}


def test_end_to_end_smoke_writes_every_file(tmp_path: Path) -> None:
    """The script writes the tables, the case reasons and one PNG contact sheet per reason."""
    paths = _write_case(tmp_path)
    spec = importlib.util.spec_from_file_location("error_script", ERROR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["error_script"] = module
    spec.loader.exec_module(module)
    out = tmp_path / "errors_out"
    status = module.main(
        [
            "--pred", str(paths["pred"]),
            "--manifest", str(paths["manifest"]),
            "--flags", str(paths["flags"]),
            "--split", str(paths["split"]),
            "--leakage-list", str(paths["leak"]),
            "--out", str(out),
        ]
    )
    assert status == 0
    for name in ("errors.csv", "lesion_stratified.csv", "dangerous_misses.csv", "quality_stratified.csv", "cases.csv"):
        assert (out / name).is_file(), name
    errors = pd.read_csv(out / "errors.csv")
    preds = pd.read_csv(paths["pred"])
    assert len(errors) == int((preds["true_grade"] != preds["pred_grade"]).sum())
    cases = pd.read_csv(out / "cases.csv")
    assert "reason" in cases.columns and CASE_MOST_CONFIDENT in set(cases["reason"])
    sheets = sorted(out.glob("contact_*.png"))
    assert sheets, "no contact sheets written"
    for sheet in sheets:
        assert sheet.read_bytes()[:8] == PNG_MAGIC
    assert set(cases["reason"]) == {p.stem[len("contact_"):] for p in sheets}
    dangerous = pd.read_csv(out / "dangerous_misses.csv")
    assert all(dangerous["pred_grade"] < 4)


def test_test_error_analysis_needs_the_leakage_list(tmp_path: Path) -> None:
    """A test predictions file without the leakage list is refused (status 2)."""
    paths = _write_case(tmp_path)
    test_pred = tmp_path / "runs" / "x" / "predictions" / "test.csv"
    split = pd.read_csv(paths["split"], dtype={"image_id": str, "subset": str})
    test_ids = ids_of_subset(split, "test")
    manifest = pd.read_csv(paths["manifest"], dtype={"image_id": str})
    truth = dict(zip(manifest["image_id"], manifest["grade"].astype(int)))
    make_predictions(test_ids, [truth[i] for i in test_ids], [truth[i] for i in test_ids], [0.9] * len(test_ids)).to_csv(test_pred, index=False)
    register_test(test_pred)
    spec = importlib.util.spec_from_file_location("error_script2", ERROR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["error_script2"] = module
    spec.loader.exec_module(module)
    status = module.main(
        ["--pred", str(test_pred), "--manifest", str(paths["manifest"]), "--split", str(paths["split"]),
         "--leakage-list", str(tmp_path / "absent.csv"), "--out", str(tmp_path / "t_out")]
    )
    assert status == 2


def test_missing_manifest_column_is_refused() -> None:
    """Joining with a manifest that lacks a required column raises ValueError."""
    preds = make_predictions(["tr000001"], [0], [0], [0.9])
    with pytest.raises(ValueError, match="lacks the columns"):
        join_predictions(preds, pd.DataFrame({"image_id": ["tr000001"]}))
