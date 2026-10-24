"""End-to-end audit on a tiny synthetic MMRDR root (no dataset needed).

Fixture layout: 5 train images (tr000001 to tr000005), 3 test images (ts000001 to
ts000003). Every readable image is a synthetic fundus with seeded vessel lines.

* ts000001 is a brightened, re-encoded copy of tr000003 (not byte-identical), so
  it is a cross-split near-duplicate found through the cosine path.
* ts000002 is a byte-identical copy of tr000001, with a different grade label.
  It is an exact duplicate and a grade disagreement.
* ts000003 is not a JPEG, so it is unreadable.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from PIL import Image, ImageEnhance

from _synthetic import make_cfg, make_fundus
from drgrade.config import DataConfig

AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit.py"


def _load_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_script", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()

# name, grade, lesion vector, how the file is made
#   ("fundus", seed) draws a synthetic fundus; ("reencode", source) brightens and
#   re-encodes the source at JPEG quality 85; ("byte_copy", source) copies the file;
#   ("unreadable", None) writes non-JPEG bytes.
FIXTURE = [
    ("tr000001", 0, [0, 0, 0, 0, 0, 0, 0], ("fundus", 1)),
    ("tr000002", 2, [1, 1, 0, 0, 0, 0, 0], ("fundus", 2)),
    ("tr000003", 1, [1, 0, 0, 0, 0, 0, 0], ("fundus", 3)),
    ("tr000004", 4, [0, 0, 0, 0, 1, 0, 0], ("fundus", 4)),
    ("tr000005", 3, [0, 1, 0, 0, 0, 0, 0], ("fundus", 5)),
    ("ts000001", 1, [1, 0, 0, 0, 0, 0, 0], ("reencode", "tr000003")),
    ("ts000002", 1, [1, 0, 0, 0, 0, 0, 0], ("byte_copy", "tr000001")),
    ("ts000003", 2, [0, 0, 0, 0, 0, 0, 0], ("unreadable", None)),
]


def _make_root(root: Path) -> None:
    """Write the synthetic MMRDR-CFP folder with images and FP.csv."""
    img_dir = root / "MMRDR-CFP" / "img"
    img_dir.mkdir(parents=True)
    lines = ["type,image,grade,lesion,lr"]
    for name, grade, lesions, (kind, argument) in FIXTURE:
        path = img_dir / f"{name}.jpg"
        if kind == "fundus":
            make_fundus(seed=argument).save(path, "JPEG", quality=95)
        elif kind == "reencode":
            source = Image.open(img_dir / f"{argument}.jpg").convert("RGB")
            ImageEnhance.Brightness(source).enhance(1.1).save(path, "JPEG", quality=85)
        elif kind == "byte_copy":
            shutil.copyfile(img_dir / f"{argument}.jpg", path)
        else:
            path.write_bytes(b"this is not a jpeg")
        lesion_text = "[" + ", ".join(str(v) for v in lesions) + "]"
        lines.append(f'FP,img/{name}.jpg,{grade},"{lesion_text}",0')
    (root / "MMRDR-CFP" / "FP.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config(root: Path, out: Path) -> DataConfig:
    return make_cfg(mmrdr_root=str(root), audit_out=str(out))


def _pair(pairs: pd.DataFrame, first: str, second: str) -> pd.Series:
    """Return the single pair row for two ids, in either order."""
    match = pairs[
        ((pairs["image_id_a"] == first) & (pairs["image_id_b"] == second))
        | ((pairs["image_id_a"] == second) & (pairs["image_id_b"] == first))
    ]
    assert len(match) == 1, (first, second)
    return match.iloc[0]


@pytest.fixture(scope="module")
def audit_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path, DataConfig]:
    base = tmp_path_factory.mktemp("audit_smoke")
    root = base / "dataset"
    _make_root(root)
    cfg = _config(root, base / "out")
    summary = audit.run_audit(cfg, workers=1, limit=None, reuse_cache=False)
    return summary, base / "out", cfg


def test_summary_and_tables_exist(audit_run: tuple) -> None:
    _, out, _ = audit_run
    for name in (
        "summary.json",
        "images.csv",
        "counts_split_grade.csv",
        "laterality_by_split_grade.csv",
        "lesion_prevalence.csv",
        "flag_counts.csv",
        "flagged_images.csv",
        "exact_duplicates.csv",
        "near_duplicate_pairs.csv",
        "dup_clusters.csv",
        "grade_disagreement_groups.csv",
        "unreadable.csv",
        "vessel_signatures.npz",
        "fig_counts_by_grade.png",
        "fig_lesion_prevalence_train.png",
        "contact_cross_split_pairs.png",
        "contact_grade_disagreement_pairs.png",
    ):
        assert (out / name).is_file(), name
    assert "phash" not in pd.read_csv(out / "images.csv").columns
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["counts"]


def test_counts_match_fixture(audit_run: tuple) -> None:
    summary, _, _ = audit_run
    assert summary["counts"]["per_split_total"] == {"train": 5, "test": 3}
    assert summary["counts"]["train_grade_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}
    assert summary["counts"]["imbalance_ratio_train"] == 1.0
    assert summary["counts"]["laterality_by_split"]["train"] == {"left": 5, "right": 0}


def test_corrupt_image_is_listed_unreadable(audit_run: tuple) -> None:
    summary, out, _ = audit_run
    assert summary["image_properties"]["unreadable_ids"] == ["ts000003"]
    unreadable = pd.read_csv(out / "unreadable.csv")
    assert list(unreadable["image_id"]) == ["ts000003"]
    assert unreadable["error"].iloc[0]


def test_cross_split_reencoded_copy_is_found_by_cosine(audit_run: tuple) -> None:
    summary, out, cfg = audit_run
    pairs = pd.read_csv(out / "near_duplicate_pairs.csv")
    row = _pair(pairs, "tr000003", "ts000001")
    assert bool(row["exact"]) is False
    assert row["split_a"] != row["split_b"]
    assert row["grade_a"] == row["grade_b"] == 1
    assert row["similarity"] >= cfg.vsig_threshold
    near = summary["duplicates"]["near"]
    assert near["pairs_by_type"]["cross_split"]["pairs"] >= 1
    assert near["pairs_by_type"]["cross_split"]["grade_match"] >= 1


def test_byte_identical_copy_is_exact_and_disagrees_on_grade(audit_run: tuple) -> None:
    summary, out, _ = audit_run
    pairs = pd.read_csv(out / "near_duplicate_pairs.csv")
    row = _pair(pairs, "tr000001", "ts000002")
    assert bool(row["exact"]) is True
    assert row["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert {int(row["grade_a"]), int(row["grade_b"])} == {0, 1}
    disagreement = pd.read_csv(out / "grade_disagreement_groups.csv")
    assert {"tr000001", "ts000002"} <= set(disagreement["image_id"])
    assert summary["duplicates"]["near"]["n_groups_with_grade_disagreement"] == 1
    assert summary["duplicates"]["near"]["n_exact_pairs"] == 1


def test_cluster_file_contains_both_pairs(audit_run: tuple) -> None:
    summary, out, _ = audit_run
    clusters = pd.read_csv(out / "dup_clusters.csv")
    for first, second in (("tr000003", "ts000001"), ("tr000001", "ts000002")):
        rows = clusters[clusters["image_id"].isin([first, second])]
        assert len(rows) == 2
        assert rows["cluster_id"].nunique() == 1
        assert set(rows["cluster_size"]) == {2}
    near = summary["duplicates"]["near"]
    assert near["n_groups"] == 2
    assert near["test_images_in_groups_with_train"] == 2


def test_flags_counted_per_split(audit_run: tuple) -> None:
    summary, _, _ = audit_run
    # ts000003 is grade 2 with no lesion flagged, so F5 fires once in test.
    assert summary["labels"]["flag_counts"]["test"]["F5_grade_ge1_no_lesion"] == 1
    assert summary["labels"]["flag_counts"]["train"]["F5_grade_ge1_no_lesion"] == 0


def test_second_run_reuses_cache(audit_run: tuple) -> None:
    first, out, cfg = audit_run
    second = audit.run_audit(cfg, workers=1, limit=None, reuse_cache=True)
    assert second["duplicates"] == first["duplicates"]
    assert second["image_properties"]["unreadable_ids"] == ["ts000003"]
    assert second["run"]["vessel_signature_cache_reused"] is True


def test_signature_cache_is_not_reused_when_settings_change(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _make_root(root)
    cfg = _config(root, tmp_path / "out")
    first = audit.run_audit(cfg, workers=1)
    assert first["run"]["vessel_signature_cache_reused"] is False
    same = audit.run_audit(cfg, workers=1, reuse_cache=True)
    assert same["run"]["vessel_signature_cache_reused"] is True
    changed = replace(cfg, vsig_ring_frac=0.8)
    other = audit.run_audit(changed, workers=1, reuse_cache=True)
    assert other["run"]["vessel_signature_cache_reused"] is False


def test_limit_run_writes_to_subfolder(audit_run: tuple) -> None:
    _, out, cfg = audit_run
    summary = audit.run_audit(cfg, workers=1, limit=3, reuse_cache=False)
    assert summary["run"]["partial_run"] is True
    assert summary["counts"]["per_split_total"]["train"] == 3
    assert (out / "limit_3" / "summary.json").is_file()


def test_interrupted_pass_resumes_to_same_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass stopped by max_seconds and resumed gives the same per-image table."""
    audit = _load_audit_module()
    root = tmp_path / "root"
    _make_root(root)
    full_cfg = _config(root, tmp_path / "full")
    audit.run_audit(full_cfg, workers=1)
    expected = pd.read_csv(tmp_path / "full" / "images.csv").sort_values("image_id")

    monkeypatch.setattr(audit, "RESUME_BATCH_SIZE", 3)
    cfg = _config(root, tmp_path / "resumed")
    with pytest.raises(audit.AuditIncomplete):
        audit.run_audit(cfg, workers=1, max_seconds=0)
    partial = pd.read_csv(tmp_path / "resumed" / audit.PARTIAL_CACHE_NAME)
    assert len(partial) == 3

    audit.run_audit(cfg, workers=1, resume=True)
    resumed = pd.read_csv(tmp_path / "resumed" / "images.csv").sort_values("image_id")
    assert not (tmp_path / "resumed" / audit.PARTIAL_CACHE_NAME).exists()
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), resumed.reset_index(drop=True)
    )
