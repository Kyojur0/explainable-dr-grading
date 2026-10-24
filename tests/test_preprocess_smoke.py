"""End-to-end preprocessing on a tiny synthetic MMRDR root (no dataset needed).

Fixture: 5 train images (tr000001 to tr000005) and 2 test images (ts000001, ts000002).
tr000005 is landscape, so its crop is padded to square. Outputs are 64 px for speed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from _synthetic import make_fundus
from drgrade.config import ConfigError

PREPROCESS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"

# name, grade, lesion vector
FIXTURE = [
    ("tr000001", 0, [0, 0, 0, 0, 0, 0, 0]),
    ("tr000002", 2, [1, 1, 0, 0, 0, 0, 0]),
    ("tr000003", 1, [1, 0, 0, 0, 0, 0, 0]),
    ("tr000004", 4, [0, 0, 0, 0, 1, 1, 0]),
    ("tr000005", 3, [0, 1, 0, 0, 0, 0, 0]),
    ("ts000001", 1, [1, 0, 0, 0, 0, 0, 0]),
    ("ts000002", 2, [0, 0, 1, 0, 0, 0, 0]),
]
EXPECTED_COLUMNS = [
    "image_id",
    "official_split",
    "grade",
    "lr",
    "les_ma",
    "les_he",
    "les_irh",
    "les_vb_irma",
    "les_nv",
    "les_vh",
    "les_rd",
    "orig_width",
    "orig_height",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "crop_failed",
    "border_level",
    "brightness",
    "contrast",
    "sharpness",
    "file",
]
TARGET = 64


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("preprocess_script", PREPROCESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = _load_script()


def _make_root(root: Path) -> None:
    """Write the synthetic MMRDR-CFP folder with images and FP.csv."""
    img_dir = root / "MMRDR-CFP" / "img"
    img_dir.mkdir(parents=True)
    lines = ["type,image,grade,lesion,lr"]
    for index, (name, grade, lesions) in enumerate(FIXTURE, start=1):
        path = img_dir / f"{name}.jpg"
        if name == "tr000005":
            canvas = Image.new("RGB", (320, 240), (0, 0, 0))
            canvas.paste(make_fundus(seed=index, size=240), (40, 0))
            canvas.save(path, "JPEG", quality=95)
        else:
            make_fundus(seed=index, size=256).save(path, "JPEG", quality=95)
        lesion_text = "[" + ", ".join(str(v) for v in lesions) + "]"
        lines.append(f'FP,img/{name}.jpg,{grade},"{lesion_text}",{index % 2}')
    (root / "MMRDR-CFP" / "FP.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_config(base: Path, name: str, root: Path, **overrides: object) -> Path:
    """Write a configs-style YAML whose outputs live under base/name."""
    values: dict[str, object] = {
        "mmrdr_root": str(root),
        "crop_threshold": 10,
        "crop_margin_frac": 0.02,
        "crop_min_area_frac": 0.30,
        "bg_detect_min": 8,
        "bg_tolerance": 10,
        "bg_corner_spread": 12,
        "target_size": TARGET,
        "circle_mask": True,
        "jpeg_quality": 90,
        "processed_dir": str(base / name / "processed"),
        "ben_graham": False,
        "ben_graham_sigma": 10,
        "vsig_decode_scale": 4,
        "vsig_canvas": 256,
        "vsig_blur_radius": 6,
        "vsig_ring_frac": 0.9,
        "vsig_grid": 64,
        "vsig_threshold": 0.93,
        "audit_out": str(base / name / "outputs" / "audit"),
        "seed": 42,
    }
    values.update(overrides)
    lines = [f"{key}: {'null' if value is None else value}" for key, value in values.items()]
    path = base / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_manifest(folder: Path) -> list[dict[str, str]]:
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("preprocess_smoke")
    root = base / "dataset"
    _make_root(root)
    return root


@pytest.fixture(scope="module")
def uninterrupted(dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One complete run, used as the reference for the other tests."""
    base = tmp_path_factory.mktemp("uninterrupted")
    config = _write_config(base, "ref", dataset)
    assert preprocess.main(["--config", str(config), "--workers", "1"]) == 0
    return base / "ref" / "processed" / f"s{TARGET}"


def test_manifest_columns_rows_and_order(uninterrupted: Path) -> None:
    with (uninterrupted / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == EXPECTED_COLUMNS
    rows = _read_manifest(uninterrupted)
    ids = [row["image_id"] for row in rows]
    assert ids == sorted(name for name, _, _ in FIXTURE)
    by_id = {row["image_id"]: row for row in rows}
    assert by_id["ts000002"]["official_split"] == "test"
    assert by_id["tr000004"]["grade"] == "4"
    assert by_id["tr000004"]["les_nv"] == "1"
    assert by_id["tr000005"]["orig_width"] == "320"
    assert by_id["tr000005"]["file"] == "tr000005.jpg"


def test_processed_files_are_square_and_complete(uninterrupted: Path) -> None:
    for row in _read_manifest(uninterrupted):
        with Image.open(uninterrupted / row["file"]) as saved:
            assert saved.size == (TARGET, TARGET)
            assert saved.mode == "RGB"


def test_no_temporary_or_partial_files_remain(uninterrupted: Path) -> None:
    assert not list(uninterrupted.glob("*.tmp"))
    assert not (uninterrupted / "manifest_partial.csv").exists()


def test_info_file_records_settings_and_counts(uninterrupted: Path) -> None:
    info = json.loads((uninterrupted / "preprocess_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "complete"
    assert info["n_images"] == len(FIXTURE)
    assert info["n_done"] == len(FIXTURE)
    assert info["n_crop_failed"] == 0
    assert info["config"]["target_size"] == TARGET
    assert info["config"]["jpeg_quality"] == 90
    assert "updated_utc" in info and "git" in info


def test_contact_figure_is_written_to_audit_sibling(uninterrupted: Path) -> None:
    # audit_out is <base>/ref/outputs/audit, so the figure goes to <base>/ref/outputs/preprocess.
    figure = uninterrupted.parents[1] / "outputs" / "preprocess" / "contact_processed.png"
    assert figure.is_file()


def test_interrupted_run_resumes_to_identical_manifest(
    dataset: Path, uninterrupted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, "resume", dataset)
    monkeypatch.setattr(preprocess, "RESUME_BATCH_SIZE", 2)

    assert preprocess.main(["--config", str(config), "--workers", "1", "--max-seconds", "0"]) == 3
    folder = tmp_path / "resume" / "processed" / f"s{TARGET}"
    assert (folder / "manifest_partial.csv").is_file()
    assert not (folder / "manifest.csv").exists()
    partial_rows = (folder / "manifest_partial.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(partial_rows) == 1 + 2  # header plus the first batch

    assert preprocess.main(["--config", str(config), "--workers", "1"]) == 0
    assert (folder / "manifest.csv").read_bytes() == (uninterrupted / "manifest.csv").read_bytes()
    assert not (folder / "manifest_partial.csv").exists()
    assert not list(folder.glob("*.tmp"))


def test_resume_skips_images_whose_jpeg_is_missing(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, "missing", dataset)
    monkeypatch.setattr(preprocess, "RESUME_BATCH_SIZE", 2)
    assert preprocess.main(["--config", str(config), "--workers", "1", "--max-seconds", "0"]) == 3
    folder = tmp_path / "missing" / "processed" / f"s{TARGET}"
    done_id = (folder / "manifest_partial.csv").read_text(encoding="utf-8").splitlines()[1].split(",")[0]
    (folder / f"{done_id}.jpg").unlink()

    assert preprocess.main(["--config", str(config), "--workers", "1"]) == 0
    rows = _read_manifest(folder)
    assert len(rows) == len(FIXTURE)
    assert (folder / f"{done_id}.jpg").is_file()


def test_changed_settings_refuse_to_resume(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, "changed", dataset)
    monkeypatch.setattr(preprocess, "RESUME_BATCH_SIZE", 2)
    assert preprocess.main(["--config", str(config), "--workers", "1", "--max-seconds", "0"]) == 3

    changed = _write_config(tmp_path, "changed", dataset, jpeg_quality=80)
    assert preprocess.main(["--config", str(changed), "--workers", "1"]) == 2
    assert preprocess.main(["--config", str(changed), "--workers", "1", "--overwrite"]) == 0


def test_zip_contains_every_image_stored(dataset: Path, tmp_path: Path) -> None:
    config = _write_config(tmp_path, "zipped", dataset)
    assert preprocess.main(["--config", str(config), "--workers", "1", "--zip"]) == 0
    zip_path = tmp_path / "zipped" / "processed" / f"s{TARGET}.zip"
    assert zip_path.is_file()
    assert not list(zip_path.parent.glob("*.tmp"))
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    expected = {f"s{TARGET}/{name}.jpg" for name, _, _ in FIXTURE}
    assert expected <= names
    assert f"s{TARGET}/manifest.csv" in names
    assert f"s{TARGET}/preprocess_info.json" in names


def test_ben_graham_run_changes_pixels_not_measures(
    dataset: Path, uninterrupted: Path, tmp_path: Path
) -> None:
    # Quality measures describe the image before normalisation (DECISIONS.md D10), so
    # switching the ablation flag must change the saved pixels but not the manifest measures.
    config = _write_config(tmp_path, "bg", dataset, ben_graham="true")
    assert preprocess.main(["--config", str(config), "--workers", "1"]) == 0
    folder = tmp_path / "bg" / "processed" / f"s{TARGET}"
    plain = {row["image_id"]: row for row in _read_manifest(uninterrupted)}
    normalised = {row["image_id"]: row for row in _read_manifest(folder)}
    for image_id, row in plain.items():
        for key in ("brightness", "contrast", "sharpness", "crop_failed"):
            assert normalised[image_id][key] == row[key], (image_id, key)
    name = "tr000002.jpg"
    with Image.open(folder / name) as bg_img, Image.open(uninterrupted / name) as plain_img:
        assert bg_img.tobytes() != plain_img.tobytes()


def test_zip_is_refused_with_limit(dataset: Path, tmp_path: Path) -> None:
    config = _write_config(tmp_path, "limited", dataset)
    with pytest.raises(SystemExit):
        preprocess.main(["--config", str(config), "--limit", "2", "--zip"])


def test_limit_run_writes_its_own_folder(dataset: Path, tmp_path: Path) -> None:
    config = _write_config(tmp_path, "lim", dataset)
    assert preprocess.main(["--config", str(config), "--workers", "1", "--limit", "3"]) == 0
    folder = tmp_path / "lim" / "processed" / f"s{TARGET}_limit3"
    assert len(_read_manifest(folder)) == 3
    assert not (tmp_path / "lim" / "processed" / f"s{TARGET}").exists()


def test_rerun_after_completion_does_no_image_work(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run on a finished folder (for example to add --zip) must not reprocess images."""
    config = _write_config(tmp_path, "rerun", dataset)
    assert preprocess.main(["--config", str(config), "--workers", "1"]) == 0
    folder = tmp_path / "rerun" / "processed" / f"s{TARGET}"
    before = (folder / "manifest.csv").read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("an image was processed again")

    monkeypatch.setattr(preprocess, "_process_image", fail)
    assert preprocess.main(["--config", str(config), "--workers", "1", "--zip"]) == 0
    assert (folder / "manifest.csv").read_bytes() == before
    assert (tmp_path / "rerun" / "processed" / f"s{TARGET}.zip").is_file()
