"""Smoke tests for scripts/evaluate.py and the comparison mode on synthetic predictions (no model, no real data)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from _prediction_data import ids_of_subset, make_predictions, register_test
from _training_data import write_tiny_dataset
from drgrade.reporting import select_run

REPO = Path(__file__).resolve().parents[1]
EVALUATE = REPO / "scripts" / "evaluate.py"
EXPECTED_FILES = ("metrics.json", "per_class.csv", "confusion_counts.csv", "confusion_norm.png", "bootstrap_ci.json")


def _module(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    dataset = write_tiny_dataset(tmp_path / "data", target_size=64, seed=2)
    split = pd.read_csv(dataset.split, dtype={"image_id": str, "subset": str})
    manifest = pd.read_csv(dataset.manifest, dtype={"image_id": str})
    val_ids = ids_of_subset(split, "val")
    test_ids = ids_of_subset(split, "test")
    truth = dict(zip(manifest["image_id"].astype(str), manifest["grade"].astype(int)))
    val_pred = [truth[i] if k % 3 else (truth[i] + 1) % 5 for k, i in enumerate(val_ids)]
    val_conf = [0.5 + 0.04 * k for k in range(len(val_ids))]
    test_pred = [truth[i] for i in test_ids]
    test_pred[0] = 0
    test_conf = [0.6] * len(test_ids)
    val_path = tmp_path / "runs" / "a" / "predictions" / "val.csv"
    val_path.parent.mkdir(parents=True)
    make_predictions(val_ids, [truth[i] for i in val_ids], val_pred, val_conf).to_csv(val_path, index=False)
    test_path = tmp_path / "runs" / "a" / "predictions" / "test.csv"
    make_predictions(test_ids, [truth[i] for i in test_ids], test_pred, test_conf).to_csv(test_path, index=False)
    register_test(test_path)
    leak = tmp_path / "leak.csv"
    pd.DataFrame(
        {"image_id": [test_ids[0]], "linked_train_ids": [val_ids[0]], "max_similarity": [0.99], "exact": [True]}
    ).to_csv(leak, index=False)
    return {"dataset": dataset, "val": val_path, "test": test_path, "leak": leak, "val_ids": val_ids, "test_ids": test_ids, "split": split}


def _args(world, *extra: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        "--manifest", str(world["dataset"].manifest),
        "--split", str(world["dataset"].split),
        "--n-boot", "20",
        "--train-majority-grade", "0",
        *extra,
    ]


def _run_evaluate(world, *extra: str) -> int:  # type: ignore[no-untyped-def]
    module = _module("evaluate_script", EVALUATE)
    return module.main(_args(world, *extra))


def test_val_evaluation_writes_every_output(world) -> None:  # type: ignore[no-untyped-def]
    """The validation evaluation writes the five files and metrics.json with the M0 block and no NaN."""
    assert _run_evaluate(world, "--pred", str(world["val"])) == 0
    out = world["val"].parent / "eval_val"
    for name in EXPECTED_FILES:
        assert (out / name).is_file(), name
    text = (out / "metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c} in metrics.json"))
    assert metrics["subset"] == "val" and metrics["n"] == len(world["val_ids"])
    assert metrics["excluding_train_copies"]["n_excluded"] == 0
    assert metrics["m0_majority_baseline"]["train_majority_grade"] == 0
    assert metrics["m0_majority_baseline"]["referable"]["auroc"] is None
    groups = metrics["by_border_group"]
    assert groups["black"]["n"] + groups["non_black"]["n"] == metrics["n"]
    confusion = pd.read_csv(out / "confusion_counts.csv", index_col=0)
    assert int(confusion.to_numpy().sum()) == metrics["n"]
    assert (out / "confusion_norm.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_test_evaluation_excludes_the_leakage_list(world) -> None:  # type: ignore[no-untyped-def]
    """On test, the listed image is excluded in the sensitivity analysis and counted in metrics.json."""
    assert _run_evaluate(world, "--pred", str(world["test"]), "--leakage-list", str(world["leak"])) == 0
    metrics = json.loads((world["test"].parent / "eval_test" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["subset"] == "test"
    assert metrics["excluding_train_copies"]["n_excluded"] == 1
    assert metrics["excluding_train_copies"]["metrics"]["n"] == metrics["n"] - 1


def test_test_evaluation_needs_the_leakage_list(world) -> None:  # type: ignore[no-untyped-def]
    """Without the leakage list, test evaluation is refused with status 2."""
    assert _run_evaluate(world, "--pred", str(world["test"]), "--leakage-list", str(world["leak"].with_name("missing.csv"))) == 2


def test_compare_mode_selects_by_qwk_then_macro_f1(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Comparison writes both formats and states the rule's choice; the selection rule is a pure function."""
    run_b = tmp_path / "runs" / "b"
    run_b.mkdir(parents=True)
    val_b = run_b / "val_predictions.csv"
    frame = pd.read_csv(world["val"], dtype={"image_id": str})
    frame.to_csv(val_b, index=False)
    (run_b / "summary.json").write_text(json.dumps({"best_val": {"qwk": 0.123}}), encoding="utf-8")
    compare_dir = tmp_path / "compare"
    module = _module("evaluate_script", EVALUATE)
    status = module.main(
        ["--compare", "--pred", str(world["val"]), str(val_b), "--split", str(world["dataset"].split), "--compare-dir", str(compare_dir)]
    )
    assert status == 0
    table = pd.read_csv(compare_dir / "val_comparison.csv")
    assert len(table) == 2
    assert set(table["run"]) == {"a", "b"}
    markdown = (compare_dir / "val_comparison.md").read_text(encoding="utf-8")
    assert "Pre-declared rule" in markdown and "selects" in markdown

    rows = [
        {"run": "x", "qwk": 0.80, "macro_f1": 0.5},
        {"run": "y", "qwk": 0.80, "macro_f1": 0.6},
        {"run": "z", "qwk": 0.79, "macro_f1": 0.9},
    ]
    assert select_run(rows)["run"] == "y"
    assert select_run([{"run": "p", "qwk": None, "macro_f1": 0.1}, {"run": "q", "qwk": 0.1, "macro_f1": 0.0}])["run"] == "q"


def test_compare_rejects_test_predictions(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The comparison is validation only."""
    module = _module("evaluate_script", EVALUATE)
    status = module.main(["--compare", "--pred", str(world["test"]), "--split", str(world["dataset"].split), "--compare-dir", str(tmp_path / "c")])
    assert status == 2


def test_comparison_without_selection_has_no_selection_line(tmp_path: Path) -> None:
    from drgrade.reporting import comparison_markdown

    rows = [
        {"run": "a", "n": 10, "accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.85, "qwk": 0.91,
         "referable_sensitivity": 0.8, "referable_miss_rate": 0.2},
    ]
    text = comparison_markdown(rows, None, "Stability and ablation runs; not candidates for selection.")
    assert "selects" not in text
    assert "not candidates" in text


def test_evaluation_refuses_unregistered_or_incomplete_files(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A test file that scripts/predict.py did not record, or a truncated file, is refused (DECISIONS.md D38)."""
    frame = pd.read_csv(world["test"], dtype={"image_id": str})
    stray = tmp_path / "runs" / "z" / "predictions" / "test.csv"
    stray.parent.mkdir(parents=True)
    frame.assign(pred_grade=0).to_csv(stray, index=False)
    assert _run_evaluate(world, "--pred", str(stray), "--leakage-list", str(world["leak"])) == 2
    short = tmp_path / "short_val.csv"
    pd.read_csv(world["val"], dtype={"image_id": str}).iloc[:-1].to_csv(short, index=False)
    assert _run_evaluate(world, "--pred", str(short)) == 2
