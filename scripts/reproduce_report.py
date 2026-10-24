"""Recompute every number in the report from the committed files in results/, and check it.

Needs no images, no GPU and no model: it reruns the project's own evaluation, calibration and
referral scripts on the saved predictions in results/predictions/ and compares each result
with the value printed in the report (results/report_expected.json).

The saved test predictions are checked against the committed access log and test locks before
they are used (src/drgrade/provenance.py), so a replaced file is refused.

Explainability numbers need the images and the model, so they are read from the committed
summaries in results/explain/ (produced by scripts/explain.py) rather than recomputed.

Usage (from the repository root):
    python scripts/reproduce_report.py            # prints a table and exits 0 if every number matches
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from drgrade.provenance import ENV_ACCESS_LOG
from drgrade.utils import setup_logging

logger = logging.getLogger(__name__)

RESULTS = Path("results")
SPLIT = Path("splits/split_v1.csv")
LEAKAGE = Path("splits/test_with_train_copy.csv")
SELECTED = "e2_effb0_512_ce"
SECONDARY = "e3_effb0_512_wce"
BASELINE = "e1_resnet18_384_wce"
N_BOOT = 1000
SEED = 42


def _script(name: str):  # type: ignore[no-untyped-def]
    """Import scripts/<name>.py as a module so its main() can be called."""
    spec = importlib.util.spec_from_file_location(f"repro_{name}", Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(step: str, main: Callable[[list[str]], int], argv: list[str]) -> None:
    status = main(argv)
    if status != 0:
        raise RuntimeError(f"{step} failed with status {status}")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stage(out: Path) -> dict[str, Path]:
    """Lay out the committed predictions as run folders, with their sidecars and test locks."""
    dirs = {}
    for run in (SELECTED, SECONDARY):
        pred = out / "runs" / run / "predictions"
        pred.mkdir(parents=True, exist_ok=True)
        for name in ("val.csv", "val.csv.meta.json", "test.csv", "test.csv.meta.json"):
            shutil.copyfile(RESULTS / "predictions" / run / name, pred / name)
        shutil.copyfile(RESULTS / f"TEST_LOCK_{run}.json", out / "runs" / run / "TEST_LOCK.json")
        dirs[run] = out / "runs" / run
    return dirs


def recompute(out: Path) -> dict[str, float]:
    """Run evaluation, calibration and referral on the committed predictions; return named numbers."""
    os.environ[ENV_ACCESS_LOG] = str(RESULTS / "test_access_log.jsonl")
    dirs = stage(out)
    evaluate = _script("evaluate").main
    calibrate = _script("calibrate").main
    referral = _script("referral").main
    manifest = RESULTS / "manifest_meta.csv"
    common = ["--manifest", str(manifest), "--split", str(SPLIT), "--leakage-list", str(LEAKAGE),
              "--n-boot", str(N_BOOT), "--seed", str(SEED)]

    # Validation, from the predictions saved during training (Table 1).
    for run in (BASELINE, SELECTED, SECONDARY):
        _run(f"evaluate {run} val (training-time)", evaluate,
             ["--pred", str(RESULTS / "predictions" / run / "val_training.csv"), "--out", str(out / "val_training" / run), *common])
    # Test (Table 2) and the CPU validation files used for calibration and referral.
    for run in (SELECTED, SECONDARY):
        for subset in ("val", "test"):
            _run(f"evaluate {run} {subset}", evaluate,
                 ["--pred", str(dirs[run] / "predictions" / f"{subset}.csv"), "--out", str(dirs[run] / f"eval_{subset}"), *common])
        _run(f"calibrate {run}", calibrate,
             ["--run", str(dirs[run]), "--test", str(dirs[run] / "predictions/test.csv"),
              "--out", str(dirs[run] / "calibration"), "--split", str(SPLIT)])
    _run(f"referral {SELECTED}", referral,
         ["--run", str(dirs[SELECTED]), "--calibration", str(dirs[SELECTED] / "calibration/calibration.json"),
          "--test", str(dirs[SELECTED] / "predictions/test.csv"),
          "--match-sensitivity-of", str(dirs[SECONDARY] / "predictions/val.csv"),
          "--out", str(dirs[SELECTED] / "referral"), "--split", str(SPLIT)])
    return extract(out, dirs)


def extract(out: Path, dirs: dict[str, Path]) -> dict[str, float]:
    """Collect the reported numbers from the freshly written outputs (and the audit and explain files)."""
    n: dict[str, float] = {}
    for run, key in ((BASELINE, "e1"), (SELECTED, "e2"), (SECONDARY, "e3")):
        h = _json(out / "val_training" / run / "metrics.json")["headline"]
        n[f"val.{key}.accuracy"] = h["accuracy"]
        n[f"val.{key}.macro_f1"] = h["macro_f1"]
        n[f"val.{key}.qwk"] = h["qwk"]
        n[f"val.{key}.referable_sensitivity"] = h["referable"]["sensitivity"]
    for run, key in ((SELECTED, "e2"), (SECONDARY, "e3")):
        m = _json(dirs[run] / "eval_test/metrics.json")
        ci = _json(dirs[run] / "eval_test/bootstrap_ci.json")["headline"]
        h = m["headline"]
        for metric in ("accuracy", "macro_f1", "qwk"):
            n[f"test.{key}.{metric}"] = h[metric]
            n[f"test.{key}.{metric}.low"] = ci[metric]["low"]
            n[f"test.{key}.{metric}.high"] = ci[metric]["high"]
        n[f"test.{key}.referable_sensitivity"] = h["referable"]["sensitivity"]
        n[f"test.{key}.referable_sensitivity.low"] = ci["referable_sensitivity"]["low"]
        n[f"test.{key}.referable_sensitivity.high"] = ci["referable_sensitivity"]["high"]
        n[f"test.{key}.referable_specificity"] = h["referable"]["specificity"]
        n[f"test.{key}.referable_auroc"] = h["referable"]["auroc"]
        n[f"test.{key}.qwk_without_copies"] = m["excluding_train_copies"]["metrics"]["qwk"]
        n[f"test.{key}.val_to_test_qwk_drop"] = n[f"val.{key}.qwk"] - h["qwk"]
        if key == "e2":
            m0 = m["m0_majority_baseline"]
            n["test.m0.accuracy"] = m0["accuracy"]
            n["test.m0.macro_f1"] = m0["macro_f1"]
            n["test.m0.qwk"] = m0["qwk"]
            n["test.e2.copies_qwk_change"] = h["qwk"] - n["test.e2.qwk_without_copies"]
            cm = np.array(h["confusion_counts"])
            true_ref = cm[2:].sum()
            n["test.e2.off_by_two_or_more"] = h["errors"]["off_by_two_or_more_rate"]
            n["test.e2.referable_misses"] = float(cm[2:, :2].sum())
            n["test.e2.referable_images"] = float(true_ref)
            n["test.e2.referable_miss_rate"] = cm[2:, :2].sum() / true_ref
            n["test.e2.referable_called_grade1"] = float(cm[2:, 1].sum())
            n["test.e2.pdr_graded_2_or_below"] = float(cm[4, :3].sum())
            n["test.e2.grade1_f1"] = h["per_class"][1]["f1"]
            n["test.e2.grade3_recall"] = h["per_class"][3]["recall"]
            groups = m["by_border_group"]
            black = np.array(groups["black"]["metrics"]["confusion_counts"])
            grey = np.array(groups["non_black"]["metrics"]["confusion_counts"])
            n["test.e2.grey_n"] = float(groups["non_black"]["n"])
            n["test.e2.grey_accuracy"] = groups["non_black"]["metrics"]["accuracy"]
            recall = np.diag(black) / black.sum(axis=1)
            n["test.e2.grey_accuracy_from_grade_mix"] = float((grey.sum(axis=1) * recall).sum() / grey.sum())
            n["test.e2.grey_referable_share"] = grey[2:].sum() / grey.sum()
            n["test.referable_share"] = true_ref / cm.sum()
        else:
            n["test.e3.grade3_recall"] = h["per_class"][3]["recall"]
    cal = _json(dirs[SELECTED] / "calibration/calibration.json")
    n["cal.e2.temperature"] = cal["temperature"]
    n["cal.e2.test_ece_before"] = cal["test"]["before"]["ece"]
    n["cal.e2.test_ece_after"] = cal["test"]["after"]["ece"]
    ref = _json(dirs[SELECTED] / "referral/referral.json")
    p, s = ref["primary"]["test"], ref["secondary"]["test"]
    n["ref.primary.misses_caught"] = float(p["referable_misses_referred"])
    n["ref.primary.share_caught"] = p["share_of_misses_referred"]
    n["ref.secondary.referred"] = s["referral_fraction"]
    n["ref.secondary.retained_miss_rate"] = s["retained"]["referable_miss_rate"]
    thr = _json(dirs[SELECTED] / "referral/referable_threshold.json")["test"]
    n["ref.matched.sensitivity"] = thr["sensitivity"]
    n["ref.matched.specificity"] = thr["specificity"]

    audit = _json(RESULTS / "audit/summary.json")
    totals = audit["counts"]["per_split_total"]
    train = audit["counts"]["train_grade_counts"]
    n["audit.images"] = float(totals["train"] + totals["test"])
    n["audit.train"] = float(totals["train"])
    n["audit.test"] = float(totals["test"])
    grade0 = audit["counts"]["per_split_grade"]["train"]["0"] + audit["counts"]["per_split_grade"]["test"]["0"]
    n["audit.grade0_share"] = grade0 / n["audit.images"]
    n["audit.imbalance_ratio"] = max(train.values()) / min(train.values())
    n["audit.resolutions"] = float(audit["image_properties"]["distinct_resolutions"])
    n["audit.repeat_groups"] = float(audit["duplicates"]["near"]["n_groups"])
    n["audit.grade1_beyond_ma_test"] = float(audit["labels"]["flag_counts"]["test"]["F2_grade1_beyond_ma"])
    n["audit.grade1_test"] = float(audit["counts"]["per_split_grade"]["test"]["1"])
    n["split.test_with_train_copy"] = float(len(pd.read_csv(LEAKAGE)))
    split = pd.read_csv(SPLIT)
    n["split.val"] = float((split["subset"] == "val").sum())

    rand = pd.read_csv(RESULTS / "explain/randomisation_summary.csv")
    stage_rows = rand.loc[rand["stage_index"] >= 0].sort_values("stage_index")
    n["xai.randomisation.classifier"] = float(stage_rows.iloc[0]["mean_spearman"])
    n["xai.randomisation.all_layers"] = float(stage_rows.iloc[-1]["mean_spearman"])
    n["xai.randomisation.unrelated"] = float(rand.loc[rand["stage_index"] == -1, "mean_spearman"].iloc[0])
    dele = pd.read_csv(RESULTS / "explain/deletion_summary.csv")
    at5 = dele.loc[dele["fraction"] == 0.05].set_index("condition")["mean_drop"]
    for cond in ("salient", "rotated", "random"):
        n[f"xai.deletion5.{cond}"] = float(at5[cond])
    by_class = pd.read_csv(RESULTS / "explain_deletion_by_class.csv")
    at5c = by_class.loc[by_class["fraction"] == 0.05].pivot(index="pred_grade", columns="condition", values="mean_drop")
    gaps = (at5c["salient"] - at5c["rotated"]).loc[[1, 2, 3]]
    n["xai.deletion5.gap_grades1to3.min"] = float(gaps.min())
    n["xai.deletion5.gap_grades1to3.max"] = float(gaps.max())
    n["xai.deletion5.grade0.salient"] = float(at5c.loc[0, "salient"])
    n["xai.deletion5.grade0.rotated"] = float(at5c.loc[0, "rotated"])
    short = _json(RESULTS / "explain/shortcut_summary.json")
    n["xai.disc_ratio"] = short["mean_disc_mass"] / short["mean_disc_area"]
    n["xai.edge_mass_minus_area"] = short["mean_edge_mass"] - short["mean_edge_area"]
    follow = _json(RESULTS / "explain_followup.json")
    n["xai.lesion_tests"] = float(follow["lesions_by_grade"]["n_tests"])
    n["xai.lesion_tests_significant_bh"] = float(follow["lesions_by_grade"]["n_bh_below_0_05"])
    return n


def compare(numbers: dict[str, float], expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in expected:
        got = numbers.get(item["name"])
        if got is None:
            rows.append({**item, "recomputed": None, "match": False})
            continue
        scaled = got * item.get("scale", 1)
        shown = round(scaled + 0.0, item["decimals"])
        ok = abs(shown - item["value"]) < 10 ** (-item["decimals"]) / 2 + 1e-12
        rows.append({**item, "recomputed": scaled, "match": bool(ok)})
    return rows


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/reproduce"), help="scratch folder (git-ignored)")
    parser.add_argument("--expected", type=Path, default=RESULTS / "report_expected.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        shutil.rmtree(args.out)
    logging.getLogger().setLevel(logging.WARNING)  # keep the console to the summary table
    numbers = recompute(args.out)
    rows = compare(numbers, _json(args.expected))
    (args.out / "report_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    width = max(len(r["label"]) for r in rows)
    print(f"{'Report number':{width}}  {'report':>9}  {'recomputed':>11}  match")
    for r in rows:
        got = "missing" if r["recomputed"] is None else f"{r['recomputed']:.{r['decimals'] + 2}f}"
        print(f"{r['label']:{width}}  {r['value']:>9}  {got:>11}  {'yes' if r['match'] else 'NO'}")
    failed = [r for r in rows if not r["match"]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} report numbers reproduced.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
