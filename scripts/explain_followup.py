"""Follow-up analysis of the explainability outputs, after an independent review (DECISIONS.md D41).

Reads only existing outputs of scripts/explain.py (no model passes) and writes:
- results/explain_lesions_by_grade.csv: heatmap measures with and without each lesion,
  within each true grade, Mann-Whitney U, with Benjamini-Hochberg and Bonferroni
  adjustment across every test in the table. `peak` is left out: it is the raw map
  maximum, whose scale follows the predicted-class logit, so it measures confidence,
  not the shape of the map.
- results/explain_deletion_by_class.csv: the deletion test by predicted class, and the
  share of images where the salient drop exceeds the rotated and random drops.
- results/explain_followup.json: summary numbers quoted in DECISIONS.md D41.

Usage:
    python scripts/explain_followup.py --run runs/e2_effb0_512_ce
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

LESIONS = ("les_ma", "les_he", "les_irh", "les_vb_irma", "les_nv", "les_vh", "les_rd")
MEASURES = ("spread_50", "edge_mass", "disc_mass")
MIN_GROUP = 5
OUT_DIR = Path("results")


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (monotone, capped at 1); NaN stays NaN."""
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    values = p[ok]
    m = values.size
    if m == 0:
        return out
    order = np.argsort(values)
    ranked = values[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    out[ok] = adjusted
    return out


def lesions_by_grade(stats: pd.DataFrame) -> pd.DataFrame:
    """Lesion present against absent, within each true grade, for each measure."""
    rows = []
    for grade, block in stats.groupby("true_grade"):
        for lesion in LESIONS:
            present = block[lesion] == 1
            for measure in MEASURES:
                a = block.loc[present, measure].dropna().to_numpy(float)
                b = block.loc[~present, measure].dropna().to_numpy(float)
                enough = a.size >= MIN_GROUP and b.size >= MIN_GROUP
                rows.append(
                    {
                        "true_grade": int(grade),
                        "lesion": lesion,
                        "measure": measure,
                        "n_with": int(a.size),
                        "n_without": int(b.size),
                        "median_with": float(np.median(a)) if enough else np.nan,
                        "median_without": float(np.median(b)) if enough else np.nan,
                        "p": float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if enough else np.nan,
                    }
                )
    table = pd.DataFrame(rows)
    p = table["p"].to_numpy(float)
    n_tests = int((~np.isnan(p)).sum())
    table["p_bh"] = bh_adjust(p)
    table["p_bonferroni"] = np.minimum(p * n_tests, 1.0)
    return table


def deletion_by_class(deletion: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean drop by predicted class, and per-fraction shares of images where salient beats the baselines."""
    frame = deletion.merge(stats[["image_id", "pred_grade"]], on="image_id", how="left", validate="many_to_one")
    if frame["pred_grade"].isna().any():
        raise ValueError("some deletion images are missing from heatmap_stats.csv")
    by_class = (
        frame.loc[frame["fraction"] > 0]
        .groupby(["pred_grade", "condition", "fraction"])["drop"]
        .agg(n="size", mean_drop="mean")
        .reset_index()
    )
    wide = frame.pivot_table(index=["image_id", "fraction"], columns="condition", values="drop").reset_index()
    wide = wide.loc[wide["fraction"] > 0]
    shares = (
        wide.assign(
            salient_gt_rotated=wide["salient"] > wide["rotated"],
            salient_gt_random=wide["salient"] > wide["random"],
        )
        .groupby("fraction")[["salient_gt_rotated", "salient_gt_random"]]
        .mean()
        .reset_index()
    )
    return by_class, shares


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    explain = args.run / "explain"
    stats = pd.read_csv(explain / "stats" / "heatmap_stats.csv", dtype={"image_id": str})
    deletion = pd.read_csv(explain / "deletion" / "deletion.csv", dtype={"image_id": str})
    rand = pd.read_csv(explain / "randomisation" / "randomisation.csv", dtype={"image_id": str})

    lesions = lesions_by_grade(stats)
    by_class, shares = deletion_by_class(deletion, stats)
    OUT_DIR.mkdir(exist_ok=True)
    lesions.to_csv(OUT_DIR / "explain_lesions_by_grade.csv", index=False)
    by_class.to_csv(OUT_DIR / "explain_deletion_by_class.csv", index=False)

    sig = lesions.loc[lesions["p_bh"] < 0.05].sort_values("p_bh")
    stage0 = rand.loc[rand["stage_index"] == 0, "spearman"]
    summary = {
        "lesions_by_grade": {
            "n_tests": int(lesions["p"].notna().sum()),
            "n_bh_below_0_05": int(len(sig)),
            "n_bonferroni_below_0_05": int((lesions["p_bonferroni"] < 0.05).sum()),
            "bh_below_0_05": sig[["true_grade", "lesion", "measure", "n_with", "n_without", "median_with", "median_without", "p", "p_bh"]].to_dict("records"),
        },
        "deletion": {
            "n_images_by_predicted_class": stats.set_index("image_id").loc[deletion["image_id"].unique(), "pred_grade"].value_counts().sort_index().to_dict(),
            "share_salient_gt_rotated_by_fraction": dict(zip(shares["fraction"], shares["salient_gt_rotated"])),
            "share_salient_gt_random_by_fraction": dict(zip(shares["fraction"], shares["salient_gt_random"])),
        },
        "randomisation_classifier_stage": {
            "n": int(stage0.size),
            "share_spearman_negative": float((stage0 < 0).mean()),
            "mean_spearman": float(stage0.mean()),
        },
    }
    (OUT_DIR / "explain_followup.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print(by_class.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
