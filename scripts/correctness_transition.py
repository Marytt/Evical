#!/usr/bin/env python
"""
correctness_transition.py -- confidence gap on correctness-degrading pairs.

For each backbone and objective, reads the main-run predictions
(runs/<MODEL>__<OBJECTIVE>/outputs/predictions_all_seeds.csv), averages head
confidence over the head seeds per item, and labels each NA/NAP pair:

  correct-to-wrong (C->W) : frozen prediction correct on NA, incorrect on NAP
  remaining               : all other pairs

Correctness of the frozen backbone does not depend on the head seed; a pair is
"correct" on a variant when the seed-averaged correctness is > 0.5.

Outputs (CSV):
  per-backbone : n_c2w, n_remaining, gap_c2w, gap_remaining,
                 one-sided Mann-Whitney p (C->W > remaining)
  pooled       : item-pooled gaps over all backbones
  test         : two-sided Wilcoxon signed-rank test over the seven
                 backbone-level differences (gap_c2w - gap_remaining)

The graded target is defined from the change in gold-answer probability and
does not use the transition label; this is a behavioural analysis, not
clinical validation.

Usage:
  python scripts/correctness_transition.py --runs_root runs --out results/correctness_transition.csv
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

OBJECTIVES = ("brier_only", "fixed", "graded")


def load_pairs(path):
    d = pd.read_csv(path, keep_default_na=False)
    d["variant"] = d["variant"].replace({"": "NA"})
    d["correct"] = d["correct"].astype(str).str.lower().isin(["true", "1"])
    d["head_conf"] = pd.to_numeric(d["head_conf"])
    ag = (d.groupby(["sample_id", "variant"])
            .agg(conf=("head_conf", "mean"), cor=("correct", "mean"),
                 n_seeds=("seed", "nunique")).reset_index())
    p = ag.pivot(index="sample_id", columns="variant", values=["conf", "cor", "n_seeds"])
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p["gap"] = p["conf_NA"] - p["conf_NAP"]
    p["c2w"] = (p["cor_NA"] > 0.5) & (p["cor_NAP"] < 0.5)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out", default="correctness_transition.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.runs_root, "*", "outputs", "predictions_all_seeds.csv")))
    if not files:
        raise SystemExit(f"no predictions_all_seeds.csv under {a.runs_root}/*/outputs/")
    rows, pooled = [], {o: [] for o in OBJECTIVES}
    for f in files:
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        model, obj = run.split("__", 1)
        if obj not in OBJECTIVES:
            continue
        p = load_pairs(f)
        pooled[obj].append(p)
        c, r = p.loc[p.c2w, "gap"], p.loc[~p.c2w, "gap"]
        rows.append(dict(level="backbone", objective=obj, model=model,
                         n_seeds=int(p["n_seeds_NA"].max()),
                         n_c2w=int(p.c2w.sum()), n_remaining=int((~p.c2w).sum()),
                         gap_c2w=c.mean(), gap_remaining=r.mean(),
                         p_mannwhitney_greater=mannwhitneyu(c, r, alternative="greater").pvalue))
    R = pd.DataFrame(rows)
    for obj in OBJECTIVES:
        if not pooled[obj]:
            continue
        P = pd.concat(pooled[obj])
        B = R[R.objective == obj]
        diff = (B.gap_c2w - B.gap_remaining).values
        rows.append(dict(level="pooled", objective=obj, model="ALL",
                         n_c2w=int(P.c2w.sum()), n_remaining=int((~P.c2w).sum()),
                         gap_c2w=P.loc[P.c2w, "gap"].mean(), gap_remaining=P.loc[~P.c2w, "gap"].mean(),
                         n_backbones=len(B), n_backbones_c2w_gt_remaining=int((diff > 0).sum()),
                         p_wilcoxon_two_sided_backbone_means=wilcoxon(diff).pvalue if len(B) > 1 else np.nan))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(out[out.level == "pooled"].round(4).to_string(index=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
