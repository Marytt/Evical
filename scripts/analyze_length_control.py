#!/usr/bin/env python
"""
control_gap_from_infer.py — final analysis of the matched-deletion control.
Reads out/control_preds_<MODEL>.csv (variant in {NA,selected,matched}) and computes,
per model and pooled over the shared item set:
  gap_selected = conf(NA) - conf(selected-NAP)
  gap_matched  = conf(NA) - conf(matched-control-NAP)
plus mean frozen-model belief change (dq) per condition and a paired Wilcoxon test.
"""
import argparse, glob, re, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/control_preds_*.csv")
    ap.add_argument("--out", default="control_gap_results.csv")
    a = ap.parse_args()
    rows = []
    ps, pm = [], []
    for fp in sorted(glob.glob(a.glob)):
        model = re.search(r"control_preds_(.+)\.csv", os.path.basename(fp)).group(1)
        d = pd.read_csv(fp, keep_default_na=False)
        d["head_conf"] = pd.to_numeric(d["head_conf"]); d["gold_prob"] = pd.to_numeric(d["gold_prob"])
        piv = d.pivot_table(index="id", columns="variant",
                            values=["head_conf","gold_prob"], aggfunc="mean")
        piv.columns = [f"{a1}_{b1}" for a1,b1 in piv.columns]
        piv = piv.dropna(subset=["head_conf_full","head_conf_selected","head_conf_matched"])
        gsel = piv["head_conf_full"] - piv["head_conf_selected"]
        gmat = piv["head_conf_full"] - piv["head_conf_matched"]
        dqs  = piv["gold_prob_full"] - piv["gold_prob_selected"]
        dqm  = piv["gold_prob_full"] - piv["gold_prob_matched"]
        try: p = wilcoxon(gsel, gmat).pvalue
        except ValueError: p = np.nan
        rows.append(dict(model=model, n=len(piv),
            gap_selected=gsel.mean(), gap_matched=gmat.mean(),
            ratio=gsel.mean()/gmat.mean() if gmat.mean() else np.nan,
            dq_selected=dqs.mean(), dq_matched=dqm.mean(),
            wilcoxon_p=p, selected_gt_matched=bool(gsel.mean() > gmat.mean())))
        ps.append(gsel.values); pm.append(gmat.values)
    assert rows, f"no files matched {a.glob}"
    R = pd.DataFrame(rows)
    PS, PM = np.concatenate(ps), np.concatenate(pm)
    R = pd.concat([R, pd.DataFrame([dict(model="POOLED", n=len(PS),
        gap_selected=PS.mean(), gap_matched=PM.mean(),
        ratio=PS.mean()/PM.mean() if PM.mean() else np.nan,
        dq_selected=np.nan, dq_matched=np.nan,
        wilcoxon_p=wilcoxon(PS,PM).pvalue, selected_gt_matched=bool(PS.mean()>PM.mean()))])],
        ignore_index=True)
    pd.set_option("display.width",200,"display.float_format",lambda x:f"{x:.4f}")
    print(R.to_string(index=False))
    nm = int(R[R.model!="POOLED"].selected_gt_matched.sum()); tot = len(R)-1
    print(f"\nselected>matched in {nm}/{tot} models; pooled ratio "
          f"{R[R.model=='POOLED'].ratio.values[0]:.2f}x, Wilcoxon p="
          f"{R[R.model=='POOLED'].wilcoxon_p.values[0]:.2e}")
    R.to_csv(a.out, index=False); print("wrote", os.path.abspath(a.out))

if __name__ == "__main__":
    main()
