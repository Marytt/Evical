#!/usr/bin/env python
"""
run_probability_baselines.py -- training-free confidence baselines.

Confidence scores computed from the frozen backbone's option-letter distribution
at the same read position as the learned head (requires predictions written
with the `opt_logits` column, i.e. features extracted with this repository's
extractor):

  max_prob    c = max_i q_f(o_i | x)
  margin      c = q_(1) - q_(2)          (top-1 minus top-2 probability)
  negentropy  c = 1 - H(q) / log K       (normalized negative entropy)

Each score is evaluated with the same metric code as the learned head
(src.evaluation.metrics.compute_all_metrics). No parameters are fitted.

Usage:
  python scripts/run_probability_baselines.py \
      --pred_glob 'runs/**/predictions_all_seeds.csv' \
      --config configs/pipeline_config.yaml \
      --out results/probability_baselines.csv
"""
import argparse, glob, json, yaml
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import compute_all_metrics


def _logits_matrix(df):
    L = [np.asarray(json.loads(s), dtype=float) for s in df["opt_logits"]]
    K = max(len(v) for v in L)
    M = np.full((len(L), K), -1e9)
    for i, v in enumerate(L):
        M[i, :len(v)] = v
    return M                                   # (n, K), padded


def _softmax(Z, T=1.0):
    Z = Z / T
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z); E[Z <= -1e8] = 0.0
    return E / E.sum(1, keepdims=True)


def _conf(name, Z, T=1.0):
    P = _softmax(Z, T)
    Ps = np.sort(P, 1)[:, ::-1]
    if name == "maxprob":   return Ps[:, 0]
    if name == "margin":    return np.clip(Ps[:, 0] - Ps[:, 1], 0, 1)
    if name == "negentropy":
        K = (P > 0).sum(1)
        H = -(np.where(P > 0, P * np.log(P + 1e-12), 0.0)).sum(1)
        return 1.0 - H / np.log(np.maximum(K, 2))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_glob", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    rows = []
    for f in sorted(glob.glob(a.pred_glob, recursive=True)):
        d = pd.read_csv(f, keep_default_na=False)
        if "opt_logits" not in d.columns:
            print(f"[skip] {f}: no opt_logits column (re-extract with the patch)")
            continue
        d["variant"] = d["variant"].replace({"": "NA"})
        d["correct"] = d["correct"].astype(str).str.lower().isin(["true", "1"])
        na = d[d.variant == "NA"].reset_index(drop=True)
        nap = d[d.variant == "NAP"].reset_index(drop=True)
        Zna, Znap = _logits_matrix(na), _logits_matrix(nap)

        model = f.split("/")[-3] if "/" in f else f
        for base in ["maxprob", "margin", "negentropy"]:
            Tuse = 1.0
            na2, nap2 = na.copy(), nap.copy()
            na2["head_conf"] = _conf(base, Zna, Tuse)
            nap2["head_conf"] = _conf(base, Znap, Tuse)
            m = compute_all_metrics(na2, nap2, cfg)
            rows.append({
                "model": model, "baseline": base,
                "acc_na": m["NA"]["accuracy"],
                "auroc_na": m["NA"]["auroc"], "ece_na": m["NA"]["ece"],
                "ucer_na": m["NA"]["ucer"],
                "gap": m["gap"]["gap"], "gap_p": m["gap"].get("p", np.nan),
                "gap_ci_lo": m["gap"]["ci_lo"], "gap_ci_hi": m["gap"]["ci_hi"],
                "cohen_d": m["gap"]["cohen_d"],
            })
            print(f"{model:28s} {base:10s} "
                  f"AUROC={m['NA']['auroc']:.3f} ECE={m['NA']['ece']:.3f} "
                  f"UCER={m['NA']['ucer']:.3f} gap={m['gap']['gap']:+.3f}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
