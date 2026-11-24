#!/usr/bin/env python
"""
reproduce_results.py -- rebuild the verified result tables from run records.

Reads the run_record.json written by run_pipeline.py / run_transfer.py and
writes portable summary tables (no paths) with explicit provenance columns:
split_seed, head_seeds, selected_layer_frac, objective, dataset, result_source.

  python scripts/reproduce_results.py --runs_root runs --transfer_root runs_transfer --out_dir results
  python scripts/reproduce_results.py ... --check      # compare with shipped results/ instead of writing

Objective naming: a run whose directory ends in "__brier_only" was trained with
ec_mode=graded and beta_ec=0 (calibration loss only); it is reported as
objective=brier_only. Directory names, not ec_mode alone, identify the objective.

ec_eligible_frac (from run_record.supervision_stats) is a TRAINING-SUPERVISION
statistic: the fraction of training pairs with a non-zero evidence-contrastive
target. It is not an evaluation metric. No "ec_fired" metric is computed.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

METRICS = ["gap", "gap_cohen_d", "acc_na", "acc_nap", "auroc_na", "auroc_nap",
           "ece_na", "ece_nap", "brier_na", "ucer_na", "ucer_nap"]
OBJECTIVES = ("brier_only", "fixed", "graded")
TOL = 1e-9


def _mean(agg, k):
    v = agg.get(k)
    if isinstance(v, dict):
        return v.get("mean", np.nan)
    return np.nan if v is None else v


def main_runs(root, dataset):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "outputs", "run_record.json"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        model, obj = run.split("__", 1)
        if obj not in OBJECTIVES:
            continue                      # e.g. ConfTuner runs are not part of this package
        r = json.load(open(f))
        row = dict(model=model, objective=obj, dataset=dataset,
                   split_seed=r["pipeline_cfg"]["seed"],
                   head_seeds=";".join(str(s) for s in r["seeds"]),
                   selected_layer_frac=r["selected_layer_frac"],
                   selected_layer_index=r["selected_layer_index"], num_layers=r["num_layers"],
                   beta_ec=r["beta_ec"], ec_mode=r["ec_mode"],
                   n_train_split=r["n_train_split"], n_val_split=r["n_val_split"],
                   n_test_pairs=r["n_test_pairs"],
                   ec_eligible_frac_train=r.get("supervision_stats", {}).get("ec_eligible_frac", np.nan),
                   result_source="main runs (split seed 42; head seeds 42/43/44)")
        row.update({m: _mean(r["aggregate"], m) for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def transfer_runs(root, main_df):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "run_record.json"))):
        r = json.load(open(f))
        run = os.path.basename(os.path.dirname(f))
        model, obj = run.split("__")[0], run.split("__")[1]
        src = main_df[(main_df.model == model) & (main_df.objective == obj)]
        row = dict(model=model, objective=obj, dataset=r["target_dataset"],
                   source_dataset=r["source_dataset"],
                   split_seed=int(src.split_seed.iloc[0]) if len(src) else np.nan,
                   head_seeds=";".join(str(s) for s in r["seeds"]),
                   selected_layer_frac=r["selected_layer_frac"],
                   selected_layer_index=r["selected_layer_index"], num_layers=r["num_layers"],
                   result_source="MediQ transfer: main-run heads applied unchanged (no fitting on target)")
        row.update({m: _mean(r["aggregate"], m) for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df):
    g = df.groupby("objective")[METRICS].mean().reindex(list(OBJECTIVES)).reset_index()
    g.insert(1, "n_backbones", df.groupby("objective").size().reindex(list(OBJECTIVES)).values)
    return g


def _compare(new, path, keys):
    old = pd.read_csv(path)
    m = new.merge(old, on=keys, suffixes=("", "_ref"), how="outer", indicator=True)
    bad = int((m["_merge"] != "both").sum())
    worst = 0.0
    for c in METRICS:
        if c in new.columns and c + "_ref" in m.columns and m[c].notna().any():
            worst = max(worst, float(np.nanmax(np.abs(m[c] - m[c + "_ref"]))))
    ok = bad == 0 and worst <= TOL
    print(f"[check] {os.path.basename(path):28s} rows={len(new):3d} unmatched={bad} max|diff|={worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--transfer_root", default=None)
    ap.add_argument("--dataset", default="medabstain-medqa-v1")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    main_df = main_runs(a.runs_root, a.dataset)
    if main_df.empty:
        sys.exit(f"no run records under {a.runs_root}/*/outputs/")
    outs = {"main/run_summary.csv": (main_df, ["model", "objective"]),
            "main/aggregate.csv": (aggregate(main_df), ["objective"])}
    if a.transfer_root:
        tr = transfer_runs(a.transfer_root, main_df)
        outs["transfer/run_summary.csv"] = (tr, ["model", "objective"])
        outs["transfer/aggregate.csv"] = (aggregate(tr), ["objective"])

    if a.check:
        ok = all(_compare(df, os.path.join(a.out_dir, p), k) for p, (df, k) in outs.items())
        sys.exit(0 if ok else 1)
    for p, (df, _) in outs.items():
        path = os.path.join(a.out_dir, p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print("wrote", path, len(df), "rows")
    pd.set_option("display.width", 200)
    print(aggregate(main_df)[["objective", "gap", "gap_cohen_d", "auroc_na", "ece_na", "ucer_na"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
