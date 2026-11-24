"""
scripts/show_results.py — read every finished run and render the paper tables.

Works two ways:

  Terminal:
      python scripts/show_results.py --out_root runs

  Jupyter (renders real DataFrames):
      import sys; sys.path.insert(0, ".")
      from scripts.show_results import load_runs, main_table, detail, health_check
      runs = load_runs("runs")
      main_table(runs)
"""

import os, json, argparse
import numpy as np
import pandas as pd


# ─── loading ─────────────────────────────────────────────────────────────────

def load_runs(out_root="runs"):
    """Return {run_name: run_record dict} for every completed run."""
    runs = {}
    if not os.path.isdir(out_root):
        print(f"No such directory: {out_root}")
        return runs
    for name in sorted(os.listdir(out_root)):
        rec = os.path.join(out_root, name, "outputs", "run_record.json")
        if os.path.exists(rec):
            runs[name] = json.load(open(rec))
    print(f"Loaded {len(runs)} completed run(s) from {out_root}")
    if not runs:
        print("  (a run only appears here after it finishes all seeds)")
    return runs


def _pm(mean, std, dec=3):
    if std is None or np.isnan(std) or std < 5e-4:
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f}±{std:.{dec}f}"


# ─── main paper table ────────────────────────────────────────────────────────

def main_table(runs, as_frame=True):
    rows = []
    for name, r in runs.items():
        a = r["aggregate"]
        method = r["ec_mode"] if r["beta_ec"] > 0 else "Brier-only"
        rows.append({
            "Model":    r["model_key"],
            "Method":   method,
            "Layer":    f'{r["selected_layer_frac"]} '
                        f'({r.get("selected_layer_index")}/{r.get("num_layers")})',
            "Acc NA":   _pm(a["acc_na"]["mean"],  a["acc_na"]["std"]),
            "Acc NAP":  _pm(a["acc_nap"]["mean"], a["acc_nap"]["std"]),
            "Gap":      _pm(a["gap"]["mean"],     a["gap"]["std"], 4),
            "Cohen d":  _pm(a["gap_cohen_d"]["mean"], a["gap_cohen_d"]["std"]),
            "AUROC NA": _pm(a["auroc_na"]["mean"], a["auroc_na"]["std"]),
            "ECE NA":   _pm(a["ece_na"]["mean"],   a["ece_na"]["std"], 4),
            "Brier NA": _pm(a["brier_na"]["mean"], a["brier_na"]["std"], 4),
            "UCER NA":  _pm(a["ucer_na"]["mean"],  a["ucer_na"]["std"], 4),
            "MajOpt":   _pm(a["majority_option_rate"]["mean"],
                            a["majority_option_rate"]["std"]),
            "PredEnt":  _pm(a["prediction_entropy_norm"]["mean"],
                            a["prediction_entropy_norm"]["std"]),
        })
    df = pd.DataFrame(rows).sort_values(["Model", "Method"]).reset_index(drop=True)
    if not as_frame:
        print(df.to_string(index=False))
    return df


# ─── per-model detail ────────────────────────────────────────────────────────

def detail(runs, run_name=None):
    """Full detail for one run (or every run if run_name is None)."""
    targets = [run_name] if run_name else list(runs)
    for nm in targets:
        r = runs[nm]
        a = r["aggregate"]
        print("=" * 78)
        print(f"{r['model_key']}   method={r['ec_mode']} "
              f"beta_ec={r['beta_ec']}   seeds={r['seeds']}")
        print("=" * 78)

        print(f"Selected layer : frac={r['selected_layer_frac']} "
              f"-> index {r.get('selected_layer_index')}/{r.get('num_layers')}")
        print("Layer sweep (validation total loss, lower is better):")
        for f, vl in sorted(r["sweep_val_losses"].items()):
            mark = "  <-- selected" if float(f) == r["selected_layer_frac"] else ""
            print(f"   frac={f}: {vl:.6f}{mark}")

        s = r.get("supervision_stats", {})
        if s:
            print(f"\nSupervision (gold_gap): mean={s['mean']:.4f} "
                  f"std={s['std']:.4f} p90={s.get('p90', float('nan')):.4f} "
                  f"EC-eligible={s['ec_eligible_frac']:.1%}")

        fa = r["frozen_test_accuracy"]
        print(f"\nFrozen accuracy (never altered by training): "
              f"NA={fa['NA']:.4f} NAP={fa['NAP']:.4f}")

        print(f"\nHeadline: gap={_pm(a['gap']['mean'], a['gap']['std'], 4)} "
              f"| Cohen d={_pm(a['gap_cohen_d']['mean'], a['gap_cohen_d']['std'])}")
        ci = r["per_seed"][0]["gap"]
        print(f"          seed-42 95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] "
              f"p={ci['p']:.2e}")

        print("\nSelective prediction (seed 42, NA):")
        print(f"{'tau':>6} {'coverage':>10} {'accuracy':>10} {'vs frozen':>11}")
        for row in r["per_seed"][0]["NA"]["selective"]:
            acc = row["accuracy"]
            if np.isnan(acc):
                print(f"{row['tau']:>6.1f} {row['coverage']:>10.1%} "
                      f"{'n/a':>10} {'':>11}")
            else:
                print(f"{row['tau']:>6.1f} {row['coverage']:>10.1%} "
                      f"{acc:>10.4f} {acc - fa['NA']:>+11.4f}")
        print()


# ─── health checks ───────────────────────────────────────────────────────────

def health_check(runs):
    """Every check a reviewer would run. All must pass before writing."""
    print("=" * 78)
    print("HEALTH CHECKS")
    print("=" * 78)
    ok_all = True

    # 1. one frozen accuracy per model
    per_model = {}
    for r in runs.values():
        fa = r["frozen_test_accuracy"]
        per_model.setdefault(r["model_key"], set()).add(
            (round(fa["NA"], 6), round(fa["NAP"], 6)))
    print("\n[1] Frozen accuracy identical across methods (prompt consistency)")
    for m, vals in per_model.items():
        ok = len(vals) == 1
        ok_all &= ok
        print(f"    {'PASS' if ok else 'FAIL'}  {m:<30} {sorted(vals)}")

    # 2. accuracy must not vary with seed
    print("\n[2] Accuracy invariant to seed (answer path untouched)")
    print("    (skipped for fine-tuning baselines — they update the base model)")
    for nm, r in runs.items():
        a = r["aggregate"]
        if r.get("trains_base_model"):
            ch = r.get("accuracy_change", {})
            print(f"    n/a   {nm:<40} fine-tunes base; "
                  f"acc change NA={ch.get('NA', float('nan')):+.4f}")
            continue
        ok = a["acc_na"]["std"] < 1e-12 and a["acc_nap"]["std"] < 1e-12
        ok_all &= ok
        print(f"    {'PASS' if ok else 'FAIL'}  {nm:<40} "
              f"std(acc_na)={a['acc_na']['std']:.2e}")

    # 3. supervision non-degenerate
    print("\n[3] Evidence supervision non-degenerate (EC-eligible >= 5%)")
    for nm, r in runs.items():
        s = r.get("supervision_stats", {})
        frac = s.get("ec_eligible_frac", 0)
        ok = frac >= 0.05
        ok_all &= ok
        print(f"    {'PASS' if ok else 'WARN'}  {nm:<40} EC-eligible={frac:.1%}")

    # 4. multi-seed
    print("\n[4] Multiple seeds (>=3)")
    for nm, r in runs.items():
        ok = len(r["seeds"]) >= 3
        ok_all &= ok
        print(f"    {'PASS' if ok else 'FAIL'}  {nm:<40} seeds={r['seeds']}")

    # 5. gap direction and significance
    print("\n[5] Confidence gap positive")
    for nm, r in runs.items():
        g = r["aggregate"]["gap"]["mean"]
        ok = g > 0
        ok_all &= ok
        print(f"    {'PASS' if ok else 'FAIL'}  {nm:<40} gap={g:+.4f}")

    # 5b. frozen accuracy plausibility (catches read-position bugs)
    print("\n[5b] Frozen accuracy above chance, and NA >= NAP")
    for m, r in {r["model_key"]: r for r in runs.values()}.items():
        fa = r["frozen_test_accuracy"]
        above = fa["NA"] > 0.25
        order = fa["NA"] >= fa["NAP"]
        ok_all_local = above and order
        print(f"    {'PASS' if ok_all_local else 'FAIL'}  {m:<30} "
              f"NA={fa['NA']:.4f} NAP={fa['NAP']:.4f}"
              f"{'  <- at chance!' if not above else ''}"
              f"{'  <- NAP>NA!' if not order else ''}")

    # 6. collapse warning
    print("\n[6] Prediction collapse (majority option rate < 0.60 preferred)")
    for nm, r in runs.items():
        mo = r["aggregate"]["majority_option_rate"]["mean"]
        print(f"    {'ok  ' if mo < 0.60 else 'WARN'}  {nm:<40} "
              f"majority_option_rate={mo:.3f}")

    print(f"\n{'ALL CRITICAL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'}")
    return ok_all


def baseline_comparison(runs, as_frame=True):
    """
    The table that supports the paper's motivating claim: fine-tuning moves the
    answer, the decoupled head does not. Accuracy is the honest comparison —
    ConfTuner emits a verbalized digit, the head emits a scalar, so gap and
    AUROC across the two carry a caveat.
    """
    rows = []
    for nm, r in runs.items():
        a  = r["aggregate"]
        fa = r["frozen_test_accuracy"]
        ft = bool(r.get("trains_base_model"))
        rows.append({
            "model":   r["model_key"],
            "method":  r["ec_mode"] if r["beta_ec"] > 0 or ft else "Brier-only",
            "updates_base": "yes" if ft else "no",
            "acc_na_frozen": round(fa["NA"], 4),
            "acc_na_after":  round(a["acc_na"]["mean"], 4),
            "acc_delta":     round(a["acc_na"]["mean"] - fa["NA"], 4),
            "gap":           round(a["gap"]["mean"], 4),
            "auroc_na":      round(a["auroc_na"]["mean"], 4),
            "ece_na":        round(a["ece_na"]["mean"], 4),
        })
    df = pd.DataFrame(rows).sort_values(
        ["model", "updates_base", "method"]).reset_index(drop=True)
    if not as_frame:
        print(df.to_string(index=False))
    return df


# ─── training curves ─────────────────────────────────────────────────────────

def history(out_root="runs", run_name=None):
    """Concatenated training_history.csv, ready for plotting."""
    frames = []
    for name in sorted(os.listdir(out_root)):
        if run_name and name != run_name:
            continue
        p = os.path.join(out_root, name, "outputs", "training_history.csv")
        if os.path.exists(p):
            h = pd.read_csv(p); h["run"] = name
            frames.append(h)
    if not frames:
        print("no training_history.csv found")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── export ──────────────────────────────────────────────────────────────────

def export(runs, save_dir="runs/_tables"):
    os.makedirs(save_dir, exist_ok=True)
    df = main_table(runs)
    df.to_csv(os.path.join(save_dir, "table_main_results.csv"), index=False)
    with open(os.path.join(save_dir, "table_main_results.tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True))
    print(f"Saved:\n  {save_dir}/table_main_results.csv"
          f"\n  {save_dir}/table_main_results.tex")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--save_dir", default="runs/_tables")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.out_root)
    if not runs:
        return
    print("\n" + "=" * 78)
    print("MAIN RESULTS (mean±std over seeds)")
    print("=" * 78)
    main_table(runs, as_frame=False)
    print()
    if args.detail:
        detail(runs)
    health_check(runs)
    print("\n" + "=" * 78)
    print("BASELINE COMPARISON — does the method move the answer?")
    print("=" * 78)
    baseline_comparison(runs, as_frame=False)
    export(runs, args.save_dir)


if __name__ == "__main__":
    main()
