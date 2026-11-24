"""
scripts/build_tables.py — collect every run_record.json into paper tables.

Usage:
  python scripts/build_tables.py --out_root runs
"""

import os, json, argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="runs")
    p.add_argument("--save_dir", default="runs/_tables")
    return p.parse_args()


def fmt(mean, std, dec=3):
    if std is None or std < 5e-4:
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f}±{std:.{dec}f}"


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    records = []
    for run in sorted(os.listdir(args.out_root)):
        rec_path = os.path.join(args.out_root, run, "outputs", "run_record.json")
        if os.path.exists(rec_path):
            records.append(json.load(open(rec_path)))

    if not records:
        raise SystemExit(f"No run_record.json found under {args.out_root}")
    print(f"Found {len(records)} runs\n")

    # ── main results table ──
    rows = []
    for r in records:
        a = r["aggregate"]
        rows.append({
            "Model":     r["model_key"],
            "Method":    r["ec_mode"] if r["beta_ec"] > 0 else "Brier-only",
            "LayerFrac": r["selected_layer_frac"],
            "LayerIdx":  f'{r.get("selected_layer_index")}/{r.get("num_layers")}',
            "Seeds":     len(r["seeds"]),
            "Acc NA":    fmt(a["acc_na"]["mean"],  a["acc_na"]["std"]),
            "Acc NAP":   fmt(a["acc_nap"]["mean"], a["acc_nap"]["std"]),
            "Gap":       fmt(a["gap"]["mean"],     a["gap"]["std"], 4),
            "Cohen d":   fmt(a["gap_cohen_d"]["mean"], a["gap_cohen_d"]["std"]),
            "AUROC NA":  fmt(a["auroc_na"]["mean"], a["auroc_na"]["std"]),
            "ECE NA":    fmt(a["ece_na"]["mean"],   a["ece_na"]["std"], 4),
            "Brier NA":  fmt(a["brier_na"]["mean"], a["brier_na"]["std"], 4),
            "UCER NA":   fmt(a["ucer_na"]["mean"],  a["ucer_na"]["std"], 4),
            "MajOptRate": fmt(a["majority_option_rate"]["mean"],
                              a["majority_option_rate"]["std"]),
            "PredEntNorm": fmt(a["prediction_entropy_norm"]["mean"],
                               a["prediction_entropy_norm"]["std"]),
        })
    main_df = pd.DataFrame(rows).sort_values(["Model", "Method"])
    main_path = os.path.join(args.save_dir, "table_main_results.csv")
    main_df.to_csv(main_path, index=False)
    print("=== MAIN RESULTS (mean±std over seeds) ===")
    print(main_df.to_string(index=False))

    # ── layer sweep table ──
    sweep_rows = []
    for r in records:
        row = {"Model": r["model_key"], "Method": r["ec_mode"],
               "Selected": r["selected_layer_frac"],
               "SelectedIdx": r.get("selected_layer_index"),
               "NumLayers": r.get("num_layers")}
        for frac, vl in r["sweep_val_losses"].items():
            row[f"val_loss@{frac}"] = round(vl, 6)
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(args.save_dir, "table_layer_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print("\n=== LAYER SWEEP (selection by lowest validation total loss) ===")
    print(sweep_df.to_string(index=False))

    # ── frozen-accuracy consistency check ──
    print("\n=== FROZEN ACCURACY CONSISTENCY ===")
    print("Every method on the same model must report identical accuracy")
    print("(the head never touches the answer path).")
    chk = {}
    for r in records:
        fa = r["frozen_test_accuracy"]
        chk.setdefault(r["model_key"], set()).add(
            (round(fa["NA"], 6), round(fa["NAP"], 6)))
    ok = True
    for model, vals in chk.items():
        status = "OK" if len(vals) == 1 else "MISMATCH"
        if len(vals) != 1:
            ok = False
        print(f"  {model:<32} {status:<9} {sorted(vals)}")
    if not ok:
        print("  [WARN] a model reports >1 frozen accuracy — check prompt consistency")

    # ── LaTeX main table ──
    latex_path = os.path.join(args.save_dir, "table_main_results.tex")
    with open(latex_path, "w") as f:
        f.write(main_df.to_latex(index=False, escape=True))

    print(f"\nSaved:")
    print(f"  {main_path}")
    print(f"  {sweep_path}")
    print(f"  {latex_path}")


if __name__ == "__main__":
    main()
