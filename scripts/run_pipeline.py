"""
scripts/run_pipeline.py — EC-Head standardized pipeline orchestrator

Protocol (identical for every model, no exceptions):
  1. Load MedAbstain paired data
  2. Extract hidden states at ALL candidate layers in ONE forward pass -> cache
  3. Fixed train/val split (seeded, same split reused for every layer & seed)
  4. Layer sweep: train one head per layer, select by LOWEST VALIDATION TOTAL LOSS
  5. Retrain final head on train+val at the selected layer, for each seed
  6. Evaluate held-out test set ONCE per seed, aggregate mean +/- std
  7. Save config, cache path, checkpoints, predictions, metrics, seeds

Usage:
  python scripts/run_pipeline.py --model Qwen2.5-7B-Instruct
  python scripts/run_pipeline.py --model Qwen2.5-7B-Instruct --seeds 42,43,44
  python scripts/run_pipeline.py --model Qwen2.5-7B-Instruct --ec_mode fixed
"""

import os, sys, json, argparse, copy
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import load_medabstain, load_dataset
from src.features.extractor import load_or_extract
from src.training.head import ConfidenceHead
from src.training.trainer import run_sweep, retrain_final
from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.utils.seeding import set_all_seeds


# ─── ARGS ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs/pipeline_config.yaml"))
    p.add_argument("--model", required=True,
                   help="Model key from config 'models' section")
    p.add_argument("--out_root", default="runs")
    p.add_argument("--cache_root", default="cache")
    p.add_argument("--seeds", default="42,43,44",
                   help="Comma-separated seeds for the final retrain")
    p.add_argument("--ec_mode", default=None,
                   help="Override config ec_mode: graded | fixed")
    p.add_argument("--beta_ec", type=float, default=None,
                   help="Override config beta_ec (use 0.0 for Brier-only ablation)")
    p.add_argument("--tag", default=None, help="Suffix for the run directory")
    p.add_argument("--dataset", default=None,
                   help="dataset key from config 'datasets' (e.g. mediq)")
    p.add_argument("--force_layer", type=float, default=None,
                   help="Skip the sweep and use this layer fraction. Use to "
                        "compare objectives at a MATCHED depth, removing the "
                        "layer confound from the fixed-vs-graded comparison.")
    p.add_argument("--strict_determinism", action="store_true",
                   help="Also enable torch.use_deterministic_algorithms")
    return p.parse_args()


# ─── SPLIT ───────────────────────────────────────────────────────────────────

def report_supervision(train_data):
    """
    FIX 7: verify the extracted supervision signal BEFORE spending epochs on it.
    A degenerate gold_gap (all zeros, or near-zero EC-eligible %) means the
    graded EC target carries no information and the run is worthless.
    """
    gg   = np.array([d["gold_gap"] for d in train_data], dtype=float)
    y_na = np.array([d["y_na"]     for d in train_data], dtype=float)
    y_np = np.array([d["y_nap"]    for d in train_data], dtype=float)
    elig = float((gg > 0).mean())

    print("\n[supervision] gold_gap = max(0, P(gold|NA) - P(gold|NAP))")
    print(f"  mean={gg.mean():.4f}  std={gg.std():.4f}  "
          f"min={gg.min():.4f}  max={gg.max():.4f}")
    for q in [50, 75, 90, 99]:
        print(f"  p{q}={np.percentile(gg, q):.4f}", end="")
    print()
    print(f"  EC-eligible (gold_gap>0): {elig:.1%}")
    print(f"  binary pattern y_NA>y_NAP: {float((y_na > y_np).mean()):.1%}")
    print(f"  train acc: NA={y_na.mean():.4f} NAP={y_np.mean():.4f}")

    if elig < 0.05:
        print("  [WARN] <5% of pairs carry any evidence signal — "
              "graded EC target is near-degenerate")
    if gg.std() < 1e-6:
        raise SystemExit("[supervision] FATAL: gold_gap has zero variance; "
                         "extraction is broken")
    return {"mean": float(gg.mean()), "std": float(gg.std()),
            "min": float(gg.min()),  "max": float(gg.max()),
            "p50": float(np.percentile(gg, 50)),
            "p90": float(np.percentile(gg, 90)),
            "ec_eligible_frac": elig,
            "binary_pattern_frac": float((y_na > y_np).mean())}


def make_split(train_data, val_frac, seed, stratify=True):
    """
    Fixed train/val split, using the PIPELINE seed (not the per-run seed) so the
    split is byte-identical across every layer and every training seed.

    When stratify=True the split is stratified on the paired correctness pattern
    (y_NA, y_NAP) -> 4 strata. Without this, the rare stratum that carries the
    entire evidence-contrastive signal (y_NA=1, y_NAP=0, ~8% of pairs) can be
    unevenly distributed, which perturbs validation loss and therefore layer
    selection.
    """
    rng = np.random.default_rng(seed)

    if not stratify:
        order = rng.permutation(len(train_data))
        n_val = int(len(train_data) * val_frac)
        return ([train_data[i] for i in order[n_val:]],
                [train_data[i] for i in order[:n_val]])

    strata = {}
    for i, d in enumerate(train_data):
        strata.setdefault((d["y_na"], d["y_nap"]), []).append(i)

    val_idx, trn_idx = [], []
    for key in sorted(strata):
        idx = np.array(strata[key])
        idx = idx[rng.permutation(len(idx))]
        n_v = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_v].tolist())
        trn_idx.extend(idx[n_v:].tolist())

    print("[split] stratified on (y_NA, y_NAP):")
    for key in sorted(strata):
        n   = len(strata[key])
        n_v = int(round(n * val_frac))
        print(f"  y_NA={int(key[0])} y_NAP={int(key[1])}: "
              f"n={n:>5} ({n/len(train_data):>6.1%})  -> val={n_v}")

    rng.shuffle(trn_idx)
    rng.shuffle(val_idx)
    return ([train_data[i] for i in trn_idx],
            [train_data[i] for i in val_idx])


# ─── EVAL ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_test(head, data, variant, frac, device, batch_size):
    head.eval()
    rows = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        h     = torch.stack([d["h"][frac] for d in batch]).to(device)
        confs = head(h).cpu().tolist()
        for d, c in zip(batch, confs):
            rows.append({
                "sample_id":  d["id"],
                "variant":    variant,
                "prediction": d["pred"],
                "gold":       d["gold"],
                "correct":    bool(d["correct"]),
                "head_conf":  c,
                "gold_prob":  d["gold_prob"],
                "opt_logits": json.dumps(d["opt_logits"]),
            })
    return rows


# ─── AGGREGATION ─────────────────────────────────────────────────────────────

def aggregate_seeds(per_seed_metrics):
    """Mean +/- std across seeds for the headline metrics."""
    def collect(path):
        out = []
        for m in per_seed_metrics:
            cur = m
            for k in path:
                cur = cur[k]
            out.append(cur)
        return np.array(out, dtype=float)

    agg = {}
    for name, path in [
        ("gap",         ("gap", "gap")),
        ("gap_p",       ("gap", "p")),
        ("gap_cohen_d", ("gap", "cohen_d")),
        ("acc_na",      ("NA",  "accuracy")),
        ("acc_nap",     ("NAP", "accuracy")),
        ("auroc_na",    ("NA",  "auroc")),
        ("auroc_nap",   ("NAP", "auroc")),
        ("ece_na",      ("NA",  "ece")),
        ("ece_nap",     ("NAP", "ece")),
        ("brier_na",    ("NA",  "brier")),
        ("ucer_na",     ("NA",  "ucer")),
        ("ucer_nap",    ("NAP", "ucer")),
        ("majority_option_rate",    ("NA", "majority_option_rate")),
        ("prediction_entropy_norm", ("NA", "prediction_entropy_norm")),
    ]:
        vals = collect(path)
        agg[name] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }
    return agg


def print_aggregate(agg, model_key, frac, seeds):
    print(f"\n{'='*74}")
    print(f"AGGREGATE OVER {len(seeds)} SEEDS: {model_key} | layer={frac}")
    print(f"seeds={seeds}")
    print(f"{'='*74}")
    print(f"{'Metric':<16} {'Mean':>10} {'Std':>9}   {'Per-seed'}")
    print("-"*74)
    for k in ["acc_na", "acc_nap", "gap", "gap_cohen_d",
              "auroc_na", "auroc_nap", "ece_na", "ece_nap",
              "brier_na", "ucer_na", "ucer_nap",
              "majority_option_rate", "prediction_entropy_norm"]:
        v = agg[k]
        per = " ".join(f"{x:.4f}" for x in v["values"])
        print(f"{k:<16} {v['mean']:>10.4f} {v['std']:>9.4f}   [{per}]")
    print(f"\nNOTE: accuracy is the FROZEN base model's accuracy. It is identical")
    print(f"      across seeds by construction — the head never touches the")
    print(f"      answer path. Any std > 0 on acc_* indicates a pipeline bug.")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    if args.model not in cfg["models"]:
        raise SystemExit(f"Unknown model key '{args.model}'. "
                         f"Available: {list(cfg['models'])}")
    model_cfg = cfg["models"][args.model]

    # config overrides (for ablations) — recorded in the saved config
    head_cfg = copy.deepcopy(cfg["head"])
    if args.ec_mode is not None:
        head_cfg["ec_mode"] = args.ec_mode
    if args.beta_ec is not None:
        head_cfg["beta_ec"] = args.beta_ec
    head_cfg["seed"] = cfg["pipeline"]["seed"]

    seeds = [int(s) for s in args.seeds.split(",")]
    mode  = head_cfg["ec_mode"] if head_cfg["beta_ec"] > 0 else "brier_only"
    run_name = f"{args.model}__{mode}" + (f"__{args.tag}" if args.tag else "")

    run_dir   = os.path.join(args.out_root, run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    out_dir   = os.path.join(run_dir, "outputs")
    cache_dir = os.path.join(args.cache_root, args.model)
    for d in [ckpt_dir, out_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'#'*74}")
    print(f"# EC-Head pipeline: {args.model}")
    print(f"# ec_mode={head_cfg['ec_mode']} beta_ec={head_cfg['beta_ec']} seeds={seeds}")
    print(f"# run_dir={run_dir}")
    print(f"{'#'*74}")

    seed_record = set_all_seeds(cfg["pipeline"]["seed"],
                               strict=args.strict_determinism)
    print(f"[pipeline] seeds set: {seed_record}")

    # ── 1. data ──
    if args.dataset:
        cfg["dataset"] = args.dataset
    paired_train, na_test, nap_test = load_dataset(cfg, cfg.get("dataset"))

    # ── 2. features (all layers, one forward pass, cached) ──
    train_data, test_na, test_nap, hidden_dim, cache_path = load_or_extract(
        paired_train, na_test, nap_test, model_cfg, cfg, cache_dir)

    frozen_acc_na  = float(np.mean([d["correct"] for d in test_na]))
    frozen_acc_nap = float(np.mean([d["correct"] for d in test_nap]))
    print(f"[pipeline] frozen test accuracy: NA={frozen_acc_na:.4f} "
          f"NAP={frozen_acc_nap:.4f}")

    n_opts = 5
    chance = 1.0 / n_opts
    if frozen_acc_na < chance + 0.05:
        print(f"[pipeline] *** WARNING: frozen NA accuracy {frozen_acc_na:.4f} is "
              f"at/near chance ({chance:.3f}) for {n_opts} options.")
        print(f"[pipeline] *** Either the model cannot do this task, or the READ "
              f"POSITION is wrong. Verify before trusting these results.")
    if frozen_acc_nap > frozen_acc_na:
        print(f"[pipeline] *** WARNING: NAP accuracy ({frozen_acc_nap:.4f}) EXCEEDS "
              f"NA ({frozen_acc_na:.4f}). Evidence removal should not help; "
              f"this usually means the model is not reading the question.")

    supervision = report_supervision(train_data)

    # ── 3. fixed split (pipeline seed, shared by all layers and all run seeds) ──
    train_set, val_set = make_split(
        train_data, cfg["pipeline"]["val_frac"], cfg["pipeline"]["seed"],
        stratify=cfg["pipeline"].get("stratify_split", True))
    print(f"[pipeline] split: train={len(train_set)} val={len(val_set)}")

    # FIX 9: resolve the concrete transformer layer index behind each fraction
    _cache_meta = torch.load(cache_path, map_location="cpu")
    layer_idxs  = _cache_meta.get("layer_idxs", {})
    num_layers  = _cache_meta.get("num_layers")
    del _cache_meta
    print(f"[pipeline] layer map (frac -> index of {num_layers}): {layer_idxs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 4. layer sweep -> select by lowest validation total loss ──
    if args.force_layer is not None:
        best_frac = args.force_layer
        if best_frac not in cfg["pipeline"]["layer_fracs"]:
            raise SystemExit(f"--force_layer {best_frac} not in "
                             f"{cfg['pipeline']['layer_fracs']}")
        print(f"\n[pipeline] LAYER FORCED to {best_frac} "
              f"(idx {layer_idxs.get(best_frac)}/{num_layers}) — sweep skipped")
        print(f"[pipeline] use this to compare objectives at matched depth")
        sweep = {"selected_frac": best_frac, "history": [],
                 **{f: {"val_loss": float("nan")} for f in cfg["pipeline"]["layer_fracs"]}}
    else:
        sweep = run_sweep(train_set, val_set, cfg["pipeline"]["layer_fracs"],
                          hidden_dim, device, head_cfg, ckpt_dir,
                          layer_idxs=layer_idxs)
        best_frac = sweep["selected_frac"]

    # ── 5. final retrain per seed on train+val, 6. evaluate test once per seed ──
    full_data = train_set + val_set if cfg["pipeline"]["retrain_on_full"] else train_set
    per_seed_metrics, per_seed_frames = [], []
    seed_records = [seed_record]
    training_history = list(sweep.get("history", []))

    for seed in seeds:
        print(f"\n{'-'*74}\n[pipeline] FINAL RETRAIN seed={seed}\n{'-'*74}")
        seed_cfg = copy.deepcopy(head_cfg)
        seed_cfg["seed"] = seed
        seed_records.append(set_all_seeds(seed, strict=args.strict_determinism))

        seed_ckpt_dir = os.path.join(ckpt_dir, f"seed{seed}")
        os.makedirs(seed_ckpt_dir, exist_ok=True)
        head, final_hist = retrain_final(
            full_data, best_frac, hidden_dim, device, seed_cfg,
            seed_ckpt_dir, layer_index=layer_idxs.get(best_frac), seed=seed)
        training_history.extend(final_hist)

        rows = (predict_test(head, test_na,  "NA",  best_frac,
                             device, head_cfg["batch_size"]) +
                predict_test(head, test_nap, "NAP", best_frac,
                             device, head_cfg["batch_size"]))
        df = pd.DataFrame(rows)
        df["seed"] = seed
        pred_path = os.path.join(out_dir, f"predictions_seed{seed}.csv")
        df.to_csv(pred_path, index=False)

        na_df  = df[df["variant"] == "NA"].sort_values("sample_id")
        nap_df = df[df["variant"] == "NAP"].sort_values("sample_id")
        m = compute_all_metrics(na_df, nap_df, cfg)
        m["seed"] = seed
        print_metrics(m, f"{args.model} (seed={seed})", best_frac)

        # frozen-accuracy invariant check
        if abs(m["NA"]["accuracy"] - frozen_acc_na) > 1e-9:
            print(f"[WARN] NA accuracy {m['NA']['accuracy']:.6f} != frozen "
                  f"{frozen_acc_na:.6f} — answer path was modified!")

        per_seed_metrics.append(m)
        per_seed_frames.append(df)

    # ── aggregate ──
    agg = aggregate_seeds(per_seed_metrics)
    print_aggregate(agg, args.model, best_frac, seeds)

    pd.concat(per_seed_frames).to_csv(
        os.path.join(out_dir, "predictions_all_seeds.csv"), index=False)

    hist_path = os.path.join(out_dir, "training_history.csv")
    pd.DataFrame(training_history).to_csv(hist_path, index=False)
    print(f"[pipeline] training history ({len(training_history)} epoch rows) "
          f"-> {hist_path}")

    # ── 7. reproducibility bundle ──
    record = {
        "model_key":     args.model,
        "model_cfg":     model_cfg,
        "head_cfg":      head_cfg,
        "pipeline_cfg":  cfg["pipeline"],
        "evaluation_cfg": cfg["evaluation"],
        "seeds":         seeds,
        "ec_mode":       head_cfg["ec_mode"],
        "beta_ec":       head_cfg["beta_ec"],
        "selected_layer_frac":  best_frac,
        "selected_layer_index":  layer_idxs.get(best_frac),
        "num_layers":            num_layers,
        "layer_frac_to_index":   {str(k): v for k, v in layer_idxs.items()},
        "supervision_stats":     supervision,
        "seed_records":           seed_records,
        "sweep_val_losses": {str(f): sweep[f]["val_loss"]
                              for f in cfg["pipeline"]["layer_fracs"]},
        "layer_forced": args.force_layer is not None,
        "frozen_test_accuracy": {"NA": frozen_acc_na, "NAP": frozen_acc_nap},
        "feature_cache":  cache_path,
        "hidden_dim":     hidden_dim,
        "n_train_pairs":  len(train_data),
        "n_train_split":  len(train_set),
        "n_val_split":    len(val_set),
        "stratified_split": cfg["pipeline"].get("stratify_split", True),
        "n_test_pairs":   len(test_na),
        "per_seed":       per_seed_metrics,
        "aggregate":      agg,
        "env": {
            "torch": torch.__version__,
            "cuda":  torch.version.cuda,
            "device": str(device),
        },
    }
    rec_path = os.path.join(out_dir, "run_record.json")
    json.dump(record, open(rec_path, "w"), indent=2)

    # single-row summary for the paper table
    summary = {
        "model": args.model, "ec_mode": mode,
        "layer_frac": best_frac,
        "layer_index": layer_idxs.get(best_frac),
        "num_layers": num_layers,
        "n_seeds": len(seeds),
    }
    for k, v in agg.items():
        summary[f"{k}_mean"] = v["mean"]
        summary[f"{k}_std"]  = v["std"]
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "summary_row.csv"), index=False)

    print(f"\n[pipeline] Saved:")
    print(f"  feature cache : {cache_path}")
    print(f"  checkpoints   : {ckpt_dir}")
    print(f"  predictions   : {out_dir}/predictions_seed*.csv")
    print(f"  run record    : {rec_path}")
    print(f"  summary row   : {out_dir}/summary_row.csv")
    print(f"  train history : {hist_path}")
    print(f"[pipeline] Done: {args.model} | layer_frac={best_frac} "
          f"(index {layer_idxs.get(best_frac)}/{num_layers}) | mode={mode}")


if __name__ == "__main__":
    main()
