"""
scripts/run_transfer.py — zero-shot cross-dataset transfer.

Takes a confidence head ALREADY TRAINED on one dataset and evaluates it on
another, with no retraining and no fitting of any kind on the target data.

This is the strongest generalisation evidence available for a probe method,
because it answers two objections at once:
  1. single-benchmark generalisation -- the head scores items from a
     distribution it never saw;
  2. paired supervision at deployment -- every target item is scored from its
     own hidden state alone. Pairing is used only to REPORT the gap, never to
     produce a confidence.

The head's layer is read from the source run's record, so target features are
extracted at exactly the same depth.

Usage:
  python scripts/run_transfer.py \
      --model Qwen2.5-7B-Instruct \
      --source_run runs/Qwen2.5-7B-Instruct__graded \
      --target_dataset mediq
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import load_dataset
from src.features.extractor import load_or_extract
from src.training.head import ConfidenceHead
from src.training.trainer import load_checkpoint
from src.evaluation.metrics import compute_all_metrics, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs/pipeline_config.yaml"))
    p.add_argument("--model", required=True)
    p.add_argument("--source_run", required=True,
                   help="run dir of the trained head, e.g. runs/X__graded")
    p.add_argument("--target_dataset", required=True,
                   help="dataset key from config 'datasets'")
    p.add_argument("--seeds", default="42,43,44",
                   help="which source seeds' heads to transfer")
    p.add_argument("--out_root",   default="runs_transfer")
    p.add_argument("--cache_root", default="cache")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    model_cfg = cfg["models"][args.model]
    seeds = [int(s) for s in args.seeds.split(",")]

    # ---- source run: what layer was the head trained at? ----
    rec_path = os.path.join(args.source_run, "outputs", "run_record.json")
    src = json.load(open(rec_path))
    frac = src["selected_layer_frac"]
    src_ds = src.get("dataset", "medabstain")
    src_method = src["ec_mode"]
    if src.get("trains_base_model"):
        raise SystemExit("source run fine-tunes the base model; transfer of a "
                         "LoRA adapter is a different experiment")
    print("#" * 74)
    print(f"# TRANSFER  {args.model}")
    print(f"# head trained on : {src_ds}  (method={src_method}, layer={frac})")
    print(f"# evaluated on    : {args.target_dataset}  (no retraining)")
    print("#" * 74)

    # ---- target features at the SAME layer ----
    cfg["dataset"] = args.target_dataset
    paired, na_test, nap_test = load_dataset(cfg, args.target_dataset)
    cache_dir = os.path.join(args.cache_root, f"{args.model}__{args.target_dataset}")
    os.makedirs(cache_dir, exist_ok=True)
    _, tgt_na, tgt_nap, hidden_dim, cache_path = load_or_extract(
        paired, na_test, nap_test, model_cfg, cfg, cache_dir)

    frozen = {"NA":  float(np.mean([d["correct"] for d in tgt_na])),
              "NAP": float(np.mean([d["correct"] for d in tgt_nap]))}
    print(f"\n[transfer] target frozen accuracy: "
          f"NA={frozen['NA']:.4f} NAP={frozen['NAP']:.4f} "
          f"(drop {frozen['NA']-frozen['NAP']:+.4f})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed, frames = [], []

    for seed in seeds:
        ck = os.path.join(args.source_run, "checkpoints", f"seed{seed}", "final_head.pt")
        if not os.path.exists(ck):
            print(f"[transfer] MISSING {ck} -- skipping seed {seed}")
            continue
        state, meta = load_checkpoint(ck, device)
        if meta.get("layer_frac") not in (None, frac):
            raise SystemExit(f"checkpoint layer {meta['layer_frac']} != run layer {frac}")

        head = ConfidenceHead(hidden_dim, cfg["head"]["hidden_dim"],
                              cfg["head"]["dropout"]).to(device)
        head.load_state_dict(state)
        head.eval()

        @torch.no_grad()
        def score(data, variant):
            rows, bs = [], cfg["head"]["batch_size"]
            for i in range(0, len(data), bs):
                b = data[i:i+bs]
                h = torch.stack([d["h"][frac] for d in b]).to(device)
                for d, c in zip(b, head(h).cpu().tolist()):
                    rows.append({"sample_id": d["id"], "variant": variant,
                                 "prediction": d["pred"], "gold": d["gold"],
                                 "correct": bool(d["correct"]),
                                 "head_conf": c, "gold_prob": d["gold_prob"],
                                 "seed": seed})
            return rows

        df = pd.DataFrame(score(tgt_na, "NA") + score(tgt_nap, "NAP"))
        na_df  = df[df.variant == "NA"].sort_values("sample_id")
        nap_df = df[df.variant == "NAP"].sort_values("sample_id")
        m = compute_all_metrics(na_df, nap_df, cfg)
        m["seed"] = seed
        print_metrics(m, f"{args.model} {src_ds}->{args.target_dataset} (seed {seed})", frac)
        per_seed.append(m); frames.append(df)

    if not per_seed:
        raise SystemExit("no source checkpoints found -- nothing transferred")

    def agg(path):
        v = []
        for m in per_seed:
            cur = m
            for k in path: cur = cur[k]
            v.append(float(cur))
        return {"mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "values": v}

    aggregate = {k: agg(p) for k, p in [
        ("gap", ("gap","gap")), ("gap_cohen_d", ("gap","cohen_d")),
        ("acc_na", ("NA","accuracy")), ("acc_nap", ("NAP","accuracy")),
        ("auroc_na", ("NA","auroc")), ("auroc_nap", ("NAP","auroc")),
        ("ece_na", ("NA","ece")), ("ece_nap", ("NAP","ece")),
        ("brier_na", ("NA","brier")), ("ucer_na", ("NA","ucer")),
        ("majority_option_rate", ("NA","majority_option_rate")),
        ("prediction_entropy_norm", ("NA","prediction_entropy_norm")),
    ]}

    name = f"{args.model}__{src_method}__{src_ds}_to_{args.target_dataset}"
    out = os.path.join(args.out_root, name, "outputs")
    os.makedirs(out, exist_ok=True)
    pd.concat(frames).to_csv(os.path.join(out, "predictions_all_seeds.csv"), index=False)

    json.dump({
        "model_key": args.model, "ec_mode": f"{src_method}_transfer",
        "beta_ec": cfg["head"]["beta_ec"], "transfer": True,
        "source_dataset": src_ds, "target_dataset": args.target_dataset,
        "source_run": args.source_run,
        "selected_layer_frac": frac,
        "selected_layer_index": src.get("selected_layer_index"),
        "num_layers": src.get("num_layers"),
        "sweep_val_losses": {},
        "frozen_test_accuracy": frozen,
        "seeds": [m["seed"] for m in per_seed],
        "note": ("Head trained on the source dataset and applied unchanged. "
                 "No fitting of any kind on the target. Each target item is "
                 "scored from its own hidden state; pairing is used only to "
                 "report the gap."),
        "per_seed": per_seed, "aggregate": aggregate,
    }, open(os.path.join(out, "run_record.json"), "w"), indent=2)

    print(f"\n{'='*74}")
    print(f"TRANSFER RESULT  {args.model}  {src_ds} -> {args.target_dataset}")
    print(f"{'='*74}")
    print(f"  gap      {aggregate['gap']['mean']:.4f} ± {aggregate['gap']['std']:.4f}")
    print(f"  AUROC NA {aggregate['auroc_na']['mean']:.4f} ± {aggregate['auroc_na']['std']:.4f}")
    print(f"  ECE NA   {aggregate['ece_na']['mean']:.4f}")
    print(f"  UCER NA  {aggregate['ucer_na']['mean']:.4f}")
    print(f"  accuracy is the target frozen model's: "
          f"NA={frozen['NA']:.4f} NAP={frozen['NAP']:.4f}")
    print(f"\nSaved: {out}/run_record.json")


if __name__ == "__main__":
    main()
