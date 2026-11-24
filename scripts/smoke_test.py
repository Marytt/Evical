"""
scripts/smoke_test.py — validate the whole pipeline on a SYNTHETIC feature
cache, with no GPU and no model download.

Builds a fake cache whose hidden states genuinely carry correctness signal at
one specific layer, then runs sweep -> selection -> retrain -> eval and checks:
  1. the sweep recovers the informative layer
  2. accuracy is byte-identical across seeds (frozen answer path invariant)
  3. the confidence gap is positive and the metrics bundle is complete

Usage:
  python scripts/smoke_test.py
"""

import os, sys, json, shutil, subprocess, hashlib
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.extractor import cache_key, cache_identity

LETTERS = ["A", "B", "C", "D", "E"]

import tempfile
TEST_ROOT   = os.path.join(tempfile.gettempdir(), "ec_smoke")
OUT_ROOT    = os.path.join(TEST_ROOT, "runs")
CACHE_ROOT  = os.path.join(TEST_ROOT, "cache")
MODEL_KEY   = "SMOKE-TEST-MODEL"
MODEL_NAME  = "smoke/test-model"
HIDDEN_DIM  = 64
N_TRAIN     = 400
N_TEST      = 200
GOOD_FRAC   = 0.8          # the layer we plant the signal in


def build_synthetic_cache(cfg, cache_path, seed=0):
    """Signal lives only at GOOD_FRAC; other layers are pure noise."""
    rng   = np.random.default_rng(seed)
    fracs = cfg["pipeline"]["layer_fracs"]

    def make_h(signal):
        """signal in [0,1] -> hidden vector; only GOOD_FRAC encodes it."""
        h = {}
        for f in fracs:
            noise = rng.normal(0, 1, HIDDEN_DIM).astype("float32")
            if f == GOOD_FRAC:
                noise[:8] += 4.0 * (signal - 0.5)   # linearly decodable
            h[f] = torch.from_numpy(noise)
        return h

    train = []
    for _ in range(N_TRAIN):
        # NA is easier than NAP: higher gold prob, more often correct
        gp_na  = float(np.clip(rng.beta(5, 3), 0, 1))
        gp_nap = float(np.clip(gp_na - abs(rng.normal(0.15, 0.10)), 0, 1))
        y_na   = float(rng.random() < gp_na)
        y_nap  = float(rng.random() < gp_nap)
        train.append({
            "h_na":  make_h(gp_na),
            "h_nap": make_h(gp_nap),
            "pred_na":  LETTERS[rng.integers(0, 5)],
            "pred_nap": LETTERS[rng.integers(0, 5)],
            "y_na": y_na, "y_nap": y_nap,
            "gp_na": gp_na, "gp_nap": gp_nap,
            "gold_gap": max(0.0, gp_na - gp_nap),
            "id_na": 0, "id_nap": 0,
        })

    # separate RNG so the synthetic option logits do not shift the main stream
    logit_rng = np.random.default_rng(seed + 1)

    def make_test(shift):
        out = []
        for i in range(N_TEST):
            gp = float(np.clip(rng.beta(5, 3) - shift, 0, 1))
            y  = float(rng.random() < gp)
            pred = LETTERS[rng.integers(0, 5)]
            out.append({
                "h": make_h(gp), "pred": pred,
                "gold": pred if y > 0.5 else LETTERS[(LETTERS.index(pred)+1) % 5],
                "correct": y, "gold_prob": gp, "id": i,
                "opt_logits": [float(x) for x in logit_rng.normal(size=len(LETTERS))],
            })
        return out

    torch.save({
        "train": train,
        "test_na":  make_test(0.00),
        "test_nap": make_test(0.15),
        "num_layers": 32, "hidden_dim": HIDDEN_DIM,
        "layer_idxs": {f: round(f*32) for f in fracs},
        "model_name": MODEL_NAME, "conf_scale": "T9",
        "identity": cache_identity(cfg["models"][MODEL_KEY], cfg),
    }, cache_path)
    print(f"[smoke] synthetic cache written: {cache_path}")
    print(f"[smoke] informative layer planted at frac={GOOD_FRAC}")


def main():
    shutil.rmtree(TEST_ROOT, ignore_errors=True)

    base_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/pipeline_config.yaml")))
    cfg = dict(base_cfg)
    cfg["models"] = {MODEL_KEY: {
        "model_name": MODEL_NAME, "prompt_mode": "chat",
        "conf_scale": "T9", "run_ec_kl": False,
    }}
    # keep the smoke test fast
    cfg["head"] = dict(base_cfg["head"])
    cfg["head"]["epochs_sweep"] = 6
    cfg["head"]["epochs_final"] = 6
    cfg["evaluation"] = dict(base_cfg["evaluation"])
    cfg["evaluation"]["n_bootstrap"] = 200

    cfg_path = os.path.join(TEST_ROOT, "smoke_config.yaml")
    os.makedirs(TEST_ROOT, exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    # plant the cache exactly where load_or_extract will look for it
    cache_dir = os.path.join(CACHE_ROOT, MODEL_KEY)
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(cfg["models"][MODEL_KEY], cfg)
    build_synthetic_cache(cfg, os.path.join(cache_dir, f"{key}.pt"))

    # stub out the network call — the cache means no data is actually needed
    stub = os.path.join(TEST_ROOT, "stub_dataset.py")
    with open(stub, "w") as f:
        f.write(
            "def load_medabstain(cfg):\n"
            "    return [], list(range(%d)), list(range(%d))\n\n"
            "def load_dataset(cfg, name=None):\n"
            "    return load_medabstain(cfg)\n" % (N_TEST, N_TEST))
    shutil.copy(os.path.join(ROOT, "src/data/dataset.py"),
                os.path.join(TEST_ROOT, "dataset_backup.py"))
    shutil.copy(stub, os.path.join(ROOT, "src/data/dataset.py"))

    try:
        cmd = [sys.executable, os.path.join(ROOT, "scripts/run_pipeline.py"),
               "--config", cfg_path, "--model", MODEL_KEY,
               "--out_root", OUT_ROOT, "--cache_root", CACHE_ROOT,
               "--seeds", "42,43,44"]
        print(f"[smoke] running: {' '.join(cmd)}\n")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-6000:])
        if r.returncode != 0:
            print("STDERR:\n", r.stderr[-4000:])
            raise SystemExit(f"[smoke] FAILED rc={r.returncode}")
    finally:
        shutil.copy(os.path.join(TEST_ROOT, "dataset_backup.py"),
                    os.path.join(ROOT, "src/data/dataset.py"))

    # ── assertions ──
    rec_path = os.path.join(OUT_ROOT, f"{MODEL_KEY}__graded",
                            "outputs", "run_record.json")
    rec = json.load(open(rec_path))
    agg = rec["aggregate"]

    print("\n" + "="*70)
    print("SMOKE TEST CHECKS")
    print("="*70)

    checks = []

    sel = rec["selected_layer_frac"]
    checks.append(("sweep recovered the informative layer "
                   f"(selected={sel}, planted={GOOD_FRAC})", sel == GOOD_FRAC))

    checks.append(("accuracy identical across seeds (frozen answer path)",
                   agg["acc_na"]["std"] == 0.0 and agg["acc_nap"]["std"] == 0.0))

    checks.append((f"confidence gap positive (mean={agg['gap']['mean']:.4f})",
                   agg["gap"]["mean"] > 0))

    checks.append((f"AUROC above chance (NA={agg['auroc_na']['mean']:.3f})",
                   agg["auroc_na"]["mean"] > 0.55))

    checks.append(("3 seeds recorded", len(rec["seeds"]) == 3
                   and len(agg["gap"]["values"]) == 3))

    checks.append(("all sweep layers evaluated",
                   len(rec["sweep_val_losses"]) == len(cfg["pipeline"]["layer_fracs"])))

    # FIX 9: concrete layer index recorded
    checks.append((f"layer index recorded "
                   f"(frac={rec['selected_layer_frac']} -> "
                   f"idx={rec.get('selected_layer_index')}/{rec.get('num_layers')})",
                   rec.get("selected_layer_index") is not None
                   and rec.get("num_layers") is not None))

    # FIX 7: supervision stats recorded
    sup = rec.get("supervision_stats", {})
    checks.append(("gold_gap supervision stats recorded",
                   all(k in sup for k in
                       ["mean", "std", "min", "max", "ec_eligible_frac"])))

    # FIX 4: stratified split recorded
    checks.append(("stratified split flag recorded",
                   rec.get("stratified_split") is True))

    # FIX 6: model-independent collapse metrics present
    checks.append(("majority_option_rate + prediction entropy reported",
                   "majority_option_rate" in agg
                   and "prediction_entropy_norm" in agg))

    # full seed registration recorded
    srs = rec.get("seed_records", [])
    checks.append(("full seed registration recorded (random/numpy/torch/cuda/cudnn)",
                   len(srs) > 0 and all(
                       k in srs[0] for k in
                       ["python_random", "numpy", "torch", "torch_cuda",
                        "cudnn_deterministic"])))

    # training history saved
    hist_p = os.path.join(OUT_ROOT, f"{MODEL_KEY}__graded",
                          "outputs", "training_history.csv")
    hist_ok = os.path.exists(hist_p)
    if hist_ok:
        import pandas as _pd
        h = _pd.read_csv(hist_p)
        need = {"phase", "layer_frac", "seed", "epoch", "train_loss",
                "val_loss", "train_brier", "train_ec", "val_gap", "lr"}
        hist_ok = need.issubset(set(h.columns)) and len(h) > 0 \
                  and set(h["phase"]) == {"sweep", "final"}
    checks.append(("training history saved with sweep+final epochs", hist_ok))

    # checkpoint carries metadata
    ck = os.path.join(OUT_ROOT, f"{MODEL_KEY}__graded",
                      "checkpoints", "seed42", "final_head.pt")
    ck_ok = os.path.exists(ck)
    if ck_ok:
        obj = torch.load(ck, map_location="cpu")
        ck_ok = (isinstance(obj, dict) and "state_dict" in obj
                 and obj.get("layer_frac") is not None
                 and obj.get("seed") == 42
                 and obj.get("config") is not None)
    checks.append(("checkpoint saved with metadata (layer/seed/epoch/config)", ck_ok))

    for name in ["predictions_seed42.csv", "predictions_all_seeds.csv",
                 "summary_row.csv", "run_record.json", "training_history.csv"]:
        p = os.path.join(OUT_ROOT, f"{MODEL_KEY}__graded", "outputs", name)
        checks.append((f"artifact saved: {name}", os.path.exists(p)))

    n_ok = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        n_ok += int(ok)

    print(f"\n{n_ok}/{len(checks)} checks passed")
    if n_ok != len(checks):
        raise SystemExit("[smoke] SOME CHECKS FAILED")
    print("[smoke] pipeline is sound — safe to run on GPU")


if __name__ == "__main__":
    main()
