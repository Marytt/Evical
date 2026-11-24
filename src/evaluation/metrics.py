"""
src/evaluation/metrics.py
Computes all 9 reported metrics + selective prediction + bootstrap CIs.
Called ONCE per model on held-out test set — never during training or selection.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, List


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    The final bin is CLOSED on the right so that a confidence of exactly 1.0 is
    counted rather than silently dropped (bins are [lo, hi) except the last,
    which is [lo, hi]).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / len(y_true))


def prediction_distribution(preds: pd.Series) -> Dict:
    """
    Model-independent collapse diagnostics. Replaces the old 'A-collapse',
    which silently missed models that collapse onto B or C instead.
      majority_option       : which option is predicted most often
      majority_option_rate  : its share (1/n_options = uniform, 1.0 = collapsed)
      prediction_entropy    : Shannon entropy (nats) of the empirical
                              distribution over predicted options
      prediction_entropy_norm: entropy / log(n_observed_options), in [0,1];
                              1.0 = uniform, 0.0 = fully collapsed
    """
    counts = preds.value_counts()
    p      = (counts / counts.sum()).values
    ent    = float(-(p * np.log(p)).sum())
    k      = len(p)
    return {
        "majority_option":      str(counts.index[0]),
        "majority_option_rate": float(p[0]),
        "prediction_entropy":   ent,
        "prediction_entropy_norm": float(ent / np.log(k)) if k > 1 else 0.0,
        "n_options_used":       int(k),
    }


def bootstrap_gap(c_na: np.ndarray, c_nap: np.ndarray,
                  n_boot: int = 2000, seed: int = 42) -> Dict:
    """Bootstrap CI for paired confidence gap."""
    rng  = np.random.default_rng(seed)
    d    = c_na - c_nap
    gap  = float(d.mean())
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    ci   = np.percentile(boot, [2.5, 97.5])
    try:
        p = float(stats.wilcoxon(c_na, c_nap).pvalue)
    except ValueError:
        p = 1.0
    cohen_d = float(gap / (d.std(ddof=1) + 1e-12))
    return {"gap": gap, "p": p, "cohen_d": cohen_d,
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1])}


def selective_prediction(y_true: np.ndarray, y_conf: np.ndarray,
                         thresholds: List[float]) -> List[Dict]:
    rows = []
    for tau in thresholds:
        mask = y_conf >= tau
        if mask.sum() > 0:
            acc = float(y_true[mask].mean())
        else:
            acc = float("nan")
        rows.append({"tau": tau, "coverage": float(mask.mean()), "accuracy": acc})
    return rows


def compute_all_metrics(na_df: pd.DataFrame, nap_df: pd.DataFrame,
                        cfg: dict) -> Dict:
    """
    Compute all 9 reported metrics for NA and NAP variants.
    Called ONCE on held-out test set.
    """
    out = {}
    ucer_thresh = cfg["evaluation"]["ucer_threshold"]
    n_boot      = cfg["evaluation"]["n_bootstrap"]
    sel_thres   = cfg["evaluation"]["selective_thresholds"]
    n_bins      = cfg["evaluation"]["ece_bins"]

    for v, df in [("NA", na_df), ("NAP", nap_df)]:
        y    = df["correct"].astype(int).values
        c    = df["head_conf"].values
        ucer = float(((~df["correct"]) & (df["head_conf"] >= ucer_thresh)).mean())

        try:
            auroc = float(roc_auc_score(y, c))
        except ValueError:
            auroc = float("nan")

        out[v] = {
            "accuracy":    float(df["correct"].mean()),
            "mean_conf":   float(c.mean()),
            "std_conf":    float(c.std()),
            "ucer":        ucer,
            "auroc":       auroc,
            "ece":         compute_ece(y, c, n_bins),
            "brier":       float(np.mean((c - y)**2)),
            "selective":   selective_prediction(y, c, sel_thres),
            **prediction_distribution(df["prediction"]),
        }

    # paired gap statistics (primary claim)
    gap_stats = bootstrap_gap(
        na_df["head_conf"].values, nap_df["head_conf"].values, n_boot)
    out["gap"] = gap_stats

    # accuracy gap (expected from evidence removal, not from training)
    acc_gap = float(na_df["correct"].mean() - nap_df["correct"].mean())
    try:
        _, p_acc = stats.wilcoxon(
            na_df["correct"].astype(int).values,
            nap_df["correct"].astype(int).values)
    except ValueError:
        p_acc = 1.0
    out["acc_gap"] = {"gap": acc_gap, "p": float(p_acc)}

    return out


def print_metrics(metrics: Dict, model_name: str, selected_frac: float):
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name} | selected_layer={selected_frac}")
    print(f"Note: accuracy = frozen base model predictions")
    print(f"{'='*70}")

    g = metrics["gap"]
    print(f"Confidence gap: {g['gap']:.4f} | p={g['p']:.2e} | "
          f"Cohen's d={g['cohen_d']:.3f} | "
          f"95% CI=[{g['ci_lo']:.4f}, {g['ci_hi']:.4f}]")

    print(f"\n{'Metric':<24} {'NA':>8} {'NAP':>8}")
    print("-"*44)
    for key in ["accuracy", "mean_conf", "ucer", "auroc", "ece", "brier",
                "majority_option_rate", "prediction_entropy_norm"]:
        na_val  = metrics["NA"].get(key, float("nan"))
        nap_val = metrics["NAP"].get(key, float("nan"))
        print(f"{key:<24} {na_val:>8.4f} {nap_val:>8.4f}")
    print(f"{'majority_option':<24} {metrics['NA']['majority_option']:>8} "
          f"{metrics['NAP']['majority_option']:>8}")

    print(f"\nSelective prediction (NA):")
    print(f"{'τ':>5} {'Coverage':>10} {'Accuracy':>10}")
    for row in metrics["NA"]["selective"]:
        acc_str = f"{row['accuracy']:.4f}" if not np.isnan(row['accuracy']) else "  n/a"
        print(f"{row['tau']:>5.1f} {row['coverage']:>10.1%} {acc_str:>10}")
