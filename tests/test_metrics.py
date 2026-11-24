import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import compute_ece, bootstrap_gap


def test_ece_perfect_calibration_is_zero():
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert compute_ece(y, p, n_bins=10) == 0.0


def test_ece_counts_confidence_exactly_one():
    # last bin is closed on the right: p == 1.0 must be counted, not dropped
    y = np.array([0.0, 0.0])
    p = np.array([1.0, 1.0])
    assert abs(compute_ece(y, p, n_bins=10) - 1.0) < 1e-12


def test_ece_hand_computed():
    y = np.array([1, 0, 1, 1], dtype=float)
    p = np.array([0.25, 0.25, 0.75, 0.75])   # bin1: |0.5-0.25|, bin2: |1.0-0.75|
    assert abs(compute_ece(y, p, n_bins=10) - 0.25) < 1e-12


def test_gap_and_paired_dz():
    rng = np.random.default_rng(0)
    c_na = rng.uniform(0.4, 0.9, 200)
    d = rng.normal(0.05, 0.02, 200)
    c_nap = c_na - d
    g = bootstrap_gap(c_na, c_nap, n_boot=200, seed=1)
    assert abs(g["gap"] - d.mean()) < 1e-12
    assert abs(g["cohen_d"] - d.mean() / d.std(ddof=1)) < 1e-6      # paired d_z
    assert g["ci_lo"] <= g["gap"] <= g["ci_hi"]
    assert g["p"] < 1e-6                                              # Wilcoxon signed-rank


def test_gap_zero_when_identical():
    c = np.linspace(0.1, 0.9, 50)
    g = bootstrap_gap(c, c.copy(), n_boot=50)
    assert g["gap"] == 0.0 and g["ci_lo"] == 0.0 and g["ci_hi"] == 0.0
