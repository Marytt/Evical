import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.head import ConfidenceHead, compute_loss

FRAC = 0.7


class _Const(torch.nn.Module):
    """Stand-in head that returns preset confidences, to test the loss algebra."""
    def __init__(self, c_na, c_nap):
        super().__init__(); self.vals = [torch.tensor(c_na), torch.tensor(c_nap)]; self.i = 0
    def forward(self, h):
        v = self.vals[self.i % 2]; self.i += 1; return v


def _batch(y_na, y_nap, t, dim=4):
    return [dict(h_na={FRAC: torch.zeros(dim)}, h_nap={FRAC: torch.zeros(dim)},
                 y_na=a, y_nap=b, gold_gap=g) for a, b, g in zip(y_na, y_nap, t)]


def _cfg(mode, beta, delta=0.1):
    return dict(alpha_cal=1.0, beta_ec=beta, ec_mode=mode, margin_delta=delta)


def test_brier_only_is_graded_with_beta_zero():
    head = _Const([0.8, 0.6], [0.5, 0.6])
    b = _batch([1.0, 0.0], [1.0, 0.0], [0.2, 0.0])
    total, brier, ec, gap = compute_loss(b, FRAC, head, torch.device("cpu"), _cfg("graded", 0.0))
    exp_brier = (((0.8-1)**2 + (0.5-1)**2) + ((0.6-0)**2 + (0.6-0)**2)) / 2 / 2
    assert abs(brier - exp_brier) < 1e-6 and abs(float(total) - exp_brier) < 1e-6


def test_graded_ec_regresses_gap_on_target():
    head = _Const([0.8, 0.6], [0.5, 0.6])          # gaps 0.3, 0.0
    b = _batch([1.0, 0.0], [1.0, 0.0], [0.2, 0.1])
    _, _, ec, gap = compute_loss(b, FRAC, head, torch.device("cpu"), _cfg("graded", 0.5))
    assert abs(ec - ((0.3-0.2)**2 + (0.0-0.1)**2) / 2) < 1e-6
    assert abs(gap - 0.15) < 1e-6


def test_fixed_margin_hinge():
    head = _Const([0.8, 0.6], [0.75, 0.3])         # gaps 0.05, 0.3
    b = _batch([1.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    _, _, ec, _ = compute_loss(b, FRAC, head, torch.device("cpu"), _cfg("fixed", 0.5, delta=0.1))
    assert abs(ec - (0.05 + 0.0) / 2) < 1e-6        # relu(0.1-0.05), relu(0.1-0.3)


def test_head_outputs_probabilities():
    torch.manual_seed(0)
    h = ConfidenceHead(input_dim=16, hidden_dim=256, dropout=0.1).eval()
    c = h(torch.randn(32, 16))
    assert c.shape == (32,) and bool(((c > 0) & (c < 1)).all())
