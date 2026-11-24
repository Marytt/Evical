"""
src/training/head.py
Confidence head architecture and loss functions.
Completely decoupled from the frozen base model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class ConfidenceHead(nn.Module):
    """
    Small MLP: hidden_state → scalar confidence ∈ [0,1].
    Trained with Brier + graded EC loss on frozen hidden states.
    Zero accuracy cost: base model weights are never updated.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.float()).squeeze(-1)


def compute_loss(batch: List[Dict], frac: float, head: ConfidenceHead,
                 device: torch.device, cfg: dict) -> tuple:
    """
    Compute Brier + EC loss for a batch of paired (NA, NAP) items.

    Selection criterion: total loss (Brier + EC) — not gap, not AUROC.
    This is what we use for validation-based layer selection.

    Args:
        batch:  list of dicts with h_na, h_nap, y_na, y_nap, gold_gap
        frac:   which layer fraction to use (key into h_na/h_nap dicts)
        head:   ConfidenceHead
        device: torch device
        cfg:    pipeline_config["head"]

    Returns:
        (total_loss, brier_loss, ec_loss, mean_conf_gap)
    """
    h_na  = torch.stack([d["h_na"][frac]  for d in batch]).to(device)
    h_nap = torch.stack([d["h_nap"][frac] for d in batch]).to(device)
    y_na  = torch.tensor([d["y_na"]    for d in batch],
                          dtype=torch.float32, device=device)
    y_nap = torch.tensor([d["y_nap"]   for d in batch],
                          dtype=torch.float32, device=device)
    t     = torch.tensor([d["gold_gap"] for d in batch],
                          dtype=torch.float32, device=device)

    c_na  = head(h_na)
    c_nap = head(h_nap)

    # Brier calibration loss (both branches)
    loss_brier = ((c_na - y_na)**2 + (c_nap - y_nap)**2).mean() / 2.0

    # Evidence-contrastive loss
    ec_mode = cfg.get("ec_mode", "graded")
    if ec_mode == "graded":
        # regress confidence gap onto gold_prob gap (rho=0.536)
        conf_gap = c_na - c_nap
        loss_ec  = ((conf_gap - t) ** 2).mean()
    else:
        # fixed margin (ablation baseline)
        delta    = cfg.get("margin_delta", 0.1)
        conf_gap = c_na - c_nap
        loss_ec  = F.relu(delta - conf_gap).mean()

    total = cfg["alpha_cal"] * loss_brier + cfg["beta_ec"] * loss_ec
    mean_gap = float((c_na - c_nap).mean().detach().cpu())
    return total, float(loss_brier), float(loss_ec), mean_gap
