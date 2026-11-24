"""
src/training/trainer.py
Layer sweep -> selection by lowest val loss -> retrain on train+val.

Protocol:
  1. Fixed stratified train/val split (same for all layers and seeds)
  2. Train one head per layer, select by lowest validation TOTAL loss (Brier+EC)
  3. Retrain the selected layer on full data (train + val)
  4. Return best head + selection record + full per-epoch training history

Every checkpoint is saved WITH metadata (layer, seed, epoch, config) so an
orphaned .pt file is still self-describing months later.
"""

import os
import json
import numpy as np
import torch
from torch.optim import AdamW
from typing import List, Dict, Tuple

from src.training.head import ConfidenceHead, compute_loss


def save_checkpoint(path: str, head: ConfidenceHead, *, layer_frac: float,
                    layer_index=None, seed=None, epoch=None, phase=None,
                    val_loss=None, cfg: dict = None, extra: dict = None):
    """Self-describing checkpoint: weights plus everything needed to place it."""
    payload = {
        "state_dict":  head.state_dict(),
        "layer_frac":  layer_frac,
        "layer_index": layer_index,
        "seed":        seed,
        "epoch":       epoch,
        "phase":       phase,
        "val_loss":    val_loss,
        "config":      cfg,
        "head_arch": {
            "input_dim":  head.net[0].in_features,
            "hidden_dim": head.net[0].out_features,
        },
        "format_version": "ckpt-v2",
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic


def load_checkpoint(path: str, device="cpu"):
    """Returns (state_dict, metadata). Tolerates bare-state_dict legacy files."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        meta = {k: v for k, v in obj.items() if k != "state_dict"}
        return obj["state_dict"], meta
    return obj, {}


def _validate(val_data: List, frac: float, head: ConfidenceHead,
              device: torch.device, cfg: dict, batch_size: int):
    """Returns (val_total, val_brier, val_ec, val_gap).
    Selection uses val_total only."""
    head.eval()
    tot, bri, ec, gaps = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            loss, b, e, gap = compute_loss(batch, frac, head, device, cfg)
            tot.append(float(loss)); bri.append(b); ec.append(e); gaps.append(gap)
    head.train()
    return (float(np.mean(tot)), float(np.mean(bri)),
            float(np.mean(ec)),  float(np.mean(gaps)))


def _train_one_layer(train_set: List, val_data: List,
                     frac: float, hidden_dim: int,
                     device: torch.device, cfg: dict,
                     epochs: int, label: str,
                     layer_index=None) -> Tuple[dict, float, List[Dict], int]:
    """Train one head for one layer.
    Returns (best_state, best_val_loss, history, best_epoch)."""
    head = ConfidenceHead(hidden_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)
    opt  = AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng  = np.random.default_rng(cfg.get("seed", 42))

    best_val_loss, best_state, best_epoch = float("inf"), None, -1
    no_improve, history = 0, []

    for epoch in range(epochs):
        order   = rng.permutation(len(train_set))
        batches = [list(order[i:i+cfg["batch_size"]])
                   for i in range(0, len(order), cfg["batch_size"])]
        head.train()
        tr_tot, tr_bri, tr_ec, tr_gap = [], [], [], []
        for batch_idx in batches:
            batch = [train_set[i] for i in batch_idx]
            opt.zero_grad()
            loss, b, e, gap = compute_loss(batch, frac, head, device, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["grad_clip"])
            opt.step()
            tr_tot.append(float(loss)); tr_bri.append(b)
            tr_ec.append(e);            tr_gap.append(gap)

        v_tot, v_bri, v_ec, v_gap = _validate(
            val_data, frac, head, device, cfg, cfg["batch_size"])

        is_best = v_tot < best_val_loss
        if is_best:
            best_val_loss = v_tot
            best_state    = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch    = epoch + 1
            no_improve    = 0
        else:
            no_improve += 1

        history.append({
            "phase": "sweep", "layer_frac": frac, "layer_index": layer_index,
            "seed": cfg.get("seed"), "epoch": epoch + 1,
            "train_loss": float(np.mean(tr_tot)),
            "train_brier": float(np.mean(tr_bri)),
            "train_ec": float(np.mean(tr_ec)),
            "train_gap": float(np.mean(tr_gap)),
            "val_loss": v_tot, "val_brier": v_bri,
            "val_ec": v_ec, "val_gap": v_gap,
            "lr": float(opt.param_groups[0]["lr"]),
            "is_best": bool(is_best),
        })

        print(f"  [{label} ep{epoch+1}] train={np.mean(tr_tot):.5f} "
              f"val={v_tot:.5f} val_gap={v_gap:.4f}{' *' if is_best else ''}")

        if no_improve >= cfg["patience"]:
            print(f"  [{label}] Early stop at epoch {epoch+1} "
                  f"(best epoch {best_epoch})")
            break

    return best_state, best_val_loss, history, best_epoch


def run_sweep(train_data: List, val_data: List,
              layer_fracs: List[float], hidden_dim: int,
              device: torch.device, cfg: dict,
              ckpt_dir: str, layer_idxs: Dict = None) -> Dict:
    """Train one head per layer; select by lowest validation total loss."""
    layer_idxs = layer_idxs or {}
    sweep, all_history = {}, []

    print(f"\n[sweep] Layers to try: {layer_fracs}")
    print(f"[sweep] Selection criterion: lowest validation total loss (Brier + EC)\n")

    for frac in layer_fracs:
        label = f"frac={frac}"
        print(f"\n[sweep] Training layer {label}")
        state, val_loss, hist, best_epoch = _train_one_layer(
            train_data, val_data, frac, hidden_dim, device,
            cfg, cfg["epochs_sweep"], label, layer_idxs.get(frac))
        all_history.extend(hist)

        head_tmp = ConfidenceHead(hidden_dim, cfg["hidden_dim"],
                                  cfg["dropout"]).to(device)
        head_tmp.load_state_dict(state)
        ckpt_path = os.path.join(ckpt_dir, f"sweep_frac{frac}.pt")
        save_checkpoint(ckpt_path, head_tmp, layer_frac=frac,
                        layer_index=layer_idxs.get(frac),
                        seed=cfg.get("seed"), epoch=best_epoch,
                        phase="sweep", val_loss=val_loss, cfg=cfg)

        sweep[frac] = {"val_loss": val_loss, "ckpt": ckpt_path,
                       "best_epoch": best_epoch}
        print(f"[sweep] frac={frac} best_val_loss={val_loss:.6f} "
              f"(epoch {best_epoch}) saved: {ckpt_path}")

    best_frac = min(sweep, key=lambda f: sweep[f]["val_loss"])
    print(f"\n[sweep] === SELECTION ===")
    for f, r in sweep.items():
        print(f"  frac={f}: val_loss={r['val_loss']:.6f}"
              f"{' <<<' if f == best_frac else ''}")
    print(f"[sweep] Selected: frac={best_frac}")

    json.dump({str(f): {"val_loss": r["val_loss"],
                        "best_epoch": r["best_epoch"]}
               for f, r in sweep.items()},
              open(os.path.join(ckpt_dir, "sweep_results.json"), "w"), indent=2)

    sweep["selected_frac"] = best_frac
    sweep["history"] = all_history
    return sweep


def retrain_final(full_data: List, best_frac: float, hidden_dim: int,
                  device: torch.device, cfg: dict,
                  ckpt_dir: str, layer_index=None,
                  seed=None) -> Tuple[ConfidenceHead, List[Dict]]:
    """
    After layer selection, retrain on full data (train + val).
    Evaluate on test ONCE after this — no further tuning.
    Returns (head, history).
    """
    print(f"\n[trainer] Retraining final head on full data "
          f"(train+val, n={len(full_data)}) | layer={best_frac}")

    head = ConfidenceHead(hidden_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)
    opt  = AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng  = np.random.default_rng(cfg.get("seed", 42))
    history = []

    for epoch in range(cfg["epochs_final"]):
        order   = rng.permutation(len(full_data))
        batches = [list(order[i:i+cfg["batch_size"]])
                   for i in range(0, len(order), cfg["batch_size"])]
        head.train()
        tot, bri, ec, gaps = [], [], [], []
        for batch_idx in batches:
            batch = [full_data[i] for i in batch_idx]
            opt.zero_grad()
            loss, b, e, gap = compute_loss(batch, best_frac, head, device, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["grad_clip"])
            opt.step()
            tot.append(float(loss)); bri.append(b); ec.append(e); gaps.append(gap)

        history.append({
            "phase": "final", "layer_frac": best_frac, "layer_index": layer_index,
            "seed": seed, "epoch": epoch + 1,
            "train_loss": float(np.mean(tot)),
            "train_brier": float(np.mean(bri)),
            "train_ec": float(np.mean(ec)),
            "train_gap": float(np.mean(gaps)),
            "val_loss": None, "val_brier": None, "val_ec": None, "val_gap": None,
            "lr": float(opt.param_groups[0]["lr"]),
            "is_best": None,   # no val set here by design
        })
        print(f"  [final ep{epoch+1}] train_loss={np.mean(tot):.5f} "
              f"gap={np.mean(gaps):.4f}")

    final_ckpt = os.path.join(ckpt_dir, "final_head.pt")
    save_checkpoint(final_ckpt, head, layer_frac=best_frac,
                    layer_index=layer_index, seed=seed,
                    epoch=cfg["epochs_final"], phase="final",
                    val_loss=None, cfg=cfg,
                    extra={"n_train_full": len(full_data)})
    print(f"[trainer] Final head saved: {final_ckpt}")
    return head, history
