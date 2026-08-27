"""
Decisive Full 50-Epoch Training Script for CausalANCScaledNet (DRDO PS 26052)
Uses LR=1e-3, T_max=50 Cosine Annealing, 1200 steps/epoch, Rebalanced SNR Sampler.
Saves checkpoints to checkpoints/pilot_scaled_capacity_v2/
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data.real_dataset_loader import get_hybrid_dataloaders
from model.causal_anc_net import build_scaled_causal_anc_model
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    records = []

    with torch.no_grad():
        for batch in val_loader:
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                enhanced, enh_r, enh_i, _ = model(noisy)
                clean_r, clean_i, _ = model.stft_forward(clean)
                losses = loss_fn(
                    pred_wav=enhanced,
                    target_wav=clean,
                    pred_real=enh_r,
                    pred_imag=enh_i,
                    target_real=clean_r,
                    target_imag=clean_i
                )
                total_loss += losses["total_loss"].item()

            clean_np = clean.cpu().numpy()
            noisy_np = noisy.cpu().numpy()
            enh_np = enhanced.cpu().numpy()

            for i in range(len(clean_np)):
                r = evaluate_batch_metrics(clean_np[i], noisy_np[i], enh_np[i])
                records.append(r)

    agg = aggregate_metric_records(records)
    agg["val_loss"] = total_loss / max(1, len(val_loader))
    return agg


def train_scaled_v2(
    save_dir: str = "checkpoints/pilot_scaled_capacity_v2",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    train_steps: int = 1200,
    val_steps: int = 200,
    hybrid_ratio: float = 0.85,
    seed: int = 42
):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Build Scaled Model
    model = build_scaled_causal_anc_model(hidden_dim=256, num_gru_layers=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("================================================================================")
    print(" DRDO PS 26052: CausalANCScaledNet Full 50-Epoch Decisive Training Run (v2)")
    print(f" Model Variant        : CausalANCScaledNet (Wide C-CRN, 32-64-128 convs, 256 GRU)")
    print(f" Trainable Parameters : {num_params:,} ({num_params*4/(1024*1024):.2f} MB FP32)")
    print(f" Save Directory       : {save_dir}")
    print(f" Schedule & Optimizer : LR = {lr} | Cosine Decay (T_max={epochs}, eta_min=1e-5) | AdamW")
    print(f" Epochs & Batch Size  : {epochs} Epochs | {train_steps} Steps/Epoch | BatchSize={batch_size}")
    print(f" Data Distribution    : Rebalanced SNR Sampler (-10 to 15 dB)")
    print("================================================================================")

    # 2. Dataloaders
    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=batch_size,
        train_epoch_size=train_steps,
        val_epoch_size=val_steps,
        hybrid_ratio=hybrid_ratio,
        snr_sampler_mode="rebalanced",
        seed=seed
    )

    # 3. Optimizer Setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_score = -999.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                enhanced, enh_r, enh_i, _ = model(noisy)
                clean_r, clean_i, _ = model.stft_forward(clean)
                losses = loss_fn(
                    pred_wav=enhanced,
                    target_wav=clean,
                    pred_real=enh_r,
                    pred_imag=enh_i,
                    target_real=clean_r,
                    target_imag=clean_i
                )
                loss = losses["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        scheduler.step()
        train_loss = total_train_loss / max(1, len(train_loader))
        val_stats = validate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        op_score = val_stats["mean_out_snr"] + 10.0 * val_stats["mean_stoi"] + 5.0 * val_stats["mean_pesq"]
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s, lr={current_lr:.1e}) | Train Loss: {train_loss:.4f} | Val Loss: {val_stats['val_loss']:.4f} | Val SNR: {val_stats['mean_out_snr']:+.2f}dB | STOI: {val_stats['mean_stoi']:.4f} | PESQ: {val_stats['mean_pesq']:.2f} | OP Score: {op_score:.2f}")

        # Checkpoint per epoch
        epoch_path = os.path.join(save_dir, f"epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "hidden_dim": 256, "num_params": num_params, "op_score": op_score}, epoch_path)

        if op_score > best_score:
            best_score = op_score
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "hidden_dim": 256, "num_params": num_params, "op_score": op_score}, best_path)
            print(f"   >>> [NEW BEST MODEL] Saved to {best_path} (OP Score: {op_score:.2f})")

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_stats["val_loss"],
            "val_out_snr": val_stats["mean_out_snr"],
            "val_stoi": val_stats["mean_stoi"],
            "val_pesq": val_stats["mean_pesq"],
            "op_score": op_score
        })

    with open(os.path.join(save_dir, "scaled_v2_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Full history and checkpoints saved to {save_dir}")


if __name__ == "__main__":
    train_scaled_v2()
