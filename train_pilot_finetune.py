"""
Pilot Fine-Tuning Script: Rebalanced SNR Distribution (DRDO PS 26052)
Fine-tunes existing epoch_50.pt weights for 15 epochs using rebalanced SNR sampling.
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
from model.causal_anc_net import build_causal_anc_model
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


def run_pilot_finetune(
    base_checkpoint: str = "checkpoints/task4_full_run/best_model.pt",
    save_dir: str = "checkpoints/pilot_rebalanced_snr",
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 16,
    train_steps: int = 1200,
    val_steps: int = 200,
    hybrid_ratio: float = 0.85,
    seed: int = 42
):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("================================================================================")
    print(" DRDO PS 26052: Bounded Pilot Fine-Tuning with Rebalanced SNR Sampling")
    print(f" Base Weights : {base_checkpoint}")
    print(f" Save Dir     : {save_dir}")
    print(f" Epochs       : {epochs} | LR = {lr} | Train Steps = {train_steps} | Seed = {seed}")
    print("================================================================================")

    # 1. Load Data with Rebalanced SNR Sampler
    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=batch_size,
        train_epoch_size=train_steps,
        val_epoch_size=val_steps,
        hybrid_ratio=hybrid_ratio,
        snr_sampler_mode="rebalanced",
        seed=seed
    )

    # 2. Build Model & Load Weights
    ckpt = torch.load(base_checkpoint, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = build_causal_anc_model(hidden_dim=hidden_dim, num_gru_layers=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f" Loaded base model weights successfully (HiddenDim={hidden_dim})")

    # 3. Optimization Setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Initial Baseline Validation
    init_stats = validate(model, val_loader, loss_fn, device)
    print(f"\n[Baseline Epoch 50 Validation on Val Split]: ValLoss={init_stats['val_loss']:.4f} | OutSNR={init_stats['mean_out_snr']:.2f}dB | STOI={init_stats['mean_stoi']:.4f} | PESQ={init_stats['mean_pesq']:.2f}")

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
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_stats['val_loss']:.4f} | Val Out SNR: {val_stats['mean_out_snr']:+.2f}dB | STOI: {val_stats['mean_stoi']:.4f} | PESQ: {val_stats['mean_pesq']:.2f} | OP Score: {op_score:.2f}")

        # Checkpoint per epoch
        epoch_path = os.path.join(save_dir, f"epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "hidden_dim": hidden_dim}, epoch_path)

        if op_score > best_score:
            best_score = op_score
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "hidden_dim": hidden_dim, "op_score": op_score}, best_path)
            print(f"   >>> [NEW BEST PILOT MODEL] Saved to {best_path}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["val_loss"],
            "val_out_snr": val_stats["mean_out_snr"],
            "val_stoi": val_stats["mean_stoi"],
            "val_pesq": val_stats["mean_pesq"],
            "op_score": op_score
        })

    with open(os.path.join(save_dir, "pilot_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nPilot fine-tuning complete. Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=1200)
    args = parser.parse_args()

    run_pilot_finetune(epochs=args.epochs, lr=args.lr, train_steps=args.train_steps)
