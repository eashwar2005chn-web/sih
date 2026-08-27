"""
DRDO PS 26052: Real-Time Tactical ANC Training Engine
Optimized for NVIDIA RTX 5060 (CUDA) using Mixed Precision (AMP FP16).
Logs SI-SNR, STOI, PESQ, SNR on validation split per epoch.
Saves best model checkpoints and export-ready weights.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from data.dataset_builder import get_dataloaders
from model.causal_anc_net import build_causal_anc_model, CausalANCNet
from model.loss import HybridANCLoss, compute_snr, compute_si_snr, compute_stoi, compute_pesq


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: str,
    epoch: int,
    grad_clip: float = 5.0
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mrstft = 0.0
    total_sisnr = 0.0
    total_complex = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mrstft += losses["mrstft_loss"].item()
        total_sisnr += losses["sisnr_loss"].item()
        total_complex += losses["complex_loss"].item()

        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0 or (batch_idx + 1) == len(train_loader):
            step_loss = total_loss / (batch_idx + 1)
            print(f"  [Epoch {epoch:02d} | Step {batch_idx+1:03d}/{len(train_loader):03d}] Loss: {step_loss:.4f} (MR-STFT: {total_mrstft/(batch_idx+1):.4f}, SI-SNR: {total_sisnr/(batch_idx+1):.4f})")

    elapsed = time.time() - start_time
    n_batches = len(train_loader)
    return {
        "loss": total_loss / n_batches,
        "mrstft": total_mrstft / n_batches,
        "sisnr": total_sisnr / n_batches,
        "complex": total_complex / n_batches,
        "time_sec": elapsed
    }


def validate_epoch(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: str
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0

    op_in, op_out, op_gains, op_stois, op_pesqs = [], [], [], [], []
    stress_in, stress_out, stress_gains, stress_stois, stress_pesqs = [], [], [], [], []
    all_gains, all_stois, all_pesqs = [], [], []

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

            # Compute objective metrics on CPU
            noisy_np = noisy.cpu().numpy()
            clean_np = clean.cpu().numpy()
            enh_np = enhanced.cpu().numpy()

            for i in range(len(clean_np)):
                raw_snr = compute_snr(clean_np[i], noisy_np[i])
                enh_snr = compute_snr(clean_np[i], enh_np[i])
                gain = enh_snr - raw_snr
                stoi_val = compute_stoi(clean_np[i], enh_np[i])
                pesq_val = compute_pesq(clean_np[i], enh_np[i])

                all_gains.append(gain)
                all_stois.append(stoi_val)
                all_pesqs.append(pesq_val)

                if raw_snr >= 0.0:  # Operational Range (0 to +15 dB SNR)
                    op_in.append(raw_snr)
                    op_out.append(enh_snr)
                    op_gains.append(gain)
                    op_stois.append(stoi_val)
                    op_pesqs.append(pesq_val)
                else:  # Degraded Channel Stress Range (-10 to 0 dB SNR)
                    stress_in.append(raw_snr)
                    stress_out.append(enh_snr)
                    stress_gains.append(gain)
                    stress_stois.append(stoi_val)
                    stress_pesqs.append(pesq_val)

    n_batches = len(val_loader)
    return {
        "val_loss": total_loss / n_batches,
        # Operational Range (DRDO Table 1: 0 to +15 dB SNR)
        "op_output_snr": float(np.mean(op_out)) if op_out else 0.0,
        "op_input_snr": float(np.mean(op_in)) if op_in else 0.0,
        "op_snr_gain": float(np.mean(op_gains)) if op_gains else 0.0,
        "op_stoi": float(np.mean(op_stois)) if op_stois else 0.0,
        "op_pesq": float(np.mean(op_pesqs)) if op_pesqs else 0.0,
        # Stress Range (DRDO Table 2: -10 to 0 dB SNR)
        "stress_snr_gain": float(np.mean(stress_gains)) if stress_gains else 0.0,
        "stress_stoi": float(np.mean(stress_stois)) if stress_stois else 0.0,
        # Overall
        "snr_gain_db": float(np.mean(all_gains)),
        "stoi": float(np.mean(all_stois)),
        "pesq": float(np.mean(all_pesqs))
    }


def main():
    parser = argparse.ArgumentParser(description="DRDO PS 26052 ANC Model Training on RTX 5060")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (tuned for 16GB RAM + RTX 5060)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden state dimension")
    parser.add_argument("--train_steps", type=int, default=1200, help="Samples per epoch for training")
    parser.add_argument("--val_steps", type=int, default=200, help="Samples for validation")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience in epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints/full_run_clamped_alpha", help="Directory to save checkpoints")
    parser.add_argument("--clean_dir", type=str, default=None, help="Optional clean speech directory")
    parser.add_argument("--dataset_mode", type=str, default="synthetic", choices=["synthetic", "real", "hybrid"], help="Dataset pipeline mode")
    parser.add_argument("--hybrid_ratio", type=float, default=0.5, help="Ratio of real to synthetic data in hybrid mode")
    parser.add_argument("--data_root", type=str, default=".", help="Root directory for downloaded real datasets")
    parser.add_argument("--sisnr_mode", type=str, default="clamped_alpha", choices=["clamped_alpha", "hinge_penalty", "standard"], help="SISNR phase handling mode")
    parser.add_argument("--hinge_weight", type=float, default=10.0, help="Hinge penalty weight for negative correlation")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    args = parser.parse_args()

    # Set global seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print("================================================================================")
    print(" DRDO PS 26052: AI/ML Tactical Adaptive Noise Cancellation Training Pipeline")
    print(f" Target Device : {args.device.upper()} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")
    print(f" Configuration : Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, HiddenDim={args.hidden_dim}, SISNRMode={args.sisnr_mode}, Mode={args.dataset_mode} (HybridRatio={args.hybrid_ratio}), Seed={args.seed}")
    print("================================================================================")

    # 1. Dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        mode=args.dataset_mode,
        batch_size=args.batch_size,
        train_epoch_size=args.train_steps,
        val_epoch_size=args.val_steps,
        clean_speech_dir=args.clean_dir,
        hybrid_ratio=args.hybrid_ratio,
        data_root=args.data_root,
        seed=args.seed
    )

    # 2. Model & Optimization
    model = build_causal_anc_model(hidden_dim=args.hidden_dim, num_gru_layers=2).to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model Architecture: Causal-CRN | Trainable Parameters: {num_params:,} ({num_params*4/(1024*1024):.2f} MB FP32)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_fn = HybridANCLoss(sisnr_mode=args.sisnr_mode, hinge_weight=args.hinge_weight).to(args.device)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    best_score = -999.0
    epochs_no_improve = 0
    history = []

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}] (LR: {scheduler.get_last_lr()[0]:.6f})")
        train_stats = train_epoch(model, train_loader, optimizer, loss_fn, scaler, args.device, epoch)
        val_stats = validate_epoch(model, val_loader, loss_fn, args.device)
        scheduler.step()

        # Checkpoint best prioritized by Operational Output SNR (DRDO Table 1 Primary Criterion)
        # Score = OP Output SNR + 10.0 * OP STOI + 5.0 * OP PESQ
        op_score = float(val_stats["op_output_snr"] + 10.0 * val_stats["op_stoi"] + 5.0 * val_stats["op_pesq"])

        print(f" -> Validation: Loss={val_stats['val_loss']:.4f} | OP Score={op_score:.2f} | OP Out SNR={val_stats['op_output_snr']:+.2f}dB (Gain: {val_stats['op_snr_gain']:+.2f}dB) | OP STOI={val_stats['op_stoi']:.4f} | OP PESQ={val_stats['op_pesq']:.2f} | Stress Gain={val_stats['stress_snr_gain']:+.2f}dB")

        # Save record
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["val_loss"],
            "op_score": op_score,
            "op_output_snr": val_stats["op_output_snr"],
            "op_snr_gain": val_stats["op_snr_gain"],
            "op_stoi": val_stats["op_stoi"],
            "op_pesq": val_stats["op_pesq"],
            "stress_snr_gain": val_stats["stress_snr_gain"],
            "snr_gain_db": val_stats["snr_gain_db"],
            "stoi": val_stats["stoi"],
            "pesq": val_stats["pesq"]
        }
        history.append(epoch_record)

        # Checkpoint latest and per-epoch snapshot
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden_dim": args.hidden_dim,
            "val_stats": val_stats
        }
        torch.save(ckpt_payload, os.path.join(args.save_dir, "latest_model.pt"))
        torch.save(ckpt_payload, os.path.join(args.save_dir, f"epoch_{epoch:02d}.pt"))

        # Checkpoint best prioritized by Operational Output SNR (DRDO Table 1 Primary Criterion)
        # Score = OP Output SNR + 10.0 * OP STOI + 5.0 * OP PESQ
        op_score = val_stats["op_output_snr"] + 10.0 * val_stats["op_stoi"] + 5.0 * val_stats["op_pesq"]
        if op_score > best_score:
            best_score = op_score
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "hidden_dim": args.hidden_dim,
                "val_stats": val_stats
            }, os.path.join(args.save_dir, "best_model.pt"))
            print(f"   [SAVED BEST MODEL] OP Score: {op_score:.2f} | OP Out SNR: {val_stats['op_output_snr']:+.2f}dB | OP STOI: {val_stats['op_stoi']:.4f} | OP PESQ: {val_stats['op_pesq']:.2f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n[Early Stopping Triggered] No improvement in operational score for {args.patience} consecutive epochs.")
                break

    # Save training history JSON
    with open(os.path.join(args.save_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n================================================================================")
    print(f" Training Complete! Best Model Saved to: {os.path.join(args.save_dir, 'best_model.pt')}")
    print("================================================================================")


if __name__ == "__main__":
    main()
