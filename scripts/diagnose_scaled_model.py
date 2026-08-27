"""
Diagnostic script for CausalANCScaledNet:
Checks per-layer gradient flow, activation ranges, mask distributions, and loss computation.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_anc_net import build_causal_anc_model, build_scaled_causal_anc_model
from model.loss import HybridANCLoss
from data.real_dataset_loader import get_hybrid_dataloaders


def run_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("================================================================================")
    print(" DRDO PS 26052: Architecture & Gradient Flow Diagnostic Audit")
    print("================================================================================")

    # 1. Instantiate both models
    base_model = build_causal_anc_model().to(device)
    scaled_model = build_scaled_causal_anc_model().to(device)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)

    # 2. Get 1 batch of data
    train_loader, _, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=16,
        train_epoch_size=16,
        snr_sampler_mode="rebalanced",
        seed=42
    )
    batch = next(iter(train_loader))
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)

    # 3. Test Scaled Model Forward
    scaled_model.train()
    scaled_model.zero_grad()

    enh_scaled, enh_r, enh_i, _ = scaled_model(noisy)
    clean_r, clean_i, _ = scaled_model.stft_forward(clean)
    noisy_r, noisy_i, _ = scaled_model.stft_forward(noisy)

    losses_scaled = loss_fn(
        pred_wav=enh_scaled,
        target_wav=clean,
        pred_real=enh_r,
        pred_imag=enh_i,
        target_real=clean_r,
        target_imag=clean_i
    )
    total_loss = losses_scaled["total_loss"]
    total_loss.backward()

    print(f"\n[Initial Loss Components - Scaled Model]:")
    for k, v in losses_scaled.items():
        print(f"  {k:15s}: {v.item():.4f}")

    # Inspect Mask Output Range
    _, _, mask_r, mask_i, _ = scaled_model.forward_spec(noisy_r, noisy_i)
    print(f"\n[Complex Mask Statistics (Tanh bounded in [-1, 1])]:")
    print(f"  mask_real: min={mask_r.min().item():+.4f}, max={mask_r.max().item():+.4f}, mean={mask_r.mean().item():+.4f}, std={mask_r.std().item():.4f}")
    print(f"  mask_imag: min={mask_i.min().item():+.4f}, max={mask_i.max().item():+.4f}, mean={mask_i.mean().item():+.4f}, std={mask_i.std().item():.4f}")

    # Inspect Per-Layer Gradient Norms
    print(f"\n[Per-Layer Gradient Norms (Scaled Model)]:")
    print(f"  {'Layer Name':45s} | {'Param Shape':22s} | {'Grad Norm':12s}")
    print("  " + "-" * 85)
    for name, param in scaled_model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            status = "OK" if 1e-5 < gnorm < 100.0 else ("VANISHING" if gnorm <= 1e-5 else "EXPLODING")
            print(f"  {name:45s} | {str(list(param.shape)):22s} | {gnorm:10.6f} ({status})")
        else:
            print(f"  {name:45s} | {str(list(param.shape)):22s} | NO GRADIENT!")

    # Check Base Model for comparison
    base_model.train()
    base_model.zero_grad()
    enh_base, base_r, base_i, _ = base_model(noisy)
    b_clean_r, b_clean_i, _ = base_model.stft_forward(clean)
    losses_base = loss_fn(
        pred_wav=enh_base,
        target_wav=clean,
        pred_real=base_r,
        pred_imag=base_i,
        target_real=b_clean_r,
        target_imag=b_clean_i
    )
    losses_base["total_loss"].backward()

    print(f"\n[Initial Loss Components - Base Model (for comparison)]:")
    for k, v in losses_base.items():
        print(f"  {k:15s}: {v.item():.4f}")


if __name__ == "__main__":
    run_diagnostics()
