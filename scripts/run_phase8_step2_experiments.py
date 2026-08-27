"""
DRDO PS 26052 — Phase 8 Step 2 Fine-Tuning Experiments & 5-Bucket Evaluation
Tests:
1. Exp 2a: Causal Frame Normalization (Per-frame GroupNorm, zero future-leak)
2. Exp 2b: Loss Fix (Hinge SI-SNR + Low-SNR loss upweighting)
3. Exp 2c: Widened Mask Bound (K=2.5 complex ratio mask scaling)
4. Exp 2d: Combined (Causal Norm + Widened Mask + Loss Weighting)

All fine-tuned for 5 epochs (1200 steps/epoch) starting from checkpoints/task4_full_run/best_model.pt.
Evaluated via eval/metrics_common.py on Held-Out Test Split (N=500).
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import get_hybrid_dataloaders, HybridRealDataset
from model.causal_anc_net import CausalANCNet, build_causal_anc_model
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def load_baseline_weights_into_model(target_model: nn.Module, baseline_ckpt_path: str):
    """Safely transfers all compatible weights from baseline checkpoint into target model."""
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    base_sd = ckpt["model_state_dict"]
    target_sd = target_model.state_dict()

    transferred = 0
    for k, v in base_sd.items():
        if k in target_sd and target_sd[k].shape == v.shape:
            target_sd[k] = v
            transferred += 1
        elif "bn" in k and hasattr(target_model, "norm_type") and target_model.norm_type == "group":
            # Map BatchNorm weight/bias to GroupNorm weight/bias
            gn_k = k.replace(".enc_bn", ".enc_bn.norm.").replace(".dec_bn", ".dec_bn.norm.")
            # If layer name format matches
            for t_k in target_sd.keys():
                if k.split(".")[0] in t_k and k.split(".")[-1] in t_k and target_sd[t_k].shape == v.shape:
                    target_sd[t_k] = v
                    transferred += 1
                    break

    target_model.load_state_dict(target_sd, strict=False)
    print(f"   [Weight Transfer] Transferred {transferred} weight tensors from {baseline_ckpt_path}")


def train_finetune_pilot(
    exp_name: str,
    save_dir: str,
    norm_type: str = "batch",
    mask_bound: float = 1.0,
    sisnr_mode: str = "clamped_alpha",
    use_loss_weighting: bool = False,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
    train_steps: int = 1200,
    val_steps: int = 100,
    device: str = "cuda"
):
    os.makedirs(save_dir, exist_ok=True)
    baseline_ckpt = "checkpoints/task4_full_run/best_model.pt"

    print(f"\n================================================================================")
    print(f" Running Phase 8 Fine-Tune Pilot: {exp_name}")
    print(f" Norm: {norm_type} | Mask Bound: {mask_bound} | SI-SNR Mode: {sisnr_mode} | Weighted Loss: {use_loss_weighting}")
    print(f" Epochs: {epochs} | Steps/Epoch: {train_steps} | LR: {lr} | Save: {save_dir}")
    print(f"================================================================================")

    # 1. Build Model
    model = CausalANCNet(
        hidden_dim=128,
        num_gru_layers=2,
        norm_type=norm_type,
        mask_bound=mask_bound
    ).to(device)

    load_baseline_weights_into_model(model, baseline_ckpt)

    # 2. Dataloaders
    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=batch_size,
        train_epoch_size=train_steps,
        val_epoch_size=val_steps,
        hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced",
        seed=42
    )

    # 3. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = HybridANCLoss(sisnr_mode=sisnr_mode, hinge_weight=5.0).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_loss = 999.0

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

                if use_loss_weighting and "snr_db" in batch:
                    # Upweight lower SNRs: w = 1.0 + clamp(-snr/10, min=0, max=1.5)
                    snr_vals = batch["snr_db"].to(device)
                    weights = 1.0 + torch.clamp(-snr_vals / 10.0, min=0.0, max=1.5)
                    loss = loss * weights.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        scheduler.step()
        train_loss = total_train_loss / max(1, len(train_loader))

        # Validate
        model.eval()
        val_loss = 0.0
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
                    val_loss += losses["total_loss"].item()
        val_loss /= max(1, len(val_loader))
        elapsed = time.time() - t0

        print(f"[{exp_name}] Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "norm_type": norm_type,
                "mask_bound": mask_bound,
                "hidden_dim": 128,
                "val_loss": val_loss
            }, best_path)

    print(f"   >>> Best model saved to {best_path} (Val Loss: {best_val_loss:.4f})")
    return model


def evaluate_candidate_model(model, device: str = "cuda"):
    model.to(device)
    model.eval()

    buckets = [
        ("Stress [-10, -5) dB", -10.0, -5.0, 7000),
        ("Stress [-5, 0) dB", -5.0, 0.0, 7500),
        ("Operational [0, 5) dB", 0.0, 5.0, 5000),
        ("Operational [5, 10) dB", 5.0, 10.0, 5500),
        ("Operational [10, 15] dB", 10.0, 15.0, 6000)
    ]

    bucket_aggs = {}
    op_records = []
    stress_records = []

    for name, min_snr, max_snr, seed in buckets:
        ds = HybridRealDataset(
            mode="hybrid",
            split="test",
            sample_rate=16000,
            segment_len_sec=2.0,
            epoch_size=100,
            snr_range=(min_snr, max_snr),
            hybrid_ratio=0.85,
            seed=seed
        )
        b_records = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()

                enh, _, _, _ = model(noisy)
                enh_np = enh.squeeze(0).cpu().numpy()

                r = evaluate_batch_metrics(clean, noisy_np, enh_np)
                b_records.append(r)
                if min_snr >= 0.0:
                    op_records.append(r)
                else:
                    stress_records.append(r)

        bucket_aggs[name] = aggregate_metric_records(b_records)

    pooled_op = aggregate_metric_records(op_records)
    pooled_stress = aggregate_metric_records(stress_records)

    return {
        "pooled_op": pooled_op,
        "pooled_stress": pooled_stress,
        "buckets": bucket_aggs
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Run 4 Fine-Tune Pilots (5 epochs each)
    m_2a = train_finetune_pilot(
        exp_name="Exp 2a (Causal Frame Norm)",
        save_dir="checkpoints/phase8_ft_causal_norm",
        norm_type="group",
        mask_bound=1.0,
        sisnr_mode="clamped_alpha",
        use_loss_weighting=False,
        epochs=5,
        device=device
    )

    m_2b = train_finetune_pilot(
        exp_name="Exp 2b (Loss Weighting & Hinge)",
        save_dir="checkpoints/phase8_ft_loss_weighting",
        norm_type="batch",
        mask_bound=1.0,
        sisnr_mode="hinge_penalty",
        use_loss_weighting=True,
        epochs=5,
        device=device
    )

    m_2c = train_finetune_pilot(
        exp_name="Exp 2c (Widened Mask K=2.5)",
        save_dir="checkpoints/phase8_ft_widened_mask",
        norm_type="batch",
        mask_bound=2.5,
        sisnr_mode="clamped_alpha",
        use_loss_weighting=False,
        epochs=5,
        device=device
    )

    m_2d = train_finetune_pilot(
        exp_name="Exp 2d (Combined Norm+Mask+Loss)",
        save_dir="checkpoints/phase8_ft_combined",
        norm_type="group",
        mask_bound=2.5,
        sisnr_mode="hinge_penalty",
        use_loss_weighting=True,
        epochs=5,
        device=device
    )

    # 2. Load Baseline for Direct 5-Way Comparison
    base_ckpt = torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")
    m_base = build_causal_anc_model(hidden_dim=128).to(device)
    m_base.load_state_dict(base_ckpt["model_state_dict"])

    print("\n================================================================================")
    print(" Evaluating All Candidate Models on Held-Out Test Split (N=500 per model)...")
    print("================================================================================")

    e_base = evaluate_candidate_model(m_base, device=device)
    e_2a = evaluate_candidate_model(m_2a, device=device)
    e_2b = evaluate_candidate_model(m_2b, device=device)
    e_2c = evaluate_candidate_model(m_2c, device=device)
    e_2d = evaluate_candidate_model(m_2d, device=device)

    # TABLE 1: Aggregate Operational Range (0 to 15 dB)
    models = [
        ("Baseline (Task 4)", e_base),
        ("2a: Causal FrameNorm", e_2a),
        ("2b: Loss Weighting", e_2b),
        ("2c: Widened Mask (K=2.5)", e_2c),
        ("2d: Combined", e_2d),
    ]

    t1 = []
    for name, e in models:
        t1.append([
            name,
            f"{e['pooled_op']['mean_out_snr']:.2f} ± {e['pooled_op']['std_out_snr']:.2f} dB",
            f"{e['pooled_op']['pass_snr_pct']:.1f}% [{e['pooled_op']['pass_snr_ci'][0]:.1f}%, {e['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{e['pooled_op']['mean_stoi']:.4f}",
            f"{e['pooled_op']['mean_pesq']:.2f}",
            f"{e['pooled_stress']['mean_out_snr']:.2f} dB ({e['pooled_stress']['mean_gain']:+.2f} dB)"
        ])

    print("\n--- TABLE 1: Phase 8 Candidate Comparison (Aggregate Operational 0-15 dB & Stress -10-0 dB) ---")
    h1 = ["Experiment / Intervention", "Operational Output SNR", "SNR >15dB Pass Rate", "Operational STOI", "PESQ (approx)", "Stress SNR (Gain)"]
    print(tabulate(t1, headers=h1, tablefmt="grid"))

    # TABLE 2: 5-Bucket Breakdown
    bucket_names = [
        "Stress [-10, -5) dB", "Stress [-5, 0) dB",
        "Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"
    ]
    t2 = []
    for b in bucket_names:
        t2.append([
            b,
            f"{e_base['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e_2a['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e_2b['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e_2c['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e_2d['buckets'][b]['mean_out_snr']:.2f} dB"
        ])
    print("\n--- TABLE 2: Disaggregated 5-Bucket Output SNR Across All Interventions ---")
    h2 = ["SNR Bucket", "Baseline", "2a (Causal Norm)", "2b (Loss Weight)", "2c (Mask K=2.5)", "2d (Combined)"]
    print(tabulate(t2, headers=h2, tablefmt="grid"))


if __name__ == "__main__":
    main()
