"""
Phase 10 Step 1 & 1b: 15-Epoch K-Screening Suite (Static & Annealed Mask Bounds)
Trains 5 candidates from scratch (15 epochs each, 1200 steps/epoch, LR=1e-3 cosine decay):
1. Static K=1.5
2. Static K=2.0
3. Static K=2.5
4. Annealed K -> 2.0 (K=1.0 for ep 1-3, ramps to 2.0 ep 4-9, holds 2.0 ep 10-15)
5. Annealed K -> 3.0 (K=1.0 for ep 1-3, ramps to 3.0 ep 4-9, holds 3.0 ep 10-15)

Evaluates all models on the held-out test split (N=500) via eval/metrics_common.py.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import get_hybrid_dataloaders, HybridRealDataset
from model.causal_anc_net import CausalANCNet, build_causal_anc_model
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def train_screen_candidate(
    cand_name: str,
    save_dir: str,
    target_k: float,
    anneal: bool = False,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 16,
    train_steps: int = 1200,
    val_steps: int = 100,
    seed: int = 42,
    device: str = "cuda"
):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    init_k = 1.0 if anneal else target_k

    print(f"\n================================================================================")
    print(f" Training Screening Candidate: {cand_name}")
    print(f" Mask Mode: polar_tanh | Target K: {target_k} | Annealed: {anneal} (Init K: {init_k})")
    print(f" Epochs: {epochs} | Steps/Epoch: {train_steps} | LR: {lr} | Save: {save_dir}")
    print(f"================================================================================")

    model = CausalANCNet(
        hidden_dim=128,
        num_gru_layers=2,
        norm_type="batch",
        mask_bound=init_k,
        mask_mode="polar_tanh"
    ).to(device)

    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=batch_size,
        train_epoch_size=train_steps,
        val_epoch_size=val_steps,
        hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced",
        seed=seed
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_score = -999.0

    for epoch in range(1, epochs + 1):
        # Update K if annealing
        if anneal:
            if epoch <= 3:
                curr_k = 1.0
            elif epoch <= 9:
                progress = (epoch - 3) / (9 - 3)
                curr_k = 1.0 + progress * (target_k - 1.0)
            else:
                curr_k = target_k
            model.mask_bound = curr_k
        else:
            curr_k = target_k

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

        # Validate
        model.eval()
        val_loss = 0.0
        val_snrs = []
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

                clean_np = clean.cpu().numpy()
                noisy_np = noisy.cpu().numpy()
                enh_np = enhanced.cpu().numpy()
                for i in range(len(clean_np)):
                    r = evaluate_batch_metrics(clean_np[i], noisy_np[i], enh_np[i])
                    val_snrs.append(r["enh_snr"])

        val_loss /= max(1, len(val_loader))
        val_snr_mean = float(np.mean(val_snrs))
        elapsed = time.time() - t0

        print(f"[{cand_name}] Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s, K={curr_k:.2f}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SNR: {val_snr_mean:+.2f} dB")

        if val_snr_mean > best_score:
            best_score = val_snr_mean
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "hidden_dim": 128,
                "mask_bound": target_k,
                "mask_mode": "polar_tanh",
                "val_snr": val_snr_mean
            }, best_path)

    print(f"   >>> Candidate {cand_name} finished. Best model saved to {best_path} (Val SNR: {best_score:.2f} dB)")
    return model


def evaluate_candidate(model, device: str = "cuda"):
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

    candidates = [
        ("Static K=1.5", "checkpoints/screen_polar_k15", 1.5, False),
        ("Static K=2.0", "checkpoints/screen_polar_k20", 2.0, False),
        ("Static K=2.5", "checkpoints/screen_polar_k25", 2.5, False),
        ("Annealed K->2.0", "checkpoints/screen_annealed_k20", 2.0, True),
        ("Annealed K->3.0", "checkpoints/screen_annealed_k30", 3.0, True),
    ]

    trained_models = {}
    for name, s_dir, k_val, anneal in candidates:
        m = train_screen_candidate(
            cand_name=name,
            save_dir=s_dir,
            target_k=k_val,
            anneal=anneal,
            epochs=15,
            device=device
        )
        trained_models[name] = m

    # Also load baseline for reference
    base_ckpt = torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")
    m_base = build_causal_anc_model(hidden_dim=128).to(device)
    m_base.load_state_dict(base_ckpt["model_state_dict"])
    trained_models["Baseline (K=1.0, 50 ep)"] = m_base

    # Load Phase 9 K=3.0 scratch for comparison
    p9_ckpt = torch.load("checkpoints/phase9_polar_k3_scratch/best_model.pt", map_location="cpu")
    m_p9 = CausalANCNet(hidden_dim=128, mask_bound=3.0, mask_mode="polar_tanh").to(device)
    m_p9.load_state_dict(p9_ckpt["model_state_dict"])
    trained_models["Phase 9 (K=3.0, 50 ep)"] = m_p9

    print("\n================================================================================")
    print(" Evaluating All Screening Candidates on Held-Out Test Split (N=500 per model)...")
    print("================================================================================")

    eval_results = {}
    for name, m in trained_models.items():
        eval_results[name] = evaluate_candidate(m, device=device)

    # TABLE 1: Operational & Stress Range Summary
    t1 = []
    for name, e in eval_results.items():
        t1.append([
            name,
            f"{e['pooled_op']['mean_out_snr']:.2f} ± {e['pooled_op']['std_out_snr']:.2f} dB",
            f"{e['pooled_op']['pass_snr_pct']:.1f}% [{e['pooled_op']['pass_snr_ci'][0]:.1f}%, {e['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{e['pooled_op']['mean_stoi']:.4f}",
            f"{e['pooled_op']['mean_pesq']:.2f}",
            f"{e['pooled_stress']['mean_out_snr']:.2f} dB ({e['pooled_stress']['mean_gain']:+.2f} dB)"
        ])

    h1 = ["Candidate", "Op Output SNR (0-15dB)", "SNR >15dB Pass Rate", "Op STOI", "PESQ", "Stress SNR (Gain)"]
    print("\n--- TABLE 1: Phase 10 Screening Suite Summary (15 Epochs Scratch vs Baselines) ---")
    print(tabulate(t1, headers=h1, tablefmt="grid"))

    # TABLE 2: Disaggregated 5-Bucket Breakdown
    bucket_names = [
        "Stress [-10, -5) dB", "Stress [-5, 0) dB",
        "Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"
    ]
    t2 = []
    for b in bucket_names:
        row = [b]
        for name in eval_results.keys():
            row.append(f"{eval_results[name]['buckets'][b]['mean_out_snr']:.2f} dB")
        t2.append(row)

    h2 = ["SNR Bucket"] + list(eval_results.keys())
    print("\n--- TABLE 2: Disaggregated 5-Bucket Output SNR Across All Candidates ---")
    print(tabulate(t2, headers=h2, tablefmt="grid"))


if __name__ == "__main__":
    main()
