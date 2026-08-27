"""
Phase 10: Full 50-Epoch Confirmatory Retraining & Fair Causal Norm Benchmark

Runs two 50-epoch scratch training runs (60,000 steps each, LR=1e-3 cosine decay to 1e-5):
1. Best Mask Candidate: Static Polar-Bounded K=1.5 (norm_type="batch", mask_mode="polar_tanh", mask_bound=1.5)
   Saved to checkpoints/phase10_polar_k15_scratch/
2. Fair Causal Norm Test: Causal GroupNorm (norm_type="group", mask_mode="component_tanh", mask_bound=1.0)
   Saved to checkpoints/phase10_causal_norm_scratch/

Evaluates all models + Baseline (811K, K=1.0) + Scaled v2 (3.24M) on held-out test split (N=500).
Re-benchmarks CPU latency and RTF under identical idle machine state.
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


def train_full_model(
    run_name: str,
    save_dir: str,
    norm_type: str = "batch",
    mask_mode: str = "polar_tanh",
    mask_bound: float = 1.5,
    epochs: int = 50,
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

    print(f"\n================================================================================")
    print(f" Starting Full 50-Epoch Retrain: {run_name}")
    print(f" Norm: {norm_type} | Mask Mode: {mask_mode} | Bound K: {mask_bound}")
    print(f" Epochs: {epochs} | Steps/Epoch: {train_steps} | LR: {lr} | Save: {save_dir}")
    print(f"================================================================================")

    model = CausalANCNet(
        hidden_dim=128,
        num_gru_layers=2,
        norm_type=norm_type,
        mask_bound=mask_bound,
        mask_mode=mask_mode
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

        # Validate
        model.eval()
        val_loss = 0.0
        val_snrs = []
        val_stois = []
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
                    val_stois.append(r["enh_stoi"])

        val_loss /= max(1, len(val_loader))
        val_snr_mean = float(np.mean(val_snrs))
        val_stoi_mean = float(np.mean(val_stois))
        op_score = val_snr_mean + 10.0 * val_stoi_mean
        elapsed = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]

        is_best = op_score > best_score
        if is_best:
            best_score = op_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "hidden_dim": 128,
                "norm_type": norm_type,
                "mask_bound": mask_bound,
                "mask_mode": mask_mode,
                "val_snr": val_snr_mean,
                "val_stoi": val_stoi_mean,
                "op_score": op_score
            }, os.path.join(save_dir, "best_model.pt"))

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_snr": val_snr_mean, "val_stoi": val_stoi_mean, "op_score": op_score
        })

        if epoch % 5 == 0 or epoch == epochs or is_best:
            best_tag = " >>> [BEST]" if is_best else ""
            print(f"[{run_name}] Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s, lr={curr_lr:.1e}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SNR: {val_snr_mean:+.2f}dB | STOI: {val_stoi_mean:.4f} | Score: {op_score:.2f}{best_tag}")

    with open(os.path.join(save_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"   >>> {run_name} training finished. Best model saved to {save_dir}/best_model.pt (Score: {best_score:.2f})")
    return model


def benchmark_cpu_latency_and_rtf(model, num_frames: int = 500):
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 128, dtype=torch.float32)
    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)

    times = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.perf_counter() - t0)

    latency_ms = float(np.mean(times)) * 1000.0
    frame_dur_ms = 128.0 / 16000.0 * 1000.0  # 8.0 ms
    rtf = latency_ms / frame_dur_ms
    return latency_ms, rtf


def evaluate_model_on_split(model, device: str = "cuda"):
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

    # Step 2: Best Screening Candidate (Polar K=1.5, 50 epochs scratch)
    m_k15 = train_full_model(
        run_name="Polar-Bounded K=1.5 (Scratch 50ep)",
        save_dir="checkpoints/phase10_polar_k15_scratch",
        norm_type="batch",
        mask_mode="polar_tanh",
        mask_bound=1.5,
        epochs=50,
        device=device
    )

    # Step 3: Fair Causal Norm Test (Causal GroupNorm, 50 epochs scratch)
    m_cnorm = train_full_model(
        run_name="Causal GroupNorm (Scratch 50ep)",
        save_dir="checkpoints/phase10_causal_norm_scratch",
        norm_type="group",
        mask_mode="component_tanh",
        mask_bound=1.0,
        epochs=50,
        device=device
    )

    # Load Baseline and Scaled Net for comprehensive comparison
    base_ckpt = torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")
    m_base = build_causal_anc_model(hidden_dim=128)
    m_base.load_state_dict(base_ckpt["model_state_dict"])

    scaled_ckpt = torch.load("checkpoints/pilot_scaled_capacity_v2/best_model.pt", map_location="cpu")
    from model.causal_anc_net import CausalANCScaledNet
    m_scaled = CausalANCScaledNet(hidden_dim=256)
    m_scaled.load_state_dict(scaled_ckpt["model_state_dict"])

    p9_ckpt = torch.load("checkpoints/phase9_polar_k3_scratch/best_model.pt", map_location="cpu")
    m_p9 = CausalANCNet(hidden_dim=128, mask_bound=3.0, mask_mode="polar_tanh")
    m_p9.load_state_dict(p9_ckpt["model_state_dict"])

    models_to_eval = {
        "Baseline (811K, BatchNorm, K=1.0)": m_base,
        "Scaled Net (3.24M, BatchNorm, K=1.0)": m_scaled,
        "Causal Norm (811K, GroupNorm, K=1.0)": m_cnorm,
        "Polar Bounded (811K, BatchNorm, K=1.5)": m_k15,
        "Phase 9 Polar (811K, BatchNorm, K=3.0)": m_p9,
    }

    print("\n================================================================================")
    print(" Measuring CPU Latency & RTF Back-to-Back on Idle State...")
    print("================================================================================")
    latencies = {}
    for name, m in models_to_eval.items():
        lat_ms, rtf = benchmark_cpu_latency_and_rtf(m)
        latencies[name] = (lat_ms, rtf)
        print(f"[{name}] Latency: {lat_ms:.3f} ms/frame | RTF: {rtf:.4f}")

    print("\n================================================================================")
    print(" Evaluating All Models on Held-Out Test Split (N=500 per model)...")
    print("================================================================================")
    eval_results = {}
    for name, m in models_to_eval.items():
        eval_results[name] = evaluate_model_on_split(m, device=device)

    # TABLE 1: Operational Summary
    t1 = []
    for name, e in eval_results.items():
        lat, rtf = latencies[name]
        t1.append([
            name,
            f"{e['pooled_op']['mean_out_snr']:.2f} ± {e['pooled_op']['std_out_snr']:.2f} dB",
            f"{e['pooled_op']['pass_snr_pct']:.1f}% [{e['pooled_op']['pass_snr_ci'][0]:.1f}%, {e['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{e['pooled_op']['mean_stoi']:.4f}",
            f"{e['pooled_op']['mean_pesq']:.2f}",
            f"{e['pooled_stress']['mean_out_snr']:.2f} dB ({e['pooled_stress']['mean_gain']:+.2f} dB)",
            f"{lat:.3f} ms",
            f"{rtf:.4f}"
        ])

    h1 = ["Model Architecture", "Op Output SNR (0-15dB)", "SNR >15dB Pass Rate", "Op STOI", "PESQ", "Stress SNR (Gain)", "CPU Latency", "CPU RTF"]
    print("\n--- TABLE 1: Phase 10 Comprehensive 5-Way Benchmark Summary ---")
    print(tabulate(t1, headers=h1, tablefmt="grid"))

    # TABLE 2: 5-Bucket Breakdown
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
    print("\n--- TABLE 2: Disaggregated 5-Bucket Output SNR Across Models ---")
    print(tabulate(t2, headers=h2, tablefmt="grid"))


if __name__ == "__main__":
    main()
