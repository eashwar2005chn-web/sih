"""
Phase 11 Step 2: Full 50-Epoch Scratch Training & Comprehensive 6-Way Benchmark
"""

import os
import sys
import time
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import get_hybrid_dataloaders, HybridRealDataset
from model.causal_anc_net import (
    CausalANCNet,
    CausalANCScaledNet,
    CausalANCTCNNet,
    build_causal_anc_model,
    build_tcn_causal_anc_model
)
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def measure_cpu_latency(model, n_frames=1000, sr=16000, hop_size=256):
    model_cpu = model.to("cpu")
    model_cpu.eval()
    audio_dur = (n_frames * hop_size) / sr
    dummy_wav = torch.randn(1, n_frames * hop_size)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_cpu(dummy_wav[:, :2048])

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model_cpu(dummy_wav)
    t1 = time.perf_counter()

    wall_time = t1 - t0
    rtf = wall_time / audio_dur
    latency_ms = (wall_time / n_frames) * 1000.0
    return latency_ms, rtf


def evaluate_model_on_split(model, device="cuda"):
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
    all_sample_records = []

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
                r["bucket"] = name
                b_records.append(r)
                all_sample_records.append(r)
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
        "buckets": bucket_aggs,
        "samples": all_sample_records
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Phase 11 Full Run on device: {device}")

    save_dir = "checkpoints/phase11_tcn_scratch"
    os.makedirs(save_dir, exist_ok=True)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Dataset & Loaders
    batch_size = 16
    train_steps = 1200
    val_steps = 100
    epochs = 50
    lr = 1e-3

    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=batch_size,
        train_epoch_size=train_steps,
        val_epoch_size=val_steps,
        hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced",
        seed=seed
    )

    # 2. Build TCN Model from Scratch
    model = build_tcn_causal_anc_model(hidden_dim=128, num_gru_layers=2, tcn_dilations=(1, 2, 4, 8, 16)).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print("\n================================================================================")
    print(" Starting Full 50-Epoch Retrain: CausalANCTCNNet (Scratch 50ep, 899K params)")
    print(f" Dilations: (1, 2, 4, 8, 16) | Receptive Field: ~496 ms feedforward + GRU")
    print(f" Epochs: {epochs} | Steps/Epoch: {train_steps} | LR: {lr} | Save: {save_dir}")
    print("================================================================================")

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

        train_loss = total_train_loss / max(train_steps, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
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
                    total_val_loss += losses["total_loss"].item()

                for i in range(clean.shape[0]):
                    c_np = clean[i].cpu().numpy()
                    n_np = noisy[i].cpu().numpy()
                    e_np = enhanced[i].cpu().numpy()
                    r = evaluate_batch_metrics(c_np, n_np, e_np)
                    records.append(r)

        val_loss = total_val_loss / max(len(val_loader), 1)
        agg = aggregate_metric_records(records)
        val_snr = agg.get("mean_out_snr", 0.0)
        val_stoi = agg.get("mean_stoi", 0.0)
        score = val_snr + 10.0 * val_stoi
        dur = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_snr": val_snr,
                "val_stoi": val_stoi,
                "op_score": score
            }, os.path.join(save_dir, "best_model.pt"))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_snr": val_snr,
            "val_stoi": val_stoi,
            "op_score": score
        })

        best_tag = " >>> [BEST]" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best or epoch == epochs:
            print(f"[CausalANCTCNNet (50ep)] Epoch [{epoch:02d}/{epochs}] ({dur:.1f}s, lr={curr_lr:.1e}) | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val SNR: {val_snr:+.2f}dB | STOI: {val_stoi:.4f} | Score: {score:.2f}{best_tag}")

        scheduler.step()

    with open(os.path.join(save_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n>>> Training finished. Best model saved to {os.path.join(save_dir, 'best_model.pt')} (Best Score: {best_score:.2f})")

    # ==============================================================================
    # 3. Comprehensive 6-Way Benchmark & Evaluation
    # ==============================================================================
    print("\n================================================================================")
    print(" Loading all 6 models for Comprehensive Back-to-Back Evaluation...")
    print("================================================================================")

    # 1. Baseline
    m_base = build_causal_anc_model(hidden_dim=128)
    m_base.load_state_dict(torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")["model_state_dict"])

    # 2. Scaled v2
    m_scaled = CausalANCScaledNet(hidden_dim=256)
    m_scaled.load_state_dict(torch.load("checkpoints/pilot_scaled_capacity_v2/best_model.pt", map_location="cpu")["model_state_dict"])

    # 3. Polar K=1.5
    m_k15 = CausalANCNet(hidden_dim=128, mask_bound=1.5, mask_mode="polar_tanh", norm_type="batch")
    m_k15.load_state_dict(torch.load("checkpoints/phase10_polar_k15_scratch/best_model.pt", map_location="cpu")["model_state_dict"])

    # 4. Causal GroupNorm
    m_gn = CausalANCNet(hidden_dim=128, mask_bound=1.0, mask_mode="component_tanh", norm_type="group")
    m_gn.load_state_dict(torch.load("checkpoints/phase10_causal_norm_scratch/best_model.pt", map_location="cpu")["model_state_dict"])

    # 5. Polar K=3.0
    m_p9 = CausalANCNet(hidden_dim=128, mask_bound=3.0, mask_mode="polar_tanh", norm_type="batch")
    m_p9.load_state_dict(torch.load("checkpoints/phase9_polar_k3_scratch/best_model.pt", map_location="cpu")["model_state_dict"])

    # 6. Phase 11 TCN
    m_tcn = build_tcn_causal_anc_model(hidden_dim=128, tcn_dilations=(1, 2, 4, 8, 16))
    m_tcn.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location="cpu")["model_state_dict"])

    models = {
        "Baseline (811K, BatchNorm, K=1.0)": m_base,
        "Causal TCN Net (899K, Dilated TCN, K=1.0)": m_tcn,
        "Scaled Net (3.24M, BatchNorm, K=1.0)": m_scaled,
        "Polar Bounded (811K, K=1.5)": m_k15,
        "Causal GroupNorm (811K, K=1.0)": m_gn,
        "Phase 9 Polar (811K, K=3.0)": m_p9,
    }

    print("\n--- Benchmarking Latency & RTF Back-to-Back on Idle State ---")
    latencies = {}
    for name, m in models.items():
        lat_ms, rtf = measure_cpu_latency(m)
        latencies[name] = (lat_ms, rtf)
        print(f"[{name}] CPU Latency: {lat_ms:.3f} ms/frame | RTF: {rtf:.4f}")

    print("\n--- Evaluating Models on Held-Out Test Split (N=500 per model) ---")
    eval_results = {}
    for name, m in models.items():
        eval_results[name] = evaluate_model_on_split(m, device=device)

    # Save per-sample test inferences to CSV
    os.makedirs("eval/test_eval_csvs", exist_ok=True)
    for name, e in eval_results.items():
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
        df_samples = pd.DataFrame(e["samples"])
        csv_file = f"eval/test_eval_csvs/{clean_name}_test_eval.csv"
        df_samples.to_csv(csv_file, index=False)

    # Summary Table
    t1 = []
    csv_summary_records = []
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
        csv_summary_records.append({
            "Model": name,
            "Op Output SNR (dB)": f"{e['pooled_op']['mean_out_snr']:.2f}",
            "Op SNR Std (dB)": f"{e['pooled_op']['std_out_snr']:.2f}",
            "SNR Pass Rate (%)": f"{e['pooled_op']['pass_snr_pct']:.1f}",
            "SNR Pass Rate CI Low (%)": f"{e['pooled_op']['pass_snr_ci'][0]:.1f}",
            "SNR Pass Rate CI High (%)": f"{e['pooled_op']['pass_snr_ci'][1]:.1f}",
            "Op STOI": f"{e['pooled_op']['mean_stoi']:.4f}",
            "Op PESQ": f"{e['pooled_op']['mean_pesq']:.2f}",
            "Stress Output SNR (dB)": f"{e['pooled_stress']['mean_out_snr']:.2f}",
            "Stress SNR Gain (dB)": f"{e['pooled_stress']['mean_gain']:+.2f}",
            "CPU Latency (ms)": f"{lat:.3f}",
            "CPU RTF": f"{rtf:.4f}"
        })

    h1 = ["Model Architecture", "Op Output SNR (0-15dB)", "SNR >15dB Pass Rate", "Op STOI", "PESQ", "Stress SNR (Gain)", "CPU Latency", "CPU RTF"]
    print("\n--- TABLE 1: Phase 11 Comprehensive 6-Way Benchmark Summary ---")
    print(tabulate(t1, headers=h1, tablefmt="grid"))

    df_summary = pd.DataFrame(csv_summary_records)
    summary_csv = "eval/phase11_models_benchmark_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved overall summary CSV to: {summary_csv}")

    # 5-Bucket Table
    bucket_names = [
        "Stress [-10, -5) dB", "Stress [-5, 0) dB",
        "Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"
    ]
    t2 = []
    bucket_csv_records = []
    for b in bucket_names:
        row = [b]
        b_dict = {"SNR Bucket": b}
        for name in eval_results.keys():
            val = eval_results[name]["buckets"][b]["mean_out_snr"]
            row.append(f"{val:.2f} dB")
            b_dict[name] = f"{val:.2f}"
        t2.append(row)
        bucket_csv_records.append(b_dict)

    h2 = ["SNR Bucket"] + list(eval_results.keys())
    print("\n--- TABLE 2: Disaggregated 5-Bucket Output SNR Across Models ---")
    print(tabulate(t2, headers=h2, tablefmt="grid"))

    df_buckets = pd.DataFrame(bucket_csv_records)
    bucket_csv = "eval/phase11_5bucket_snr_breakdown.csv"
    df_buckets.to_csv(bucket_csv, index=False)
    print(f"Saved 5-bucket CSV to: {bucket_csv}")


if __name__ == "__main__":
    main()
