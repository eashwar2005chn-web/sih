"""
Phase 10 Final Evaluation & CSV Export Suite
Evaluates all models on held-out test split (N=500 per model) and re-benchmarks CPU latency/RTF.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from data.real_dataset_loader import verify_eval_splits_frozen
from model.causal_anc_net import CausalANCNet, CausalANCScaledNet, build_causal_anc_model
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def measure_cpu_latency(model, n_frames: int = 1000, sr: int = 16000, hop_size: int = 128):
    """Measures exact per-frame compute latency and RTF on CPU."""
    model_cpu = model.to("cpu")
    model_cpu.eval()
    audio_dur = (n_frames * hop_size) / sr
    dummy_wav = torch.randn(1, n_frames * hop_size)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_cpu(dummy_wav[:, :1024])

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model_cpu(dummy_wav)
    t1 = time.perf_counter()

    wall_time = t1 - t0
    rtf = wall_time / audio_dur
    latency_ms = (wall_time / n_frames) * 1000.0
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
    # Guard: every published metric was measured on the frozen val/test file lists.
    # Raises if they have drifted (e.g. a new data source leaked into an eval split).
    verify_eval_splits_frozen()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Baseline
    base_ckpt = torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")
    m_base = build_causal_anc_model(hidden_dim=128)
    m_base.load_state_dict(base_ckpt["model_state_dict"])

    # Load Scaled v2
    scaled_ckpt = torch.load("checkpoints/pilot_scaled_capacity_v2/best_model.pt", map_location="cpu")
    m_scaled = CausalANCScaledNet(hidden_dim=256)
    m_scaled.load_state_dict(scaled_ckpt["model_state_dict"])

    # Load Phase 9 Polar K=3.0
    p9_ckpt = torch.load("checkpoints/phase9_polar_k3_scratch/best_model.pt", map_location="cpu")
    m_p9 = CausalANCNet(hidden_dim=128, mask_bound=3.0, mask_mode="polar_tanh")
    m_p9.load_state_dict(p9_ckpt["model_state_dict"])

    # Load Phase 10 Polar K=1.5
    p10_k15_ckpt = torch.load("checkpoints/phase10_polar_k15_scratch/best_model.pt", map_location="cpu")
    m_k15 = CausalANCNet(hidden_dim=128, mask_bound=1.5, mask_mode="polar_tanh", norm_type="batch")
    m_k15.load_state_dict(p10_k15_ckpt["model_state_dict"])

    # Load Phase 10 Causal GroupNorm
    p10_gn_ckpt = torch.load("checkpoints/phase10_causal_norm_scratch/best_model.pt", map_location="cpu")
    m_gn = CausalANCNet(hidden_dim=128, mask_bound=1.0, mask_mode="component_tanh", norm_type="group")
    m_gn.load_state_dict(p10_gn_ckpt["model_state_dict"])

    models = {
        "Baseline (811K, BatchNorm, K=1.0)": m_base,
        "Scaled Net (3.24M, BatchNorm, K=1.0)": m_scaled,
        "Causal GroupNorm (811K, K=1.0)": m_gn,
        "Polar Bounded (811K, K=1.5)": m_k15,
        "Phase 9 Polar (811K, K=3.0)": m_p9,
    }

    print("\n================================================================================")
    print(" Benchmarking Latency & RTF Back-to-Back on Idle State...")
    print("================================================================================")
    latencies = {}
    for name, m in models.items():
        lat_ms, rtf = measure_cpu_latency(m)
        latencies[name] = (lat_ms, rtf)
        print(f"[{name}] CPU Latency: {lat_ms:.3f} ms/frame | RTF: {rtf:.4f}")

    print("\n================================================================================")
    print(" Evaluating Models on Held-Out Test Split (N=500 per model)...")
    print("================================================================================")
    eval_results = {}
    for name, m in models.items():
        eval_results[name] = evaluate_model_on_split(m, device=device)

    # Save per-sample test evaluations to CSV
    os.makedirs("eval/test_eval_csvs", exist_ok=True)
    for name, e in eval_results.items():
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
        df_samples = pd.DataFrame(e["samples"])
        csv_file = f"eval/test_eval_csvs/{clean_name}_test_eval.csv"
        df_samples.to_csv(csv_file, index=False)
        print(f"Saved test samples CSV: {csv_file}")

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
    print("\n--- TABLE 1: Phase 10 Comprehensive 5-Way Benchmark Summary ---")
    print(tabulate(t1, headers=h1, tablefmt="grid"))

    df_summary = pd.DataFrame(csv_summary_records)
    summary_csv = "eval/phase10_models_benchmark_summary.csv"
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
    bucket_csv = "eval/phase10_5bucket_snr_breakdown.csv"
    df_buckets.to_csv(bucket_csv, index=False)
    print(f"Saved 5-bucket CSV to: {bucket_csv}")


if __name__ == "__main__":
    main()
