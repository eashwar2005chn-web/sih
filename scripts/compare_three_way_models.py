"""
Three-Way Comprehensive Comparison:
1. Baseline Model (Task 4 Epoch 50, ~816K params)
2. Data-Rebalanced Pilot (~816K params, fine-tuned 15 epochs)
3. Scaled Capacity Pilot (CausalANCScaledNet, ~3.35M params, trained 20 epochs from scratch)

Measures exact CPU Latency & RTF and evaluates on Held-Out Test Split using eval/metrics_common.py.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model, build_scaled_causal_anc_model
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
    return rtf, latency_ms


def evaluate_model_on_held_out(model, device: str = "cuda"):
    model.to(device)
    model.eval()

    # 1. Operational Range [0, 15] dB (N=300 pooled)
    op_buckets = [
        ("Operational [0, 5) dB", 0.0, 5.0, 5000),
        ("Operational [5, 10) dB", 5.0, 10.0, 5500),
        ("Operational [10, 15] dB", 10.0, 15.0, 6000)
    ]
    op_records = []
    bucket_aggs = {}

    for name, min_snr, max_snr, seed in op_buckets:
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
                op_records.append(r)
        bucket_aggs[name] = aggregate_metric_records(b_records)

    pooled_op = aggregate_metric_records(op_records)

    # 2. Stress Range [-10, 0] dB (N=200 pooled)
    stress_buckets = [
        ("Stress [-10, -5) dB", -10.0, -5.0, 7000),
        ("Stress [-5, 0) dB", -5.0, 0.0, 7500)
    ]
    stress_records = []
    for name, min_snr, max_snr, seed in stress_buckets:
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
                stress_records.append(r)
        bucket_aggs[name] = aggregate_metric_records(b_records)

    pooled_stress = aggregate_metric_records(stress_records)

    return {
        "pooled_op": pooled_op,
        "pooled_stress": pooled_stress,
        "buckets": bucket_aggs
    }


def run_three_way_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_base = "checkpoints/task4_full_run/best_model.pt"
    ckpt_rebal = "checkpoints/pilot_rebalanced_snr/best_model.pt"
    ckpt_scaled = "checkpoints/pilot_scaled_capacity/best_model.pt"

    print("================================================================================")
    print(" DRDO PS 26052: Standardized Three-Way Model Architecture & Performance Audit")
    print("================================================================================")

    # 1. Load Model 1: Baseline
    m1_dict = torch.load(ckpt_base, map_location="cpu")
    model_base = build_causal_anc_model(hidden_dim=m1_dict.get("hidden_dim", 128))
    model_base.load_state_dict(m1_dict["model_state_dict"])
    p_base = sum(p.numel() for p in model_base.parameters())

    # 2. Load Model 2: Rebalanced Pilot
    m2_dict = torch.load(ckpt_rebal, map_location="cpu")
    model_rebal = build_causal_anc_model(hidden_dim=m2_dict.get("hidden_dim", 128))
    model_rebal.load_state_dict(m2_dict["model_state_dict"])
    p_rebal = sum(p.numel() for p in model_rebal.parameters())

    # 3. Load Model 3: Scaled Pilot
    m3_dict = torch.load(ckpt_scaled, map_location="cpu")
    model_scaled = build_scaled_causal_anc_model(hidden_dim=m3_dict.get("hidden_dim", 256))
    model_scaled.load_state_dict(m3_dict["model_state_dict"])
    p_scaled = sum(p.numel() for p in model_scaled.parameters())

    # Measure CPU Latency & RTF
    print("[1/3] Measuring CPU Latency & Real-Time Factor...")
    rtf_base, lat_base = measure_cpu_latency(model_base)
    rtf_rebal, lat_rebal = measure_cpu_latency(model_rebal)
    rtf_scaled, lat_scaled = measure_cpu_latency(model_scaled)

    # Evaluate on Held-out test set
    print("[2/3] Evaluating Models on Held-Out Test Split...")
    eval_base = evaluate_model_on_held_out(model_base, device=device)
    eval_rebal = evaluate_model_on_held_out(model_rebal, device=device)
    eval_scaled = evaluate_model_on_held_out(model_scaled, device=device)

    # Table 1: Model Hardware Profile
    hw_table = [
        ["Architecture Variant", "CausalANCNet (Baseline)", "CausalANCNet (Fine-Tuned)", "CausalANCScaledNet (Wide C-CRN)"],
        ["Parameters", f"{p_base:,} ({p_base*4/1e6:.2f} MB)", f"{p_rebal:,} ({p_rebal*4/1e6:.2f} MB)", f"{p_scaled:,} ({p_scaled*4/1e6:.2f} MB)"],
        ["Encoder Conv Channels", "[16, 32, 64]", "[16, 32, 64]", "[32, 64, 128]"],
        ["Recurrent Core", "2-Layer GRU (Hidden=128)", "2-Layer GRU (Hidden=128)", "2-Layer GRU (Hidden=256)"],
        ["Training Regime", "50 Epochs (Uniform -5 to 15dB)", "15 Epochs FT (Rebalanced -10 to 15dB)", "20 Epochs Scratch (Rebalanced -10 to 15dB)"],
        ["CPU Latency / Frame", f"{lat_base:.3f} ms / frame", f"{lat_rebal:.3f} ms / frame", f"{lat_scaled:.3f} ms / frame"],
        ["CPU Real-Time Factor (RTF)", f"{rtf_base:.4f} (Target < 0.100)", f"{rtf_rebal:.4f} (Target < 0.100)", f"{rtf_scaled:.4f} (Target < 0.100)"],
        ["Algorithmic Lookahead", "0.0 ms (Strictly Causal)", "0.0 ms (Strictly Causal)", "0.0 ms (Strictly Causal)"]
    ]

    # Table 2: Aggregate Operational Performance (0 to 15 dB)
    op_table = [
        [
            "Output SNR (Mean ± Std)",
            f"{eval_base['pooled_op']['mean_out_snr']:.2f} ± {eval_base['pooled_op']['std_out_snr']:.2f} dB",
            f"{eval_rebal['pooled_op']['mean_out_snr']:.2f} ± {eval_rebal['pooled_op']['std_out_snr']:.2f} dB",
            f"{eval_scaled['pooled_op']['mean_out_snr']:.2f} ± {eval_scaled['pooled_op']['std_out_snr']:.2f} dB"
        ],
        [
            "Output SNR >15.0 dB Pass Rate",
            f"{eval_base['pooled_op']['pass_snr_pct']:.1f}% [{eval_base['pooled_op']['pass_snr_ci'][0]:.1f}%, {eval_base['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{eval_rebal['pooled_op']['pass_snr_pct']:.1f}% [{eval_rebal['pooled_op']['pass_snr_ci'][0]:.1f}%, {eval_rebal['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{eval_scaled['pooled_op']['pass_snr_pct']:.1f}% [{eval_scaled['pooled_op']['pass_snr_ci'][0]:.1f}%, {eval_scaled['pooled_op']['pass_snr_ci'][1]:.1f}%]"
        ],
        [
            "STOI Intelligibility",
            f"{eval_base['pooled_op']['mean_stoi']:.4f} ± {eval_base['pooled_op']['std_stoi']:.4f}",
            f"{eval_rebal['pooled_op']['mean_stoi']:.4f} ± {eval_rebal['pooled_op']['std_stoi']:.4f}",
            f"{eval_scaled['pooled_op']['mean_stoi']:.4f} ± {eval_scaled['pooled_op']['std_stoi']:.4f}"
        ],
        [
            "STOI >0.850 Pass Rate",
            f"{eval_base['pooled_op']['pass_stoi_pct']:.1f}%",
            f"{eval_rebal['pooled_op']['pass_stoi_pct']:.1f}%",
            f"{eval_scaled['pooled_op']['pass_stoi_pct']:.1f}%"
        ],
        [
            "PESQ Quality (approx)",
            f"{eval_base['pooled_op']['mean_pesq']:.2f} ± {eval_base['pooled_op']['std_pesq']:.2f}",
            f"{eval_rebal['pooled_op']['mean_pesq']:.2f} ± {eval_rebal['pooled_op']['std_pesq']:.2f}",
            f"{eval_scaled['pooled_op']['mean_pesq']:.2f} ± {eval_scaled['pooled_op']['std_pesq']:.2f}"
        ]
    ]

    # Table 3: Stress Regime Performance (-10 to 0 dB)
    stress_table = [
        [
            "Stress Output SNR",
            f"{eval_base['pooled_stress']['mean_out_snr']:.2f} ± {eval_base['pooled_stress']['std_out_snr']:.2f} dB",
            f"{eval_rebal['pooled_stress']['mean_out_snr']:.2f} ± {eval_rebal['pooled_stress']['std_out_snr']:.2f} dB",
            f"{eval_scaled['pooled_stress']['mean_out_snr']:.2f} ± {eval_scaled['pooled_stress']['std_out_snr']:.2f} dB"
        ],
        [
            "Stress SNR Gain",
            f"{eval_base['pooled_stress']['mean_gain']:+.2f} dB",
            f"{eval_rebal['pooled_stress']['mean_gain']:+.2f} dB",
            f"{eval_scaled['pooled_stress']['mean_gain']:+.2f} dB"
        ],
        [
            "Stress STOI",
            f"{eval_base['pooled_stress']['mean_stoi']:.4f} ± {eval_base['pooled_stress']['std_stoi']:.4f}",
            f"{eval_rebal['pooled_stress']['mean_stoi']:.4f} ± {eval_rebal['pooled_stress']['std_stoi']:.4f}",
            f"{eval_scaled['pooled_stress']['mean_stoi']:.4f} ± {eval_scaled['pooled_stress']['std_stoi']:.4f}"
        ],
        [
            "Stress PESQ (approx)",
            f"{eval_base['pooled_stress']['mean_pesq']:.2f} ± {eval_base['pooled_stress']['std_pesq']:.2f}",
            f"{eval_rebal['pooled_stress']['mean_pesq']:.2f} ± {eval_rebal['pooled_stress']['std_pesq']:.2f}",
            f"{eval_scaled['pooled_stress']['mean_pesq']:.2f} ± {eval_scaled['pooled_stress']['std_pesq']:.2f}"
        ]
    ]

    print("\n--- TABLE 1: Model Hardware & Efficiency Profile ---")
    print(tabulate(hw_table, headers=["Hardware & Architecture Parameter", "Baseline Model", "Rebalanced Model", "Scaled Capacity Model"], tablefmt="grid"))

    print("\n--- TABLE 2: Aggregate Operational Range Performance (0 to 15 dB SNR Pooled, N=300) ---")
    headers_op = ["Metric / Parameter", "Baseline (50 Epochs)", "Rebalanced Pilot (15 Epochs FT)", "Scaled Capacity Pilot (20 Epochs)"]
    print(tabulate(op_table, headers=headers_op, tablefmt="grid"))

    print("\n--- TABLE 3: Degraded Channel Stress Regime Performance (-10 to 0 dB SNR Pooled, N=200) ---")
    print(tabulate(stress_table, headers=headers_op, tablefmt="grid"))

    # Table 4: Disaggregated 5-Bucket Output SNR Comparison
    bucket_names = [
        "Stress [-10, -5) dB", "Stress [-5, 0) dB",
        "Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"
    ]
    b_rows = []
    for b in bucket_names:
        b_rows.append([
            b,
            f"{eval_base['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{eval_rebal['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{eval_scaled['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{eval_base['buckets'][b]['mean_stoi']:.4f}",
            f"{eval_rebal['buckets'][b]['mean_stoi']:.4f}",
            f"{eval_scaled['buckets'][b]['mean_stoi']:.4f}"
        ])
    print("\n--- TABLE 4: Disaggregated 5-Bucket Breakdown Comparison ---")
    b_headers = ["SNR Bucket", "Base Out SNR", "Rebalanced Out SNR", "Scaled Out SNR", "Base STOI", "Rebalanced STOI", "Scaled STOI"]
    print(tabulate(b_rows, headers=b_headers, tablefmt="grid"))


if __name__ == "__main__":
    run_three_way_comparison()
