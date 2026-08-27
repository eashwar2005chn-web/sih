"""
Final Decisive Four-Way Model Evaluation Script (DRDO PS 26052)
Evaluates:
1. Baseline Model (Task 4 Full Run: 811K params, 50 epochs, LR=1e-3)
2. Data-Rebalanced Pilot (811K params, 15 epochs FT, LR=1e-4)
3. Scaled Capacity v1 (3.24M params, 20 epochs scratch, LR=3e-4)
4. Scaled Capacity v2 (3.24M params, 50 epochs scratch, LR=1e-3)

Uses eval/metrics_common.py across the Held-Out Test Split.
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


def evaluate_model_on_split(model, device: str = "cuda"):
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


def run_four_way_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_base = "checkpoints/task4_full_run/best_model.pt"
    ckpt_rebal = "checkpoints/pilot_rebalanced_snr/best_model.pt"
    ckpt_v1 = "checkpoints/pilot_scaled_capacity/best_model.pt"
    ckpt_v2 = "checkpoints/pilot_scaled_capacity_v2/best_model.pt"

    print("================================================================================")
    print(" DRDO PS 26052: Comprehensive Four-Way Model Performance & Compliance Audit")
    print("================================================================================")

    # 1. Baseline
    d1 = torch.load(ckpt_base, map_location="cpu")
    m1 = build_causal_anc_model(hidden_dim=d1.get("hidden_dim", 128))
    m1.load_state_dict(d1["model_state_dict"])
    p1 = sum(p.numel() for p in m1.parameters())

    # 2. Rebalanced Pilot
    d2 = torch.load(ckpt_rebal, map_location="cpu")
    m2 = build_causal_anc_model(hidden_dim=d2.get("hidden_dim", 128))
    m2.load_state_dict(d2["model_state_dict"])
    p2 = sum(p.numel() for p in m2.parameters())

    # 3. Scaled v1
    d3 = torch.load(ckpt_v1, map_location="cpu")
    m3 = build_scaled_causal_anc_model(hidden_dim=d3.get("hidden_dim", 256))
    m3.load_state_dict(d3["model_state_dict"])
    p3 = sum(p.numel() for p in m3.parameters())

    # 4. Scaled v2
    d4 = torch.load(ckpt_v2, map_location="cpu")
    m4 = build_scaled_causal_anc_model(hidden_dim=d4.get("hidden_dim", 256))
    m4.load_state_dict(d4["model_state_dict"])
    p4 = sum(p.numel() for p in m4.parameters())

    print("\n[1/3] Measuring CPU Latencies...")
    rtf1, lat1 = measure_cpu_latency(m1)
    rtf2, lat2 = measure_cpu_latency(m2)
    rtf3, lat3 = measure_cpu_latency(m3)
    rtf4, lat4 = measure_cpu_latency(m4)

    print("[2/3] Evaluating Models on Held-Out Test Split (N=500 each)...")
    e1 = evaluate_model_on_split(m1, device=device)
    e2 = evaluate_model_on_split(m2, device=device)
    e3 = evaluate_model_on_split(m3, device=device)
    e4 = evaluate_model_on_split(m4, device=device)

    # TABLE 1: Hardware & Configuration
    t1 = [
        ["Model Architecture", "CausalANCNet (811K)", "CausalANCNet (811K)", "CausalANCScaled (3.24M)", "CausalANCScaled (3.24M)"],
        ["Parameters", f"{p1:,}", f"{p2:,}", f"{p3:,}", f"{p4:,}"],
        ["Training Budget", "50 Epochs (Scratch)", "15 Epochs (Fine-Tune)", "20 Epochs (Scratch)", "50 Epochs (Scratch)"],
        ["Initial LR", "1.0e-3", "1.0e-4", "3.0e-4", "1.0e-3"],
        ["SNR Sampler", "Uniform [-5, 15] dB", "Rebalanced [-10, 15] dB", "Rebalanced [-10, 15] dB", "Rebalanced [-10, 15] dB"],
        ["CPU Latency / Frame", f"{lat1:.3f} ms", f"{lat2:.3f} ms", f"{lat3:.3f} ms", f"{lat4:.3f} ms"],
        ["CPU RTF", f"{rtf1:.4f}", f"{rtf2:.4f}", f"{rtf3:.4f}", f"{rtf4:.4f}"],
        ["Lookahead", "0.0 ms", "0.0 ms", "0.0 ms", "0.0 ms"]
    ]

    # TABLE 2: Aggregate Operational Range (0 to 15 dB)
    t2 = [
        [
            "Output SNR (Mean ± Std)",
            f"{e1['pooled_op']['mean_out_snr']:.2f} ± {e1['pooled_op']['std_out_snr']:.2f} dB",
            f"{e2['pooled_op']['mean_out_snr']:.2f} ± {e2['pooled_op']['std_out_snr']:.2f} dB",
            f"{e3['pooled_op']['mean_out_snr']:.2f} ± {e3['pooled_op']['std_out_snr']:.2f} dB",
            f"{e4['pooled_op']['mean_out_snr']:.2f} ± {e4['pooled_op']['std_out_snr']:.2f} dB"
        ],
        [
            "Output SNR >15.0 dB Pass Rate",
            f"{e1['pooled_op']['pass_snr_pct']:.1f}% [{e1['pooled_op']['pass_snr_ci'][0]:.1f}%, {e1['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{e2['pooled_op']['pass_snr_pct']:.1f}% [{e2['pooled_op']['pass_snr_ci'][0]:.1f}%, {e2['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{e3['pooled_op']['pass_snr_pct']:.1f}% [{e3['pooled_op']['pass_snr_ci'][0]:.1f}%, {e3['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{e4['pooled_op']['pass_snr_pct']:.1f}% [{e4['pooled_op']['pass_snr_ci'][0]:.1f}%, {e4['pooled_op']['pass_snr_ci'][1]:.1f}%]"
        ],
        [
            "STOI Intelligibility",
            f"{e1['pooled_op']['mean_stoi']:.4f} ± {e1['pooled_op']['std_stoi']:.4f}",
            f"{e2['pooled_op']['mean_stoi']:.4f} ± {e2['pooled_op']['std_stoi']:.4f}",
            f"{e3['pooled_op']['mean_stoi']:.4f} ± {e3['pooled_op']['std_stoi']:.4f}",
            f"{e4['pooled_op']['mean_stoi']:.4f} ± {e4['pooled_op']['std_stoi']:.4f}"
        ],
        [
            "STOI >0.850 Pass Rate",
            f"{e1['pooled_op']['pass_stoi_pct']:.1f}%",
            f"{e2['pooled_op']['pass_stoi_pct']:.1f}%",
            f"{e3['pooled_op']['pass_stoi_pct']:.1f}%",
            f"{e4['pooled_op']['pass_stoi_pct']:.1f}%"
        ],
        [
            "PESQ Quality (approx)",
            f"{e1['pooled_op']['mean_pesq']:.2f} ± {e1['pooled_op']['std_pesq']:.2f}",
            f"{e2['pooled_op']['mean_pesq']:.2f} ± {e2['pooled_op']['std_pesq']:.2f}",
            f"{e3['pooled_op']['mean_pesq']:.2f} ± {e3['pooled_op']['std_pesq']:.2f}",
            f"{e4['pooled_op']['mean_pesq']:.2f} ± {e4['pooled_op']['std_pesq']:.2f}"
        ]
    ]

    # TABLE 3: Stress Range (-10 to 0 dB)
    t3 = [
        [
            "Stress Output SNR",
            f"{e1['pooled_stress']['mean_out_snr']:.2f} ± {e1['pooled_stress']['std_out_snr']:.2f} dB",
            f"{e2['pooled_stress']['mean_out_snr']:.2f} ± {e2['pooled_stress']['std_out_snr']:.2f} dB",
            f"{e3['pooled_stress']['mean_out_snr']:.2f} ± {e3['pooled_stress']['std_out_snr']:.2f} dB",
            f"{e4['pooled_stress']['mean_out_snr']:.2f} ± {e4['pooled_stress']['std_out_snr']:.2f} dB"
        ],
        [
            "Stress SNR Gain",
            f"{e1['pooled_stress']['mean_gain']:+.2f} dB",
            f"{e2['pooled_stress']['mean_gain']:+.2f} dB",
            f"{e3['pooled_stress']['mean_gain']:+.2f} dB",
            f"{e4['pooled_stress']['mean_gain']:+.2f} dB"
        ],
        [
            "Stress STOI",
            f"{e1['pooled_stress']['mean_stoi']:.4f} ± {e1['pooled_stress']['std_stoi']:.4f}",
            f"{e2['pooled_stress']['mean_stoi']:.4f} ± {e2['pooled_stress']['std_stoi']:.4f}",
            f"{e3['pooled_stress']['mean_stoi']:.4f} ± {e3['pooled_stress']['std_stoi']:.4f}",
            f"{e4['pooled_stress']['mean_stoi']:.4f} ± {e4['pooled_stress']['std_stoi']:.4f}"
        ],
        [
            "Stress PESQ (approx)",
            f"{e1['pooled_stress']['mean_pesq']:.2f} ± {e1['pooled_stress']['std_pesq']:.2f}",
            f"{e2['pooled_stress']['mean_pesq']:.2f} ± {e2['pooled_stress']['std_pesq']:.2f}",
            f"{e3['pooled_stress']['mean_pesq']:.2f} ± {e3['pooled_stress']['std_pesq']:.2f}",
            f"{e4['pooled_stress']['mean_pesq']:.2f} ± {e4['pooled_stress']['std_pesq']:.2f}"
        ]
    ]

    print("\n--- TABLE 1: Model Configurations & Hardware Efficiency ---")
    headers_cfg = ["Parameter", "Baseline (Task 4)", "Data-Rebalanced Pilot", "Scaled Pilot v1", "Scaled Model v2 (Decisive)"]
    print(tabulate(t1, headers=headers_cfg, tablefmt="grid"))

    print("\n--- TABLE 2: Aggregate Operational Range (0 to 15 dB Input SNR, N=300) ---")
    print(tabulate(t2, headers=headers_cfg, tablefmt="grid"))

    print("\n--- TABLE 3: Degraded Channel Stress Range (-10 to 0 dB Input SNR, N=200) ---")
    print(tabulate(t3, headers=headers_cfg, tablefmt="grid"))

    # TABLE 4: Disaggregated 5-Bucket Breakdown
    bucket_names = [
        "Stress [-10, -5) dB", "Stress [-5, 0) dB",
        "Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"
    ]
    t4 = []
    for b in bucket_names:
        t4.append([
            b,
            f"{e1['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e2['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e3['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e4['buckets'][b]['mean_out_snr']:.2f} dB",
            f"{e1['buckets'][b]['mean_stoi']:.4f}",
            f"{e4['buckets'][b]['mean_stoi']:.4f}"
        ])
    print("\n--- TABLE 4: Disaggregated 5-Bucket Output SNR & STOI Breakdown ---")
    b_headers = ["SNR Bucket", "Baseline SNR", "Rebal Pilot SNR", "Scaled v1 SNR", "Scaled v2 SNR", "Base STOI", "Scaled v2 STOI"]
    print(tabulate(t4, headers=b_headers, tablefmt="grid"))

    return {
        "m1": e1, "m2": e2, "m3": e3, "m4": e4,
        "lat": (lat1, lat2, lat3, lat4),
        "rtf": (rtf1, rtf2, rtf3, rtf4)
    }


if __name__ == "__main__":
    run_four_way_evaluation()
