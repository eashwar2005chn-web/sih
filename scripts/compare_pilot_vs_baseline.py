"""
Direct Comparison: Task 4 Baseline (Epoch 50) vs Pilot Fine-Tuned (Rebalanced SNR)
Evaluates both checkpoints on the exact same Held-Out Test Split across:
1. Aggregate Operational Range (0 to 15 dB Input SNR)
2. Degraded Channel Stress Range (-10 to 0 dB Input SNR)
3. Disaggregated 5-Bucket Breakdown
"""

import os
import sys
import json
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def evaluate_checkpoint_on_split(
    checkpoint_path: str,
    device: str = "cuda"
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = build_causal_anc_model(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Aggregate Operational Range [0, 15] dB (N=300 pooled)
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


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_ckpt = "checkpoints/task4_full_run/best_model.pt"
    pilot_ckpt = "checkpoints/pilot_rebalanced_snr/best_model.pt"

    print("================================================================================")
    print(" DRDO PS 26052: Standardized Comparison — Baseline vs Pilot Rebalanced SNR Model")
    print(f" Base Checkpoint  : {base_ckpt}")
    print(f" Pilot Checkpoint : {pilot_ckpt}")
    print("================================================================================")

    print("\n[1/2] Evaluating Base Checkpoint (Epoch 50)...")
    base_res = evaluate_checkpoint_on_split(base_ckpt, device=device)

    print("\n[2/2] Evaluating Pilot Checkpoint (Rebalanced SNR)...")
    pilot_res = evaluate_checkpoint_on_split(pilot_ckpt, device=device)

    # 1. Headline Operational Comparison
    op_table = [
        [
            "Aggregate Operational Output SNR (0-15 dB)",
            f"{base_res['pooled_op']['mean_out_snr']:.2f} ± {base_res['pooled_op']['std_out_snr']:.2f} dB",
            f"{pilot_res['pooled_op']['mean_out_snr']:.2f} ± {pilot_res['pooled_op']['std_out_snr']:.2f} dB",
            f"{pilot_res['pooled_op']['mean_out_snr'] - base_res['pooled_op']['mean_out_snr']:+.2f} dB"
        ],
        [
            "Operational Output SNR >15.0 dB Pass Rate",
            f"{base_res['pooled_op']['pass_snr_pct']:.1f}% [95% CI: {base_res['pooled_op']['pass_snr_ci'][0]:.1f}%, {base_res['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{pilot_res['pooled_op']['pass_snr_pct']:.1f}% [95% CI: {pilot_res['pooled_op']['pass_snr_ci'][0]:.1f}%, {pilot_res['pooled_op']['pass_snr_ci'][1]:.1f}%]",
            f"{pilot_res['pooled_op']['pass_snr_pct'] - base_res['pooled_op']['pass_snr_pct']:+.1f}%"
        ],
        [
            "Aggregate Operational STOI",
            f"{base_res['pooled_op']['mean_stoi']:.4f} ± {base_res['pooled_op']['std_stoi']:.4f}",
            f"{pilot_res['pooled_op']['mean_stoi']:.4f} ± {pilot_res['pooled_op']['std_stoi']:.4f}",
            f"{pilot_res['pooled_op']['mean_stoi'] - base_res['pooled_op']['mean_stoi']:+.4f}"
        ],
        [
            "Operational STOI >0.850 Pass Rate",
            f"{base_res['pooled_op']['pass_stoi_pct']:.1f}% [95% CI: {base_res['pooled_op']['pass_stoi_ci'][0]:.1f}%, {base_res['pooled_op']['pass_stoi_ci'][1]:.1f}%]",
            f"{pilot_res['pooled_op']['pass_stoi_pct']:.1f}% [95% CI: {pilot_res['pooled_op']['pass_stoi_ci'][0]:.1f}%, {pilot_res['pooled_op']['pass_stoi_ci'][1]:.1f}%]",
            f"{pilot_res['pooled_op']['pass_stoi_pct'] - base_res['pooled_op']['pass_stoi_pct']:+.1f}%"
        ],
        [
            "Aggregate Operational PESQ (approx)",
            f"{base_res['pooled_op']['mean_pesq']:.2f} ± {base_res['pooled_op']['std_pesq']:.2f}",
            f"{pilot_res['pooled_op']['mean_pesq']:.2f} ± {pilot_res['pooled_op']['std_pesq']:.2f}",
            f"{pilot_res['pooled_op']['mean_pesq'] - base_res['pooled_op']['mean_pesq']:+.2f}"
        ]
    ]

    # 2. Stress Range Comparison
    stress_table = [
        [
            "Stress Regime Output SNR (-10 to 0 dB)",
            f"{base_res['pooled_stress']['mean_out_snr']:.2f} ± {base_res['pooled_stress']['std_out_snr']:.2f} dB",
            f"{pilot_res['pooled_stress']['mean_out_snr']:.2f} ± {pilot_res['pooled_stress']['std_out_snr']:.2f} dB",
            f"{pilot_res['pooled_stress']['mean_out_snr'] - base_res['pooled_stress']['mean_out_snr']:+.2f} dB"
        ],
        [
            "Stress Regime SNR Gain",
            f"{base_res['pooled_stress']['mean_gain']:+.2f} dB",
            f"{pilot_res['pooled_stress']['mean_gain']:+.2f} dB",
            f"{pilot_res['pooled_stress']['mean_gain'] - base_res['pooled_stress']['mean_gain']:+.2f} dB"
        ],
        [
            "Stress Regime STOI",
            f"{base_res['pooled_stress']['mean_stoi']:.4f} ± {base_res['pooled_stress']['std_stoi']:.4f}",
            f"{pilot_res['pooled_stress']['mean_stoi']:.4f} ± {pilot_res['pooled_stress']['std_stoi']:.4f}",
            f"{pilot_res['pooled_stress']['mean_stoi'] - base_res['pooled_stress']['mean_stoi']:+.4f}"
        ],
        [
            "Stress Regime STOI >0.850 Pass Rate",
            f"{base_res['pooled_stress']['pass_stoi_pct']:.1f}%",
            f"{pilot_res['pooled_stress']['pass_stoi_pct']:.1f}%",
            f"{pilot_res['pooled_stress']['pass_stoi_pct'] - base_res['pooled_stress']['pass_stoi_pct']:+.1f}%"
        ],
        [
            "Stress Regime PESQ (approx)",
            f"{base_res['pooled_stress']['mean_pesq']:.2f} ± {base_res['pooled_stress']['std_pesq']:.2f}",
            f"{pilot_res['pooled_stress']['mean_pesq']:.2f} ± {pilot_res['pooled_stress']['std_pesq']:.2f}",
            f"{pilot_res['pooled_stress']['mean_pesq'] - base_res['pooled_stress']['mean_pesq']:+.2f}"
        ]
    ]

    print("\n--- TABLE 1: Aggregate Operational Range (0 to 15 dB SNR Pooled, N=300) ---")
    headers = ["Metric / Parameter", "Baseline (Task 4 Epoch 50)", "Pilot (Rebalanced SNR)", "Delta"]
    print(tabulate(op_table, headers=headers, tablefmt="grid"))

    print("\n--- TABLE 2: Degraded Channel Stress Regime (-10 to 0 dB SNR Pooled, N=200) ---")
    print(tabulate(stress_table, headers=headers, tablefmt="grid"))

    # Print Per-Bucket Breakdown
    bucket_names = [
        "Stress [-10, -5) dB", "Stress [-5, 0) dB",
        "Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"
    ]
    b_rows = []
    for b in bucket_names:
        b_base = base_res["buckets"][b]
        b_pilot = pilot_res["buckets"][b]
        b_rows.append([
            b,
            f"{b_base['mean_out_snr']:.2f} dB",
            f"{b_pilot['mean_out_snr']:.2f} dB",
            f"{b_pilot['mean_out_snr'] - b_base['mean_out_snr']:+.2f} dB",
            f"{b_base['mean_stoi']:.4f}",
            f"{b_pilot['mean_stoi']:.4f}",
            f"{b_pilot['mean_stoi'] - b_base['mean_stoi']:+.4f}",
            f"{b_base['pass_snr_pct']:.1f}%",
            f"{b_pilot['pass_snr_pct']:.1f}%"
        ])

    print("\n--- TABLE 3: Disaggregated 5-Bucket Breakdown Comparison ---")
    b_headers = ["SNR Bucket", "Base Out SNR", "Pilot Out SNR", "SNR Delta", "Base STOI", "Pilot STOI", "STOI Delta", "Base >15dB Pass", "Pilot >15dB Pass"]
    print(tabulate(b_rows, headers=b_headers, tablefmt="grid"))


if __name__ == "__main__":
    run_comparison()
