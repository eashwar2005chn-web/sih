"""
Detailed SNR-Bucket Tail Risk & Percentile Analysis on Held-Out Test Split
Uses standardized eval/metrics_common.py.
"""

import os
import sys
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def analyze_bucket_tails(
    checkpoint_path: str = "checkpoints/task4_full_run/best_model.pt",
    n_samples_per_bucket: int = 100,
    split: str = "test"
):
    print("================================================================================")
    print(" DRDO PS 26052: Disaggregated SNR Bucket Tail & Distribution Audit")
    print(f" Checkpoint   : {checkpoint_path}")
    print(f" Data Split   : {split.upper()} (MD5 partitioned)")
    print(f" Bucket Scale : N = {n_samples_per_bucket} samples per bucket (Total N = {n_samples_per_bucket * 5})")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = build_causal_anc_model(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    buckets = [
        ("Stress: [-10, -5) dB", -10.0, -5.0),
        ("Stress: [-5, 0) dB", -5.0, 0.0),
        ("Operational: [0, 5) dB", 0.0, 5.0),
        ("Operational: [5, 10) dB", 5.0, 10.0),
        ("Operational: [10, 15] dB", 10.0, 15.0)
    ]

    summary_rows = []
    tail_rows = []
    bucket_results = {}

    for name, min_snr, max_snr in buckets:
        test_ds = HybridRealDataset(
            mode="hybrid",
            split=split,
            sample_rate=16000,
            segment_len_sec=2.0,
            epoch_size=n_samples_per_bucket,
            snr_range=(min_snr, max_snr),
            hybrid_ratio=0.85,
            seed=5000 + int(abs(min_snr) * 100)
        )

        records = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                item = test_ds[i]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()

                enh, _, _, _ = model(noisy)
                enh_np = enh.squeeze(0).cpu().numpy()

                records.append(evaluate_batch_metrics(clean, noisy_np, enh_np))

        agg = aggregate_metric_records(records)
        bucket_results[name] = agg

        out_snrs = np.array([r["enh_snr"] for r in records])
        stois = np.array([r["enh_stoi"] for r in records])
        pesqs = np.array([r["enh_pesq"] for r in records])

        p5_out = float(np.percentile(out_snrs, 5))
        min_out = float(np.min(out_snrs))
        p5_stoi = float(np.percentile(stois, 5))
        min_stoi = float(np.min(stois))
        p5_pesq = float(np.percentile(pesqs, 5))
        min_pesq = float(np.min(pesqs))

        summary_rows.append([
            name,
            f"N={agg['n']}",
            f"{agg['mean_in_snr']:+.2f} dB",
            f"{agg['mean_out_snr']:+.2f} ± {agg['std_out_snr']:.2f} dB",
            f"{agg['mean_gain']:+.2f} dB",
            f"{agg['mean_stoi']:.4f} ± {agg['std_stoi']:.4f}",
            f"{agg['mean_pesq']:.2f} ± {agg['std_pesq']:.2f}"
        ])

        tail_rows.append([
            name,
            f"N={agg['n']}",
            f"{p5_out:+.2f} dB (Min: {min_out:+.2f})",
            f"{agg['pass_snr_pct']:.1f}% [{agg['pass_snr_ci'][0]:.1f}%, {agg['pass_snr_ci'][1]:.1f}%]",
            f"{p5_stoi:.4f} (Min: {min_stoi:.4f})",
            f"{agg['pass_stoi_pct']:.1f}% [{agg['pass_stoi_ci'][0]:.1f}%, {agg['pass_stoi_ci'][1]:.1f}%]",
            f"{p5_pesq:.2f} (Min: {min_pesq:.2f})",
            f"{agg['pass_pesq_pct']:.1f}% [{agg['pass_pesq_ci'][0]:.1f}%, {agg['pass_pesq_ci'][1]:.1f}%]"
        ])

    print("\n--- TABLE A: Mean ± Std Performance Across SNR Buckets (Consolidated Metrics) ---")
    headers_a = ["Input SNR Bucket", "Samples", "Avg Input SNR", "Output SNR (Mean ± Std)", "Avg SNR Gain", "STOI (Mean ± Std)", "PESQ approx (Mean ± Std)"]
    print(tabulate(summary_rows, headers=headers_a, tablefmt="grid"))

    print("\n--- TABLE B: Worst-Case Tail (5th Percentile & Min) & Target Pass Rates ---")
    headers_b = ["Input SNR Bucket", "Samples", "Output SNR P5 (Min)", "SNR >15dB Pass (95% CI)", "STOI P5 (Min)", "STOI >0.85 Pass (95% CI)", "PESQ P5 (Min)", "PESQ >2.5 Pass (95% CI)"]
    print(tabulate(tail_rows, headers=headers_b, tablefmt="grid"))

    return bucket_results


if __name__ == "__main__":
    analyze_bucket_tails(n_samples_per_bucket=100)
