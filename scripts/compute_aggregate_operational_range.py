"""
Compute Aggregate Operational Range Metrics (0 to 15 dB Input SNR Pooled)
Uses standardized eval/metrics_common.py for exact evaluation.
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


def compute_pooled_operational_metrics(
    checkpoint_path: str = "checkpoints/task4_full_run/best_model.pt",
    samples_per_bucket: int = 100,
    split: str = "test"
):
    print("================================================================================")
    print(" DRDO PS 26052: Consolidated Pooled Operational Range Evaluation (0 to 15 dB SNR)")
    print(f" Checkpoint   : {checkpoint_path}")
    print(f" Data Split   : {split.upper()} (Held-Out Test Split)")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = build_causal_anc_model(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    op_buckets = [
        ("Operational [0, 5) dB", 0.0, 5.0, 5000),
        ("Operational [5, 10) dB", 5.0, 10.0, 5500),
        ("Operational [10, 15] dB", 10.0, 15.0, 6000)
    ]

    records = []
    bucket_results = {}

    for name, min_snr, max_snr, seed in op_buckets:
        ds = HybridRealDataset(
            mode="hybrid",
            split=split,
            sample_rate=16000,
            segment_len_sec=2.0,
            epoch_size=samples_per_bucket,
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
                records.append(r)

        bucket_results[name] = aggregate_metric_records(b_records)

    pooled_agg = aggregate_metric_records(records)

    table_data = [
        ["Total Pooled Samples (N)", f"{pooled_agg['n']} samples across [0, 15] dB"],
        ["Average Input SNR", f"{pooled_agg['mean_in_snr']:.2f} dB (Full Range: 0.0 to 15.0 dB)"],
        ["Average Output SNR", f"{pooled_agg['mean_out_snr']:.2f} ± {pooled_agg['std_out_snr']:.2f} dB (SEM: ±{pooled_agg['sem_out_snr']:.3f} dB) [95% CI: {pooled_agg['ci_out_snr_95'][0]:.2f} to {pooled_agg['ci_out_snr_95'][1]:.2f} dB]"],
        ["Output SNR >15.0 dB Pass Rate", f"{pooled_agg['pass_snr_pct']:.1f}% ({pooled_agg['pass_snr_count']}/{pooled_agg['n']} passing) [95% CI: {pooled_agg['pass_snr_ci'][0]:.1f}% to {pooled_agg['pass_snr_ci'][1]:.1f}%]"],
        ["Average STOI Intelligibility", f"{pooled_agg['mean_stoi']:.4f} ± {pooled_agg['std_stoi']:.4f} (SEM: ±{pooled_agg['sem_stoi']:.4f}) [95% CI: {pooled_agg['ci_stoi_95'][0]:.4f} to {pooled_agg['ci_stoi_95'][1]:.4f}]"],
        ["STOI >0.850 Pass Rate", f"{pooled_agg['pass_stoi_pct']:.1f}% ({pooled_agg['pass_stoi_count']}/{pooled_agg['n']} passing) [95% CI: {pooled_agg['pass_stoi_ci'][0]:.1f}% to {pooled_agg['pass_stoi_ci'][1]:.1f}%]"],
        ["Average PESQ Quality (approx)", f"{pooled_agg['mean_pesq']:.2f} ± {pooled_agg['std_pesq']:.2f} (SEM: ±{pooled_agg['sem_pesq']:.3f}) [95% CI: {pooled_agg['ci_pesq_95'][0]:.2f} to {pooled_agg['ci_pesq_95'][1]:.2f}]"],
        ["PESQ >2.50 Pass Rate", f"{pooled_agg['pass_pesq_pct']:.1f}% ({pooled_agg['pass_pesq_count']}/{pooled_agg['n']} passing) [95% CI: {pooled_agg['pass_pesq_ci'][0]:.1f}% to {pooled_agg['pass_pesq_ci'][1]:.1f}%]"],
        ["Average SNR Gain", f"{pooled_agg['mean_gain']:+.2f} ± {pooled_agg['std_gain']:.2f} dB"]
    ]

    print("\n" + tabulate(table_data, headers=["Metric / Parameter", "Consolidated Value"], tablefmt="grid"))
    return pooled_agg, bucket_results


if __name__ == "__main__":
    compute_pooled_operational_metrics(samples_per_bucket=100)
