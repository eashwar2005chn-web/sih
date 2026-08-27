"""
Compare N=100 vs N=250 Evaluation on [10, 15] dB Operational Bucket
Calculates Mean, Std, SEM, 95% CIs, and Binomial Pass Rates with Wilson Score CIs.
"""

import os
import sys
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr, compute_stoi, compute_pesq


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    """Compute Wilson score interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96 if confidence == 0.95 else 2.576
    p_hat = successes / total
    denominator = 1 + (z**2) / total
    centre_adj = p_hat + (z**2) / (2 * total)
    spread = z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)
    lower = max(0.0, (centre_adj - spread) / denominator)
    upper = min(1.0, (centre_adj + spread) / denominator)
    return (lower * 100.0, upper * 100.0)


def evaluate_high_snr_sample_sizes(
    checkpoint_path: str = "checkpoints/task4_full_run/best_model.pt",
    seed: int = 4242
):
    print("================================================================================")
    print(" DRDO PS 26052: Controlled Sample-Size Audit (N=100 vs N=250 on [10, 15] dB Bucket)")
    print(f" Checkpoint   : {checkpoint_path}")
    print(f" Data Split   : HELD-OUT TEST SPLIT (MD5 partitioned)")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = build_causal_anc_model(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_sizes = [100, 250]
    results = {}

    for n in sample_sizes:
        ds = HybridRealDataset(
            mode="hybrid",
            split="test",
            sample_rate=16000,
            segment_len_sec=2.0,
            epoch_size=n,
            snr_range=(10.0, 15.0),
            hybrid_ratio=0.85,
            seed=seed
        )

        in_snrs, out_snrs, gains, stois, pesqs = [], [], [], [], []

        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()

                enh, _, _, _ = model(noisy)
                enh_np = enh.squeeze(0).cpu().numpy()

                raw_snr = compute_snr(clean, noisy_np)
                enh_snr = compute_snr(clean, enh_np)

                in_snrs.append(raw_snr)
                out_snrs.append(enh_snr)
                gains.append(enh_snr - raw_snr)
                stois.append(compute_stoi(clean, enh_np))
                pesqs.append(compute_pesq(clean, enh_np))

        out_snrs = np.array(out_snrs)
        gains = np.array(gains)
        stois = np.array(stois)
        pesqs = np.array(pesqs)

        mean_out = np.mean(out_snrs)
        std_out = np.std(out_snrs, ddof=1)
        sem_out = std_out / np.sqrt(n)

        mean_stoi = np.mean(stois)
        std_stoi = np.std(stois, ddof=1)
        sem_stoi = std_stoi / np.sqrt(n)

        mean_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs, ddof=1)
        sem_pesq = std_pesq / np.sqrt(n)

        snr_pass_count = int(np.sum(out_snrs >= 15.0))
        snr_pass_pct = (snr_pass_count / n) * 100.0
        snr_ci_low, snr_ci_high = wilson_score_interval(snr_pass_count, n, 0.95)

        stoi_pass_count = int(np.sum(stois >= 0.850))
        stoi_pass_pct = (stoi_pass_count / n) * 100.0
        stoi_ci_low, stoi_ci_high = wilson_score_interval(stoi_pass_count, n, 0.95)

        pesq_pass_count = int(np.sum(pesqs >= 2.50))
        pesq_pass_pct = (pesq_pass_count / n) * 100.0
        pesq_ci_low, pesq_ci_high = wilson_score_interval(pesq_pass_count, n, 0.95)

        results[n] = {
            "mean_out": mean_out, "std_out": std_out, "sem_out": sem_out,
            "snr_pass_pct": snr_pass_pct, "snr_ci": (snr_ci_low, snr_ci_high),
            "mean_stoi": mean_stoi, "std_stoi": std_stoi, "sem_stoi": sem_stoi,
            "stoi_pass_pct": stoi_pass_pct, "stoi_ci": (stoi_ci_low, stoi_ci_high),
            "mean_pesq": mean_pesq, "std_pesq": std_pesq, "sem_pesq": sem_pesq,
            "pesq_pass_pct": pesq_pass_pct, "pesq_ci": (pesq_ci_low, pesq_ci_high)
        }

    table_data = [
        ["Output SNR (Mean ± Std)", f"{results[100]['mean_out']:.2f} ± {results[100]['std_out']:.2f} dB (SEM: ±{results[100]['sem_out']:.3f})", f"{results[250]['mean_out']:.2f} ± {results[250]['std_out']:.2f} dB (SEM: ±{results[250]['sem_out']:.3f})"],
        ["Output SNR >15dB Pass Rate", f"{results[100]['snr_pass_pct']:.1f}% [95% CI: {results[100]['snr_ci'][0]:.1f}%, {results[100]['snr_ci'][1]:.1f}%]", f"{results[250]['snr_pass_pct']:.1f}% [95% CI: {results[250]['snr_ci'][0]:.1f}%, {results[250]['snr_ci'][1]:.1f}%]"],
        ["STOI Intelligibility (Mean ± Std)", f"{results[100]['mean_stoi']:.4f} ± {results[100]['std_stoi']:.4f} (SEM: ±{results[100]['sem_stoi']:.4f})", f"{results[250]['mean_stoi']:.4f} ± {results[250]['std_stoi']:.4f} (SEM: ±{results[250]['sem_stoi']:.4f})"],
        ["STOI >0.850 Pass Rate", f"{results[100]['stoi_pass_pct']:.1f}% [95% CI: {results[100]['stoi_ci'][0]:.1f}%, {results[100]['stoi_ci'][1]:.1f}%]", f"{results[250]['stoi_pass_pct']:.1f}% [95% CI: {results[250]['stoi_ci'][0]:.1f}%, {results[250]['stoi_ci'][1]:.1f}%]"],
        ["PESQ Quality approx (Mean ± Std)", f"{results[100]['mean_pesq']:.2f} ± {results[100]['std_pesq']:.2f} (SEM: ±{results[100]['sem_pesq']:.3f})", f"{results[250]['mean_pesq']:.2f} ± {results[250]['std_pesq']:.2f} (SEM: ±{results[250]['sem_pesq']:.3f})"],
        ["PESQ >2.50 Pass Rate", f"{results[100]['pesq_pass_pct']:.1f}% [95% CI: {results[100]['pesq_ci'][0]:.1f}%, {results[100]['pesq_ci'][1]:.1f}%]", f"{results[250]['pesq_pass_pct']:.1f}% [95% CI: {results[250]['pesq_ci'][0]:.1f}%, {results[250]['pesq_ci'][1]:.1f}%]"]
    ]

    print("\n--- Comparative Evaluation: N=100 vs N=250 on [10, 15] dB Operational Bucket ---")
    headers = ["Evaluation Metric", "Sample Size N = 100", "Sample Size N = 250"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    evaluate_high_snr_sample_sizes()
