"""
Large-Sample Bootstrap Confidence Interval Evaluation on [10, 15] dB Operational Bucket
Evaluates N=250 held-out test samples and computes 95% and 99% bootstrap confidence intervals.
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


def evaluate_high_snr_confidence_interval(
    checkpoint_path: str = "checkpoints/task4_full_run/best_model.pt",
    n_test_samples: int = 250
):
    print("================================================================================")
    print(" DRDO PS 26052: High-SNR Operational Bucket [10, 15] dB Statistical Audit")
    print(f" Checkpoint   : {checkpoint_path}")
    print(f" Sample Size  : N = {n_test_samples} strictly held-out test mixtures")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = build_causal_anc_model(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Held-out test dataset strictly within [10.0, 15.0] dB
    test_ds = HybridRealDataset(
        mode="hybrid",
        split="test",
        sample_rate=16000,
        segment_len_sec=2.0,
        epoch_size=n_test_samples,
        snr_range=(10.0, 15.0),
        hybrid_ratio=0.85,
        seed=7777
    )

    in_snrs, out_snrs, gains, stois, pesqs = [], [], [], [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
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
    sem_out = std_out / np.sqrt(len(out_snrs))

    # Bootstrap 95% & 99% CIs (10,000 resamples)
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(out_snrs, size=len(out_snrs), replace=True)) for _ in range(10000)]
    ci_95_low, ci_95_high = np.percentile(boot_means, [2.5, 97.5])
    ci_99_low, ci_99_high = np.percentile(boot_means, [0.5, 99.5])

    pass_rate = np.mean(out_snrs >= 15.0) * 100.0

    table_data = [
        ["Sample Size (N)", f"{len(out_snrs)} held-out test audio clips"],
        ["Average Input SNR", f"{np.mean(in_snrs):.2f} dB (Std: {np.std(in_snrs):.2f} dB)"],
        ["Average Output SNR (Mean)", f"{mean_out:.2f} dB"],
        ["Standard Deviation (Std)", f"±{std_out:.2f} dB"],
        ["Standard Error (SEM)", f"±{sem_out:.3f} dB"],
        ["95% Bootstrap Confidence Interval", f"[{ci_95_low:.2f} dB, {ci_95_high:.2f} dB]"],
        ["99% Bootstrap Confidence Interval", f"[{ci_99_low:.2f} dB, {ci_99_high:.2f} dB]"],
        ["DRDO >15.0 dB Pass Rate", f"{pass_rate:.1f}% of evaluated clips"],
        ["Average STOI Intelligibility", f"{np.mean(stois):.4f} (Std: ±{np.std(stois):.4f})"],
        ["Average PESQ Quality (approx.)", f"{np.mean(pesqs):.2f} (Std: ±{np.std(pesqs):.2f})"]
    ]

    print("\n" + tabulate(table_data, headers=["Statistical Parameter", "Empirical Measurement on Held-Out Test Split"], tablefmt="grid"))


if __name__ == "__main__":
    evaluate_high_snr_confidence_interval(n_test_samples=250)
