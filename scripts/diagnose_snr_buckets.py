"""
Diagnostic Script: SNR-Bucket Analysis (DRDO PS 26052)
Evaluates Output SNR, SNR Gain, STOI, and PESQ across discretized input SNR buckets:
[-10 to -5 dB], [-5 to 0 dB], [0 to 5 dB], [5 to 10 dB], [10 to 15 dB].
"""

import os
import sys
import argparse
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr, compute_stoi, compute_pesq


def analyze_snr_buckets(checkpoint_path: str, n_samples_per_bucket: int = 50, split: str = "test", mode: str = "hybrid", hybrid_ratio: float = 0.85):
    print("================================================================================")
    print(" DRDO PS 26052: Input SNR Bucket Breakdown Diagnostic")
    print(f" Checkpoint : {checkpoint_path}")
    print(f" Data Split : {split.upper()} (Held-out, strictly isolated files)")
    print(f" Mode       : {mode} (HybridRatio={hybrid_ratio})")
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

    table_data = []

    for name, min_snr, max_snr in buckets:
        # Create dataset sampled strictly from the specified split within this specific SNR bucket
        dataset = HybridRealDataset(
            mode=mode,
            split=split,
            sample_rate=16000,
            segment_len_sec=2.0,
            epoch_size=n_samples_per_bucket,
            snr_range=(min_snr, max_snr),
            hybrid_ratio=hybrid_ratio,
            seed=2000 + int(abs(min_snr) * 10)
        )

        in_snrs = []
        out_snrs = []
        snr_gains = []
        stois = []
        pesqs = []

        with torch.no_grad():
            for i in range(len(dataset)):
                item = dataset[i]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()

                enhanced, _, _, _ = model(noisy)
                enh_np = enhanced.squeeze(0).cpu().numpy()

                raw_snr = compute_snr(clean, noisy_np)
                enh_snr = compute_snr(clean, enh_np)

                in_snrs.append(raw_snr)
                out_snrs.append(enh_snr)
                snr_gains.append(enh_snr - raw_snr)
                stois.append(compute_stoi(clean, enh_np))
                pesqs.append(compute_pesq(clean, enh_np))

        mean_in = np.mean(in_snrs)
        mean_out = np.mean(out_snrs)
        mean_gain = np.mean(snr_gains)
        mean_stoi = np.mean(stois)
        mean_pesq = np.mean(pesqs)

        table_data.append([
            name,
            f"{mean_in:+.2f} dB",
            f"{mean_out:+.2f} dB",
            f"{mean_gain:+.2f} dB",
            f"{mean_stoi:.4f}",
            f"{mean_pesq:.2f}"
        ])

    headers = ["Input SNR Bucket", "Avg Input SNR", "Avg Output SNR", "Avg SNR Gain", "STOI", "PESQ (approx.)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/task4_full_run/best_model.pt")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--hybrid_ratio", type=float, default=0.85)
    args = parser.parse_args()
    analyze_snr_buckets(args.checkpoint, n_samples_per_bucket=args.samples, split=args.split, hybrid_ratio=args.hybrid_ratio)
