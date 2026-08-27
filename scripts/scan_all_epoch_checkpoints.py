"""
Scan All Saved Checkpoints from Task 4 Full Run Across Aggregate Operational Range
Evaluates epoch_05, epoch_10, epoch_15, epoch_20, epoch_25, epoch_30, epoch_35, epoch_40, epoch_45, epoch_50
on the pooled Aggregate Operational Range (0 to 15 dB Input SNR).
"""

import os
import sys
import glob
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr, compute_stoi, compute_pesq


def scan_checkpoints():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = "checkpoints/task4_full_run"
    epochs_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Pre-generate evaluation test dataset (N=150 pooled samples from [0, 15] dB)
    op_ds = HybridRealDataset(
        mode="hybrid",
        split="test",
        sample_rate=16000,
        segment_len_sec=2.0,
        epoch_size=150,
        snr_range=(0.0, 15.0),
        hybrid_ratio=0.85,
        seed=4242
    )

    stress_ds = HybridRealDataset(
        mode="hybrid",
        split="test",
        sample_rate=16000,
        segment_len_sec=2.0,
        epoch_size=75,
        snr_range=(-10.0, -0.01),
        hybrid_ratio=0.85,
        seed=9999
    )

    rows = []

    for ep in epochs_to_test:
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{ep:02d}.pt")
        if not os.path.exists(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_causal_anc_model(hidden_dim=ckpt.get("hidden_dim", 128)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 1. Operational Range [0, 15] dB
        op_out_snrs, op_stois, op_pesqs = [], [], []
        with torch.no_grad():
            for i in range(len(op_ds)):
                item = op_ds[i]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()

                enh, _, _, _ = model(noisy)
                enh_np = enh.squeeze(0).cpu().numpy()

                op_out_snrs.append(compute_snr(clean, enh_np))
                op_stois.append(compute_stoi(clean, enh_np))
                op_pesqs.append(compute_pesq(clean, enh_np))

        # 2. Stress Range [-10, 0] dB
        stress_gains = []
        with torch.no_grad():
            for i in range(len(stress_ds)):
                item = stress_ds[i]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()

                enh, _, _, _ = model(noisy)
                enh_np = enh.squeeze(0).cpu().numpy()

                raw_snr = compute_snr(clean, noisy_np)
                enh_snr = compute_snr(clean, enh_np)
                stress_gains.append(enh_snr - raw_snr)

        op_out_snrs = np.array(op_out_snrs)
        mean_op_snr = np.mean(op_out_snrs)
        pass_op_snr = np.mean(op_out_snrs >= 15.0) * 100.0
        mean_stoi = np.mean(op_stois)
        mean_pesq = np.mean(op_pesqs)
        mean_stress_gain = np.mean(stress_gains)

        rows.append([
            f"Epoch {ep:02d}",
            f"{mean_op_snr:.2f} dB",
            f"{pass_op_snr:.1f}%",
            f"{mean_stoi:.4f}",
            f"{mean_pesq:.2f}",
            f"{mean_stress_gain:+.2f} dB"
        ])

    print("================================================================================")
    print(" DRDO PS 26052: Epoch-by-Epoch Aggregate Performance Scan (0 to 15 dB SNR)")
    print("================================================================================")
    headers = ["Checkpoint", "Aggregate Op Out SNR", "SNR >15dB Pass %", "Aggregate STOI", "Aggregate PESQ (approx)", "Stress SNR Gain"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    scan_checkpoints()
