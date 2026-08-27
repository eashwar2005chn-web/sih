"""
Phase 9 Step 0: Oracle Complex Ratio Mask Phase-Distortion vs Magnitude-Bounding Diagnostic
Evaluates:
1. Raw Oracle Mask (Unconstrained M = S / Y)
2. Per-component linear clamp [-1, 1] (M_r = clamp(M_r, -1, 1), M_i = clamp(M_i, -1, 1))
3. Per-component Tanh (Current Model Mask: M_r = tanh(M_r), M_i = tanh(M_i)) -> Distorts Phase & Compresses Magnitude
4. Polar Magnitude Tanh with EXACT Phase Preserved (M = tanh(|M|) * e^{j theta}) -> K=1.0
5. Polar Magnitude Tanh with K=2.0 (M = 2.0 * tanh(|M|/2.0) * e^{j theta})
6. Polar Magnitude Tanh with K=3.0 (M = 3.0 * tanh(|M|/3.0) * e^{j theta})
7. Polar Magnitude Tanh with K=5.0 (M = 5.0 * tanh(|M|/5.0) * e^{j theta})
8. Polar Linear Clamp |M| <= 1.0 (M = min(1.0, |M|) * e^{j theta})

Evaluates across all 5 standard test buckets on the held-out test split (N=500).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr


def run_step0_oracle_polar_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/task4_full_run/best_model.pt"

    print("================================================================================")
    print(" DRDO PS 26052: Phase 9 Step 0 — Oracle Polar Mask & Phase Distortion Diagnostic")
    print("================================================================================")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_causal_anc_model(hidden_dim=ckpt.get("hidden_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    buckets = [
        ("Stress [-10, -5) dB", -10.0, -5.0, 7000),
        ("Stress [-5, 0) dB", -5.0, 0.0, 7500),
        ("Operational [0, 5) dB", 0.0, 5.0, 5000),
        ("Operational [5, 10) dB", 5.0, 10.0, 5500),
        ("Operational [10, 15] dB", 10.0, 15.0, 6000)
    ]

    results_table = []

    for b_name, min_snr, max_snr, seed in buckets:
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

        n_samples = len(ds)
        snrs_raw = []
        snrs_comp_clamp1 = []
        snrs_comp_tanh = []
        snrs_polar_tanh1 = []
        snrs_polar_tanh2 = []
        snrs_polar_tanh3 = []
        snrs_polar_tanh5 = []
        snrs_polar_clamp1 = []
        snrs_model = []

        with torch.no_grad():
            for i in range(n_samples):
                item = ds[i]
                clean_t = item["clean"].unsqueeze(0).to(device)
                noisy_t = item["noisy"].unsqueeze(0).to(device)
                orig_len = clean_t.shape[-1]

                # Model prediction
                pred_wav, _, _, _ = model(noisy_t)

                clean_r, clean_i, _ = model.stft_forward(clean_t)
                noisy_r, noisy_i, _ = model.stft_forward(noisy_t)

                # M_oracle = S_clean / Y_noisy = (S * Y*) / (|Y|^2 + eps)
                denom = noisy_r ** 2 + noisy_i ** 2 + 1e-8
                m_r = (clean_r * noisy_r + clean_i * noisy_i) / denom
                m_i = (clean_i * noisy_r - clean_r * noisy_i) / denom

                mag = torch.sqrt(m_r ** 2 + m_i ** 2 + 1e-12)
                phase_cos = m_r / mag
                phase_sin = m_i / mag

                clean_np = item["clean"].numpy()
                model_np = pred_wav.squeeze(0).cpu().numpy()

                # Helper to apply mask & iSTFT
                def reconstruct_snr(mr, mi):
                    sr = noisy_r * mr - noisy_i * mi
                    si = noisy_r * mi + noisy_i * mr
                    wav = model.istft_forward(sr, si, original_len=orig_len).squeeze(0).cpu().numpy()
                    return compute_snr(clean_np, wav)

                # 1. Raw Oracle
                snrs_raw.append(reconstruct_snr(m_r, m_i))

                # 2. Per-component Linear Clamp [-1, 1]
                mr_cc = torch.clamp(m_r, -1.0, 1.0)
                mi_cc = torch.clamp(m_i, -1.0, 1.0)
                snrs_comp_clamp1.append(reconstruct_snr(mr_cc, mi_cc))

                # 3. Per-component Tanh (Current Model Mask Head)
                mr_ct = torch.tanh(m_r)
                mi_ct = torch.tanh(m_i)
                snrs_comp_tanh.append(reconstruct_snr(mr_ct, mi_ct))

                # 4. Polar Tanh(mag) * phase (K=1.0)
                mag_pt1 = torch.tanh(mag)
                mr_pt1 = mag_pt1 * phase_cos
                mi_pt1 = mag_pt1 * phase_sin
                snrs_polar_tanh1.append(reconstruct_snr(mr_pt1, mi_pt1))

                # 5. Polar Tanh K=2.0
                mag_pt2 = 2.0 * torch.tanh(mag / 2.0)
                mr_pt2 = mag_pt2 * phase_cos
                mi_pt2 = mag_pt2 * phase_sin
                snrs_polar_tanh2.append(reconstruct_snr(mr_pt2, mi_pt2))

                # 6. Polar Tanh K=3.0
                mag_pt3 = 3.0 * torch.tanh(mag / 3.0)
                mr_pt3 = mag_pt3 * phase_cos
                mi_pt3 = mag_pt3 * phase_sin
                snrs_polar_tanh3.append(reconstruct_snr(mr_pt3, mi_pt3))

                # 7. Polar Tanh K=5.0
                mag_pt5 = 5.0 * torch.tanh(mag / 5.0)
                mr_pt5 = mag_pt5 * phase_cos
                mi_pt5 = mag_pt5 * phase_sin
                snrs_polar_tanh5.append(reconstruct_snr(mr_pt5, mi_pt5))

                # 8. Polar Linear Clamp |M| <= 1.0
                mag_pc1 = torch.clamp(mag, max=1.0)
                mr_pc1 = mag_pc1 * phase_cos
                mi_pc1 = mag_pc1 * phase_sin
                snrs_polar_clamp1.append(reconstruct_snr(mr_pc1, mi_pc1))

                snrs_model.append(compute_snr(clean_np, model_np))

        results_table.append([
            b_name,
            f"{np.mean(snrs_comp_tanh):.2f} dB",      # Component Tanh (Current Model)
            f"{np.mean(snrs_polar_tanh1):.2f} dB",     # Polar Tanh K=1 (Phase Preserved)
            f"{np.mean(snrs_polar_clamp1):.2f} dB",    # Polar Clamp |M|<=1
            f"{np.mean(snrs_polar_tanh2):.2f} dB",     # Polar Tanh K=2
            f"{np.mean(snrs_polar_tanh3):.2f} dB",     # Polar Tanh K=3
            f"{np.mean(snrs_polar_tanh5):.2f} dB",     # Polar Tanh K=5
            f"{np.mean(snrs_comp_clamp1):.2f} dB",     # Comp Clamp [-1, 1]
            f"{np.mean(snrs_model):.2f} dB"            # Actual Baseline Model
        ])

    headers = [
        "Bucket",
        "Comp Tanh (Current)",
        "Polar Tanh (K=1)",
        "Polar Clamp (|M|<=1)",
        "Polar Tanh (K=2)",
        "Polar Tanh (K=3)",
        "Polar Tanh (K=5)",
        "Comp Clamp [-1,1]",
        "Baseline Model"
    ]
    print("\n--- ORACLE MASK RECONSTRUCTION SNR UNDER VARIOUS MASK HEAD NONLINEARITIES ---")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run_step0_oracle_polar_diagnostic()
