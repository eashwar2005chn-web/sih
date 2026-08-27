"""
Step 1 No-Training Diagnostics for DRDO PS 26052
Runs 1a (Pre-clamp loss saturation), 1b (Oracle mask magnitude & truncation), and 1c (BatchNorm audit)
on checkpoints/task4_full_run/best_model.pt across the 5 standard test buckets.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr, compute_si_snr


def run_step1_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/task4_full_run/best_model.pt"

    print("================================================================================")
    print(" DRDO PS 26052: Step 1 No-Training Diagnostics Audit")
    print(f" Checkpoint: {ckpt_path}")
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

    # --- 1a & 1b Data Structures ---
    clamp_results = []
    oracle_results = []

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
        # 1a tracking
        n_dot_le_zero = 0
        pre_clamp_sisnrs = []
        post_clamp_sisnrs = []
        dot_products = []

        # 1b tracking
        oracle_mags = []
        oracle_max_comps = []
        oracle_snrs_raw = []
        oracle_snrs_bounded_1 = []
        oracle_snrs_tanh = []
        model_snrs = []

        with torch.no_grad():
            for i in range(n_samples):
                item = ds[i]
                clean_t = item["clean"].unsqueeze(0).to(device)
                noisy_t = item["noisy"].unsqueeze(0).to(device)

                # Forward through model
                pred_wav, _, _, _ = model(noisy_t)

                # --- 1a: Pre-clamp SI-SNR analysis ---
                pred_zm = pred_wav - torch.mean(pred_wav, dim=-1, keepdim=True)
                target_zm = clean_t - torch.mean(clean_t, dim=-1, keepdim=True)
                dot = torch.sum(pred_zm * target_zm, dim=-1).item()
                target_energy = torch.sum(target_zm ** 2, dim=-1).item() + 1e-8
                pred_energy = torch.sum(pred_zm ** 2, dim=-1).item() + 1e-8

                dot_products.append(dot)
                if dot <= 0.0:
                    n_dot_le_zero += 1

                # Unconstrained projection SI-SNR
                s_target_unconstrained = (dot / target_energy) * target_zm
                e_noise_unconstrained = pred_zm - s_target_unconstrained
                unconstrained_sisnr = 10.0 * np.log10(
                    (torch.sum(s_target_unconstrained ** 2).item() + 1e-8) /
                    (torch.sum(e_noise_unconstrained ** 2).item() + 1e-8)
                )
                pre_clamp_sisnrs.append(unconstrained_sisnr)

                # Clamped alpha projection SI-SNR
                alpha_clamped = max(0.0, dot) / target_energy
                s_target_clamped = alpha_clamped * target_zm
                e_noise_clamped = pred_zm - s_target_clamped
                clamped_sisnr = 10.0 * np.log10(
                    (torch.sum(s_target_clamped ** 2).item() + 1e-8) /
                    (torch.sum(e_noise_clamped ** 2).item() + 1e-8)
                )
                post_clamp_sisnrs.append(clamped_sisnr)

                # --- 1b: Oracle mask analysis ---
                clean_r, clean_i, _ = model.stft_forward(clean_t)
                noisy_r, noisy_i, _ = model.stft_forward(noisy_t)

                # S_clean = clean_r + j*clean_i, Y_noisy = noisy_r + j*noisy_i
                # M_oracle = S_clean / Y_noisy = (S_clean * Y_noisy*) / (|Y_noisy|^2 + eps)
                denom = noisy_r ** 2 + noisy_i ** 2 + 1e-8
                m_oracle_r = (clean_r * noisy_r + clean_i * noisy_i) / denom
                m_oracle_i = (clean_i * noisy_r - clean_r * noisy_i) / denom

                mag_oracle = torch.sqrt(m_oracle_r ** 2 + m_oracle_i ** 2)
                max_comp = torch.maximum(torch.abs(m_oracle_r), torch.abs(m_oracle_i))

                oracle_mags.append(mag_oracle.cpu().numpy().flatten())
                oracle_max_comps.append(max_comp.cpu().numpy().flatten())

                # Reconstruct time-domain audio with:
                # (1) Raw Oracle Mask
                clean_np = item["clean"].numpy()
                noisy_np = item["noisy"].numpy()
                model_np = pred_wav.squeeze(0).cpu().numpy()

                s_pred_r = noisy_r * m_oracle_r - noisy_i * m_oracle_i
                s_pred_i = noisy_r * m_oracle_i + noisy_i * m_oracle_r
                wav_oracle_raw = model.istft_forward(s_pred_r, s_pred_i, original_len=clean_t.shape[-1]).squeeze(0).cpu().numpy()

                # (2) Component-clamped [-1, 1] Oracle Mask
                m_clamp_r = torch.clamp(m_oracle_r, -1.0, 1.0)
                m_clamp_i = torch.clamp(m_oracle_i, -1.0, 1.0)
                s_pred_cr = noisy_r * m_clamp_r - noisy_i * m_clamp_i
                s_pred_ci = noisy_r * m_clamp_i + noisy_i * m_clamp_r
                wav_oracle_bounded = model.istft_forward(s_pred_cr, s_pred_ci, original_len=clean_t.shape[-1]).squeeze(0).cpu().numpy()

                # (3) Tanh(Oracle Mask)
                m_tanh_r = torch.tanh(m_oracle_r)
                m_tanh_i = torch.tanh(m_oracle_i)
                s_pred_tr = noisy_r * m_tanh_r - noisy_i * m_tanh_i
                s_pred_ti = noisy_r * m_tanh_i + noisy_i * m_tanh_r
                wav_oracle_tanh = model.istft_forward(s_pred_tr, s_pred_ti, original_len=clean_t.shape[-1]).squeeze(0).cpu().numpy()

                oracle_snrs_raw.append(compute_snr(clean_np, wav_oracle_raw))
                oracle_snrs_bounded_1.append(compute_snr(clean_np, wav_oracle_bounded))
                oracle_snrs_tanh.append(compute_snr(clean_np, wav_oracle_tanh))
                model_snrs.append(compute_snr(clean_np, model_np))

        # Flatten oracle arrays
        all_oracle_mags = np.concatenate(oracle_mags)
        all_max_comps = np.concatenate(oracle_max_comps)

        clamp_results.append([
            b_name,
            f"{n_dot_le_zero}/{n_samples} ({100.0*n_dot_le_zero/n_samples:.1f}%)",
            f"{np.mean(pre_clamp_sisnrs):.2f} dB",
            f"{np.mean(post_clamp_sisnrs):.2f} dB",
            f"{np.min(dot_products):.4f}",
            f"{np.mean(dot_products):.4f}"
        ])

        oracle_results.append([
            b_name,
            f"{np.median(all_oracle_mags):.2f}",
            f"{np.percentile(all_oracle_mags, 90):.2f}",
            f"{np.percentile(all_oracle_mags, 99):.2f}",
            f"{100.0 * np.mean(all_oracle_mags > 1.0):.1f}%",
            f"{100.0 * np.mean(all_max_comps > 1.0):.1f}%",
            f"{np.mean(oracle_snrs_raw):.2f} dB",
            f"{np.mean(oracle_snrs_bounded_1):.2f} dB",
            f"{np.mean(oracle_snrs_tanh):.2f} dB",
            f"{np.mean(model_snrs):.2f} dB"
        ])

    print("\n--- DIAGNOSTIC 1A: Pre-Clamp SI-SNR Loss Saturation Audit ---")
    headers_1a = ["Bucket", "Clamped (dot <= 0)", "Mean Pre-Clamp SI-SNR", "Mean Post-Clamp SI-SNR", "Min Dot Prod", "Mean Dot Prod"]
    print(tabulate(clamp_results, headers=headers_1a, tablefmt="grid"))

    print("\n--- DIAGNOSTIC 1B: Oracle Complex Ratio Mask Distribution & Truncation Ceiling ---")
    headers_1b = ["Bucket", "Median |M|", "P90 |M|", "P99 |M|", "% |M| > 1.0", "% Comp > 1.0", "Oracle Raw SNR", "Oracle [-1,1] SNR", "Oracle Tanh SNR", "Actual Model SNR"]
    print(tabulate(oracle_results, headers=headers_1b, tablefmt="grid"))

    # --- 1C: Normalization Audit ---
    print("\n--- DIAGNOSTIC 1C: Normalization & Streaming Causality Audit ---")
    print("Inspecting BatchNorm layers in CausalANCNet:")
    bn_layers = [
        ("enc_bn1", model.enc_bn1),
        ("enc_bn2", model.enc_bn2),
        ("enc_bn3", model.enc_bn3),
        ("dec_bn3", model.dec_bn3),
        ("dec_bn2", model.dec_bn2),
    ]
    bn_data = []
    for name, layer in bn_layers:
        rm = layer.running_mean.cpu().numpy()
        rv = layer.running_var.cpu().numpy()
        bn_data.append([
            name,
            f"{layer.num_features} channels",
            f"Mean: {np.mean(rm):.4f}, Range: [{np.min(rm):.4f}, {np.max(rm):.4f}]",
            f"Mean: {np.mean(rv):.4f}, Range: [{np.min(rv):.4f}, {np.max(rv):.4f}]",
            f"momentum={layer.momentum}, eps={layer.eps}"
        ])
    print(tabulate(bn_data, headers=["Layer", "Channels", "Running Mean", "Running Variance", "Config"], tablefmt="grid"))

    # Compare batch evaluation vs sequential frame-by-frame chunk evaluation
    print("\nAuditing BatchNorm Evaluation Mode vs Train Mode Difference:")
    print("1. Evaluation Harness Status: model.eval() is ACTIVE during eval/metrics_common.py evaluation.")
    print("2. In model.eval(), BatchNorm2d uses frozen running_mean and running_var (does not compute batch statistics over future frames).")
    print("3. Training vs Inference Discrepancy: During training (model.train()), BatchNorm2d computed mean/var pooled across (B, F, T).")
    print("   This means during training, normalization of frame t used batch statistics containing frames t+1...T (future frames).")
    print("   Replacing BatchNorm2d with causal GroupNorm (per-frame across channels) or causal Cumulative LayerNorm eliminates this train/eval mismatch.")


if __name__ == "__main__":
    run_step1_diagnostics()
