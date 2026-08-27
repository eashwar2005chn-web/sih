"""
Deterministic Verification Script: Alignment Between In-Training Validation and Standalone Eval
Runs both scoring paths on the exact same checkpoint and compares all metrics side-by-side to full precision.
"""

import os
import sys
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import get_hybrid_dataloaders
from model.causal_anc_net import build_causal_anc_model
from model.loss import HybridANCLoss, compute_snr, compute_stoi, compute_pesq
from train import validate_epoch


def verify_alignment(checkpoint_path: str):
    print("================================================================================")
    print(" DRDO PS 26052: Verification of Scoring Alignment & Determinism")
    print(f" Target Checkpoint: {checkpoint_path}")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 128)

    model = build_causal_anc_model(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)

    # 1. Instantiate the canonical validation DataLoader
    _, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid",
        batch_size=16,
        val_epoch_size=80,
        hybrid_ratio=0.85,
        data_root="."
    )

    # 2. Run In-Training Validation Path (Run 1)
    print("\n[Step 1] Running In-Training Validation Loop (Run 1)...")
    val_run1 = validate_epoch(model, val_loader, loss_fn, device)

    # 3. Run In-Training Validation Path (Run 2 - checking determinism across consecutive runs)
    print("[Step 2] Running In-Training Validation Loop (Run 2 - verifying zero stochastic drift)...")
    val_run2 = validate_epoch(model, val_loader, loss_fn, device)

    # 4. Run Standalone Direct Metric Evaluation on the same validation dataset
    print("[Step 3] Running Standalone Evaluation on the same validation set...")
    val_ds = val_loader.dataset
    op_in, op_out, op_gain, op_stoi, op_pesq = [], [], [], [], []
    stress_gain, stress_stoi = [], []

    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            noisy = item["noisy"].unsqueeze(0).to(device)
            clean_np = item["clean"].numpy()
            noisy_np = item["noisy"].numpy()

            enh, _, _, _ = model(noisy)
            enh_np = enh.squeeze(0).cpu().numpy()

            raw_snr = compute_snr(clean_np, noisy_np)
            enh_snr = compute_snr(clean_np, enh_np)
            gain = enh_snr - raw_snr
            stoi_val = compute_stoi(clean_np, enh_np)
            pesq_val = compute_pesq(clean_np, enh_np)

            if raw_snr >= 0.0:
                op_in.append(raw_snr)
                op_out.append(enh_snr)
                op_gain.append(gain)
                op_stoi.append(stoi_val)
                op_pesq.append(pesq_val)
            else:
                stress_gain.append(gain)
                stress_stoi.append(stoi_val)

    standalone_results = {
        "op_output_snr": float(np.mean(op_out)),
        "op_input_snr": float(np.mean(op_in)),
        "op_snr_gain": float(np.mean(op_gain)),
        "op_stoi": float(np.mean(op_stoi)),
        "op_pesq": float(np.mean(op_pesq)),
        "stress_snr_gain": float(np.mean(stress_gain)),
        "stress_stoi": float(np.mean(stress_stoi))
    }

    # 5. Side-by-Side Direct Comparison Table
    table_data = [
        ["Operational Output SNR (dB)", f"{val_run1['op_output_snr']:.6f} dB", f"{val_run2['op_output_snr']:.6f} dB", f"{standalone_results['op_output_snr']:.6f} dB", f"{abs(val_run1['op_output_snr'] - standalone_results['op_output_snr']):.6f}"],
        ["Operational Input SNR (dB)", f"{val_run1['op_input_snr']:.6f} dB", f"{val_run2['op_input_snr']:.6f} dB", f"{standalone_results['op_input_snr']:.6f} dB", f"{abs(val_run1['op_input_snr'] - standalone_results['op_input_snr']):.6f}"],
        ["Operational SNR Gain (dB)", f"{val_run1['op_snr_gain']:+.6f} dB", f"{val_run2['op_snr_gain']:+.6f} dB", f"{standalone_results['op_snr_gain']:+.6f} dB", f"{abs(val_run1['op_snr_gain'] - standalone_results['op_snr_gain']):.6f}"],
        ["Operational STOI Score", f"{val_run1['op_stoi']:.6f}", f"{val_run2['op_stoi']:.6f}", f"{standalone_results['op_stoi']:.6f}", f"{abs(val_run1['op_stoi'] - standalone_results['op_stoi']):.6f}"],
        ["Operational PESQ Score", f"{val_run1['op_pesq']:.6f}", f"{val_run2['op_pesq']:.6f}", f"{standalone_results['op_pesq']:.6f}", f"{abs(val_run1['op_pesq'] - standalone_results['op_pesq']):.6f}"],
        ["Stress SNR Gain (dB)", f"{val_run1['stress_snr_gain']:+.6f} dB", f"{val_run2['stress_snr_gain']:+.6f} dB", f"{standalone_results['stress_snr_gain']:+.6f} dB", f"{abs(val_run1['stress_snr_gain'] - standalone_results['stress_snr_gain']):.6f}"],
        ["Stress STOI Score", f"{val_run1['stress_stoi']:.6f}", f"{val_run2['stress_stoi']:.6f}", f"{standalone_results['stress_stoi']:.6f}", f"{abs(val_run1['stress_stoi'] - standalone_results['stress_stoi']):.6f}"]
    ]

    headers = ["Metric Name", "In-Training Run 1", "In-Training Run 2", "Standalone Eval", "Absolute Delta (|Run1 - Standalone|)"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    max_delta = max(abs(val_run1[k] - standalone_results[k]) for k in standalone_results.keys())
    if max_delta < 1e-3:
        print(f"\n[VERIFICATION PASSED] In-Training Validation and Standalone Eval match to <0.001 dB precision! (Max FP16 batch delta: {max_delta:.6f})")
    else:
        print(f"\n[VERIFICATION FAILED] Discrepancy detected: Max Delta = {max_delta}")


if __name__ == "__main__":
    checkpoint = "checkpoints/diagnostic_real_corpus_ext/best_model.pt"
    verify_alignment(checkpoint)
