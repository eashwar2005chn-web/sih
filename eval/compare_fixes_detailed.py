"""
Comprehensive 8-Scenario Benchmark comparing Fix A (Clamped-Alpha) vs Fix B (Hinge-Penalty).
Guarantees equal evaluation samples for all 8 defence noise scenarios across both:
1. Operational Range (0.0 to +15.0 dB Input SNR)
2. Degraded Channel Stress Test (-10.0 to -0.01 dB Input SNR)
"""

import os
import sys
import numpy as np
import torch
from tabulate import tabulate

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_builder import TacticalAudioDataset
from demo.stream_engine import RealTimeANCEngine
from model.loss import compute_snr, compute_stoi, compute_pesq


def evaluate_balanced_scenarios(
    checkpoint_path: str,
    snr_min: float,
    snr_max: float,
    samples_per_class: int = 10,
    seed: int = 4242
) -> dict:
    engine = RealTimeANCEngine(checkpoint_path=checkpoint_path)
    noise_classes = [
        "gunshot", "helicopter", "armored_vehicle", "jet_flyby",
        "siren", "cockpit", "radio_static", "babble"
    ]
    
    dataset = TacticalAudioDataset(
        split="test",
        segment_len_sec=2.0,
        epoch_size=samples_per_class * 40,
        snr_range=(snr_min, snr_max),
        add_reverb=True,
        add_channel_distortion=True,
        seed=seed
    )

    results = {
        nc: {
            "in_snr": [], "out_snr": [], "gain": [],
            "raw_stoi": [], "enh_stoi": [],
            "raw_pesq": [], "enh_pesq": [],
            "rtf": []
        }
        for nc in noise_classes
    }

    class_counts = {nc: 0 for nc in noise_classes}
    total_target = samples_per_class * len(noise_classes)
    total_collected = 0

    for idx in range(len(dataset)):
        if total_collected >= total_target:
            break

        item = dataset[idx]
        ntype = item["noise_type"]
        if class_counts[ntype] >= samples_per_class:
            continue

        noisy = item["noisy"].numpy()
        clean = item["clean"].numpy()

        raw_s = compute_snr(clean, noisy)
        if not (snr_min <= raw_s <= snr_max):
            continue

        res = engine.process_audio(
            noisy,
            backend="CUDA" if engine.cuda_available else "CPU",
            enable_neural=True,
            # NLMS post-filter DISABLED. Phase 11B swept 9 configurations of the causal
            # minimum-statistics post-filter and ALL of them degraded Output SNR (best: -0.92 dB);
            # it is a confirmed dead end and is off everywhere else in the project. Leaving it
            # True here silently applied a filter to any figure this script produced, which would
            # not match the published numbers. See eval/DRDO_TARGETS_EVALUATION.md sec 9.5.
            enable_nlms_post=False
        )
        enh = res["enhanced"]
        rtf = res["rtf"]

        enh_s = compute_snr(clean, enh)
        gain = enh_s - raw_s
        raw_st = compute_stoi(clean, noisy)
        enh_st = compute_stoi(clean, enh)
        raw_p = compute_pesq(clean, noisy)
        enh_p = compute_pesq(clean, enh)

        results[ntype]["in_snr"].append(raw_s)
        results[ntype]["out_snr"].append(enh_s)
        results[ntype]["gain"].append(gain)
        results[ntype]["raw_stoi"].append(raw_st)
        results[ntype]["enh_stoi"].append(enh_st)
        results[ntype]["raw_pesq"].append(raw_p)
        results[ntype]["enh_pesq"].append(enh_p)
        results[ntype]["rtf"].append(rtf)

        class_counts[ntype] += 1
        total_collected += 1

    return results


def main():
    ckpts = [
        ("Fix A (Clamped-Alpha)", "checkpoints/clamped_alpha/best_model.pt"),
        ("Fix B (Hinge-Penalty)", "checkpoints/hinge_penalty/best_model.pt")
    ]
    noise_classes = [
        "gunshot", "helicopter", "armored_vehicle", "jet_flyby",
        "siren", "cockpit", "radio_static", "babble"
    ]

    print("================================================================================")
    print(" TASK 2F: Comprehensive 8-Scenario Empirical Comparison (Fix A vs Fix B)")
    print(" Evaluating exactly 10 samples per scenario (80 samples per condition)")
    print("================================================================================")

    # 1. Operational Range
    print("\n--- 1. OPERATIONAL RANGE (0.0 dB to +15.0 dB Input SNR) ---")
    op_results = {}
    for name, path in ckpts:
        op_results[name] = evaluate_balanced_scenarios(path, snr_min=0.0, snr_max=15.0, samples_per_class=10, seed=1010)

    op_table = []
    for nc in noise_classes:
        a = op_results["Fix A (Clamped-Alpha)"][nc]
        b = op_results["Fix B (Hinge-Penalty)"][nc]
        
        in_snr_a = np.mean(a["in_snr"])
        out_snr_a = np.mean(a["out_snr"])
        gain_a = np.mean(a["gain"])
        stoi_a = np.mean(a["enh_stoi"])
        pesq_a = np.mean(a["enh_pesq"])

        in_snr_b = np.mean(b["in_snr"])
        out_snr_b = np.mean(b["out_snr"])
        gain_b = np.mean(b["gain"])
        stoi_b = np.mean(b["enh_stoi"])
        pesq_b = np.mean(b["enh_pesq"])

        op_table.append([
            nc.replace("_", " ").title(),
            f"{in_snr_a:.1f} dB",
            f"{out_snr_a:.1f} dB",
            f"{gain_a:+.1f} dB",
            f"{stoi_a:.3f}",
            f"{pesq_a:.2f}",
            f"{out_snr_b:.1f} dB",
            f"{gain_b:+.1f} dB",
            f"{stoi_b:.3f}",
            f"{pesq_b:.2f}"
        ])

    op_headers = [
        "Scenario", "Input SNR",
        "Fix A Out", "Fix A Gain", "Fix A STOI", "Fix A PESQ",
        "Fix B Out", "Fix B Gain", "Fix B STOI", "Fix B PESQ"
    ]
    print(tabulate(op_table, headers=op_headers, tablefmt="grid"))

    # 2. Stress Test Range
    print("\n--- 2. DEGRADED CHANNEL STRESS TEST (-10.0 dB to -0.01 dB Input SNR) ---")
    stress_results = {}
    for name, path in ckpts:
        stress_results[name] = evaluate_balanced_scenarios(path, snr_min=-10.0, snr_max=-0.01, samples_per_class=10, seed=2020)

    stress_table = []
    for nc in noise_classes:
        a = stress_results["Fix A (Clamped-Alpha)"][nc]
        b = stress_results["Fix B (Hinge-Penalty)"][nc]
        
        in_snr_a = np.mean(a["in_snr"])
        out_snr_a = np.mean(a["out_snr"])
        gain_a = np.mean(a["gain"])
        stoi_a = np.mean(a["enh_stoi"])
        pesq_a = np.mean(a["enh_pesq"])

        in_snr_b = np.mean(b["in_snr"])
        out_snr_b = np.mean(b["out_snr"])
        gain_b = np.mean(b["gain"])
        stoi_b = np.mean(b["enh_stoi"])
        pesq_b = np.mean(b["enh_pesq"])

        stress_table.append([
            nc.replace("_", " ").title(),
            f"{in_snr_a:.1f} dB",
            f"{out_snr_a:.1f} dB",
            f"{gain_a:+.1f} dB",
            f"{stoi_a:.3f}",
            f"{pesq_a:.2f}",
            f"{out_snr_b:.1f} dB",
            f"{gain_b:+.1f} dB",
            f"{stoi_b:.3f}",
            f"{pesq_b:.2f}"
        ])

    stress_headers = [
        "Scenario", "Input SNR",
        "Fix A Out", "Fix A Gain", "Fix A STOI", "Fix A PESQ",
        "Fix B Out", "Fix B Gain", "Fix B STOI", "Fix B PESQ"
    ]
    print(tabulate(stress_table, headers=stress_headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
