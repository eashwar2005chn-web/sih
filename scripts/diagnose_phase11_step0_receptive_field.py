"""
Phase 11 Step 0: Empirical Temporal Receptive Field Diagnostic & Noise Periodicity Analysis
"""

import os
import sys
import glob
import numpy as np
import torch
import soundfile as sf
import scipy.signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from data.defence_noise_generator import DefenceNoiseGenerator


def measure_empirical_receptive_field(model, dataset, device="cpu", num_samples=50):
    """
    Measures the baseline model's effective temporal receptive field by perturbing
    input frame (t - k) and measuring the L2 perturbation of the predicted mask at frame t.
    """
    model.eval()
    model.to(device)

    lags = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128]
    perturbation_magnitudes = {k: [] for k in lags}

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            noisy_wav = item["noisy"].unsqueeze(0).to(device)  # (1, T_samples)

            # STFT
            n_real, n_imag, _ = model.stft_forward(noisy_wav)  # (1, F, T_frames)
            num_frames = n_real.shape[-1]
            if num_frames < 120:
                continue

            # Unperturbed mask
            _, _, mask_r_orig, mask_i_orig, _ = model.forward_spec(n_real, n_imag)
            m_orig = torch.complex(mask_r_orig, mask_i_orig)  # (1, F, T)

            # Evaluate at several interior target frames t
            target_frames = [f for f in [60, 80, 100] if f < num_frames - 5]
            for t_target in target_frames:
                base_val = m_orig[:, :, t_target]  # (1, F)

                # Reference perturbation at lag k=1
                for k in lags:
                    if t_target - k < 0:
                        continue
                    # Create perturbed spectrogram: zero out frame (t_target - k)
                    n_real_pert = n_real.clone()
                    n_imag_pert = n_imag.clone()
                    n_real_pert[:, :, t_target - k] = 0.0
                    n_imag_pert[:, :, t_target - k] = 0.0

                    _, _, mr_p, mi_p, _ = model.forward_spec(n_real_pert, n_imag_pert)
                    m_pert = torch.complex(mr_p, mi_p)
                    pert_val = m_pert[:, :, t_target]

                    diff = torch.norm(pert_val - base_val, p=2).item()
                    perturbation_magnitudes[k].append(diff)

    mean_perturbations = {k: float(np.mean(perturbation_magnitudes[k])) if perturbation_magnitudes[k] else 0.0 for k in lags}
    ref_k1 = mean_perturbations[1] if mean_perturbations[1] > 0 else 1e-6
    rel_perturbations = {k: mean_perturbations[k] / ref_k1 for k in lags}

    # Find cutoff lag where relative effect drops below 10% (0.10)
    cutoff_lag = lags[-1]
    for k in sorted(lags):
        if rel_perturbations[k] < 0.10:
            cutoff_lag = k
            break

    hop_ms = (model.hop_length / 16000.0) * 1000.0  # 16 ms @ hop 256
    erf_ms = cutoff_lag * hop_ms

    return {
        "lags": lags,
        "mean_diff": mean_perturbations,
        "relative_diff": rel_perturbations,
        "cutoff_lag_frames": cutoff_lag,
        "hop_ms": hop_ms,
        "erf_ms": erf_ms
    }


def analyze_tactical_noise_periodicity(sr=16000):
    """
    Analyzes dominant periodicities in tactical noise classes (Jet, Tank, Rotor, Drone, Gunshot)
    using autocorrelation and power spectral peak spacing.
    """
    noise_profiles = {}
    gen = DefenceNoiseGenerator(sample_rate=sr)

    # 1. Synthetic Tactical Profiles
    generators = {
        "Synthetic Helicopter Rotor": lambda: gen.generate_helicopter_rotor(duration_sec=3.0),
        "Synthetic Armored Tank Engine": lambda: gen.generate_armored_vehicle_engine(duration_sec=3.0),
        "Synthetic Fighter Jet Flyby": lambda: gen.generate_jet_flyby(duration_sec=3.0),
        "Synthetic Gunfire Burst": lambda: gen.generate_impulsive_gunshot(duration_sec=3.0)
    }

    for name, gen_fn in generators.items():
        wav = gen_fn()
        # Autocorrelation
        wav_center = wav[sr : 2 * sr]  # 1-sec clip
        wav_center = wav_center - np.mean(wav_center)
        autocorr = np.correlate(wav_center, wav_center, mode="full")
        autocorr = autocorr[len(autocorr)//2 :]
        autocorr = autocorr / (autocorr[0] + 1e-12)

        # Find peaks between 20 Hz and 1000 Hz (lag: sr/1000 to sr/20 -> 16 to 800 samples)
        min_lag = int(sr / 1000)  # 16 samples = 1 ms (1000 Hz)
        max_lag = int(sr / 20)    # 800 samples = 50 ms (20 Hz)
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) > 0:
            peak_idx = np.argmax(search_region) + min_lag
            peak_val = autocorr[peak_idx]
            f0 = sr / peak_idx
            period_ms = (peak_idx / sr) * 1000.0
        else:
            f0 = 0.0
            period_ms = 0.0
            peak_val = 0.0

        noise_profiles[name] = {
            "dominant_f0_hz": f0,
            "period_ms": period_ms,
            "period_frames_at_16ms_hop": period_ms / 16.0,
            "periodicity_strength": peak_val
        }

    # 2. Real Tactical Files (if available)
    real_files = glob.glob("datasets/tactical_noise/**/*.wav", recursive=True)
    if not real_files:
        real_files = glob.glob("data/tactical_noise/**/*.wav", recursive=True)

    if real_files:
        sample_files = real_files[:10]
        f0_list, per_list, str_list = [], [], []
        for rf in sample_files:
            try:
                data, file_sr = sf.read(rf)
                if len(data.shape) > 1:
                    data = data[:, 0]
                if file_sr != sr:
                    data = scipy.signal.resample_poly(data, sr, file_sr)
                if len(data) > sr:
                    clip = data[:sr] - np.mean(data[:sr])
                    ac = np.correlate(clip, clip, mode="full")
                    ac = ac[len(ac)//2 :]
                    ac = ac / (ac[0] + 1e-12)
                    min_lag = int(sr / 1000)
                    max_lag = int(sr / 20)
                    srch = ac[min_lag:max_lag]
                    if len(srch) > 0:
                        p_idx = np.argmax(srch) + min_lag
                        f0_list.append(sr / p_idx)
                        per_list.append((p_idx / sr) * 1000.0)
                        str_list.append(ac[p_idx])
            except Exception:
                pass

        if f0_list:
            noise_profiles["Real Tactical Corpus (MAD/AudioSet Avg)"] = {
                "dominant_f0_hz": float(np.mean(f0_list)),
                "period_ms": float(np.mean(per_list)),
                "period_frames_at_16ms_hop": float(np.mean(per_list)) / 16.0,
                "periodicity_strength": float(np.mean(str_list))
            }

    return noise_profiles


def main():
    print("================================================================================")
    print(" PHASE 11 STEP 0: TEMPORAL RECEPTIVE FIELD DIAGNOSTIC & NOISE PERIODICITY")
    print("================================================================================")

    # 1. Load Baseline Model
    base_ckpt = torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")
    model = build_causal_anc_model(hidden_dim=128)
    model.load_state_dict(base_ckpt["model_state_dict"])
    print("Loaded Baseline CausalANCNet (811K params).")

    # 2. Load Dataset
    ds = HybridRealDataset(mode="hybrid", split="test", epoch_size=100, seed=5000)
    print(f"Loaded test dataset ({len(ds)} samples).")

    # 3. Measure Effective Temporal Receptive Field
    print("\nMeasuring Empirical Effective Receptive Field (ERF)...")
    erf_results = measure_empirical_receptive_field(model, ds, device="cpu", num_samples=40)

    print("\n--- Empirical Receptive Field Decay Curve ---")
    print(f"{'Lag (frames)':<15}{'Lag (ms)':<15}{'Mean L2 Mask Diff':<22}{'Relative Impact (% of k=1)':<25}")
    print("-" * 75)
    for k in erf_results["lags"]:
        lag_ms = k * erf_results["hop_ms"]
        diff = erf_results["mean_diff"][k]
        rel = erf_results["relative_diff"][k] * 100.0
        print(f"{k:<15}{lag_ms:<15.1f}{diff:<22.5f}{rel:<25.2f}%")

    print("-" * 75)
    print(f"Empirical ERF Cutoff (<10% threshold): {erf_results['cutoff_lag_frames']} frames ({erf_results['erf_ms']:.1f} ms)")

    # 4. Analyze Target Noise Periodicity
    print("\nAnalyzing Target Tactical Noise Periodicity...")
    noise_profiles = analyze_tactical_noise_periodicity()

    print("\n--- Tactical Noise Periodicity Summary ---")
    print(f"{'Noise Type':<35}{'Dominant f0 (Hz)':<20}{'Period (ms)':<15}{'Period (frames)':<18}{'Periodicity (corr)':<20}")
    print("-" * 105)
    for name, p in noise_profiles.items():
        print(f"{name:<35}{p['dominant_f0_hz']:<20.1f}{p['period_ms']:<15.2f}{p['period_frames_at_16ms_hop']:<18.2f}{p['periodicity_strength']:<20.3f}")

    print("-" * 105)

    # 5. Synthesis
    max_noise_period_ms = max(p["period_ms"] for p in noise_profiles.values())
    print("\n--- Step 0 Diagnostic Synthesis ---")
    print(f"1. Model Empirical Temporal Receptive Field: {erf_results['erf_ms']:.1f} ms ({erf_results['cutoff_lag_frames']} frames)")
    print(f"2. Max Fundamental Period across Noise Types: {max_noise_period_ms:.2f} ms")
    if erf_results["erf_ms"] >= max_noise_period_ms * 3:
        print(f"3. Assessment: The baseline model's GRU already captures {erf_results['erf_ms']:.1f} ms of left context, which spans {erf_results['erf_ms']/max_noise_period_ms:.1f}x the dominant acoustic period.")
        print("   However, local convolutional feature extraction prior to the GRU is restricted to kernel size (3,3) causal convs (receptive field = ~6 frames / 96 ms in the CNN encoder).")
        print("   A dilated causal TCN stack will test whether explicit multi-scale temporal convolutions over 300-500 ms provide structural predictive advantage over pure recurrent memory.")
    else:
        print(f"3. Assessment: ERF is {erf_results['erf_ms']:.1f} ms, potentially shorter than multi-cycle harmonics.")


if __name__ == "__main__":
    main()
