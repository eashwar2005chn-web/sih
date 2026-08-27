"""
Phase 11 Step 1: Causality Verification & Back-to-Back CPU Latency/RTF Profiling
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_anc_net import build_causal_anc_model, build_tcn_causal_anc_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_strict_causality(model):
    """
    Verifies that model output at frame t depends ONLY on frames <= t.
    Modifies input at frame t_future > t and checks if frame t changes.
    """
    model.eval()
    model.to("cpu")

    B, T_samples = 1, 32000  # 2.0 sec
    x1 = torch.randn(B, T_samples)
    x2 = x1.clone()

    # Modify strictly the second half of the audio (from sample 20000 onwards)
    split_sample = 20000
    x2[:, split_sample:] = torch.randn(B, T_samples - split_sample)

    with torch.no_grad():
        out1, _, _, _ = model(x1)
        out2, _, _, _ = model(x2)

    # In STFT with win_length=512, hop=256, frames strictly before split_sample should be identical
    # Calculate sample index safe margin: split_sample - win_length
    safe_sample = split_sample - model.win_length
    diff = torch.max(torch.abs(out1[:, :safe_sample] - out2[:, :safe_sample])).item()
    is_causal = diff < 1e-5
    return is_causal, diff


def benchmark_cpu_latency(model, n_frames=1000, sr=16000, hop_size=256):
    model.eval()
    model.to("cpu")
    audio_dur = (n_frames * hop_size) / sr
    dummy_wav = torch.randn(1, n_frames * hop_size)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_wav[:, :2048])

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_wav)
    t1 = time.perf_counter()

    wall_time = t1 - t0
    rtf = wall_time / audio_dur
    latency_ms = (wall_time / n_frames) * 1000.0
    return latency_ms, rtf


def main():
    print("================================================================================")
    print(" PHASE 11 STEP 1: ARCHITECTURE PROFILE & STRICT CAUSALITY VERIFICATION")
    print("================================================================================")

    # 1. Models
    m_base = build_causal_anc_model(hidden_dim=128)
    m_tcn = build_tcn_causal_anc_model(hidden_dim=128, tcn_dilations=(1, 2, 4, 8, 16))

    params_base = count_parameters(m_base)
    params_tcn = count_parameters(m_tcn)

    print(f"Baseline CausalANCNet Parameters : {params_base:,}")
    print(f"CausalANCTCNNet Parameters       : {params_tcn:,} (+{params_tcn - params_base:,} params)")

    # 2. Causality Verification
    print("\n--- Verifying Causality ---")
    causal_base, diff_base = verify_strict_causality(m_base)
    causal_tcn, diff_tcn = verify_strict_causality(m_tcn)

    print(f"Baseline Causality Check: {'PASSED' if causal_base else 'FAILED'} (Max future leakage diff: {diff_base:.2e})")
    print(f"TCN Stack Causality Check: {'PASSED' if causal_tcn else 'FAILED'} (Max future leakage diff: {diff_tcn:.2e})")

    # 3. CPU Latency and RTF Benchmark
    print("\n--- Back-to-Back CPU Latency & RTF Benchmark (1000 frames) ---")
    lat_base, rtf_base = benchmark_cpu_latency(m_base)
    lat_tcn, rtf_tcn = benchmark_cpu_latency(m_tcn)

    print(f"{'Model':<30}{'Parameters':<15}{'CPU Latency/Frame':<20}{'CPU RTF':<15}{'DRDO Target':<15}")
    print("-" * 95)
    print(f"{'Baseline CausalANCNet':<30}{params_base:<15,}{lat_base:<20.3f} ms{rtf_base:<15.4f}{'< 1.0 ms / < 0.100':<15}")
    print(f"{'CausalANCTCNNet (5 dilations)':<30}{params_tcn:<15,}{lat_tcn:<20.3f} ms{rtf_tcn:<15.4f}{'< 1.0 ms / < 0.100':<15}")
    print("-" * 95)

    if lat_tcn < 1.0 and rtf_tcn < 0.100:
        print(">>> CPU Compute Profile: PASSED! Well within real-time DRDO budget (>10x margin). Ready for Step 2.")
    else:
        print(">>> WARNING: CPU profile exceeds budget! Needs channel/depth adjustment.")


if __name__ == "__main__":
    main()
