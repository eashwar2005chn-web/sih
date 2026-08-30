"""
Section A.5 — Audit the "silent substitution" bug class across the metric stack.

Two instances found by code inspection in model/loss.py:
  (i)  compute_stoi(): if pystoi raises, silently returns a *correlation proxy*
       0.5*(r+1) that is NOT STOI, with no log or flag.
  (ii) compute_pesq(): if n_frames <= 0, returns the literal 2.5 -- which is exactly
       the project's PESQ pass threshold (>2.50).

This script instruments both paths and replays the real N=500 held-out test set to
count how many reported values actually came from a fallback rather than the real
metric. Read-only; changes nothing.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model.loss as L
from data.real_dataset_loader import HybridRealDataset

BUCKETS = [
    ("Stress [-10, -5) dB", -10.0, -5.0, 7000),
    ("Stress [-5, 0) dB", -5.0, 0.0, 7500),
    ("Operational [0, 5) dB", 0.0, 5.0, 5000),
    ("Operational [5, 10) dB", 5.0, 10.0, 5500),
    ("Operational [10, 15] dB", 10.0, 15.0, 6000),
]

counters = {"stoi_calls": 0, "stoi_fallback": 0, "pesq_calls": 0, "pesq_hardcoded": 0,
            "pesq_real_itu": 0, "pesq_surrogate": 0}
fallback_examples = []


def instrumented_stoi(clean, enhanced, sr=16000):
    counters["stoi_calls"] += 1
    if L.HAS_PYSTOI:
        try:
            import pystoi
            return float(pystoi.stoi(clean, enhanced, sr, extended=False))
        except Exception as e:
            counters["stoi_fallback"] += 1
            if len(fallback_examples) < 6:
                fallback_examples.append(f"STOI fallback fired: {type(e).__name__}: {e}")
    r = np.corrcoef(clean, enhanced)[0, 1]
    return float(np.clip(0.5 * (r + 1.0), 0.0, 1.0))


def instrumented_pesq(clean, enhanced, sr=16000):
    counters["pesq_calls"] += 1
    min_len = min(len(clean), len(enhanced))
    c = clean[:min_len].astype(np.float32)
    e = enhanced[:min_len].astype(np.float32)
    if L.HAS_PESQ:
        counters["pesq_real_itu"] += 1
        return L.compute_pesq(c, e, sr)
    n_frames = (min_len - 512) // 256
    if n_frames <= 0:
        counters["pesq_hardcoded"] += 1
        if len(fallback_examples) < 12:
            fallback_examples.append(f"PESQ hardcoded 2.5 fired: min_len={min_len}")
        return 2.5
    counters["pesq_surrogate"] += 1
    return L.compute_pesq(c, e, sr)


def main():
    print("=" * 78)
    print("SECTION A.5 - SILENT-FALLBACK ACTIVATION AUDIT (replay of real N=500 test set)")
    print("=" * 78)
    print(f"  pystoi available : {L.HAS_PYSTOI}")
    print(f"  ITU pesq available: {L.HAS_PESQ}")

    import torch
    from model.causal_anc_net import build_causal_anc_model
    ck = torch.load("checkpoints/task4_full_run/best_model.pt", map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)

    for name, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000,
                               segment_len_sec=2.0, epoch_size=100,
                               snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
        with torch.no_grad():
            for i in range(len(ds)):
                it = ds[i]
                clean = it["clean"].numpy()
                noisy = it["noisy"].numpy()
                enh, _, _, _ = m(it["noisy"].unsqueeze(0).to(dev))
                enh = enh.squeeze(0).cpu().numpy()
                # replicate exactly what evaluate_batch_metrics computes
                instrumented_stoi(clean, noisy)
                instrumented_stoi(clean, enh)
                instrumented_pesq(clean, noisy)
                instrumented_pesq(clean, enh)
        print(f"  [done] {name}")

    print("\n--- RESULTS ---")
    print(f"  compute_stoi calls           : {counters['stoi_calls']}")
    print(f"  ...silently used CORR PROXY  : {counters['stoi_fallback']}"
          f"   ({100*counters['stoi_fallback']/max(counters['stoi_calls'],1):.2f}%)")
    print(f"  compute_pesq calls           : {counters['pesq_calls']}")
    print(f"  ...returned hardcoded 2.5    : {counters['pesq_hardcoded']}"
          f"   ({100*counters['pesq_hardcoded']/max(counters['pesq_calls'],1):.2f}%)")
    print(f"  ...used real ITU-T P.862     : {counters['pesq_real_itu']}")
    print(f"  ...used Bark surrogate       : {counters['pesq_surrogate']}")
    if fallback_examples:
        print("\n  fallback examples:")
        for x in fallback_examples:
            print("   ", x)
    else:
        print("\n  No fallback ever fired on the real test set.")

    verdict_stoi = "CLEAN" if counters["stoi_fallback"] == 0 else "CONTAMINATED"
    verdict_pesq = "CLEAN" if counters["pesq_hardcoded"] == 0 else "CONTAMINATED"
    print(f"\n  VERDICT  reported STOI values : {verdict_stoi}")
    print(f"  VERDICT  reported PESQ values : {verdict_pesq} (surrogate-vs-ITU caveat is separate)")


if __name__ == "__main__":
    main()
