"""
Section A.7 — Reconcile the two aggregate SNR numbers in generated eval output
(6.96 dB in eval/DRDO_TARGETS_EVALUATION.html vs 12.75 dB headline), and characterize
run-to-run variance of the locked eval configuration under fixed seeds.

Isolates the ONE variable that plausibly explains most of the gap (the NLMS post-filter,
which the older harness enables and the locked harness does not) by running the same
utterances through both paths.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr, compute_stoi

BUCKETS = [
    ("Operational [0, 5) dB", 0.0, 5.0, 5000),
    ("Operational [5, 10) dB", 5.0, 10.0, 5500),
    ("Operational [10, 15] dB", 10.0, 15.0, 6000),
]
CKPT = "checkpoints/task4_full_run/best_model.pt"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)

    from demo.stream_engine import RealTimeANCEngine
    eng = RealTimeANCEngine(checkpoint_path=CKPT)

    print("=" * 90)
    print("A.7 RECONCILIATION: locked eval path vs older evaluate_drdo_targets.py path")
    print("=" * 90)
    print(f"  device={dev}  checkpoint={CKPT}")

    # ---------- Part 1: run-to-run variance of the LOCKED path, fixed seeds ----------
    print("\n--- Part 1: run-to-run variance, locked config, 3 repeats, fixed seeds ---")
    runs = []
    for rep in range(3):
        vals = []
        for name, lo, hi, seed in BUCKETS:
            ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000,
                                   segment_len_sec=2.0, epoch_size=100,
                                   snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
            with torch.no_grad():
                for i in range(len(ds)):
                    it = ds[i]
                    enh, _, _, _ = m(it["noisy"].unsqueeze(0).to(dev))
                    vals.append(compute_snr(it["clean"].numpy(), enh.squeeze(0).cpu().numpy()))
        runs.append(np.array(vals))
        print(f"  repeat {rep+1}: N={len(vals)}  mean Output SNR = {np.mean(vals):.6f} dB")
    d12 = float(np.max(np.abs(runs[0] - runs[1])))
    d13 = float(np.max(np.abs(runs[0] - runs[2])))
    print(f"  max per-utterance |run1-run2| = {d12:.3e}")
    print(f"  max per-utterance |run1-run3| = {d13:.3e}")
    print(f"  spread of means = {max(np.mean(r) for r in runs) - min(np.mean(r) for r in runs):.3e} dB")
    print(f"  VERDICT: {'fully deterministic' if max(d12,d13)==0 else 'non-deterministic - see spread'}")

    # ---------- Part 2: isolate the NLMS variable on identical utterances ----------
    print("\n--- Part 2: same utterances, direct model vs stream-engine WITH NLMS ---")
    direct, with_nlms, without_nlms = [], [], []
    t0 = time.time()
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
                direct.append(compute_snr(clean, enh.squeeze(0).cpu().numpy()))
                r_on = eng.process_audio(noisy, backend="CPU", enable_neural=True, enable_nlms_post=True)
                with_nlms.append(compute_snr(clean, r_on["enhanced"]))
                r_off = eng.process_audio(noisy, backend="CPU", enable_neural=True, enable_nlms_post=False)
                without_nlms.append(compute_snr(clean, r_off["enhanced"]))
    print(f"  ({time.time()-t0:.0f}s, N={len(direct)})")
    print(f"  direct model forward (locked path) : {np.mean(direct):7.2f} dB")
    print(f"  stream engine, NLMS OFF            : {np.mean(without_nlms):7.2f} dB")
    print(f"  stream engine, NLMS ON             : {np.mean(with_nlms):7.2f} dB")
    print(f"  --> cost attributable to NLMS      : {np.mean(with_nlms)-np.mean(without_nlms):+7.2f} dB")
    print(f"  --> stream-vs-direct pipeline delta: {np.mean(without_nlms)-np.mean(direct):+7.2f} dB")


if __name__ == "__main__":
    main()
