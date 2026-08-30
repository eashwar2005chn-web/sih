"""
Section C.1 Step 0 — Independent re-derivation of the mask-shape ceiling.

The project's strongest diagnostic (Phase 8) reported that the IDEAL complex ratio mask,
pushed through the model's tanh activation, caps output SNR at ~12.8 dB -- matching the
trained model's ceiling almost exactly -- while the SAME ideal mask under a LINEAR CLAMP to
the SAME [-1, 1] bound reaches 23-31 dB.

If that is right, the ceiling is not the bound magnitude at all (which Phases 9-10 swept via
K, and which always hurt). It is the *curvature of tanh inside* the bound. tanh(1)=0.76, so
a mask that should pass a bin at unity gain only ever passes 76% of it; energy is
systematically removed from speech-dominated bins.

This probe recomputes it from scratch, on the VALIDATION split (never used for any prior
decision), so the conclusion does not depend on trusting the earlier run.

No model, no training -- this is a property of the output parameterisation alone.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.loss import compute_snr

N_FFT, HOP = 512, 256
BUCKETS = [
    ("Stress [-10,-5)", -10.0, -5.0, 91000),
    ("Stress [-5,0)", -5.0, 0.0, 91500),
    ("Op [0,5)", 0.0, 5.0, 92000),
    ("Op [5,10)", 5.0, 10.0, 92500),
    ("Op [10,15]", 10.0, 15.0, 93000),
]
PER_BUCKET = 40


def stft(x, win):
    return torch.stft(x, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)


def istft(X, win, n):
    return torch.istft(X, N_FFT, HOP, N_FFT, win, center=True, length=n)


def apply_and_score(clean_w, noisy_w, win, shaper):
    """Build the ideal cRM from clean/noisy, reshape it with `shaper`, reconstruct, score."""
    S = stft(clean_w, win)
    Y = stft(noisy_w, win)
    # Ideal complex ratio mask M = S / Y (guard only the truly-zero bins).
    denom = Y.real ** 2 + Y.imag ** 2
    safe = denom > 1e-10
    Mr = torch.zeros_like(Y.real)
    Mi = torch.zeros_like(Y.imag)
    Mr[safe] = ((S.real * Y.real + S.imag * Y.imag)[safe]) / denom[safe]
    Mi[safe] = ((S.imag * Y.real - S.real * Y.imag)[safe]) / denom[safe]

    Mr2, Mi2 = shaper(Mr, Mi)
    Er = Mr2 * Y.real - Mi2 * Y.imag
    Ei = Mr2 * Y.imag + Mi2 * Y.real
    out = istft(torch.complex(Er, Ei), win, clean_w.shape[-1])
    return compute_snr(clean_w.squeeze(0).numpy(), out.squeeze(0).numpy())


SHAPERS = {
    "ideal (no shaping)":        lambda r, i: (r, i),
    "tanh  [current model]":     lambda r, i: (torch.tanh(r), torch.tanh(i)),
    "linear clamp [-1,1]":       lambda r, i: (r.clamp(-1, 1), i.clamp(-1, 1)),
    "hard-sigmoid-ish (x/1+|x|)": lambda r, i: (r / (1 + r.abs()), i / (1 + i.abs())),
    "clamp [-2,2]":              lambda r, i: (r.clamp(-2, 2), i.clamp(-2, 2)),
    "tanh K=2 (K*tanh(x/K))":    lambda r, i: (2 * torch.tanh(r / 2), 2 * torch.tanh(i / 2)),
}


def main():
    win = torch.hann_window(N_FFT)
    print("=" * 96)
    print("ORACLE MASK-SHAPE PROBE  (validation split, ideal cRM, no trained model involved)")
    print("=" * 96)

    results = {k: {b[0]: [] for b in BUCKETS} for k in SHAPERS}

    for name, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(mode="hybrid", split="val", sample_rate=16000,
                               segment_len_sec=2.0, epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
        for i in range(len(ds)):
            it = ds[i]
            c = it["clean"].unsqueeze(0)
            y = it["noisy"].unsqueeze(0)
            if not np.any(it["noisy"].numpy()):
                continue
            for k, fn in SHAPERS.items():
                results[k][name].append(apply_and_score(c, y, win, fn))
        print(f"  [done] {name}")

    hdr = f"\n{'mask shaping':30}" + "".join(f"{b[0]:>18}" for b in BUCKETS) + f"{'MEAN':>10}"
    print(hdr)
    print("-" * len(hdr))
    means = {}
    for k in SHAPERS:
        row = f"{k:30}"
        allv = []
        for b in BUCKETS:
            v = results[k][b[0]]
            row += f"{np.mean(v):>18.2f}"
            allv += v
        means[k] = float(np.mean(allv))
        row += f"{means[k]:>10.2f}"
        print(row)

    print("\n" + "=" * 96)
    t = means["tanh  [current model]"]
    c = means["linear clamp [-1,1]"]
    print(f"  tanh (what the model uses)      : {t:6.2f} dB")
    print(f"  linear clamp, SAME [-1,1] bound : {c:6.2f} dB")
    print(f"  headroom unlocked by shape alone: {c - t:+6.2f} dB")
    print()
    if c - t > 5.0:
        print("  => CONFIRMED. The ceiling is the SHAPE of the squashing function, not the bound.")
        print("     Phases 9-10 swept the BOUND (K=1.5..3.0) and always hurt, because widening")
        print("     the bound permits noise amplification. Changing the SHAPE at a FIXED unit")
        print("     bound is a different intervention entirely -- and it has never been tested.")
    else:
        print("  => NOT confirmed on this split. Do not proceed with the linear-clamp retrain.")


if __name__ == "__main__":
    main()
