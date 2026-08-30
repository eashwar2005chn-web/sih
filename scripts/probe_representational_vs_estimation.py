"""
Phase 17 — Is the 12.75 dB ceiling REPRESENTATIONAL or ESTIMATION-limited?

Phase 8 concluded: "pushing the ideal mask through tanh caps SNR at ~12.9 dB, which matches
the trained model exactly, therefore the tanh parameterisation IS the ceiling."

That inference contains a logical error. tanh(M_ideal) is NOT the set of masks a
tanh-output network can produce. The network emits tanh(f(x)) for a learned f, and f can
output any real number -- tanh(3)=0.995, tanh(5)=0.9999. So the REACHABLE SET of a
tanh-output head is the open interval (-1, 1), which is essentially the same reachable set
as a linear clamp to [-1, 1].

The oracle already measured that set: clamp[-1,1] = 28.52 dB.

If so, the trained model at 12.75 dB is nowhere near its representational ceiling, and the
gap is OPTIMISATION / ESTIMATION -- which is fixable by training, not by re-parameterising.

This probe settles it three ways:
  (A) SNR reachable by tanh-output head  = clamp ideal mask to tanh's reachable range
  (B) What the TRAINED model's masks actually look like vs the ideal mask
  (C) Whether the trained model's pre-activations are saturating
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr

N_FFT, HOP = 512, 256
CKPT = "checkpoints/task4_full_run/best_model.pt"
BUCKETS = [("[-10,-5)", -10.0, -5.0, 91000), ("[-5,0)", -5.0, 0.0, 91500),
           ("[0,5)", 0.0, 5.0, 92000), ("[5,10)", 5.0, 10.0, 92500),
           ("[10,15]", 10.0, 15.0, 93000)]
PER_BUCKET = 30


def ideal_mask(S, Y):
    den = Y.real ** 2 + Y.imag ** 2
    ok = den > 1e-10
    Mr = torch.zeros_like(Y.real); Mi = torch.zeros_like(Y.imag)
    Mr[ok] = ((S.real * Y.real + S.imag * Y.imag)[ok]) / den[ok]
    Mi[ok] = ((S.imag * Y.real - S.real * Y.imag)[ok]) / den[ok]
    return Mr, Mi


def recon(Mr, Mi, Y, win, n):
    Er = Mr * Y.real - Mi * Y.imag
    Ei = Mr * Y.imag + Mi * Y.real
    return torch.istft(torch.complex(Er, Ei), N_FFT, HOP, N_FFT, win, center=True, length=n)


def main():
    win = torch.hann_window(N_FFT)
    ck = torch.load(CKPT, map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128)
    m.load_state_dict(ck["model_state_dict"]); m.eval()

    rows = {k: [] for k in ["model", "tanh(M*)", "clamp(M*) [reachable]", "M* raw"]}
    mag_model, mag_ideal_clipped, sat_frac = [], [], []

    print("=" * 92)
    print("PHASE 17 — representational vs estimation ceiling (validation split)")
    print("=" * 92)

    for name, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(mode="hybrid", split="val", sample_rate=16000, segment_len_sec=2.0,
                               epoch_size=PER_BUCKET, snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
        for i in range(len(ds)):
            it = ds[i]
            if not np.any(it["noisy"].numpy()):
                continue
            c = it["clean"].unsqueeze(0); y = it["noisy"].unsqueeze(0)
            n = c.shape[-1]
            S = torch.stft(c, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
            Y = torch.stft(y, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
            Mr, Mi = ideal_mask(S, Y)
            cn = c.squeeze(0).numpy()

            with torch.no_grad():
                _, _, mr_hat, mi_hat, _ = m.forward_spec(Y.real, Y.imag, None)

            rows["M* raw"].append(compute_snr(cn, recon(Mr, Mi, Y, win, n).squeeze(0).numpy()))
            rows["tanh(M*)"].append(compute_snr(cn, recon(torch.tanh(Mr), torch.tanh(Mi), Y, win, n).squeeze(0).numpy()))
            # tanh's reachable set == (-1,1); use 0.999 to stay strictly inside it
            rows["clamp(M*) [reachable]"].append(
                compute_snr(cn, recon(Mr.clamp(-0.999, 0.999), Mi.clamp(-0.999, 0.999), Y, win, n).squeeze(0).numpy()))
            rows["model"].append(compute_snr(cn, recon(mr_hat, mi_hat, Y, win, n).squeeze(0).numpy()))

            mag_model.append(float(torch.sqrt(mr_hat ** 2 + mi_hat ** 2).mean()))
            tgt = torch.sqrt(Mr.clamp(-1, 1) ** 2 + Mi.clamp(-1, 1) ** 2)
            mag_ideal_clipped.append(float(tgt.mean()))
            sat_frac.append(float((torch.sqrt(mr_hat ** 2 + mi_hat ** 2) > 0.95).float().mean()))
        print(f"  [done] {name}")

    print(f"\n{'configuration':32} {'mean SNR':>10}")
    print("-" * 46)
    for k in ["model", "tanh(M*)", "clamp(M*) [reachable]", "M* raw"]:
        print(f"{k:32} {np.mean(rows[k]):>10.2f} dB")

    reach = np.mean(rows["clamp(M*) [reachable]"])
    got = np.mean(rows["model"])
    tanhm = np.mean(rows["tanh(M*)"])

    print("\n" + "=" * 92)
    print(f"  What a tanh-output head CAN represent : {reach:6.2f} dB")
    print(f"  What the trained model achieves       : {got:6.2f} dB")
    print(f"  UNREALISED headroom inside the SAME architecture: {reach - got:+6.2f} dB")
    print(f"\n  (Phase 8's tanh(M*) figure, for reference: {tanhm:6.2f} dB -- this is NOT the")
    print( "   architecture's limit, it is just 'the ideal mask squashed', which the network")
    print( "   never computes. The network learns f and emits tanh(f), reaching any value in (-1,1).)")
    print(f"\n  Mean |mask| produced by model : {np.mean(mag_model):.4f}")
    print(f"  Mean |mask| of clipped ideal  : {np.mean(mag_ideal_clipped):.4f}")
    print(f"  Fraction of model bins > 0.95 : {np.mean(sat_frac):.4f}")
    if np.mean(mag_model) < np.mean(mag_ideal_clipped) - 0.03:
        print("  => Model is UNDER-shooting the ideal mask magnitude: it is not saturating tanh,")
        print("     it is estimating conservatively. That is an ESTIMATION gap, not a bound.")
    print("=" * 92)


if __name__ == "__main__":
    main()
