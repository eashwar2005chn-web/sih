"""
Oracle VAD / speech-gating probe — ceiling analysis for "find the voice, keep it, kill the rest".

The trained model already emits a complex ratio mask, i.e. a per-time-frequency-bin gain.
That IS speech/noise discrimination at 257 bins x every 16 ms. This probe asks whether a
COARSER, EXPLICIT voice-activity decision could add anything on top of it, and measures the
absolute ceiling by using an ORACLE detector derived from the clean reference. A real detector
can only do worse, so whatever this reports is an upper bound.

Four conditions, all on the P22-LONG model of record:
  BASE   enhanced output, unmodified
  A      oracle HARD frame gate  - zero every STFT frame the clean signal says has no speech
  B      oracle SOFT frame gate  - attenuate those frames by -20 dB instead of zeroing
  C      oracle BINARY TF mask   - per-BIN oracle keep/discard on the noisy input (not the
                                   model output); the ceiling of ideal binary bin selection

Also reports how much output energy currently sits in speech-absent frames, which is the
quantity idea A would be removing. If that is already near zero there is nothing to gain.

VALIDATION split only. The test set is not touched.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr
from eval.metrics_common import evaluate_batch_metrics

N_FFT, HOP = 512, 256
CKPT = "checkpoints/phase22_LONG150K/best_model.pt"
VAD_DB = -40.0          # frame is "speech" if its clean energy is within 40 dB of the peak frame
BUCKETS = [
    ("Stress [-10,-5)", -10.0, -5.0, 91000),
    ("Stress [-5,0)",   -5.0,  0.0,  91500),
    ("Op [0,5)",         0.0,  5.0,  92000),
    ("Op [5,10)",        5.0, 10.0,  92500),
    ("Op [10,15]",      10.0, 15.0,  93000),
]
PER_BUCKET = 40


def stft(x, win):
    return torch.stft(x, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)


def istft(X, win, n):
    return torch.istft(X, N_FFT, HOP, N_FFT, win, center=True, length=n)


def oracle_frame_vad(clean_w, win):
    """Per-frame speech/no-speech from the CLEAN signal. Returns bool [T]."""
    S = stft(clean_w, win)
    e = (S.real ** 2 + S.imag ** 2).sum(dim=0)          # energy per frame
    if float(e.max()) <= 0:
        return torch.ones_like(e, dtype=torch.bool)
    e_db = 10.0 * torch.log10(e / e.max() + 1e-12)
    return e_db > VAD_DB


def main():
    dev = "cpu"
    ck = torch.load(CKPT, map_location="cpu")
    model = build_causal_anc_model(hidden_dim=128, mask_mode="component_tanh", mask_bound=1.0)
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(dev)
    win = torch.hann_window(N_FFT)

    print("=" * 92)
    print("ORACLE VAD / SPEECH-GATING PROBE  —  ceiling on explicit voice detection")
    print("=" * 92)
    print(f"  model      : {CKPT} (epoch {ck.get('epoch')}, val_snr {ck.get('val_snr'):.2f})")
    print(f"  split      : VALIDATION   VAD threshold: peak {VAD_DB:+.0f} dB per frame")
    print(f"  detector   : ORACLE (derived from the clean reference) -> upper bound\n", flush=True)

    rows = {k: [] for k in ("base", "A", "B", "C")}
    frac_silent, frac_energy_in_silence = [], []

    for label, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(split="val", mode="hybrid", epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), seed=seed, hybrid_ratio=0.85)
        b = {k: [] for k in ("base", "A", "B", "C")}
        for i in range(len(ds)):
            s = ds[i]
            clean = s["clean"].numpy().astype(np.float64)
            noisy = s["noisy"].numpy().astype(np.float64)
            with torch.no_grad():
                enh = model(s["noisy"].unsqueeze(0))[0][0].numpy().astype(np.float64)

            cw = torch.from_numpy(clean).float()
            nw = torch.from_numpy(noisy).float()
            ew = torch.from_numpy(enh).float()
            n = len(clean)

            vad = oracle_frame_vad(cw, win)                     # [T] bool
            E = stft(ew, win)
            keep = vad.unsqueeze(0)                             # [1,T]

            # how much of the model's output energy sits in speech-absent frames
            e_fr = (E.real ** 2 + E.imag ** 2).sum(dim=0)
            tot = float(e_fr.sum()) + 1e-20
            frac_silent.append(float((~vad).float().mean()))
            frac_energy_in_silence.append(float(e_fr[~vad].sum()) / tot)

            a = istft(E * keep, win, n).numpy().astype(np.float64)
            b_soft = istft(E * torch.where(keep, torch.ones_like(E.real),
                                           torch.full_like(E.real, 0.1)), win, n).numpy().astype(np.float64)

            # C: oracle BINARY per-bin mask applied to the NOISY input
            S, Y = stft(cw, win), stft(nw, win)
            snr_bin = (S.real ** 2 + S.imag ** 2) / ((Y - S).real ** 2 + (Y - S).imag ** 2 + 1e-12)
            c = istft(Y * (snr_bin > 1.0), win, n).numpy().astype(np.float64)

            b["base"].append(compute_snr(clean, enh))
            b["A"].append(compute_snr(clean, a))
            b["B"].append(compute_snr(clean, b_soft))
            b["C"].append(compute_snr(clean, c))

        print(f"  {label:18s}  base {np.mean(b['base']):+6.2f}  "
              f"A_hard {np.mean(b['A']):+6.2f}  B_soft {np.mean(b['B']):+6.2f}  "
              f"C_binTF {np.mean(b['C']):+6.2f}", flush=True)
        for k in rows:
            rows[k].extend(b[k])

    print("\n" + "=" * 92)
    print("AGGREGATE (val, N=%d)" % len(rows["base"]))
    print("=" * 92)
    base = np.mean(rows["base"])
    for k, name in [("base", "BASE   model output, unmodified"),
                    ("A", "A      oracle HARD frame gate"),
                    ("B", "B      oracle SOFT frame gate (-20 dB)"),
                    ("C", "C      oracle BINARY per-bin TF mask")]:
        m = np.mean(rows[k])
        print(f"  {name:42s} {m:+7.2f} dB   ({m - base:+.2f} vs base)")
    print()
    print(f"  frames the oracle calls speech-absent      : {100*np.mean(frac_silent):.1f}%")
    print(f"  model output energy sitting in those frames: {100*np.mean(frac_energy_in_silence):.3f}%")
    print("=" * 92)


if __name__ == "__main__":
    main()
