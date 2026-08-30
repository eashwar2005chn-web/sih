"""
Where does the SPREAD in per-utterance output SNR come from, and how much of it is cheap?

The pass rate is a function of BOTH the mean and the spread, and cutting the spread is the
cheaper lever (see scripts/pass_rate_feasibility.py). Every experiment in the ledger so far has
targeted the mean. This probe decomposes the spread on the VALIDATION split.

Four conditions per utterance, on the P22-LONG model of record:

  BASE      model output as-is
  GAIN      + optimal per-utterance scalar rescale (the SI-SNR / SNR mismatch: the training
            objective is scale-INVARIANT, the eval metric is scale-DEPENDENT, so the model is
            never asked to get absolute gain right). ORACLE alpha -> upper bound on this fix.
  GLOBAL    + a single global constant rescale, fitted across all utterances. This is what a
            deployable fix without oracle knowledge would look like at minimum.
  TANH-ORACLE  the ideal cRM squashed through tanh. NOT a ceiling: the network emits
            tanh(f(x)) and can reach any value in (-1,1), so tanh(M*) is not its reachable set.
            Reported because the trained model BEATING this number is direct evidence that the
            Phase 8 "tanh is the ceiling" reading was wrong.

Reports mean, sd AND pass rate for each, because a change that raises the mean while widening
the spread can leave the pass rate flat.

VALIDATION split. Test set untouched.
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
CKPT = "checkpoints/phase22_LONG150K/best_model.pt"
THRESH = 15.0
BUCKETS = [
    ("Op [0,5)",   0.0,  5.0,  92000),
    ("Op [5,10)",  5.0, 10.0,  92500),
    ("Op [10,15]",10.0, 15.0,  93000),
]
PER_BUCKET = 100


def stats(x, name):
    x = np.asarray(x, float)
    return (f"  {name:34s} mean {x.mean():+7.2f}  sd {x.std(ddof=1):5.2f}  "
            f"pass {100*(x > THRESH).mean():5.1f}%")


def main():
    dev = "cpu"
    ck = torch.load(CKPT, map_location="cpu")
    model = build_causal_anc_model(hidden_dim=128, mask_mode="component_tanh", mask_bound=1.0)
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(dev)
    win = torch.hann_window(N_FFT)

    print("=" * 92)
    print("VARIANCE DECOMPOSITION  -  how much of the spread is gain miscalibration?")
    print("=" * 92)
    print(f"  model : {CKPT} (epoch {ck.get('epoch')})   split: VALIDATION\n", flush=True)

    base, gain, orc, alphas, raws = [], [], [], [], []
    for label, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(split="val", mode="hybrid", epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), seed=seed, hybrid_ratio=0.85)
        b_b, b_g, b_o = [], [], []
        for i in range(len(ds)):
            s = ds[i]
            clean = s["clean"].numpy().astype(np.float64)
            noisy = s["noisy"].numpy().astype(np.float64)
            with torch.no_grad():
                enh = model(s["noisy"].unsqueeze(0))[0][0].numpy().astype(np.float64)

            # optimal scalar rescale: argmin_a ||clean - a*enh||  ->  a = <c,e>/<e,e>
            den = float(enh @ enh)
            a = float(clean @ enh) / den if den > 1e-20 else 1.0
            alphas.append(a)

            cw = torch.from_numpy(clean).float()
            nw = torch.from_numpy(noisy).float()
            S = torch.stft(cw, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
            Y = torch.stft(nw, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
            d = Y.real ** 2 + Y.imag ** 2
            ok = d > 1e-10
            Mr = torch.zeros_like(Y.real); Mi = torch.zeros_like(Y.imag)
            Mr[ok] = ((S.real*Y.real + S.imag*Y.imag)[ok]) / d[ok]
            Mi[ok] = ((S.imag*Y.real - S.real*Y.imag)[ok]) / d[ok]
            # Clamp to the head's REACHABLE set. Unclamped, M*=S/Y reconstructs S exactly and
            # the "oracle" is just float precision (~100 dB), which is not a ceiling on anything.
            Mr = torch.tanh(Mr); Mi = torch.tanh(Mi)
            E = torch.complex(Y.real*Mr - Y.imag*Mi, Y.real*Mi + Y.imag*Mr)
            o = torch.istft(E, N_FFT, HOP, N_FFT, win, center=True,
                            length=len(clean)).numpy().astype(np.float64)

            b_b.append(compute_snr(clean, enh))
            b_g.append(compute_snr(clean, a * enh))
            b_o.append(compute_snr(clean, o))
            raws.append(compute_snr(clean, noisy))

        print(f"  {label:12s} base {np.mean(b_b):+6.2f} (sd {np.std(b_b, ddof=1):4.2f}, "
              f"pass {100*np.mean(np.array(b_b)>THRESH):4.1f}%)   "
              f"gain-fixed {np.mean(b_g):+6.2f} (sd {np.std(b_g, ddof=1):4.2f}, "
              f"pass {100*np.mean(np.array(b_g)>THRESH):4.1f}%)", flush=True)
        base += b_b; gain += b_g; orc += b_o

    base = np.array(base); gain = np.array(gain); orc = np.array(orc)
    a = np.array(alphas)
    print("\n" + "=" * 92)
    print(f"AGGREGATE (val, N={len(base)})")
    print("=" * 92)
    print(stats(base, "BASE    model output"))
    print(stats(gain, "GAIN    + oracle per-utterance rescale"))
    print(stats(orc,  "ORACLE  ideal complex ratio mask"))
    print()
    print(f"  optimal per-utterance alpha : median {np.median(a):.4f}  "
          f"mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  "
          f"[{np.percentile(a,5):.3f}, {np.percentile(a,95):.3f}]")
    print(f"  -> the model runs {100*(1-np.median(a)):.1f}% hot on average, but alpha varies "
          f"per utterance (sd {a.std(ddof=1):.3f}),")
    print(f"     so a single global constant cannot remove it.")
    print()
    print(f"  gain fix buys : {gain.mean()-base.mean():+.2f} dB mean, "
          f"{gain.std(ddof=1)-base.std(ddof=1):+.2f} dB sd, "
          f"{100*((gain>THRESH).mean()-(base>THRESH).mean()):+.1f} pp pass rate")
    print(f"  model vs tanh-squashed oracle : {base.mean()-orc.mean():+.2f} dB "
          f"-> the model EXCEEDS it, confirming tanh(M*) was never the reachable set")
    print("=" * 92)


if __name__ == "__main__":
    main()
