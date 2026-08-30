"""
Feasibility analysis: are the two SNR targets attainable AT ALL by this architecture?

Distinguishes two very different questions:
  (1) Aggregate Output SNR > 15.0 dB  -- how far is the mean, and what is the hard ceiling?
  (2) Per-utterance pass rate > 80%   -- what mean would even be REQUIRED, and can the
      best-possible mask (the oracle over the architecture's reachable set) reach it?

(2) is the decisive one. If the ORACLE -- the ideal complex ratio mask restricted to what a
tanh head can actually emit, i.e. the best this architecture could ever do with perfect
estimation -- does not clear 80%, then no amount of training or capacity can, and the target
is unattainable rather than merely unreached.
"""
import os
import sys
import csv
import math

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.loss import compute_snr

N_FFT, HOP = 512, 256
OP_BUCKETS = [("Operational [0, 5) dB", 0.0, 5.0, 5000),
              ("Operational [5, 10) dB", 5.0, 10.0, 5500),
              ("Operational [10, 15] dB", 10.0, 15.0, 6000)]
PER_BUCKET = 100
CSV17 = "eval/test_eval_csvs/Phase17_LongTrain_811K_test_eval.csv"


def oracle_reachable(clean_w, noisy_w, win):
    """Ideal cRM restricted to a tanh head's reachable set (-1,1). Best possible for this net."""
    S = torch.stft(clean_w, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
    Y = torch.stft(noisy_w, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
    den = Y.real ** 2 + Y.imag ** 2
    ok = den > 1e-10
    Mr = torch.zeros_like(Y.real); Mi = torch.zeros_like(Y.imag)
    Mr[ok] = ((S.real * Y.real + S.imag * Y.imag)[ok]) / den[ok]
    Mi[ok] = ((S.imag * Y.real - S.real * Y.imag)[ok]) / den[ok]
    Mr = Mr.clamp(-0.999, 0.999); Mi = Mi.clamp(-0.999, 0.999)
    Er = Mr * Y.real - Mi * Y.imag
    Ei = Mr * Y.imag + Mi * Y.real
    out = torch.istft(torch.complex(Er, Ei), N_FFT, HOP, N_FFT, win,
                      center=True, length=clean_w.shape[-1])
    return compute_snr(clean_w.squeeze(0).numpy(), out.squeeze(0).numpy())


def main():
    print("=" * 92)
    print("FEASIBILITY OF THE TWO SNR TARGETS")
    print("=" * 92)

    rows = [r for r in csv.DictReader(open(CSV17)) if r["bucket"].startswith("Operational")]
    s = np.array([float(r["enh_snr"]) for r in rows])
    print(f"\nPhase 17 achieved (N={len(s)}): mean {s.mean():.2f} dB, sd {s.std(ddof=1):.2f}, "
          f"pass>15dB {100*(s>15).mean():.1f}%")

    print("\n--- per-bucket pass rates now ---")
    for name, _, _, _ in OP_BUCKETS:
        b = np.array([float(r["enh_snr"]) for r in rows if r["bucket"] == name])
        print(f"  {name:26} mean {b.mean():6.2f} dB   pass {100*(b>15).mean():5.1f}%")

    # What mean would 80% pass require, holding the observed spread?
    sd = s.std(ddof=1)
    z80 = 0.8416  # P(X > 15) = 0.80  =>  mean = 15 + z * sd
    need = 15.0 + z80 * sd
    print(f"\n--- what 80% pass would REQUIRE ---")
    print(f"  observed spread sd = {sd:.2f} dB")
    print(f"  for 80% of utterances to exceed 15 dB, mean must be ~{need:.2f} dB")
    print(f"  that is {need - s.mean():+.2f} dB beyond Phase 17, and {need - 12.75:+.2f} dB beyond baseline")

    # THE decisive test: best possible mask for this architecture.
    print(f"\n--- ORACLE CEILING: best mask a tanh head could ever emit (N={3*PER_BUCKET}) ---")
    win = torch.hann_window(N_FFT)
    orc = []
    for name, lo, hi, seed in OP_BUCKETS:
        ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000,
                               segment_len_sec=2.0, epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
        vals = []
        for i in range(len(ds)):
            it = ds[i]
            if not np.any(it["noisy"].numpy()):
                continue
            vals.append(oracle_reachable(it["clean"].unsqueeze(0), it["noisy"].unsqueeze(0), win))
        orc += vals
        v = np.array(vals)
        print(f"  {name:26} oracle mean {v.mean():6.2f} dB   oracle pass {100*(v>15).mean():5.1f}%")
    orc = np.array(orc)
    print(f"  {'ALL OPERATIONAL':26} oracle mean {orc.mean():6.2f} dB   oracle pass {100*(orc>15).mean():5.1f}%")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    gap_snr = 15.0 - s.mean()
    print(f"  [1] SNR > 15.0 dB   : now {s.mean():.2f}, gap {gap_snr:+.2f} dB. "
          f"Oracle ceiling {orc.mean():.2f} dB.")
    print(f"      -> headroom remaining between achieved and ceiling: {orc.mean()-s.mean():.2f} dB")
    print(f"      -> ATTAINABLE: {'YES' if orc.mean() > 15.0 else 'NO'} "
          f"(the ceiling {'exceeds' if orc.mean()>15 else 'is below'} the target)")
    op = 100 * (orc > 15).mean()
    print(f"\n  [2] pass rate > 80% : now {100*(s>15).mean():.1f}%. Oracle pass rate {op:.1f}%.")
    if op < 80.0:
        print(f"      -> ATTAINABLE: NO. Even a PERFECT mask in this architecture reaches only")
        print(f"         {op:.1f}%, short of 80%. Training/capacity cannot close this; it is a")
        print(f"         property of the task+metric, not of the model.")
    else:
        print(f"      -> ATTAINABLE in principle: oracle clears 80%, so the gap is estimation.")


if __name__ == "__main__":
    main()
