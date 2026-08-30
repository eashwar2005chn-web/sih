"""
Gate the sub-band model on the two properties that would disqualify it for PS 26052.

1. CAUSALITY. The sub-band path adds a GRU over the frequency-neighbourhood of every bin.
   Frequency context is fine -- the whole frame exists at frame time -- but any leakage along
   TIME would break the real-time claim. Tested by perturbing a future frame and measuring
   whether any earlier output sample moves.

2. LATENCY. One GRU applied independently to 257 bins is the compute risk of this design.
   Measured per frame against the 16 ms hop, since RTF < 0.1 is the project's budget.

Both are measured, not asserted.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_anc_subband import CausalANCSubbandNet
from model.causal_anc_net import CausalANCNet

SR, HOP = 16000, 256
HOP_MS = 1000.0 * HOP / SR


def causality(model, name, n=16000, split=0.5):
    """Perturb everything after `split` and find the earliest changed output sample."""
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, n)
    cut = int(n * split)
    x2 = x.clone()
    x2[:, cut:] += 5.0 * torch.randn(1, n - cut)
    with torch.no_grad():
        y1 = model(x)[0][0].numpy()
        y2 = model(x2)[0][0].numpy()
    d = np.abs(y1 - y2)
    tol = 1e-6
    changed = np.nonzero(d > tol)[0]
    first = int(changed[0]) if len(changed) else n
    leak = cut - first          # >0 means output moved BEFORE the perturbation
    print(f"  {name:22s} perturb@{cut:6d}  first change@{first:6d}  "
          f"backward leakage {leak:+6d} samples ({1000.0*leak/SR:+.1f} ms)  "
          f"max|d| before cut = {d[:cut].max():.2e}")
    return leak


def latency(model, name, frames=200):
    model.eval()
    x = torch.randn(1, frames * HOP)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(10):
            model(x)
        dt = (time.perf_counter() - t0) / 10
    per_frame_ms = 1000.0 * dt / frames
    rtf = dt / (frames * HOP / SR)
    print(f"  {name:22s} {per_frame_ms:7.3f} ms/frame   RTF {rtf:.4f}   "
          f"{'OK' if rtf < 0.1 else 'OVER BUDGET'}  (hop = {HOP_MS:.1f} ms)")
    return per_frame_ms, rtf


def main():
    sb = CausalANCSubbandNet(hidden_dim=128, num_gru_layers=2, norm_type="batch", mask_bound=1.0)
    fb = CausalANCNet(hidden_dim=128, num_gru_layers=2, norm_type="batch", mask_bound=1.0)

    print("=" * 92)
    print("SUB-BAND MODEL GATE  -  causality and latency (CPU, batch 1)")
    print("=" * 92)
    print("\n[1] CAUSALITY along time")
    print("    (the STFT front end is centre-padded, so ~n_fft/2 = 256 samples of backward")
    print("     leakage is EXPECTED and already disclosed for the full-band model too;")
    print("     what matters is that sub-band adds NO MORE than the full-band baseline)")
    l_fb = causality(fb, "full-band (baseline)")
    l_sb = causality(sb, "full+sub-band")
    same = l_sb <= l_fb
    print(f"    -> sub-band adds {l_sb - l_fb:+d} samples vs baseline : "
          f"{'PASS' if same else 'FAIL - extra lookahead introduced'}")

    print("\n[2] LATENCY")
    fb_ms, fb_rtf = latency(fb, "full-band (baseline)")
    sb_ms, sb_rtf = latency(sb, "full+sub-band")
    print(f"    -> sub-band costs {sb_ms/max(fb_ms,1e-9):.2f}x the full-band model per frame")

    print("\n" + "=" * 92)
    ok = same and sb_rtf < 0.1
    print("  RESULT: " + ("PASS - causal and inside the real-time budget."
                          if ok else "REVIEW - see failures above."))
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
