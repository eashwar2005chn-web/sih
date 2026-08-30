"""
Diagnostic: is there a train/eval objective mismatch costing us free dB?

HybridANCLoss optimises SI-SNR, which is SCALE-INVARIANT: it is indifferent to a constant
gain error on the output. The reported metric, compute_snr(), is scale-DEPENDENT: it
computes 10log10(||s||^2 / ||s_hat - s||^2), so a gain error is charged as error energy.

If the model's output carries a systematic gain error, we are being penalised at eval for
something training never asked it to fix -- and a single scalar per utterance would recover
it. This measures exactly how much.

Reports, on the VALIDATION split:
  - SNR as reported
  - SNR after the OPTIMAL per-utterance rescale  alpha* = <clean,enh>/<enh,enh>
  - the distribution of alpha* (is the model systematically hot or quiet?)
  - the same after a SINGLE global alpha fitted across all utterances (deployable as a
    constant, no per-utterance oracle knowledge required)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr

CKPT = "checkpoints/phase17_long_train/best_model.pt"
BUCKETS = [("[-10,-5)", -10.0, -5.0, 91000), ("[-5,0)", -5.0, 0.0, 91500),
           ("[0,5)", 0.0, 5.0, 92000), ("[5,10)", 5.0, 10.0, 92500),
           ("[10,15]", 10.0, 15.0, 93000)]
PER_BUCKET = 60


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = build_causal_anc_model(hidden_dim=128, mask_mode="component_tanh", mask_bound=1.0)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)

    print("=" * 88)
    print("SCALE-MISMATCH PROBE (Phase 17 model, validation split)")
    print("=" * 88)
    print(f"  checkpoint epoch {ck.get('epoch')}, val_snr {ck.get('val_snr'):.2f}\n")

    rows = []
    for name, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(mode="hybrid", split="val", sample_rate=16000,
                               segment_len_sec=2.0, epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
        for i in range(len(ds)):
            it = ds[i]
            if not np.any(it["noisy"].numpy()):
                continue
            c = it["clean"].numpy()
            with torch.no_grad():
                e = m(it["noisy"].unsqueeze(0).to(dev))[0].squeeze(0).float().cpu().numpy()
            L = min(len(c), len(e)); c, e = c[:L], e[:L]
            den = float(np.dot(e, e)) + 1e-12
            alpha = float(np.dot(c, e)) / den
            rows.append({"bucket": name, "clean": c, "enh": e, "alpha": alpha,
                         "snr": compute_snr(c, e),
                         "snr_opt": compute_snr(c, alpha * e)})
        print(f"  [done] {name}")

    alphas = np.array([r["alpha"] for r in rows])
    snr = np.array([r["snr"] for r in rows])
    snr_opt = np.array([r["snr_opt"] for r in rows])

    # a SINGLE global scalar, fitted once, deployable without oracle knowledge
    grid = np.linspace(0.80, 1.40, 241)
    best_g, best_v = 1.0, -1e9
    for g in grid:
        v = float(np.mean([compute_snr(r["clean"], g * r["enh"]) for r in rows]))
        if v > best_v:
            best_g, best_v = float(g), v

    print(f"\n  N = {len(rows)}")
    print(f"  {'as reported':34} {snr.mean():7.2f} dB")
    print(f"  {'+ optimal PER-UTTERANCE rescale':34} {snr_opt.mean():7.2f} dB   "
          f"({snr_opt.mean()-snr.mean():+.2f} dB, oracle upper bound)")
    print(f"  {'+ SINGLE global scalar a=%.3f' % best_g:34} {best_v:7.2f} dB   "
          f"({best_v-snr.mean():+.2f} dB, deployable today)")

    print(f"\n  alpha* distribution: mean {alphas.mean():.4f}  median {np.median(alphas):.4f}  "
          f"sd {alphas.std():.4f}")
    if alphas.mean() < 0.97:
        print("  => model output is systematically HOT (too loud); eval charges the excess as error.")
    elif alphas.mean() > 1.03:
        print("  => model output is systematically QUIET; eval charges the deficit as error.")
    else:
        print("  => no large systematic gain bias.")

    print(f"\n  {'bucket':12} {'as reported':>13} {'per-utt rescale':>17} {'gain':>8} {'mean alpha*':>12}")
    for name, _, _, _ in BUCKETS:
        b = [r for r in rows if r["bucket"] == name]
        if not b:
            continue
        a = np.array([r["snr"] for r in b]); o = np.array([r["snr_opt"] for r in b])
        al = np.mean([r["alpha"] for r in b])
        print(f"  {name:12} {a.mean():>10.2f} dB {o.mean():>14.2f} dB {o.mean()-a.mean():>+7.2f} {al:>12.4f}")

    print("\n" + "=" * 88)
    if best_v - snr.mean() > 0.15:
        print(f"  ACTIONABLE: a single constant gain of {best_g:.3f} recovers {best_v-snr.mean():+.2f} dB")
        print( "  for free -- no retraining, no architecture change. This is a genuine")
        print( "  train/eval objective mismatch (SI-SNR is scale-invariant, the metric is not).")
    else:
        print("  No meaningful free dB from rescaling; the objective mismatch is not costing us.")


if __name__ == "__main__":
    main()
