"""
Section A.3 — Export before/after audio clips spanning all 5 SNR buckets so a human can
sanity-check for artifacts (musical noise, pumping, speech dropouts) that STOI/SNR/PESQ
would not surface.

Exports, for each bucket, 3 utterances x 3 files (noisy / enhanced / clean reference),
plus a manifest.csv giving the measured metrics for each clip so listening impressions
can be tied back to numbers.
"""
import os
import sys
import csv
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr, compute_stoi

CKPT = "checkpoints/task4_full_run/best_model.pt"
OUTDIR = "eval/listening_clips"
PER_BUCKET = 3

BUCKETS = [
    ("stress_-10_-5", -10.0, -5.0, 7000),
    ("stress_-5_0", -5.0, 0.0, 7500),
    ("op_0_5", 0.0, 5.0, 5000),
    ("op_5_10", 5.0, 10.0, 5500),
    ("op_10_15", 10.0, 15.0, 6000),
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)

    rows = []
    for tag, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000,
                               segment_len_sec=2.0, epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
        with torch.no_grad():
            for i in range(len(ds)):
                it = ds[i]
                clean = it["clean"].numpy()
                noisy = it["noisy"].numpy()
                enh, _, _, _ = m(it["noisy"].unsqueeze(0).to(dev))
                enh = enh.squeeze(0).cpu().numpy()

                stem = f"{tag}_{i:02d}"
                for kind, sig in [("noisy", noisy), ("enhanced", enh), ("clean", clean)]:
                    p = os.path.join(OUTDIR, f"{stem}_{kind}.wav")
                    peak = float(np.max(np.abs(sig))) or 1.0
                    sf.write(p, (sig / max(peak, 1e-9) * 0.95).astype(np.float32), 16000)

                rows.append({
                    "clip": stem,
                    "bucket": tag,
                    "noise_source": it["noise_type"],
                    "speech_source": it["speech_source"],
                    "target_snr_db": round(float(it["snr_db"]), 2),
                    "measured_in_snr_db": round(compute_snr(clean, noisy), 2),
                    "out_snr_db": round(compute_snr(clean, enh), 2),
                    "in_stoi": round(compute_stoi(clean, noisy), 4),
                    "out_stoi": round(compute_stoi(clean, enh), 4),
                })
        print(f"  [exported] {tag}")

    man = os.path.join(OUTDIR, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)*3} wav files + {man}")
    print("\n  clip                 bucket           tgt   in_snr  out_snr  in_stoi out_stoi")
    for r in rows:
        print(f"  {r['clip']:20} {r['bucket']:15} {r['target_snr_db']:6.1f} {r['measured_in_snr_db']:7.2f} "
              f"{r['out_snr_db']:8.2f} {r['in_stoi']:8.3f} {r['out_stoi']:8.3f}")
    print("\n  LISTEN FOR: musical noise / warbling in the stress clips, speech onset clipping,")
    print("  pumping between words, and whether residual noise is steady rather than fluttering.")


if __name__ == "__main__":
    main()
