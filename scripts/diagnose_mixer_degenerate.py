"""
Phase 14 Step 1: Root-cause diagnosis of degenerate SNR utterances in the held-out N=500 test set.

Regenerates the exact same test items produced by scripts/evaluate_phase10_final.py
(same buckets, same seeds, same epoch_size) and instruments the mixing stage to find
WHY some utterances end up with measured input SNR far outside their labeled bucket.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from data.defence_noise_generator import mix_speech_and_noise
from model.loss import compute_snr

BUCKETS = [
    ("Stress [-10, -5) dB", -10.0, -5.0, 7000),
    ("Stress [-5, 0) dB", -5.0, 0.0, 7500),
    ("Operational [0, 5) dB", 0.0, 5.0, 5000),
    ("Operational [5, 10) dB", 5.0, 10.0, 5500),
    ("Operational [10, 15] dB", 10.0, 15.0, 6000),
]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def main():
    rows = []
    for name, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(
            mode="hybrid", split="test", sample_rate=16000, segment_len_sec=2.0,
            epoch_size=100, snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed,
        )
        for i in range(len(ds)):
            # Replicate __getitem__ deterministically so we can inspect intermediates.
            r = np.random.RandomState(ds.split_seed + i)
            clean_src, speech_meta = ds._get_clean_speech(i, rng=r)
            noise_src, noise_meta = ds._get_noise(rng=r)
            snr_db = float(r.uniform(lo, hi))

            noisy, clean_mixed, scaled_noise = mix_speech_and_noise(
                clean_src, noise_src, target_snr_db=snr_db
            )
            snr_at_mix = compute_snr(clean_mixed, noisy)

            item = ds[i]
            final_snr = compute_snr(item["clean"].numpy(), item["noisy"].numpy())

            rows.append({
                "bucket": name, "idx": i, "target_snr": snr_db,
                "snr_at_mix": snr_at_mix, "final_snr": final_snr,
                "noise_rms": rms(noise_src), "speech_rms": rms(clean_src),
                "scaled_noise_rms": rms(scaled_noise),
                "noise_meta": noise_meta, "speech_meta": speech_meta,
                "noise_is_allzero": bool(np.all(noise_src == 0)),
                "noise_len": len(noise_src),
            })
        print(f"[done] {name}")

    import csv
    out = "eval/mixer_degenerate_diagnosis.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")

    # ---- Analysis ----
    print("\n=== Utterances with final measured SNR >3 dB outside labeled bucket ===")
    bad = []
    for row in rows:
        lo, hi = [(b[1], b[2]) for b in BUCKETS if b[0] == row["bucket"]][0]
        if row["final_snr"] > hi + 3.0 or row["final_snr"] < lo - 3.0:
            bad.append(row)
    print(f"count = {len(bad)} / {len(rows)}")
    for b in sorted(bad, key=lambda x: -x["final_snr"])[:25]:
        print(f"  {b['bucket']:24} idx={b['idx']:3d} target={b['target_snr']:7.2f} "
              f"mix={b['snr_at_mix']:8.2f} final={b['final_snr']:9.2f} "
              f"noise_rms={b['noise_rms']:.3e} allzero={b['noise_is_allzero']} src={b['noise_meta'][:28]}")

    print("\n=== Correlation: is degeneracy explained by near-silent noise? ===")
    allz = [r for r in rows if r["noise_is_allzero"]]
    tiny = [r for r in rows if not r["noise_is_allzero"] and r["noise_rms"] < 1e-4]
    print(f"  all-zero noise clips : {len(allz)}")
    print(f"  near-silent (<1e-4)  : {len(tiny)}")
    print(f"  degenerate (>3dB off): {len(bad)}")
    ov = [r for r in bad if r["noise_is_allzero"] or r["noise_rms"] < 1e-4]
    print(f"  degenerate AND silent-noise: {len(ov)}")

    print("\n=== Does mixing itself hit target (before RIR/bandpass)? ===")
    dev_mix = [abs(r["snr_at_mix"] - r["target_snr"]) for r in rows if r["noise_rms"] >= 1e-4]
    dev_fin = [abs(r["final_snr"] - r["target_snr"]) for r in rows if r["noise_rms"] >= 1e-4]
    if dev_mix:
        print(f"  healthy-noise rows: |mix-target| mean={np.mean(dev_mix):.3f} max={np.max(dev_mix):.3f}")
        print(f"  healthy-noise rows: |final-target| mean={np.mean(dev_fin):.3f} max={np.max(dev_fin):.3f}")


if __name__ == "__main__":
    main()
