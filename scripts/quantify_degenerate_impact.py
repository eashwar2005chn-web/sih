"""
Phase 14 Step 2: Quantify the impact of degenerate test utterances on reported metrics.

Joins eval/mixer_degenerate_diagnosis.csv (row-aligned, bit-exact with the original
eval run) against the per-model test CSVs, and recomputes aggregate + per-bucket
statistics under three filters:

  ORIGINAL : all 500 utterances (what every phase report currently uses)
  CLEAN-A  : exclude only silent-noise utterances (measured mixing bug)
  CLEAN-B  : exclude any utterance whose measured input SNR falls >3 dB outside
             its labeled bucket range (bug + post-channel band-limiting drift)
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BUCKETS = [
    ("Stress [-10, -5) dB", -10.0, -5.0),
    ("Stress [-5, 0) dB", -5.0, 0.0),
    ("Operational [0, 5) dB", 0.0, 5.0),
    ("Operational [5, 10) dB", 5.0, 10.0),
    ("Operational [10, 15] dB", 10.0, 15.0),
]
TOL = 3.0

MODELS = {
    "Baseline 811K": "eval/test_eval_csvs/Baseline_811K_BatchNorm_K1.0_test_eval.csv",
    "Causal TCN 899K": "eval/test_eval_csvs/Causal_TCN_Net_899K_Dilated_TCN_K1.0_test_eval.csv",
}


def load():
    diag = list(csv.DictReader(open("eval/mixer_degenerate_diagnosis.csv")))
    flags = []
    for row in diag:
        lo, hi = [(b[1], b[2]) for b in BUCKETS if b[0] == row["bucket"]][0]
        snr = float(row["final_snr"])
        flags.append({
            "bucket": row["bucket"],
            "silent": row["noise_is_allzero"] == "True",
            "out_of_band": (snr > hi + TOL) or (snr < lo - TOL),
        })
    return flags


def stats(rows):
    if not rows:
        return None
    s = np.array([float(r["enh_snr"]) for r in rows])
    t = np.array([float(r["enh_stoi"]) for r in rows])
    p = np.array([float(r["enh_pesq"]) for r in rows])
    g = np.array([float(r["snr_gain"]) for r in rows])
    return {
        "n": len(rows), "snr": s.mean(), "sd": s.std(ddof=1) if len(s) > 1 else 0.0,
        "pass": 100.0 * (s > 15.0).mean(), "stoi": t.mean(), "pesq": p.mean(), "gain": g.mean(),
    }


def main():
    flags = load()
    n_silent = sum(f["silent"] for f in flags)
    n_oob = sum(f["out_of_band"] for f in flags)
    print(f"Degenerate accounting over N=500:")
    print(f"  silent-noise (mixing bug)          : {n_silent}")
    print(f"  >3 dB outside labeled bucket       : {n_oob}")
    print(f"  (silent rows are a subset of above): {sum(f['silent'] and f['out_of_band'] for f in flags)}")

    for mname, path in MODELS.items():
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == len(flags)
        for r, f in zip(rows, flags):
            r["_silent"] = f["silent"]
            r["_oob"] = f["out_of_band"]

        print("\n" + "=" * 100)
        print(f"MODEL: {mname}")
        print("=" * 100)

        filters = [
            ("ORIGINAL", lambda r: True),
            ("CLEAN-A (drop silent)", lambda r: not r["_silent"]),
            ("CLEAN-B (drop >3dB off)", lambda r: not r["_oob"]),
        ]

        print(f"\n-- Aggregate: Operational (0-15 dB buckets) --")
        print(f"{'filter':26} {'N':>4} {'OutSNR':>16} {'pass>15':>9} {'STOI':>8} {'PESQ':>6} {'gain':>7}")
        for label, fn in filters:
            sub = [r for r in rows if r["bucket"].startswith("Operational") and fn(r)]
            st = stats(sub)
            print(f"{label:26} {st['n']:>4} {st['snr']:>8.2f} +/-{st['sd']:<5.2f} {st['pass']:>8.1f}% {st['stoi']:>8.4f} {st['pesq']:>6.2f} {st['gain']:>+7.2f}")

        print(f"\n-- Aggregate: Stress (-10-0 dB buckets) --")
        print(f"{'filter':26} {'N':>4} {'OutSNR':>16} {'pass>15':>9} {'STOI':>8} {'PESQ':>6} {'gain':>7}")
        for label, fn in filters:
            sub = [r for r in rows if r["bucket"].startswith("Stress") and fn(r)]
            st = stats(sub)
            print(f"{label:26} {st['n']:>4} {st['snr']:>8.2f} +/-{st['sd']:<5.2f} {st['pass']:>8.1f}% {st['stoi']:>8.4f} {st['pesq']:>6.2f} {st['gain']:>+7.2f}")

        print(f"\n-- Per-bucket Output SNR --")
        hdr = f"{'bucket':24}" + "".join(f"{l:>26}" for l, _ in filters)
        print(hdr)
        for bname, _, _ in BUCKETS:
            line = f"{bname:24}"
            for label, fn in filters:
                sub = [r for r in rows if r["bucket"] == bname and fn(r)]
                st = stats(sub)
                line += f"{st['snr']:>15.2f} dB (n={st['n']:>3})" if st else f"{'--':>26}"
            print(line)

        print(f"\n-- Per-bucket STOI --")
        print(hdr)
        for bname, _, _ in BUCKETS:
            line = f"{bname:24}"
            for label, fn in filters:
                sub = [r for r in rows if r["bucket"] == bname and fn(r)]
                st = stats(sub)
                line += f"{st['stoi']:>19.4f} (n={st['n']:>3})" if st else f"{'--':>26}"
            print(line)


if __name__ == "__main__":
    main()
