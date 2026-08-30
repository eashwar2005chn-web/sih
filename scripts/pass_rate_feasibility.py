"""
What would it actually take to reach an 85% per-utterance pass rate (Output SNR > 15 dB)?

Works from the committed per-utterance CSV of the model of record, so every number here is
recomputed from measured data rather than assumed.

Two levers move a pass rate, and they are NOT equally expensive:
  - raise the MEAN  (every experiment in the ledger so far has targeted this)
  - shrink the SPREAD (never targeted; the ledger has no experiment aimed at variance)

Reports the required mean at several spreads, the per-bucket decomposition of where failures
actually are, and how far the oracle ceiling sits above each bucket.
"""
import os
import sys
import numpy as np
import csv as _csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CSV = "eval/test_eval_csvs/Phase22_Long150k_811K_test_eval.csv"
THRESH = 15.0
TARGETS = [0.80, 0.85]
OP_BUCKETS = ["Operational [0, 5) dB", "Operational [5, 10) dB", "Operational [10, 15] dB"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main():
    with open(CSV, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    op = [r for r in rows if r["bucket"] in OP_BUCKETS]
    x = np.array([float(r["enh_snr"]) for r in op])
    mu, sd = x.mean(), x.std(ddof=1)
    k = int((x > THRESH).sum())
    lo, hi = wilson(k, len(x))

    print("=" * 92)
    print("PASS-RATE FEASIBILITY  -  what it takes to reach 80% / 85% (Output SNR > 15 dB)")
    print("=" * 92)
    print(f"  source : {CSV}")
    print(f"  scope  : operational buckets, N={len(x)}")
    print(f"  now    : mean {mu:.2f} dB   sd {sd:.2f}   pass {100*k/len(x):.1f}% "
          f"[{lo:.1f}, {hi:.1f}]\n")

    # ---- lever 1 / lever 2 trade-off, assuming the shape stays roughly normal
    print("  Required MEAN to hit the target, as a function of the SPREAD:")
    print(f"  {'sd':>6} | {'80%':>18} | {'85%':>18}")
    print("  " + "-" * 50)
    for s in [sd, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5]:
        cells = []
        for t in TARGETS:
            from scipy.stats import norm
            need = THRESH + norm.ppf(t) * s
            cells.append(f"{need:7.2f} dB ({need-mu:+5.2f})")
        tag = " <- today" if abs(s - sd) < 1e-9 else ""
        print(f"  {s:6.2f} | {cells[0]:>18} | {cells[1]:>18}{tag}")

    # ---- empirical, distribution-free: shift the whole distribution
    print("\n  Distribution-free check (shift every utterance by a constant, no shape assumption):")
    for t in TARGETS:
        need = 0.0
        while need < 30.0:
            if (x + need > THRESH).mean() >= t:
                break
            need += 0.01
        print(f"    {int(t*100)}% pass requires a uniform shift of {need:+.2f} dB "
              f"-> mean {mu+need:.2f} dB")

    # ---- where the failures actually are
    print("\n  Where the failures are (operational buckets):")
    print(f"  {'bucket':26} {'N':>4} {'mean':>8} {'sd':>6} {'pass%':>7} {'fails':>6}")
    print("  " + "-" * 62)
    for b in OP_BUCKETS:
        s = np.array([float(r["enh_snr"]) for r in rows if r["bucket"] == b])
        p = 100 * (s > THRESH).mean()
        print(f"  {b:26} {len(s):4d} {s.mean():8.2f} {s.std(ddof=1):6.2f} {p:6.1f}% "
              f"{int((s<=THRESH).sum()):6d}")

    fails = [r for r in op if float(r["enh_snr"]) <= THRESH]
    print(f"\n  total operational failures: {len(fails)} of {len(op)}")
    for b in OP_BUCKETS:
        n = sum(1 for r in fails if r["bucket"] == b)
        print(f"    {b:26} {n:4d}  ({100.0*n/max(1,len(fails)):.1f}% of all failures)")

    # ---- how far below threshold are the failures?
    d = THRESH - np.array([float(r["enh_snr"]) for r in fails])
    print(f"\n  Shortfall of failing utterances (dB below 15):")
    for q in [50, 75, 90, 100]:
        print(f"    {q:3d}th pct : {np.percentile(d, q):6.2f} dB")
    print(f"    within 1 dB of passing : {int((d <= 1).sum())} utterances "
          f"({100*(d<=1).mean():.1f}% of failures)")
    print(f"    within 3 dB of passing : {int((d <= 3).sum())} utterances "
          f"({100*(d<=3).mean():.1f}% of failures)")
    print("=" * 92)


if __name__ == "__main__":
    main()
