"""
Section A.1 (continued) — Did the content leakage actually inflate reported metrics?

Splits the N=500 evaluated test set into LEAKED (clean-speech or noise file is a
bit-identical duplicate of a training file) vs CLEAN (neither is), and compares model
performance on each. Reports before/after for BOTH the included and excluded case,
per the project's standing rule on exclusion proposals.

Also reports the speaker-overlap case separately, since ~100% of real-speech test
items share a speaker with training and therefore cannot be excluded by subsetting.
"""
import os
import sys
import csv
import json
import math
import hashlib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import RealAudioManifest

MODELS = {
    "Baseline 811K": "eval/test_eval_csvs/Baseline_811K_BatchNorm_K1.0_test_eval.csv",
    "Causal TCN 899K": "eval/test_eval_csvs/Causal_TCN_Net_899K_Dilated_TCN_K1.0_test_eval.csv",
}


def content_hash(path):
    try:
        a, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if a.ndim > 1:
        a = a.mean(axis=1)
    h = hashlib.sha256()
    h.update(str(sr).encode())
    h.update(np.ascontiguousarray(a.astype(np.float32)).tobytes())
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - hw), 100 * (c + hw))


def mean_ci(x, z=1.96):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else 0.0, 0.0, 0.0)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return (m, m - z * se, m + z * se)


def main():
    mf = RealAudioManifest()
    leaked = set()
    for cat in ["clean_speech", "generic_noise", "tactical_noise", "rirs"]:
        h_tr = {}
        for p in mf.get_split_files(cat, "train"):
            h = content_hash(p)
            if h:
                h_tr.setdefault(h, []).append(p)
        for p in mf.get_split_files(cat, "test"):
            h = content_hash(p)
            if h and h in h_tr:
                leaked.add(os.path.basename(p))

    rows_diag = list(csv.DictReader(open("eval/mixer_degenerate_diagnosis.csv")))

    def base_of(meta):
        return meta.split(":", 1)[1] if ":" in meta else None

    flags = []
    for r in rows_diag:
        sb, nb = base_of(r["speech_meta"]), base_of(r["noise_meta"])
        flags.append({
            "leaked": (sb is not None and sb in leaked) or (nb is not None and nb in leaked),
            "silent": r["noise_is_allzero"] == "True",
        })

    print("=" * 96)
    print("LEAKAGE IMPACT ON REPORTED METRICS  (leaked = exact-duplicate train audio in speech or noise)")
    print("=" * 96)

    summary = {}
    for mname, path in MODELS.items():
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == len(flags)
        for r, f in zip(rows, flags):
            r.update(f)

        print(f"\n### {mname}")
        summary[mname] = {}

        for scope, pred in [("Operational (0-15 dB)", lambda r: r["bucket"].startswith("Operational")),
                            ("Stress (-10-0 dB)", lambda r: r["bucket"].startswith("Stress"))]:
            sub = [r for r in rows if pred(r) and not r["silent"]]   # drop known mixer-bug rows
            lk = [r for r in sub if r["leaked"]]
            cl = [r for r in sub if not r["leaked"]]
            print(f"\n  {scope}   (excluding 7 known silent-noise rows)")
            print(f"  {'subset':22} {'N':>4} {'OutSNR (95% CI)':>28} {'pass>15':>9} {'CI':>16} {'STOI':>8}")
            block = {}
            for lbl, grp in [("ALL (as reported)", sub), ("LEAKED", lk), ("CLEAN (no dupes)", cl)]:
                if not grp:
                    continue
                s = np.array([float(r["enh_snr"]) for r in grp])
                t = np.array([float(r["enh_stoi"]) for r in grp])
                m, lo, hi = mean_ci(s)
                k = int((s > 15).sum())
                plo, phi = wilson(k, len(s))
                print(f"  {lbl:22} {len(grp):>4} {m:>10.2f} [{lo:6.2f},{hi:6.2f}] {100*k/len(s):>8.1f}% "
                      f"[{plo:5.1f},{phi:5.1f}] {t.mean():>8.4f}")
                block[lbl] = {"n": len(grp), "snr": round(m, 3), "snr_ci": [round(lo, 3), round(hi, 3)],
                              "pass_pct": round(100 * k / len(s), 2), "pass_ci": [round(plo, 2), round(phi, 2)],
                              "stoi": round(float(t.mean()), 4)}
            # difference test
            if lk and cl:
                a = np.array([float(r["enh_snr"]) for r in lk])
                b = np.array([float(r["enh_snr"]) for r in cl])
                diff = a.mean() - b.mean()
                se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                print(f"  --> LEAKED minus CLEAN = {diff:+.2f} dB   (95% CI [{diff-1.96*se:+.2f}, {diff+1.96*se:+.2f}])"
                      f"   {'SIGNIFICANT' if abs(diff) > 1.96*se else 'not significant'}")
                block["leaked_minus_clean_db"] = round(float(diff), 3)
                block["leaked_minus_clean_ci"] = [round(float(diff - 1.96 * se), 3), round(float(diff + 1.96 * se), 3)]
                block["significant"] = bool(abs(diff) > 1.96 * se)
            summary[mname][scope] = block

    with open("eval/leakage_metric_impact.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote eval/leakage_metric_impact.json")


if __name__ == "__main__":
    main()
