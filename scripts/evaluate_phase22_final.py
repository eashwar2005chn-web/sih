"""
Phase 17 — THE single pre-committed test-set evaluation of the long-trained model.

Protocol is byte-identical to scripts/evaluate_phase10_final.py, which produced every
published baseline number: same 5 buckets, same seeds, same epoch_size=100 per bucket
(N=500), same metric functions, same bucket-label grouping.

Per Section B.1 this runs ONCE and is logged as final. It is not re-run after any
further tweak.
"""
import os
import sys
import csv
import json
import math

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from data.real_dataset_loader import verify_eval_splits_frozen
from model.causal_anc_net import build_causal_anc_model
from eval.metrics_common import evaluate_batch_metrics

BUCKETS = [
    ("Stress [-10, -5) dB", -10.0, -5.0, 7000),
    ("Stress [-5, 0) dB", -5.0, 0.0, 7500),
    ("Operational [0, 5) dB", 0.0, 5.0, 5000),
    ("Operational [5, 10) dB", 5.0, 10.0, 5500),
    ("Operational [10, 15] dB", 10.0, 15.0, 6000),
]

NEW_CKPT = "checkpoints/phase22_LONG150K/best_model.pt"
OUT_CSV = "eval/test_eval_csvs/Phase22_Long150k_811K_test_eval.csv"

# Published baseline, recomputed from the committed CSV (Phase 13/14 verified).
BASE = {"op_snr": 12.75, "op_sd": 5.35, "op_pass": 28.0, "op_stoi": 0.8990,
        "op_pesq": 3.69, "stress_snr": 4.59, "stress_gain": 6.84}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mean_ci(x, z=1.96):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - z * se, m + z * se


def main():
    # Guard: every published metric was measured on the frozen val/test file lists.
    # Raises if they have drifted (e.g. a new data source leaked into an eval split).
    verify_eval_splits_frozen()
    dev = "cpu"
    ck = torch.load(NEW_CKPT, map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128, mask_mode="component_tanh", mask_bound=1.0)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)

    print("=" * 96)
    print("PHASE 22 - FINAL TEST-SET EVALUATION (single pre-committed run, N=500)")
    print("=" * 96)
    print(f"  checkpoint : {NEW_CKPT} (epoch {ck.get('epoch')}, val_snr {ck.get('val_snr'):.2f})")
    print(f"  protocol   : identical to evaluate_phase10_final.py\n")

    recs = []
    with torch.no_grad():
        for name, lo, hi, seed in BUCKETS:
            ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000,
                                   segment_len_sec=2.0, epoch_size=100,
                                   snr_range=(lo, hi), hybrid_ratio=0.85, seed=seed)
            for i in range(len(ds)):
                it = ds[i]
                enh, _, _, _ = m(it["noisy"].unsqueeze(0).to(dev))
                r = evaluate_batch_metrics(it["clean"].numpy(), it["noisy"].numpy(),
                                           enh.squeeze(0).cpu().numpy())
                r["bucket"] = name
                recs.append(r)
            print(f"  [done] {name}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\n  wrote {OUT_CSV}")

    op = [r for r in recs if r["bucket"].startswith("Operational")]
    st = [r for r in recs if r["bucket"].startswith("Stress")]

    s = np.array([r["enh_snr"] for r in op])
    t = np.array([r["enh_stoi"] for r in op])
    p = np.array([r["enh_pesq"] for r in op])
    g = np.array([r["snr_gain"] for r in op])
    m_, lo_, hi_ = mean_ci(s)
    k = int((s > 15.0).sum())
    plo, phi = wilson(k, len(s))

    ss = np.array([r["enh_snr"] for r in st])
    sg = np.array([r["snr_gain"] for r in st])

    print("\n" + "=" * 96)
    print("OPERATIONAL RANGE (0-15 dB input, N=%d)" % len(op))
    print("=" * 96)
    print(f"  {'metric':22} {'baseline':>14} {'Phase 17':>22} {'delta':>10}")
    print(f"  {'Output SNR':22} {BASE['op_snr']:>11.2f} dB {m_:>13.2f} dB [{lo_:.2f},{hi_:.2f}] {m_-BASE['op_snr']:>+9.2f}")
    print(f"  {'Pass rate >15 dB':22} {BASE['op_pass']:>13.1f}% {100*k/len(s):>13.1f}% [{plo:.1f},{phi:.1f}] {100*k/len(s)-BASE['op_pass']:>+9.1f}")
    print(f"  {'STOI':22} {BASE['op_stoi']:>14.4f} {t.mean():>22.4f} {t.mean()-BASE['op_stoi']:>+10.4f}")
    print(f"  {'PESQ (proxy)':22} {BASE['op_pesq']:>14.2f} {p.mean():>22.2f} {p.mean()-BASE['op_pesq']:>+10.2f}")
    print(f"  {'mean SNR gain':22} {'-':>14} {g.mean():>19.2f} dB")

    print("\nSTRESS RANGE (-10-0 dB input, N=%d)" % len(st))
    print(f"  {'Output SNR':22} {BASE['stress_snr']:>11.2f} dB {ss.mean():>19.2f} dB {ss.mean()-BASE['stress_snr']:>+9.2f}")
    print(f"  {'mean SNR gain':22} {BASE['stress_gain']:>11.2f} dB {sg.mean():>19.2f} dB {sg.mean()-BASE['stress_gain']:>+9.2f}")

    print("\nPER-BUCKET Output SNR")
    print(f"  {'bucket':26} {'Phase 17':>12} {'STOI':>9}")
    per = {}
    for name, _, _, _ in BUCKETS:
        b = [r for r in recs if r["bucket"] == name]
        bs = float(np.mean([r["enh_snr"] for r in b]))
        bt = float(np.mean([r["enh_stoi"] for r in b]))
        per[name] = {"snr": round(bs, 3), "stoi": round(bt, 4), "n": len(b)}
        print(f"  {name:26} {bs:>9.2f} dB {bt:>9.4f}")

    print("\n" + "=" * 96)
    print("VERDICT vs the two compliance targets")
    print("=" * 96)
    snr_ok = m_ > 15.0
    pass_ok = (100 * k / len(s)) > 80.0
    print(f"  Aggregate Output SNR > 15.0 dB : {m_:.2f} dB  -> {'MET' if snr_ok else 'NOT MET'}")
    print(f"  Per-utterance pass rate > 80%  : {100*k/len(s):.1f}%  -> {'MET' if pass_ok else 'NOT MET'}")
    print(f"  STOI > 0.850                   : {t.mean():.4f}  -> {'MET' if t.mean()>0.85 else 'NOT MET'}")
    print(f"  PESQ(proxy) > 2.50             : {p.mean():.2f}  -> {'MET' if p.mean()>2.5 else 'NOT MET'}")
    if snr_ok and pass_ok:
        print("\n  => TARGET MET ON BOTH SNR METRICS.")
    elif snr_ok or pass_ok:
        print(f"\n  => PARTIAL: {'SNR met, pass rate not met' if snr_ok else 'pass rate met, SNR not met'}.")
    else:
        print("\n  => Neither SNR metric met.")

    json.dump({
        "checkpoint": NEW_CKPT, "epoch": ck.get("epoch"), "val_snr": ck.get("val_snr"),
        "n_operational": len(op), "n_stress": len(st),
        "op_snr": round(float(m_), 3), "op_snr_ci": [round(lo_, 3), round(hi_, 3)],
        "op_pass_rate": round(100 * k / len(s), 2), "op_pass_ci": [round(plo, 2), round(phi, 2)],
        "op_stoi": round(float(t.mean()), 4), "op_pesq": round(float(p.mean()), 3),
        "op_gain": round(float(g.mean()), 3),
        "stress_snr": round(float(ss.mean()), 3), "stress_gain": round(float(sg.mean()), 3),
        "per_bucket": per, "baseline_reference": BASE,
        "target_snr_met": bool(snr_ok), "target_pass_met": bool(pass_ok),
    }, open("eval/phase22_final_test_results.json", "w"), indent=2)
    print("\n  wrote eval/phase22_final_test_results.json")


if __name__ == "__main__":
    main()
