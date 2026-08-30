"""
Section A.1 (continued) — Full, uncapped content-duplication audit, mapped onto the
actual N=500 held-out test utterances that every phase was evaluated on.

Answers the question that matters: of the 500 test items actually evaluated, how many
drew a clean-speech file or a noise file whose *audio content* is bit-identical to a
file in the training manifest?
"""
import os
import sys
import csv
import json
import hashlib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import RealAudioManifest


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


def build(paths):
    d = {}
    for p in paths:
        h = content_hash(p)
        if h:
            d.setdefault(h, []).append(p)
    return d


def main():
    mf = RealAudioManifest()
    out = {}

    print("=" * 78)
    print("FULL CONTENT-DUPLICATION AUDIT (uncapped)")
    print("=" * 78)

    leaked_basenames = set()
    for cat in ["clean_speech", "generic_noise", "tactical_noise", "rirs"]:
        tr = mf.get_split_files(cat, "train")
        te = mf.get_split_files(cat, "test")
        h_tr = build(tr)
        h_te = build(te)
        shared = set(h_tr) & set(h_te)
        n_te_leaked = sum(len(h_te[h]) for h in shared)
        print(f"\n{cat}: train={len(tr)} test={len(te)}")
        print(f"  distinct shared audio hashes : {len(shared)}")
        print(f"  TEST files that are exact dupes of a TRAIN file: {n_te_leaked} "
              f"({100*n_te_leaked/max(len(te),1):.1f}% of the {cat} test pool)")
        for h in list(shared)[:4]:
            print(f"    e.g. TEST {os.path.basename(h_te[h][0])}  ==  TRAIN {os.path.basename(h_tr[h][0])}")
        out[cat] = {
            "n_train": len(tr), "n_test": len(te),
            "shared_hashes": len(shared),
            "test_files_leaked": n_te_leaked,
            "test_pool_leaked_pct": round(100 * n_te_leaked / max(len(te), 1), 2),
        }
        for h in shared:
            for p in h_te[h]:
                leaked_basenames.add(os.path.basename(p))

    # ---- Map onto the actual N=500 evaluated test items ----
    diag_path = "eval/mixer_degenerate_diagnosis.csv"
    if not os.path.exists(diag_path):
        print(f"\n[skip] {diag_path} not found; run scripts/diagnose_mixer_degenerate.py first.")
        return

    rows = list(csv.DictReader(open(diag_path)))
    print("\n" + "=" * 78)
    print(f"IMPACT ON THE ACTUAL N={len(rows)} EVALUATED TEST SET")
    print("=" * 78)

    def base_of(meta):
        # meta looks like "real:<file>" / "real_tactical:<file>" / "synth_<type>"
        if ":" in meta:
            return meta.split(":", 1)[1]
        return None

    n_speech_leak = n_noise_leak = n_either = 0
    n_synth_speech = n_synth_noise = 0
    per_bucket = {}
    for r in rows:
        sb = base_of(r["speech_meta"])
        nb = base_of(r["noise_meta"])
        if sb is None:
            n_synth_speech += 1
        if nb is None:
            n_synth_noise += 1
        s_leak = sb is not None and sb in leaked_basenames
        n_leak = nb is not None and nb in leaked_basenames
        n_speech_leak += s_leak
        n_noise_leak += n_leak
        either = s_leak or n_leak
        n_either += either
        b = r["bucket"]
        per_bucket.setdefault(b, {"n": 0, "leak": 0})
        per_bucket[b]["n"] += 1
        per_bucket[b]["leak"] += either

    print(f"  test items using SYNTHETIC speech (no real file)      : {n_synth_speech}")
    print(f"  test items using SYNTHETIC noise  (no real file)      : {n_synth_noise}")
    print(f"  test items whose CLEAN SPEECH is a train-content dupe : {n_speech_leak}")
    print(f"  test items whose NOISE       is a train-content dupe  : {n_noise_leak}")
    print(f"  test items contaminated by EITHER channel             : {n_either} "
          f"({100*n_either/len(rows):.1f}% of N={len(rows)})")
    print("\n  per bucket:")
    for b, v in per_bucket.items():
        print(f"    {b:26} {v['leak']:3d}/{v['n']:3d}  ({100*v['leak']/v['n']:.0f}%)")

    out["evaluated_test_set"] = {
        "n": len(rows),
        "synthetic_speech_items": n_synth_speech,
        "synthetic_noise_items": n_synth_noise,
        "items_speech_content_leaked": n_speech_leak,
        "items_noise_content_leaked": n_noise_leak,
        "items_either_leaked": n_either,
        "items_either_leaked_pct": round(100 * n_either / len(rows), 2),
        "per_bucket": per_bucket,
    }

    with open("eval/content_leakage_audit.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote eval/content_leakage_audit.json")


if __name__ == "__main__":
    main()
