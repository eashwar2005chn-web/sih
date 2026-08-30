"""
Section A.1 — Train/test split leakage audit.

Checks three independent leakage channels between the training manifest and the
held-out test manifest:
  (1) FILE-level overlap (path identity)
  (2) SPEAKER-level overlap (LibriSpeech/VCTK speaker IDs parsed from paths)
  (3) CONTENT-level duplication (SHA-256 of decoded PCM, catches same audio
      stored under different filenames, which path-based checks cannot see)

Reports explicit counts. Makes no changes.
"""
import os
import re
import sys
import json
import hashlib
import collections

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import RealAudioManifest

CATEGORIES = ["clean_speech", "generic_noise", "tactical_noise", "rirs"]


def speaker_id(path: str):
    """
    Extract a speaker identifier from a clean-speech path.
    LibriSpeech layout: .../<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    VCTK layout:        .../pXXX/pXXX_YYY.wav
    Returns None if no speaker convention is recognised.
    """
    base = os.path.basename(path)
    m = re.match(r"^(\d+)-\d+-\d+", base)          # LibriSpeech
    if m:
        return f"librispeech:{m.group(1)}"
    m = re.match(r"^(p\d+)_\d+", base)              # VCTK / VoiceBank
    if m:
        return f"vctk:{m.group(1)}"
    m = re.match(r"^(p\d+)", base)
    if m:
        return f"vctk:{m.group(1)}"
    parts = path.replace("\\", "/").split("/")
    for p in reversed(parts[:-1]):
        if p.isdigit():
            return f"librispeech:{p}"
        if re.fullmatch(r"p\d+", p):
            return f"vctk:{p}"
    return None


def content_hash(path: str):
    """SHA-256 over decoded mono PCM, so identical audio under different names collides."""
    try:
        a, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = np.ascontiguousarray(a.astype(np.float32))
    h = hashlib.sha256()
    h.update(str(sr).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def main():
    mf = RealAudioManifest()
    print("=" * 78)
    print("SECTION A.1 — SPLIT LEAKAGE AUDIT")
    print("=" * 78)
    print("\nManifest inventory:")
    for cat, srcs in mf.get_summary().items():
        tot = sum(srcs.values())
        print(f"  {cat:16} total={tot:6d}  " + ", ".join(f"{k}={v}" for k, v in srcs.items()))

    report = {}

    for cat in CATEGORIES:
        tr = set(mf.get_split_files(cat, "train"))
        va = set(mf.get_split_files(cat, "val"))
        te = set(mf.get_split_files(cat, "test"))
        print(f"\n--- {cat} ---")
        print(f"  counts: train={len(tr)} val={len(va)} test={len(te)}")

        # (1) FILE-level
        f_tr_te = tr & te
        f_tr_va = tr & va
        f_va_te = va & te
        print(f"  [1] FILE overlap  train AND test={len(f_tr_te)}  train AND val={len(f_tr_va)}  val AND test={len(f_va_te)}")

        report[cat] = {
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "file_overlap_train_test": len(f_tr_te),
            "file_overlap_train_val": len(f_tr_va),
            "file_overlap_val_test": len(f_va_te),
        }

        # (2) SPEAKER-level (clean speech only)
        if cat == "clean_speech":
            s_tr = {s for s in (speaker_id(f) for f in tr) if s}
            s_te = {s for s in (speaker_id(f) for f in te) if s}
            s_va = {s for s in (speaker_id(f) for f in va) if s}
            unparsed = sum(1 for f in (tr | va | te) if speaker_id(f) is None)
            ov = s_tr & s_te
            print(f"  [2] SPEAKER  train={len(s_tr)} val={len(s_va)} test={len(s_te)} unparsed_files={unparsed}")
            print(f"      train AND test speakers = {len(ov)}"
                  + (f"   <-- LEAKAGE" if ov else "   (clean)"))
            if ov:
                ex = sorted(ov)[:12]
                print(f"      examples: {ex}")
                frac = len([f for f in te if speaker_id(f) in ov]) / max(len(te), 1)
                print(f"      fraction of TEST files whose speaker also appears in TRAIN: {100*frac:.1f}%")
                report[cat]["speaker_overlap_train_test"] = len(ov)
                report[cat]["test_files_with_train_speaker_pct"] = round(100 * frac, 2)
            else:
                report[cat]["speaker_overlap_train_test"] = 0

        # (3) CONTENT-level hash (noise + clean; sample-capped for runtime)
        CAP = 400
        tr_l, te_l = sorted(tr)[:CAP], sorted(te)[:CAP]
        h_tr = {}
        for f in tr_l:
            h = content_hash(f)
            if h:
                h_tr.setdefault(h, []).append(f)
        dupes = []
        for f in te_l:
            h = content_hash(f)
            if h and h in h_tr:
                dupes.append((f, h_tr[h][0]))
        print(f"  [3] CONTENT hash (train n={len(tr_l)} vs test n={len(te_l)}, capped at {CAP}): "
              f"{len(dupes)} exact-audio duplicates"
              + ("   <-- LEAKAGE" if dupes else "   (clean)"))
        for a, b in dupes[:5]:
            print(f"      DUP: {os.path.basename(a)}  ==  {os.path.basename(b)}")
        report[cat]["content_dupes_train_test"] = len(dupes)
        report[cat]["content_hash_sampled_train"] = len(tr_l)
        report[cat]["content_hash_sampled_test"] = len(te_l)

    # gunshot_kaggle asymmetry
    print("\n--- Distribution-shift note (not leakage) ---")
    tac_tr = mf.get_split_files("tactical_noise", "train")
    tac_te = mf.get_split_files("tactical_noise", "test")
    g_tr = [f for f in tac_tr if "gunshot" in f.replace("\\", "/")]
    g_te = [f for f in tac_te if "gunshot" in f.replace("\\", "/")]
    print(f"  gunshot_kaggle files in TRAIN={len(g_tr)}  in TEST={len(g_te)}")
    print("  (excluded from train by license policy; if present in test this is unseen-source shift,")
    print("   which makes the test HARDER, not easier — opposite of leakage)")
    report["gunshot_kaggle_train"] = len(g_tr)
    report["gunshot_kaggle_test"] = len(g_te)

    with open("eval/split_leakage_audit.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote eval/split_leakage_audit.json")


if __name__ == "__main__":
    main()
