"""
Verify the speaker-disjoint training regime delivers what it claims, before spending a retrain.

Three properties must hold SIMULTANEOUSLY once dev-clean is excluded from train and
LibriSpeech train-clean-100 is pinned to it:

  1. val/test file lists are UNCHANGED  -> new numbers stay comparable to every published one
  2. ZERO speaker overlap between train and test
  3. ZERO content overlap (sha256 of decoded audio) between train and test clean speech

Property 1 is what keeps the ledger valid; 2 and 3 are what the retrain is for.
Reports the resulting training-corpus size in hours.
"""
import hashlib
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.real_dataset_loader as L
from scripts.audit_speaker_overlap import librispeech_speaker, voicebank_speaker


def spk(p):
    return librispeech_speaker(p) or voicebank_speaker(p)


def audio_sha(p):
    try:
        a, _ = sf.read(p, dtype="float32")
    except Exception:
        return None
    if a.ndim > 1:
        a = a.mean(axis=1)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def hours(files):
    t = 0.0
    for p in files:
        try:
            i = sf.info(p)
            t += i.frames / i.samplerate
        except Exception:
            pass
    return t / 3600.0


def main():
    print("=" * 90)
    print("SPEAKER-DISJOINT TRAINING - PRE-RETRAIN VERIFICATION")
    print("=" * 90)

    excl = L.set_speaker_disjoint_training(True)
    print(f"  EXCLUDED_FROM_TRAIN = {sorted(excl)}")
    print(f"  PINNED_SOURCES      = {L.PINNED_SOURCES}\n")

    m = L.RealAudioManifest()
    tr = m.get_split_files("clean_speech", "train")
    te = m.get_split_files("clean_speech", "test")
    va = m.get_split_files("clean_speech", "val")

    ok = True

    # 1 -- evaluation sets unchanged
    try:
        L.verify_eval_splits_frozen()
        print("  [1] val/test byte-identical to frozen manifest ......... PASS")
    except RuntimeError as e:
        ok = False
        print("  [1] val/test byte-identical to frozen manifest ......... FAIL")
        print(f"      {e}")

    # 2 -- speakers
    s_tr = {spk(p) for p in tr} - {None}
    s_te = {spk(p) for p in te} - {None}
    shared = s_tr & s_te
    print(f"  [2] speaker overlap train x test ....................... "
          f"{'PASS' if not shared else 'FAIL'}   "
          f"(train {len(s_tr)} spk, test {len(s_te)} spk, shared {len(shared)})")
    if shared:
        ok = False
        print(f"      shared: {sorted(shared)[:8]}")

    # 3 -- decoded-audio content
    print("      hashing test clean speech for content comparison ...", flush=True)
    h_te = {}
    for p in te:
        h = audio_sha(p)
        if h:
            h_te.setdefault(h, p)
    print("      hashing train clean speech (this takes a minute) ...", flush=True)
    hit = 0
    seen = set()
    for p in tr:
        h = audio_sha(p)
        if h in h_te and h not in seen:
            seen.add(h)
            hit += 1
    pct = 100.0 * hit / max(1, len(h_te))
    print(f"  [3] content overlap train x test ....................... "
          f"{'PASS' if hit == 0 else 'FAIL'}   "
          f"({hit} of {len(h_te)} distinct test clips = {pct:.2f}%)")
    if hit:
        ok = False

    print(f"\n  training corpus : {len(tr):6d} clean-speech files   {hours(tr):.2f} h")
    print(f"  val / test      : {len(va):6d} / {len(te):6d} files (unchanged)")
    print("\n" + "=" * 90)
    print("  RESULT: " + ("PASS - safe to retrain; results stay comparable to the ledger."
                          if ok else "FAIL - do NOT retrain until resolved."))
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
