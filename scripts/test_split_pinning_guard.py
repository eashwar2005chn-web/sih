"""
Adversarial test of the split-pinning guard.

Two claims must hold before a large corpus is added to training:

  1. PINNED  : adding ~2000 new clean-speech files pinned to 'train' leaves val/test IDENTICAL.
  2. UNPINNED: the SAME files, unpinned, DO contaminate val/test -- and the guard CATCHES it.

Claim 2 is the negative control. Without it, claim 1 proves nothing: a guard that never fires
would also "pass". Uses synthetic silent wavs in a temp dir; touches no real data.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.real_dataset_loader as L

N_FAKE = 2000


def build_fake_corpus(root):
    """N_FAKE tiny wavs laid out like LibriSpeech: <root>/<speaker>/<chapter>/<utt>.wav"""
    sig = np.zeros(1600, dtype=np.float32)
    for i in range(N_FAKE):
        d = os.path.join(root, f"{1000 + i // 40}", f"{i // 8}")
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, f"{1000 + i // 40}-{i}.wav"), sig, 16000)


def fingerprint_with(src_dir, pinned):
    """Fingerprint val/test with the fake source present, pinned or not."""
    dirs = {c: dict(s) for c, s in L.DEFAULT_DATASET_DIRS.items()}
    dirs["clean_speech"]["librispeech_train100"] = src_dir
    saved = dict(L.PINNED_SOURCES)
    try:
        L.PINNED_SOURCES.clear()
        if pinned:
            L.PINNED_SOURCES["librispeech_train100"] = "train"
        m = L.RealAudioManifest(dataset_dirs=dirs)
        return L.eval_split_fingerprint(m), m
    finally:
        L.PINNED_SOURCES.clear()
        L.PINNED_SOURCES.update(saved)


def main():
    base = L.eval_split_fingerprint()
    tmp = tempfile.mkdtemp(prefix="splitguard_")
    ok = True
    try:
        corpus = os.path.join(tmp, "train-clean-100")
        build_fake_corpus(corpus)
        print("=" * 88)
        print(f"SPLIT-PINNING GUARD TEST   ({N_FAKE} synthetic clean-speech files)")
        print("=" * 88)
        print(f"  baseline   val clean_speech = {base['val']['clean_speech']['n_files']}"
              f"   test clean_speech = {base['test']['clean_speech']['n_files']}\n")

        fp_p, m_p = fingerprint_with(corpus, pinned=True)
        n_train_p = len(m_p.get_split_files("clean_speech", "train"))
        same = all(fp_p[s]["clean_speech"]["sha256"] == base[s]["clean_speech"]["sha256"]
                   for s in ("val", "test"))
        print(f"  [1] PINNED to train")
        print(f"      train clean_speech files : {n_train_p}  (baseline 2546 + {N_FAKE})")
        print(f"      val  clean_speech        : {fp_p['val']['clean_speech']['n_files']}")
        print(f"      test clean_speech        : {fp_p['test']['clean_speech']['n_files']}")
        print(f"      val/test unchanged       : {'YES  <-- required' if same else 'NO  ** FAIL **'}")
        ok &= same

        fp_u, _ = fingerprint_with(corpus, pinned=False)
        leaked_v = fp_u["val"]["clean_speech"]["n_files"] - base["val"]["clean_speech"]["n_files"]
        leaked_t = fp_u["test"]["clean_speech"]["n_files"] - base["test"]["clean_speech"]["n_files"]
        drifted = any(fp_u[s]["clean_speech"]["sha256"] != base[s]["clean_speech"]["sha256"]
                      for s in ("val", "test"))
        print(f"\n  [2] NEGATIVE CONTROL - same files, NOT pinned")
        print(f"      files leaked into val    : {leaked_v}")
        print(f"      files leaked into test   : {leaked_t}")
        print(f"      guard detects drift      : {'YES  <-- required' if drifted else 'NO  ** FAIL **'}")
        ok &= drifted

        print("\n" + "=" * 88)
        print("  RESULT: " + ("PASS - pinning holds val/test fixed, and the guard fires without it."
                              if ok else "FAIL - see above."))
        print("=" * 88)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
