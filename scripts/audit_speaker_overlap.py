"""
Speaker-overlap audit across the train / val / test splits.

`RealAudioManifest.get_split_files()` assigns each file to a split by hashing its FILE PATH.
That guarantees a file appears in exactly one split, but says nothing about the SPEAKER: a
given speaker's utterances are scattered across all three splits. For speech enhancement the
relevant generalisation question is whether the model has heard the VOICE before, not whether
it has heard the sentence before.

This audit reports, per split, the distinct LibriSpeech speaker IDs and how many of the test
speakers were also present during training. It writes eval/speaker_overlap_audit.json.

No model, no training. This is a property of the split function alone.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import RealAudioManifest


def librispeech_speaker(path):
    """LibriSpeech layout: .../dev-clean/<speaker>/<chapter>/<utt>.flac"""
    parts = os.path.normpath(path).split(os.sep)
    for tag in ("dev-clean", "train-clean-100", "train-clean-360", "train-other-500"):
        if tag in parts:
            i = parts.index(tag)
            if i + 1 < len(parts):
                return f"{tag}/{parts[i + 1]}"
    return None


def voicebank_speaker(path):
    """VoiceBank layout: files named p226_001.wav -> speaker p226."""
    b = os.path.basename(path)
    return f"voicebank/{b.split('_')[0]}" if b.startswith("p") and "_" in b else None


def speakers_in(manifest, split):
    out = {}
    for p in manifest.get_split_files("clean_speech", split):
        spk = librispeech_speaker(p) or voicebank_speaker(p)
        if spk:
            out[spk] = out.get(spk, 0) + 1
    return out


def main():
    m = RealAudioManifest()
    tr, va, te = (speakers_in(m, s) for s in ("train", "val", "test"))

    seen_te = sorted(set(te) & set(tr))
    seen_va = sorted(set(va) & set(tr))
    unseen_te = sorted(set(te) - set(tr))

    print("=" * 84)
    print("SPEAKER-OVERLAP AUDIT  (clean_speech, split by MD5 of file path)")
    print("=" * 84)
    print(f"  distinct speakers   train {len(tr):4d}   val {len(va):4d}   test {len(te):4d}")
    print(f"  TEST speakers also seen in TRAIN : {len(seen_te):4d} of {len(te)}"
          f"  ({100.0*len(seen_te)/max(1,len(te)):.1f}%)")
    print(f"  VAL  speakers also seen in TRAIN : {len(seen_va):4d} of {len(va)}"
          f"  ({100.0*len(seen_va)/max(1,len(va)):.1f}%)")
    print(f"  TEST speakers UNSEEN in training : {len(unseen_te):4d}")
    if unseen_te:
        print(f"     {unseen_te}")
    print()
    if len(unseen_te) == 0 and len(te) > 0:
        print("  VERDICT: the held-out test split is FILE-disjoint but NOT SPEAKER-disjoint.")
        print("           Every test speaker was heard during training. Reported test metrics")
        print("           therefore measure same-speaker performance and are an OPTIMISTIC")
        print("           estimate of performance on speakers the model has never heard.")
    else:
        print("  VERDICT: test split contains speakers unseen in training.")
    print("=" * 84)

    out = {
        "split_function": "md5(filepath) % 100 -> train <80, val 80-90, test >=90",
        "n_speakers": {"train": len(tr), "val": len(va), "test": len(te)},
        "test_speakers_seen_in_train": len(seen_te),
        "test_speakers_unseen_in_train": len(unseen_te),
        "val_speakers_seen_in_train": len(seen_va),
        "unseen_test_speakers": unseen_te,
        "speaker_disjoint": len(unseen_te) == len(te) and len(te) > 0,
    }
    os.makedirs("eval", exist_ok=True)
    json.dump(out, open("eval/speaker_overlap_audit.json", "w"), indent=2)
    print("  wrote eval/speaker_overlap_audit.json")


if __name__ == "__main__":
    main()
