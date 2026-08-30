"""
Freeze (or verify) the val/test file lists.

Every published metric in this project was measured on a specific set of val/test files. Adding
a new data source, renaming a directory, or touching the split function can change which files
land in those splits -- silently, and in a way that makes new numbers incomparable to old ones
while still looking valid.

This script captures a sha256 over the sorted val/test file lists, per category, and stores it
in eval/frozen_eval_manifest.json. `--verify` re-computes and compares.

    python scripts/freeze_eval_splits.py            # verify against the stored manifest
    python scripts/freeze_eval_splits.py --freeze   # (re)write the manifest -- deliberate act

Re-freezing invalidates comparability with every previously published number. Do it only when
you intend that, and say so in the evaluation report.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import (
    FROZEN_EVAL_MANIFEST,
    PINNED_SOURCES,
    RealAudioManifest,
    eval_split_fingerprint,
    verify_eval_splits_frozen,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="(re)write the manifest instead of verifying against it")
    a = ap.parse_args()

    m = RealAudioManifest()
    fp = eval_split_fingerprint(m)

    print("=" * 88)
    print("EVALUATION SPLIT " + ("FREEZE" if a.freeze else "VERIFY"))
    print("=" * 88)
    for split in ("val", "test"):
        tot = sum(v["n_files"] for v in fp[split].values())
        print(f"  {split:5s} {tot:6d} files   " +
              "  ".join(f"{k}={v['n_files']}" for k, v in fp[split].items() if v["n_files"]))
    print(f"\n  pinned sources: {PINNED_SOURCES or '(none)'}")
    for src, sp in PINNED_SOURCES.items():
        n = len(m.manifest.get("clean_speech", {}).get(src, []))
        print(f"    {src:24s} -> {sp:5s}  ({n} files currently present)")
    print()

    if a.freeze:
        if os.path.isfile(FROZEN_EVAL_MANIFEST):
            with open(FROZEN_EVAL_MANIFEST, "r", encoding="utf-8") as fh:
                prev = json.load(fh)
            print("  !! A manifest already exists (frozen "
                  f"{prev.get('frozen_utc', '?')}). Overwriting it invalidates comparability")
            print("     with every number measured under it.")
        os.makedirs(os.path.dirname(FROZEN_EVAL_MANIFEST), exist_ok=True)
        payload = {
            "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": ("Reference val/test file lists. Every published metric was measured on "
                        "these exact files. verify_eval_splits_frozen() raises if they change."),
            "split_function": "md5(filepath) % 100 -> train <80, val 80-90, test >=90",
            "pinned_sources": dict(PINNED_SOURCES),
            "fingerprint": fp,
        }
        with open(FROZEN_EVAL_MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  WROTE {FROZEN_EVAL_MANIFEST}")
        print("=" * 88)
        return 0

    try:
        verify_eval_splits_frozen()
    except RuntimeError as e:
        print("  FAIL\n")
        print(e)
        print("=" * 88)
        return 1
    print("  PASS - val/test are byte-identical to the frozen manifest.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
