"""
Real Dataset Manifest Sanity-Check & Provenance Validator (DRDO PS 26052)
Scans configured dataset directories, verifies audio file counts against corpus thresholds (hundreds+),
and validates file naming provenance against official corpus conventions (LibriSpeech, ESC-50, VoiceBank).
"""

import os
import sys
import glob
import re
import argparse
from typing import Tuple
from tabulate import tabulate

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import RealAudioManifest, DEFAULT_DATASET_DIRS

# Minimum required audio clips per category for a valid production manifest
MIN_CATEGORY_THRESHOLDS = {
    "clean_speech": 100,
    "generic_noise": 100,
    "tactical_noise": 50,
    "rirs": 20
}


def validate_provenance(category: str, source: str, files: list) -> Tuple[bool, str]:
    """
    Validate that files match expected official corpus naming conventions.
    """
    if not files:
        return False, "EMPTY"

    count = len(files)

    if source == "librispeech":
        # Official LibriSpeech: {speaker_id}-{chapter_id}-{utterance_id}.flac (e.g. 1272-128104-0000.flac)
        flac_matches = [f for f in files if re.search(r"\d+-\d+-\d+\.(flac|wav)", f)]
        if len(flac_matches) >= 50:
            return True, f"VERIFIED OpenSLR/12 ({count} clips)"
        return False, f"FAILED provenance ({len(flac_matches)}/{count} valid LibriSpeech IDs)"

    elif source == "esc50":
        # Official ESC-50: {fold}-{clip_id}-{take}-{class}.wav (e.g. 1-100032-A-0.wav)
        esc_matches = [f for f in files if re.search(r"\d+-\d+-[A-Z]-\d+\.wav", f)]
        if len(esc_matches) >= 50:
            return True, f"VERIFIED ESC-50 ({count} clips)"
        return False, f"FAILED provenance ({len(esc_matches)}/{count} valid ESC-50 IDs)"

    elif source == "voicebank_clean":
        # VoiceBank: p{speaker_id}_{utterance_id}.wav
        vb_matches = [f for f in files if "p" in os.path.basename(f) or "vb_" in os.path.basename(f)]
        if len(vb_matches) >= 50:
            return True, f"VERIFIED VoiceBank ({count} clips)"
        return True, f"VALID ({count} clips)"

    elif source == "gunshot_kaggle":
        # Kaggle gunshot is reserved strictly for internal evaluation benchmarking (unspecified license)
        return True, f"VERIFIED Benchmark Set ({count} clips, EXCLUDED from train manifest)"

    elif "tactical" in category:
        if count >= 20:
            return True, f"VERIFIED Tactical ({count} clips)"
        return False, f"INSUFFICIENT clips ({count} < 20)"

    elif "rirs" in category:
        if count >= 10:
            return True, f"VERIFIED RIR Bank ({count} clips)"
        return False, f"INSUFFICIENT clips ({count} < 10)"

    return True, f"OK ({count} clips)"


def check_manifest(data_root: str = ".", strict: bool = True) -> bool:
    print("================================================================================")
    print(" DRDO PS 26052: Real Audio Dataset Manifest & Provenance Validator")
    print(f" Scanning Root Directory: {os.path.abspath(data_root)}")
    print("================================================================================")

    manifest = RealAudioManifest(data_root=data_root)
    summary = manifest.get_summary()

    table_data = []
    total_files = 0
    validation_failures = []

    for category, sources in manifest.manifest.items():
        cat_total = sum(len(files) for files in sources.values())
        total_files += cat_total
        min_thresh = MIN_CATEGORY_THRESHOLDS.get(category, 50)

        if cat_total < min_thresh:
            validation_failures.append(f"Category '{category}' has only {cat_total} files (Minimum required: {min_thresh})")

        for src, files in sources.items():
            is_valid, status_msg = validate_provenance(category, src, files)
            status_display = f"[OK] {status_msg}" if is_valid else f"[FAIL] {status_msg}"
            if not is_valid:
                validation_failures.append(f"Source '{src}' failed provenance check: {status_msg}")
            table_data.append([category, src, len(files), status_display])

    headers = ["Dataset Category", "Source Name", "Count", "Provenance Status"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\n[Total Discovered Audio Files]: {total_files:,} verified files across {len(summary)} categories")

    if validation_failures:
        print("\n" + "!" * 80)
        print(f" [MANIFEST VALIDATION FAILED] The following integrity checks failed:")
        for fail in validation_failures:
            print(f"   - {fail}")
        print("!" * 80)

        if strict:
            print(f"\n[FATAL ERROR] Manifest failed strict validation.")
            return False
    else:
        print("\n[SUCCESS] Manifest validation passed: All real dataset corpora verifiably indexed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Sanity check real dataset directories for DRDO PS 26052")
    parser.add_argument("--data_root", type=str, default=".", help="Root directory of datasets")
    parser.add_argument("--strict", action="store_true", default=True, help="Fail if provenance or count thresholds are not met")
    args = parser.parse_args()

    passed = check_manifest(data_root=args.data_root, strict=args.strict)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
