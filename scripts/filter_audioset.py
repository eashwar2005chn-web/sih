"""
AudioSet Segment Filter for Tactical Defence AI (DRDO PS 26052)
Filters AudioSet segments CSV using the official AudioSet ontology JSON.
Selects tactical combat & generic ambient labels, capping at N clips per label,
and outputs a clean CSV formatted with start/end timestamps ready for yt-dlp downloading.
"""

import os
import csv
import json
import argparse
from typing import Dict, List, Set, Tuple, Optional


# Verified against official AudioSet class_labels_indices.csv (Google Research)
DEFAULT_TACTICAL_LABELS = [
    "/m/032s66",    # Gunshot, gunfire (Index: 427)
    "/m/04zjc",     # Machine gun (Index: 428)
    "/m/0_1c",      # Artillery fire (Index: 430)
    "/m/014zdl",    # Explosion (Index: 426)
    "/m/09ct_",     # Helicopter (Index: 339)
    "/m/04229",     # Jet engine (Index: 337)
    "/m/0k5j",      # Aircraft (Index: 335)
    "/m/03kmc9",    # Siren (Index: 396)
    "/m/0dgbq",     # Civil defense siren (Index: 397)
    "/m/01h82_",    # Engine knocking (Index: 350)
    "/t/dd00067"    # Heavy engine / low frequency (Index: 349)
]


def load_ontology(ontology_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse ontology.json into id_to_name and name_to_id mappings.
    """
    if not os.path.exists(ontology_path):
        return {}, {}

    with open(ontology_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_name = {}
    name_to_id = {}
    for item in data:
        mid = item["id"]
        name = item["name"]
        id_to_name[mid] = name
        name_to_id[name.lower()] = mid

    return id_to_name, name_to_id


def filter_audioset_segments(
    segments_csv: str,
    ontology_path: Optional[str],
    target_labels: List[str],
    output_csv: str,
    cap_per_label: int = 150
) -> int:
    """
    Filter AudioSet segments CSV and save matching rows.
    """
    id_to_name, name_to_id = load_ontology(ontology_path) if ontology_path else ({}, {})

    # Resolve target label names to MIDs if needed
    resolved_mids = set()
    for lbl in target_labels:
        lbl_clean = lbl.strip()
        if lbl_clean.startswith("/m/"):
            resolved_mids.add(lbl_clean)
        elif lbl_clean.lower() in name_to_id:
            resolved_mids.add(name_to_id[lbl_clean.lower()])
        else:
            resolved_mids.add(lbl_clean)

    print(f"[AudioSet Filter] Target Label IDs ({len(resolved_mids)}): {list(resolved_mids)}")

    label_counts = {mid: 0 for mid in resolved_mids}
    matched_rows = []

    with open(segments_csv, "r", encoding="utf-8", errors="ignore") as f:
        # Skip header lines beginning with #
        lines = [line for line in f if not line.startswith("#")]
        reader = csv.reader(lines, skipinitialspace=True)
        
        for row in reader:
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            start_sec = float(row[1].strip())
            end_sec = float(row[2].strip())
            labels_str = row[3].strip().replace('"', "")
            row_labels = [l.strip() for l in labels_str.split(",")]

            # Check if any target label matches
            for mid in row_labels:
                if mid in resolved_mids and label_counts[mid] < cap_per_label:
                    lbl_name = id_to_name.get(mid, mid)
                    matched_rows.append({
                        "ytid": ytid,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "label_id": mid,
                        "label_name": lbl_name,
                        "url": f"https://www.youtube.com/watch?v={ytid}"
                    })
                    label_counts[mid] += 1
                    break

    # Write output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ytid", "start_sec", "end_sec", "label_id", "label_name", "url"])
        writer.writeheader()
        writer.writerows(matched_rows)

    print(f"\n[Summary] Filtered {len(matched_rows)} clips written to: {output_csv}")
    for mid, count in label_counts.items():
        name = id_to_name.get(mid, mid)
        print(f" - {name} ({mid}): {count} clips")

    return len(matched_rows)


def main():
    parser = argparse.ArgumentParser(description="Filter AudioSet segments for Tactical Defence AI")
    parser.add_argument("--segments_csv", type=str, required=True, help="Path to AudioSet balanced_train_segments.csv")
    parser.add_argument("--ontology", type=str, default=None, help="Path to ontology.json")
    parser.add_argument("--labels", type=str, nargs="+", default=DEFAULT_TACTICAL_LABELS, help="List of label names or MIDs")
    parser.add_argument("--output_csv", type=str, default="datasets/tactical_noise/audioset_filtered.csv", help="Destination output CSV")
    parser.add_argument("--cap_per_label", type=int, default=150, help="Max clips per label class")
    args = parser.parse_args()

    filter_audioset_segments(
        segments_csv=args.segments_csv,
        ontology_path=args.ontology,
        target_labels=args.labels,
        output_csv=args.output_csv,
        cap_per_label=args.cap_per_label
    )


if __name__ == "__main__":
    main()
