import urllib.request
import csv

url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as response:
    lines = [line.decode("utf-8") for line in response.readlines()]
    reader = csv.reader(lines)
    header = next(reader)
    labels = {row[1]: (row[0], row[2]) for row in reader if len(row) >= 3}

targets = [
    "/m/03m5k",   # Gunshot, gunfire
    "/m/02z550",  # Machine gun
    "/m/012ffv",  # Artillery fire
    "/m/012n7d",  # Explosion
    "/m/09ct_",   # Helicopter
    "/m/0k4j",    # Aircraft
    "/m/03q5t",   # Siren
    "/m/07r04",   # Truck
    "/m/07p55fl"  # Engine knocking
]

print("=== CHECKING AUDIOSER TARGET LABELS ===")
for t in targets:
    if t in labels:
        print(f"{t:12s} -> Index: {labels[t][0]:3s} | Name: \"{labels[t][1]}\"")
    else:
        print(f"{t:12s} -> NOT FOUND IN OFFICIAL AUDIOSET METADATA!")

print("\n=== KEYWORD SEARCH FOR ALL RELEVANT COMBAT/TACTICAL CLASSES ===")
keywords = ["gun", "artillery", "explosion", "burst", "weapon", "helicopter", "siren", "aircraft", "rotor", "drone", "engine"]
for mid, (idx, name) in labels.items():
    if any(k in name.lower() for k in keywords):
        print(f"Index: {idx:3s} | MID: {mid:12s} | Name: \"{name}\"")
