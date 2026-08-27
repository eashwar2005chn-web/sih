"""
Live Progress Monitor & CSV Exporter.
Reads all checkpoints and active logs, displays the latest training metrics in terminal,
and saves a live snapshot to eval/live_progress.csv.
"""

import os
import glob
import json
import pandas as pd
from tabulate import tabulate

def monitor():
    csv_dir = "eval/training_progress_csvs"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs("eval", exist_ok=True)

    json_files = glob.glob("checkpoints/**/train_history.json", recursive=True) + \
                 glob.glob("checkpoints/**/history.json", recursive=True) + \
                 glob.glob("checkpoints/**/*history.json", recursive=True)
    json_files = sorted(list(set(json_files)))

    records = []
    for jf in json_files:
        run_name = os.path.dirname(os.path.relpath(jf, "checkpoints")).replace("\\", "_")
        if not run_name:
            continue
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            if not data or not isinstance(data, list):
                continue
            df = pd.DataFrame(data)
            csv_path = os.path.join(csv_dir, f"{run_name}_history.csv")
            df.to_csv(csv_path, index=False)

            last = df.iloc[-1]
            ep = int(last.get("epoch", len(df)))
            t_loss = last.get("train_loss", 0.0)
            v_loss = last.get("val_loss", 0.0)
            v_snr = last.get("val_snr", (last.get("val_out_snr", 0.0)))
            v_stoi = last.get("val_stoi", 0.0)
            score = last.get("op_score", 0.0)

            records.append({
                "Run / Model": run_name,
                "Current Epoch": f"{ep}/50",
                "Train Loss": f"{t_loss:.4f}",
                "Val Loss": f"{v_loss:.4f}",
                "Val SNR (dB)": f"{v_snr:+.2f}",
                "Val STOI": f"{v_stoi:.4f}",
                "Score": f"{score:.2f}",
                "Status": "COMPLETED" if ep >= 50 else f"RUNNING ({ep}/50)"
            })
        except Exception:
            pass

    if records:
        df_live = pd.DataFrame(records)
        df_live.to_csv("eval/live_progress.csv", index=False)
        print("\n================================================================================")
        print(" LIVE TRAINING PROGRESS (Snapshot saved to eval/live_progress.csv)")
        print("================================================================================")
        print(tabulate(df_live, headers="keys", tablefmt="grid", showindex=False))
    else:
        print("No active training history found yet.")

if __name__ == "__main__":
    monitor()
