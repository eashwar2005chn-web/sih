"""
Utility script to export all phase training histories and screening results into CSV format.
"""

import os
import glob
import json
import pandas as pd

def main():
    csv_dir = "eval/training_progress_csvs"
    os.makedirs(csv_dir, exist_ok=True)

    checkpoints_dir = "checkpoints"
    json_files = glob.glob(f"{checkpoints_dir}/**/train_history.json", recursive=True)

    summary_records = []

    for jf in json_files:
        rel_path = os.path.relpath(jf, checkpoints_dir)
        run_name = os.path.dirname(rel_path).replace("\\", "_")
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            if not data:
                continue
            df = pd.DataFrame(data)
            csv_path = os.path.join(csv_dir, f"{run_name}_history.csv")
            df.to_csv(csv_path, index=False)
            print(f"Exported: {csv_path} ({len(df)} epochs)")

            latest = df.iloc[-1]
            best_val_snr = df["val_snr"].max() if "val_snr" in df else (df["val_out_snr"].max() if "val_out_snr" in df else None)
            best_op = df["op_score"].max() if "op_score" in df else None
            summary_records.append({
                "Run Name": run_name,
                "Total Epochs": len(df),
                "Final Train Loss": f"{latest.get('train_loss', 0.0):.4f}",
                "Final Val Loss": f"{latest.get('val_loss', 0.0):.4f}",
                "Best Val SNR (dB)": f"{best_val_snr:.2f}" if best_val_snr is not None else "N/A",
                "Best Score": f"{best_op:.2f}" if best_op is not None else "N/A",
                "CSV File": csv_path
            })
        except Exception as e:
            print(f"Error processing {jf}: {e}")

    summary_df = pd.DataFrame(summary_records)
    summary_csv = "eval/training_runs_overview.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved overall summary CSV to: {summary_csv}")
    print("\n--- Current Training History Overview ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
