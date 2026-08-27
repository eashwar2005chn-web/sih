import json
import numpy as np
from tabulate import tabulate

with open("checkpoints/pilot_scaled_capacity/scaled_pilot_history.json") as f:
    history = json.load(f)

table_rows = []
for h in history:
    table_rows.append([
        f"Epoch {h['epoch']:02d}",
        f"{h['train_loss']:.4f}",
        f"{h['val_loss']:.4f}",
        f"{h['val_out_snr']:+.2f} dB",
        f"{h['val_stoi']:.4f}",
        f"{h['val_pesq']:.2f}",
        f"{h['op_score']:.2f}"
    ])

print(tabulate(table_rows, headers=["Epoch", "Train Loss", "Val Loss", "Val Out SNR", "Val STOI", "Val PESQ", "Val OP Score"], tablefmt="grid"))

# Compute slopes for last 5 epochs (16-20)
epochs_last5 = np.array([16, 17, 18, 19, 20])
train_loss_last5 = np.array([h["train_loss"] for h in history[15:]])
val_loss_last5 = np.array([h["val_loss"] for h in history[15:]])
val_snr_last5 = np.array([h["val_out_snr"] for h in history[15:]])
val_stoi_last5 = np.array([h["val_stoi"] for h in history[15:]])

slope_train = np.polyfit(epochs_last5, train_loss_last5, 1)[0]
slope_val_loss = np.polyfit(epochs_last5, val_loss_last5, 1)[0]
slope_val_snr = np.polyfit(epochs_last5, val_snr_last5, 1)[0]
slope_val_stoi = np.polyfit(epochs_last5, val_stoi_last5, 1)[0]

print(f"\n--- Linear Slopes over Epochs 16-20 (per epoch) ---")
print(f"Train Loss Slope : {slope_train:+.4f} / epoch")
print(f"Val Loss Slope   : {slope_val_loss:+.4f} / epoch")
print(f"Val Out SNR Slope: {slope_val_snr:+.4f} dB / epoch")
print(f"Val STOI Slope   : {slope_val_stoi:+.5f} / epoch")
