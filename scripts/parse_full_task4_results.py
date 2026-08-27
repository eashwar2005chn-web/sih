import os, glob, json
import torch
from tabulate import tabulate

with open('checkpoints/task4_full_run/train_history.json') as f:
    hist = json.load(f)

table = []
best_score = -999.0
best_ep = None

for h in hist:
    ep = h['epoch']
    score = h['op_score']
    if score > best_score:
        best_score = score
        best_ep = ep
    table.append([
        ep,
        f"{h['train_loss']:.4f}",
        f"{h['val_loss']:.4f}",
        f"{score:.2f}",
        f"{h['op_output_snr']:+.2f} dB",
        f"{h['op_snr_gain']:+.2f} dB",
        f"{h['op_stoi']:.4f}",
        f"{h['op_pesq']:.2f}",
        f"{h['stress_snr_gain']:+.2f} dB"
    ])

headers = ["Epoch", "Train Loss", "Val Loss", "OP Score", "OP Out SNR", "OP Gain", "OP STOI", "OP PESQ", "Stress Gain"]

print("==========================================================================================")
print(" DRDO PS 26052: TASK 4 FULL 50-EPOCH TRAINING RUN PROGRESS REPORT")
print(f" Best Model Checkpoint: Epoch {best_ep} (OP Score: {best_score:.2f})")
print("==========================================================================================")
print(tabulate(table, headers=headers, tablefmt="grid"))
