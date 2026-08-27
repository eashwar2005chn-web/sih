import os, glob
import torch
from tabulate import tabulate

ckpts = sorted(glob.glob('checkpoints/task4_full_run/epoch_*.pt'))
table = []
best_score = -999.0
best_ep = None

for p in ckpts[:20]:
    ckpt = torch.load(p, map_location='cpu')
    ep = ckpt['epoch']
    stats = ckpt['val_stats']
    op_score = stats['op_output_snr'] + 10.0 * stats['op_stoi'] + 5.0 * stats['op_pesq']
    if op_score > best_score:
        best_score = op_score
        best_ep = ep
    table.append([
        ep,
        f"{stats['val_loss']:.4f}",
        f"{op_score:.2f}",
        f"{stats['op_output_snr']:+.2f} dB",
        f"{stats['op_snr_gain']:+.2f} dB",
        f"{stats['op_stoi']:.4f}",
        f"{stats['op_pesq']:.2f}",
        f"{stats['stress_snr_gain']:+.2f} dB",
        f"{stats['stress_stoi']:.4f}"
    ])

headers = ["Epoch", "Val Loss", "OP Score", "OP Out SNR", "OP SNR Gain", "OP STOI", "OP PESQ", "Stress Gain", "Stress STOI"]
print("=== MILESTONE 1 (EPOCHS 1 - 20) PROGRESS REPORT ===")
print(tabulate(table, headers=headers, tablefmt="grid"))
print(f"\nCurrent Best Model so far: Epoch {best_ep} (OP Score: {best_score:.2f})")
