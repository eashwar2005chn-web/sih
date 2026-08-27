import json
from tabulate import tabulate

with open('checkpoints/controlled_replication/train_history.json') as f:
    hist = json.load(f)

table = []
for h in hist:
    table.append([
        h['epoch'],
        f"{h['train_loss']:.4f}",
        f"{h['val_loss']:.4f}",
        f"{h['op_output_snr']:.2f} dB",
        f"{h['op_snr_gain']:+.2f} dB",
        f"{h['op_stoi']:.4f}",
        f"{h['op_pesq']:.2f}",
        f"{h['stress_snr_gain']:+.2f} dB"
    ])

headers = ["Epoch", "Train Loss", "Val Loss", "OP Out SNR", "OP SNR Gain", "OP STOI", "OP PESQ", "Stress Gain"]
print(tabulate(table, headers=headers, tablefmt="grid"))
