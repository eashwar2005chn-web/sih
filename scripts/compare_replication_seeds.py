import json
from tabulate import tabulate

with open('checkpoints/controlled_replication/train_history.json') as f1:
    h42 = json.load(f1)

with open('checkpoints/controlled_replication_seed1234/train_history.json') as f2:
    h1234 = json.load(f2)

table = []
for d1, d2 in zip(h42, h1234):
    ep = d1['epoch']
    table.append([
        ep,
        f"{d1['val_loss']:.4f}",
        f"{d2['val_loss']:.4f}",
        f"{d1['op_output_snr']:+.2f} dB",
        f"{d2['op_output_snr']:+.2f} dB",
        f"{d1['op_stoi']:.4f}",
        f"{d2['op_stoi']:.4f}",
        f"{d1['op_pesq']:.2f}",
        f"{d2['op_pesq']:.2f}",
        f"{d1['stress_snr_gain']:+.2f} dB",
        f"{d2['stress_snr_gain']:+.2f} dB"
    ])

headers = [
    "Epoch",
    "ValLoss (S:42)", "ValLoss (S:1234)",
    "OP SNR (S:42)", "OP SNR (S:1234)",
    "STOI (S:42)", "STOI (S:1234)",
    "PESQ (S:42)", "PESQ (S:1234)",
    "Stress (S:42)", "Stress (S:1234)"
]

print(tabulate(table, headers=headers, tablefmt="grid"))
