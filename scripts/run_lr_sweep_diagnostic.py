"""
Diagnostic LR Sweep for CausalANCScaledNet (3-4 epochs per rate):
Tests LR = 1e-3, 3e-4, 1e-4, 5e-4 to compare loss decay trajectories.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_anc_net import build_scaled_causal_anc_model
from model.loss import HybridANCLoss
from data.real_dataset_loader import get_hybrid_dataloaders
from eval.metrics_common import evaluate_batch_metrics, aggregate_metric_records


def run_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4]
    epochs_test = 4
    train_steps = 1200
    val_steps = 200

    results = {}

    print("================================================================================")
    print(" DRDO PS 26052: Diagnostic LR Sweep for CausalANCScaledNet (3.24M Params)")
    print(f" Testing LRs: {learning_rates} | Epochs per run: {epochs_test} | Steps: {train_steps}")
    print("================================================================================")

    for lr in learning_rates:
        print(f"\n---> Testing LR = {lr} ...")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = build_scaled_causal_anc_model(hidden_dim=256, num_gru_layers=2).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_test, eta_min=1e-5)
        loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        train_loader, val_loader, _ = get_hybrid_dataloaders(
            mode="hybrid",
            batch_size=16,
            train_epoch_size=train_steps,
            val_epoch_size=val_steps,
            hybrid_ratio=0.85,
            snr_sampler_mode="rebalanced",
            seed=42
        )

        history = []
        for ep in range(1, epochs_test + 1):
            model.train()
            total_train_loss = 0.0
            t0 = time.time()
            for batch in train_loader:
                noisy = batch["noisy"].to(device, non_blocking=True)
                clean = batch["clean"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                    enh, enh_r, enh_i, _ = model(noisy)
                    clean_r, clean_i, _ = model.stft_forward(clean)
                    losses = loss_fn(
                        pred_wav=enh,
                        target_wav=clean,
                        pred_real=enh_r,
                        pred_imag=enh_i,
                        target_real=clean_r,
                        target_imag=clean_i
                    )
                    loss = losses["total_loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            scheduler.step()
            train_loss = total_train_loss / max(1, len(train_loader))

            # Validation
            model.eval()
            total_val_loss = 0.0
            val_snrs = []
            val_stois = []
            with torch.no_grad():
                for batch in val_loader:
                    noisy = batch["noisy"].to(device, non_blocking=True)
                    clean = batch["clean"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16):
                        enh, enh_r, enh_i, _ = model(noisy)
                        clean_r, clean_i, _ = model.stft_forward(clean)
                        losses = loss_fn(
                            pred_wav=enh,
                            target_wav=clean,
                            pred_real=enh_r,
                            pred_imag=enh_i,
                            target_real=clean_r,
                            target_imag=clean_i
                        )
                        total_val_loss += losses["total_loss"].item()

                    c_np = clean.cpu().numpy()
                    n_np = noisy.cpu().numpy()
                    e_np = enh.cpu().numpy()
                    for i in range(len(c_np)):
                        r = evaluate_batch_metrics(c_np[i], n_np[i], e_np[i])
                        val_snrs.append(r["enh_snr"])
                        val_stois.append(r["enh_stoi"])

            val_loss = total_val_loss / max(1, len(val_loader))
            val_snr = float(np.mean(val_snrs))
            val_stoi = float(np.mean(val_stois))
            elapsed = time.time() - t0

            print(f"  [LR {lr:.0e}] Ep {ep}/{epochs_test} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SNR: {val_snr:+.2f}dB | STOI: {val_stoi:.4f}")
            history.append({
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_snr": val_snr,
                "val_stoi": val_stoi
            })

        results[lr] = history

    print("\n--- Summary: Val Loss & SNR after 4 Epochs across LRs ---")
    summary_table = []
    for lr, hist in results.items():
        summary_table.append([
            f"{lr:.0e}",
            f"{hist[0]['val_loss']:.4f}",
            f"{hist[1]['val_loss']:.4f}",
            f"{hist[2]['val_loss']:.4f}",
            f"{hist[3]['val_loss']:.4f}",
            f"{hist[3]['val_snr']:+.2f} dB",
            f"{hist[3]['val_stoi']:.4f}"
        ])
    print(tabulate(summary_table, headers=["Learning Rate", "Ep 1 ValLoss", "Ep 2 ValLoss", "Ep 3 ValLoss", "Ep 4 ValLoss", "Ep 4 Val SNR", "Ep 4 STOI"], tablefmt="grid"))


if __name__ == "__main__":
    run_sweep()
