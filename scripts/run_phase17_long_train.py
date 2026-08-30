"""
Phase 17 / Section C.7 — Is the 12.75 dB plateau optimisation-limited?

Trains the EXACT published baseline architecture (CausalANCNet, 811K params,
component_tanh, K=1.0, BatchNorm), the same HybridANCLoss, and the same data pipeline.

WHAT IS DIFFERENT FROM THE PUBLISHED BASELINE (disclose all three when reporting):
  1. Training length : 40 ep x 12000 samples = 30,000 optimizer steps
                       (every prior "full 50-epoch" run was 3,750 steps)
  2. LR recipe       : 500-step linear warmup, peak 5e-4 (not 1e-3), cosine -> 1e-5.
                       The first attempt at this length diverged to NaN at epoch ~4 using
                       the original flat 1e-3; that recipe had never seen more than ~75
                       steps per epoch at high LR, and 750/epoch destabilised it.
  3. NaN recovery    : non-finite batches are skipped; if the weights themselves go
                       non-finite the last good epoch snapshot is restored and LR is
                       halved. Prevents one bad batch from destroying hours of compute.

Nothing about the architecture, mask, loss or data is changed, so any gain is attributable
to the training budget/recipe alone. Checkpoint selection uses the VALIDATION split only.
"""
import os
import sys
import copy
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.optim import AdamW

from data.real_dataset_loader import get_hybrid_dataloaders
from model.causal_anc_net import CausalANCNet
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics

EPOCHS = 40
TRAIN_SAMPLES = 12000
VAL_SAMPLES = 200
BATCH = 16
PEAK_LR = 5e-4
MIN_LR = 1e-5
WARMUP_STEPS = 500
SEED = 42
SAVE_DIR = "checkpoints/phase17_long_train"


def lr_at(step, total_steps):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    p = min(1.0, max(0.0, p))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * p))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(os.cpu_count() or 8)
    device = "cpu"  # CUDA unusable here: sm_120 GPU vs sm_90-max torch build

    print(f"[Phase17] threads={torch.get_num_threads()} device={device}")
    print(f"[Phase17] {EPOCHS} ep x {TRAIN_SAMPLES} samples = "
          f"{EPOCHS*TRAIN_SAMPLES//BATCH:,} optimizer steps (prior runs: 3,750)")
    print(f"[Phase17] warmup={WARMUP_STEPS} peak_lr={PEAK_LR} -> {MIN_LR}")

    model = CausalANCNet(hidden_dim=128, num_gru_layers=2, norm_type="batch",
                         mask_bound=1.0, mask_mode="component_tanh").to(device)

    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid", batch_size=BATCH, train_epoch_size=TRAIN_SAMPLES,
        val_epoch_size=VAL_SAMPLES, hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced", seed=SEED)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(device)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    gstep = 0
    best_score = -999.0
    last_good = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot, nb, skipped = 0.0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            cur_lr = lr_at(gstep, total_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            opt.zero_grad(set_to_none=True)
            enhanced, enh_r, enh_i, _ = model(noisy)
            clean_r, clean_i, _ = model.stft_forward(clean)
            losses = loss_fn(pred_wav=enhanced, target_wav=clean,
                             pred_real=enh_r, pred_imag=enh_i,
                             target_real=clean_r, target_imag=clean_i)
            loss = losses["total_loss"]

            if not torch.isfinite(loss):
                skipped += 1
                gstep += 1
                continue

            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                gstep += 1
                continue

            opt.step()
            tot += loss.item()
            nb += 1
            gstep += 1

        # Weight-level integrity check
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[Phase17] !! weights non-finite at epoch {epoch}; restoring last good "
                  f"snapshot and halving peak LR")
            model.load_state_dict(last_good)
            globals()["PEAK_LR"] = PEAK_LR * 0.5
            continue

        train_loss = tot / max(1, nb)

        model.eval()
        vs, vt = [], []
        with torch.no_grad():
            for b in val_loader:
                n = b["noisy"].to(device)
                c = b["clean"]
                enh, _, _, _ = model(n)
                e = enh.cpu().numpy()
                cn = c.numpy()
                nn_ = b["noisy"].numpy()
                for i in range(len(e)):
                    if not np.all(np.isfinite(e[i])):
                        continue
                    r = evaluate_batch_metrics(cn[i], nn_[i], e[i])
                    vs.append(r["enh_snr"])
                    vt.append(r["enh_stoi"])

        val_snr = float(np.mean(vs)) if vs else float("nan")
        val_stoi = float(np.mean(vt)) if vt else float("nan")
        score = val_snr + 10.0 * val_stoi

        is_best = np.isfinite(score) and score > best_score
        if is_best:
            best_score = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "hidden_dim": 128, "norm_type": "batch", "mask_bound": 1.0,
                        "mask_mode": "component_tanh", "val_snr": val_snr,
                        "val_stoi": val_stoi, "op_score": score},
                       os.path.join(SAVE_DIR, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())

        history.append({"epoch": epoch, "train_loss": train_loss, "val_snr": val_snr,
                        "val_stoi": val_stoi, "op_score": score, "skipped": skipped,
                        "lr": cur_lr})
        json.dump(history, open(os.path.join(SAVE_DIR, "train_history.json"), "w"), indent=2)

        tag = " >>> [BEST]" if is_best else ""
        print(f"[Phase17] Epoch [{epoch:02d}/{EPOCHS}] ({time.time()-t0:.0f}s, lr={cur_lr:.2e}) "
              f"| Train {train_loss:.4f} | Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} "
              f"| Score {score:.2f} | skipped {skipped}{tag}", flush=True)

    print(f"[Phase17] done. best val op_score={best_score:.2f} -> {SAVE_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
