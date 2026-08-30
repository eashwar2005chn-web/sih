"""
Phase 18 / Section C.7 — Long GPU training run (C7-L2).

Same 811K CausalANCNet, same component_tanh K=1.0 mask, same HybridANCLoss, same data
pipeline as the published baseline. The ONLY differences:

    training length : 150,000 optimizer steps
                      (published baseline: 3,750   |  Phase 17 CPU run: 30,000)
    LR recipe       : 1000-step warmup -> peak 5e-4 -> cosine to 1e-5
    precision       : AMP fp16 on CUDA
    dataloading     : num_workers=8 (data gen is 96 ms/batch vs 18 ms GPU compute;
                      with 0 workers the GPU would idle ~84% of the time)
    robustness      : non-finite batches skipped, weights snapshot/restored on corruption

Enabled by the cu128 upgrade (torch 2.11.0, sm_120 kernels) -- 13.2x faster per step.

Checkpoint selection uses the VALIDATION split only. Test set untouched.
Usage:  python scripts/run_phase18_gpu_long_train.py [--smoke]
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.optim import AdamW

from data.real_dataset_loader import get_hybrid_dataloaders
from model.causal_anc_net import CausalANCNet
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics

EPOCHS = 75
TRAIN_SAMPLES = 32000
VAL_SAMPLES = 200
BATCH = 16
PEAK_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 1000
SEED = 42
WORKERS = 8
SAVE_DIR = "checkpoints/phase18_gpu_long"


def lr_at(step, total):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = min(1.0, max(0.0, (step - WARMUP) / max(1, total - WARMUP)))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 tiny epochs to validate the setup")
    args = ap.parse_args()

    epochs = 2 if args.smoke else EPOCHS
    train_samples = 3200 if args.smoke else TRAIN_SAMPLES
    save_dir = SAVE_DIR + ("_smoke" if args.smoke else "")

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _p = torch.randn(16, 16, device=dev); _ = (_p @ _p).sum().item()
    except Exception as e:
        print(f"[Phase18] CUDA probe failed ({e}); falling back to CPU")
        dev = "cpu"
    use_amp = (dev == "cuda")

    print(f"[Phase18] device={dev} amp={use_amp} workers={WORKERS}")
    print(f"[Phase18] {epochs} ep x {train_samples} samples = "
          f"{epochs*train_samples//BATCH:,} optimizer steps")

    model = CausalANCNet(hidden_dim=128, num_gru_layers=2, norm_type="batch",
                         mask_bound=1.0, mask_mode="component_tanh").to(dev)

    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid", batch_size=BATCH, train_epoch_size=train_samples,
        val_epoch_size=VAL_SAMPLES, hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced", seed=SEED, num_workers=WORKERS)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    loss_fn = HybridANCLoss(sisnr_mode="clamped_alpha").to(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = epochs * max(1, len(train_loader))
    gstep, best, hist = 0, -999.0, []
    last_good = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        tot, nb, skipped = 0.0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            cur_lr = lr_at(gstep, total_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            noisy = batch["noisy"].to(dev, non_blocking=True)
            clean = batch["clean"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                enhanced, enh_r, enh_i, _ = model(noisy)
                clean_r, clean_i, _ = model.stft_forward(clean)
                losses = loss_fn(pred_wav=enhanced, target_wav=clean,
                                 pred_real=enh_r, pred_imag=enh_i,
                                 target_real=clean_r, target_imag=clean_i)
                loss = losses["total_loss"]

            if not torch.isfinite(loss):
                skipped += 1; gstep += 1; continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; gstep += 1
                scaler.update(); continue
            scaler.step(opt)
            scaler.update()

            tot += loss.item(); nb += 1; gstep += 1

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[Phase18] !! non-finite weights at epoch {epoch}; restoring snapshot")
            model.load_state_dict(last_good)
            continue

        train_loss = tot / max(1, nb)

        model.eval()
        vs, vt = [], []
        with torch.no_grad():
            for b in val_loader:
                enh, _, _, _ = model(b["noisy"].to(dev))
                e = enh.float().cpu().numpy()
                cn, nn_ = b["clean"].numpy(), b["noisy"].numpy()
                for i in range(len(e)):
                    if not np.all(np.isfinite(e[i])):
                        continue
                    r = evaluate_batch_metrics(cn[i], nn_[i], e[i])
                    vs.append(r["enh_snr"]); vt.append(r["enh_stoi"])

        val_snr = float(np.mean(vs)) if vs else float("nan")
        val_stoi = float(np.mean(vt)) if vt else float("nan")
        score = val_snr + 10.0 * val_stoi

        is_best = np.isfinite(score) and score > best
        if is_best:
            best = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "hidden_dim": 128, "norm_type": "batch", "mask_bound": 1.0,
                        "mask_mode": "component_tanh", "val_snr": val_snr,
                        "val_stoi": val_stoi, "op_score": score},
                       os.path.join(save_dir, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())

        hist.append({"epoch": epoch, "train_loss": train_loss, "val_snr": val_snr,
                     "val_stoi": val_stoi, "op_score": score, "skipped": skipped,
                     "lr": cur_lr, "sec": round(time.time() - t0, 1)})
        json.dump(hist, open(os.path.join(save_dir, "train_history.json"), "w"), indent=2)

        print(f"[Phase18] Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s, lr={cur_lr:.2e}) "
              f"| Train {train_loss:.4f} | Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} "
              f"| Score {score:.2f} | skip {skipped}" + (" >>> [BEST]" if is_best else ""),
              flush=True)

    print(f"[Phase18] done. best val op_score={best:.2f} -> {save_dir}/best_model.pt")


if __name__ == "__main__":
    main()
