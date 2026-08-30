"""
Phase 20 — Direct mask supervision (single pre-committed comparison).

Recipe IDENTICAL to Phase 17 (C7-L1), which scored 14.35 dB val / 14.35 dB test:
    30,000 optimizer steps, fp32, warmup 500 -> peak 5e-4 -> cosine 1e-5, batch 16, seed 42.
Phase 17 is therefore the CONTROL and the loss is the only variable.

ONE arm, ONE lambda, chosen by GRADIENT MATCHING rather than by sweeping, so this is a
single comparison and not a best-of-N search:
    |grad| HybridANCLoss   = 13.011
    |grad| MaskSupervision =  1.037
    lambda for 50% of the existing gradient = 6.27   <-- used here

Rationale: the measured bottleneck is estimation accuracy (the tanh head can represent
27.57 dB; the trained model realises 10.39 dB). Every existing loss term supervises a
SUMMED quantity -- waveform or spectrogram -- so per-bin gradient is diluted. The ideal
complex ratio mask is available in closed form, so estimation can be supervised directly,
per time-frequency bin.

Checkpoint selection on the VALIDATION split. Test set untouched.
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
from model.loss import HybridANCLoss, MaskSupervisionLoss
from eval.metrics_common import evaluate_batch_metrics

EPOCHS = 30
TRAIN_SAMPLES = 16000
VAL_SAMPLES = 200
BATCH = 16
PEAK_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 500
SEED = 42
WORKERS = 8
LAMBDA_MASK = 6.27
SAVE_DIR = "checkpoints/phase20_mask_supervision"
CONTROL_VAL = 14.35


def lr_at(step, total):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = min(1.0, max(0.0, (step - WARMUP) / max(1, total - WARMUP)))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * p))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    epochs = 2 if a.smoke else EPOCHS
    train_samples = 1600 if a.smoke else TRAIN_SAMPLES
    save_dir = SAVE_DIR + ("_smoke" if a.smoke else "")

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Phase20] device={dev} (fp32)  lambda_mask={LAMBDA_MASK}")
    print(f"[Phase20] {epochs} ep x {train_samples} samples = "
          f"{epochs*train_samples//BATCH:,} steps   CONTROL = Phase 17 @ {CONTROL_VAL} dB", flush=True)

    model = CausalANCNet(hidden_dim=128, num_gru_layers=2, norm_type="batch",
                         mask_bound=1.0, mask_mode="component_tanh").to(dev)
    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid", batch_size=BATCH, train_epoch_size=train_samples,
        val_epoch_size=VAL_SAMPLES, hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced", seed=SEED, num_workers=WORKERS)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    hyb = HybridANCLoss(sisnr_mode="clamped_alpha").to(dev)
    msk = MaskSupervisionLoss().to(dev)

    total_steps = epochs * max(1, len(train_loader))
    gstep, best, hist = 0, -999.0, []
    last_good = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train(); tot = 0.0; tot_m = 0.0; nb = 0; skipped = 0
        t0 = time.time()
        for batch in train_loader:
            cur_lr = lr_at(gstep, total_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            noisy = batch["noisy"].to(dev, non_blocking=True)
            clean = batch["clean"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            n_r, n_i, _ = model.stft_forward(noisy)
            c_r, c_i, _ = model.stft_forward(clean)
            enh_r, enh_i, mask_r, mask_i, _ = model.forward_spec(n_r, n_i, None)
            enh_wav = model.istft_forward(enh_r, enh_i, noisy.shape[-1])

            l_base = hyb(pred_wav=enh_wav, target_wav=clean, pred_real=enh_r,
                         pred_imag=enh_i, target_real=c_r, target_imag=c_i)["total_loss"]
            l_mask = msk(mask_r, mask_i, n_r, n_i, c_r, c_i)
            loss = l_base + LAMBDA_MASK * l_mask

            if not torch.isfinite(loss):
                skipped += 1; gstep += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; gstep += 1; continue
            opt.step()
            tot += loss.item(); tot_m += float(l_mask); nb += 1; gstep += 1

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[Phase20] !! non-finite weights ep{epoch}; restoring")
            model.load_state_dict(last_good); continue

        model.eval(); vs, vt = [], []
        with torch.no_grad():
            for b in val_loader:
                e = model(b["noisy"].to(dev))[0].float().cpu().numpy()
                cn, nn_ = b["clean"].numpy(), b["noisy"].numpy()
                for i in range(len(e)):
                    if not np.all(np.isfinite(e[i])):
                        continue
                    r = evaluate_batch_metrics(cn[i], nn_[i], e[i])
                    vs.append(r["enh_snr"]); vt.append(r["enh_stoi"])
        val_snr = float(np.mean(vs)); val_stoi = float(np.mean(vt))
        score = val_snr + 10.0 * val_stoi
        is_best = np.isfinite(score) and score > best
        if is_best:
            best = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "hidden_dim": 128, "norm_type": "batch", "mask_bound": 1.0,
                        "mask_mode": "component_tanh", "val_snr": val_snr,
                        "val_stoi": val_stoi, "op_score": score,
                        "lambda_mask": LAMBDA_MASK},
                       os.path.join(save_dir, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())

        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb),
                     "mask_loss": tot_m / max(1, nb), "val_snr": val_snr,
                     "val_stoi": val_stoi, "op_score": score, "skipped": skipped})
        json.dump(hist, open(os.path.join(save_dir, "train_history.json"), "w"), indent=2)
        print(f"[Phase20] Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s, lr={cur_lr:.2e}) "
              f"| Train {tot/max(1,nb):.4f} | Mask {tot_m/max(1,nb):.4f} "
              f"| Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} | Score {score:.2f} "
              f"| skip {skipped}" + (" >>> [BEST]" if is_best else ""), flush=True)

    b = max(hist, key=lambda x: x["op_score"])
    d = b["val_snr"] - CONTROL_VAL
    print("\n" + "=" * 88)
    print(f"PHASE 20 RESULT  best val SNR {b['val_snr']:+.2f} dB (ep {b['epoch']}), "
          f"STOI {b['val_stoi']:.4f}")
    print(f"  vs CONTROL (Phase 17, identical recipe) {CONTROL_VAL:.2f} dB : {d:+.2f} dB")
    print(f"  -> {'PROMOTE to a single test evaluation' if d > 1.0 else 'DO NOT PROMOTE'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
