"""
Phase 23 — FAIR capacity test (C5-C1).

The Phase 5-7 conclusion "capacity ruled out" was drawn at 3,750 optimizer steps, where no
model of any size could have shown a benefit. This re-tests it at a budget where the 811K
model demonstrably still improves.

Recipe is IDENTICAL to P22-LONG (which scored 15.00 dB val / 15.26 dB test with the 811K
model), so CAPACITY is the only variable:
    150,000 steps, fp32, warmup 500 -> peak 5e-4 -> cosine 1e-5, batch 16, seed 42,
    workers 8 with the worker_init_fn fix (0% duplicate samples, verified).

  model    : CausalANCScaledNet, hidden_dim 256  -> 3,235,842 params (4x the 811K baseline)
  control  : P22-LONG, 811K params, same recipe  -> 15.00 dB val

Motivation: P22 finished with train loss at -0.25 while validation oscillated in hundredths
(14.96-15.00 over its final 10 epochs). A model that fits training data hard while
validation is pinned is the textbook signature of a capacity limit, not a compute limit.
Separately, the architecture can represent 27.57 dB (probe_representational_vs_estimation)
while realising ~15, so the headroom is estimation quality.

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
from model.causal_anc_net import CausalANCScaledNet
from model.loss import HybridANCLoss
from eval.metrics_common import evaluate_batch_metrics

EPOCHS = 75
TRAIN_SAMPLES = 32000
VAL_SAMPLES = 200
BATCH = 16
PEAK_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 500
SEED = 42
WORKERS = 8
SAVE_DIR = "checkpoints/phase23_capacity_3M"
CONTROL_VAL = 15.00   # P22-LONG, 811K, identical recipe


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

    model = CausalANCScaledNet(hidden_dim=256).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[Phase23] device={dev} fp32 workers={WORKERS}")
    print(f"[Phase23] CausalANCScaledNet hidden=256 -> {n_par:,} params (control: 811,522)")
    print(f"[Phase23] {epochs} ep x {train_samples} samples = {epochs*train_samples//BATCH:,} steps")
    print(f"[Phase23] CONTROL = P22-LONG (811K, identical recipe) @ {CONTROL_VAL:.2f} dB val", flush=True)

    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid", batch_size=BATCH, train_epoch_size=train_samples,
        val_epoch_size=VAL_SAMPLES, hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced", seed=SEED, num_workers=WORKERS)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    hyb = HybridANCLoss(sisnr_mode="clamped_alpha").to(dev)

    total_steps = epochs * max(1, len(train_loader))
    gstep, best, hist = 0, -999.0, []
    last_good = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train(); tot = 0.0; nb = 0; skipped = 0
        t0 = time.time()
        for batch in train_loader:
            cur_lr = lr_at(gstep, total_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            noisy = batch["noisy"].to(dev, non_blocking=True)
            clean = batch["clean"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            enh_wav, enh_r, enh_i, _ = model(noisy)
            c_r, c_i, _ = model.stft_forward(clean)
            loss = hyb(pred_wav=enh_wav, target_wav=clean, pred_real=enh_r,
                       pred_imag=enh_i, target_real=c_r, target_imag=c_i)["total_loss"]

            if not torch.isfinite(loss):
                skipped += 1; gstep += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; gstep += 1; continue
            opt.step()
            tot += loss.item(); nb += 1; gstep += 1

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[Phase23] !! non-finite weights ep{epoch}; restoring")
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
                        "hidden_dim": 256, "arch": "CausalANCScaledNet",
                        "val_snr": val_snr, "val_stoi": val_stoi, "op_score": score},
                       os.path.join(save_dir, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())

        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb), "val_snr": val_snr,
                     "val_stoi": val_stoi, "op_score": score, "skipped": skipped})
        json.dump(hist, open(os.path.join(save_dir, "train_history.json"), "w"), indent=2)
        print(f"[Phase23] Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s) "
              f"| Train {tot/max(1,nb):.4f} | Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} "
              f"| Score {score:.2f} | skip {skipped}" + (" >>> [BEST]" if is_best else ""), flush=True)

    b = max(hist, key=lambda x: x["op_score"])
    d = b["val_snr"] - CONTROL_VAL
    print("\n" + "=" * 88)
    print("PHASE 23 RESULT - fair capacity test, 3.24M vs 811K at identical budget")
    print("=" * 88)
    print(f"  3.24M model   : {b['val_snr']:+7.2f} dB  STOI {b['val_stoi']:.4f}  (ep {b['epoch']})")
    print(f"  811K control  : {CONTROL_VAL:+7.2f} dB  [P22-LONG, identical recipe]")
    print(f"  delta         : {d:+7.2f} dB  -> "
          f"{'PROMOTE to a single test evaluation' if d > 1.0 else 'DO NOT PROMOTE'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
