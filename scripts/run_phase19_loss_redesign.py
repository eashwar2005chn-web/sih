"""
Phase 19 / Section C.2 — Loss redesign.

Recipe is IDENTICAL to Phase 17 (C7-L1), which scored 14.35 dB val / 14.35 dB test:
    30,000 optimizer steps, fp32, warmup 500 -> peak 5e-4 -> cosine 1e-5, batch 16, seed 42.
Phase 17 therefore serves as the CONTROL -- the loss is the only variable changed.

fp32 deliberately, not AMP: Phase 18 skipped 63 batches to fp16 gradient overflow and lost
to Phase 17 despite 5x the steps, so precision is held at the known-good setting here.

Arms:
  C2-A  + direct scale-dependent SNR loss (lambda_snr=1.0)
  C2-B  + direct SNR loss AND input-SNR-weighted sampling (hard_weight=3.0)

Motivation (scripts/probe_scale_mismatch.py): the Phase 17 model runs ~8% hot; the optimal
per-utterance rescale recovers +1.06 dB but a single global constant recovers only +0.06 dB,
so the gain error is utterance-dependent and can only be learned, not post-corrected.

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
from model.loss import ScaleAwareANCLoss
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

ARMS = {
    "C2-A_snr_loss": dict(lambda_snr=1.0, hard_weight=1.0),
    "C2-B_snr_plus_weighted": dict(lambda_snr=1.0, hard_weight=3.0),
}


def lr_at(step, total):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = min(1.0, max(0.0, (step - WARMUP) / max(1, total - WARMUP)))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * p))


def train_arm(name, cfg, epochs, train_samples):
    save_dir = f"checkpoints/phase19_{name}"
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*88}\n[Phase19] ARM {name}  {cfg}  device={dev} (fp32)\n{'='*88}", flush=True)

    model = CausalANCNet(hidden_dim=128, num_gru_layers=2, norm_type="batch",
                         mask_bound=1.0, mask_mode="component_tanh").to(dev)
    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid", batch_size=BATCH, train_epoch_size=train_samples,
        val_epoch_size=VAL_SAMPLES, hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced", seed=SEED, num_workers=WORKERS)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    loss_fn = ScaleAwareANCLoss(sisnr_mode="clamped_alpha", **cfg).to(dev)

    total_steps = epochs * max(1, len(train_loader))
    gstep, best, hist = 0, -999.0, []
    last_good = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train(); tot = nb = skipped = 0; tot = 0.0
        t0 = time.time()
        for batch in train_loader:
            cur_lr = lr_at(gstep, total_steps)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            noisy = batch["noisy"].to(dev, non_blocking=True)
            clean = batch["clean"].to(dev, non_blocking=True)
            snr_db = batch["snr_db"].to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            enhanced, enh_r, enh_i, _ = model(noisy)
            clean_r, clean_i, _ = model.stft_forward(clean)
            losses = loss_fn(pred_wav=enhanced, target_wav=clean,
                             pred_real=enh_r, pred_imag=enh_i,
                             target_real=clean_r, target_imag=clean_i, snr_db=snr_db)
            loss = losses["total_loss"]
            if not torch.isfinite(loss):
                skipped += 1; gstep += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; gstep += 1; continue
            opt.step()
            tot += loss.item(); nb += 1; gstep += 1

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[Phase19] !! non-finite weights ep{epoch}; restoring"); model.load_state_dict(last_good); continue

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
                        "val_stoi": val_stoi, "op_score": score, "loss_cfg": cfg},
                       os.path.join(save_dir, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())
        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb), "val_snr": val_snr,
                     "val_stoi": val_stoi, "op_score": score, "skipped": skipped})
        json.dump(hist, open(os.path.join(save_dir, "train_history.json"), "w"), indent=2)
        print(f"[{name}] Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s, lr={cur_lr:.2e}) "
              f"| Train {tot/max(1,nb):.4f} | Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} "
              f"| Score {score:.2f} | skip {skipped}" + (" >>> [BEST]" if is_best else ""), flush=True)

    b = max(hist, key=lambda x: x["op_score"])
    return {"arm": name, "cfg": cfg, "best_epoch": b["epoch"],
            "best_val_snr": round(b["val_snr"], 4), "best_val_stoi": round(b["val_stoi"], 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ep = 2 if a.smoke else EPOCHS
    ts = 1600 if a.smoke else TRAIN_SAMPLES

    out = {}
    for name, cfg in ARMS.items():
        out[name] = train_arm(name, cfg, ep, ts)
        json.dump(out, open("eval/phase19_summary.json", "w"), indent=2)

    print("\n" + "=" * 88)
    print("PHASE 19 SUMMARY (val split).  CONTROL = Phase 17, identical recipe: 14.35 dB")
    print("=" * 88)
    for k, v in out.items():
        d = v["best_val_snr"] - 14.35
        print(f"  {k:26} val SNR {v['best_val_snr']:+7.2f} dB  STOI {v['best_val_stoi']:.4f}  "
              f"vs control {d:+6.2f} dB  -> {'PROMOTE' if d > 1.0 else 'do not promote'}")


if __name__ == "__main__":
    main()
