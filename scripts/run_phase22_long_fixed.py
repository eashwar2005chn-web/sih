"""
Phase 21 — Re-run on the FIXED data pipeline, with an internal control.

Phases 18/19/20 were invalidated: they ran at num_workers=8, where all workers inherited an
identical copy of the train-split RandomState and produced 76.4% duplicate samples, while
their control (Phase 17) ran at num_workers=0 with 0% duplicates. Fixed via worker_init_fn
(verified 0% duplicates at num_workers=8).

This run does NOT trust Phase 17's 14.35 dB as a control any more, because that number was
produced on a different data path. Instead it trains its OWN control under exactly the
settings the treatment arm uses. Everything is shared: pipeline, workers, precision, seed,
schedule, step count. The loss is the only variable.

  ARM A  CONTROL  : HybridANCLoss alone            (replicates the Phase 17 recipe)
  ARM B  TREATMENT: + direct per-bin mask supervision, lambda=6.27
                    (lambda fixed in advance by gradient matching, not swept)

30,000 steps, fp32, warmup 500 -> peak 5e-4 -> cosine 1e-5, batch 16, seed 42, workers 8.
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

EPOCHS = 75
TRAIN_SAMPLES = 32000
VAL_SAMPLES = 200
BATCH = 16
PEAK_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 500
SEED = 42
WORKERS = 8
LAMBDA_MASK = 6.27

ARMS = [("LONG150K", 0.0)]


def lr_at(step, total):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = min(1.0, max(0.0, (step - WARMUP) / max(1, total - WARMUP)))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * p))


def train_arm(name, lam, epochs, train_samples):
    save_dir = f"checkpoints/phase22_{name}"
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*88}\n[Phase22] ARM {name}  lambda_mask={lam}  device={dev} fp32 "
          f"workers={WORKERS}\n{'='*88}", flush=True)

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
        model.train(); tot = 0.0; nb = 0; skipped = 0
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

            loss = hyb(pred_wav=enh_wav, target_wav=clean, pred_real=enh_r,
                       pred_imag=enh_i, target_real=c_r, target_imag=c_i)["total_loss"]
            if lam > 0:
                loss = loss + lam * msk(mask_r, mask_i, n_r, n_i, c_r, c_i)

            if not torch.isfinite(loss):
                skipped += 1; gstep += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; gstep += 1; continue
            opt.step()
            tot += loss.item(); nb += 1; gstep += 1

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[Phase22] !! non-finite weights ep{epoch}; restoring")
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
                        "val_stoi": val_stoi, "op_score": score, "lambda_mask": lam},
                       os.path.join(save_dir, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())

        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb), "val_snr": val_snr,
                     "val_stoi": val_stoi, "op_score": score, "skipped": skipped})
        json.dump(hist, open(os.path.join(save_dir, "train_history.json"), "w"), indent=2)
        print(f"[{name}] Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s) "
              f"| Train {tot/max(1,nb):.4f} | Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} "
              f"| Score {score:.2f} | skip {skipped}" + (" >>> [BEST]" if is_best else ""), flush=True)

    b = max(hist, key=lambda x: x["op_score"])
    return {"arm": name, "lambda_mask": lam, "best_epoch": b["epoch"],
            "best_val_snr": round(b["val_snr"], 4), "best_val_stoi": round(b["val_stoi"], 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ep = 2 if a.smoke else EPOCHS
    ts = 1600 if a.smoke else TRAIN_SAMPLES

    out = {}
    for name, lam in ARMS:
        out[name] = train_arm(name, lam, ep, ts)
        json.dump(out, open("eval/phase22_summary.json", "w"), indent=2)

    r = out["LONG150K"]
    CTRL = 14.17  # P21-A internal control, SAME fixed pipeline, 30,000 steps
    d = r["best_val_snr"] - CTRL
    print("\n" + "=" * 88)
    print("PHASE 22 RESULT — 150,000 steps on the FIXED data pipeline (fp32)")
    print("=" * 88)
    print(f"  150k steps          : {r['best_val_snr']:+7.2f} dB  STOI {r['best_val_stoi']:.4f}  (ep {r['best_epoch']})")
    print(f"  P21-A control 30k   : {CTRL:+7.2f} dB   [same pipeline, same recipe, 5x fewer steps]")
    print(f"  delta               : {d:+7.2f} dB  -> "
          f"{'PROMOTE to a single test evaluation' if d > 1.0 else 'DO NOT PROMOTE'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
