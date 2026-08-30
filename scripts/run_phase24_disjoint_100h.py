"""
Phase 24 - the 100-hour speaker-disjoint retrain.

Recipe is IDENTICAL to P22-LONG (15.00 dB val / 15.26 dB test) in every respect:
    150,000 steps, fp32, warmup 500 -> peak 5e-4 -> cosine 1e-5, batch 16, seed 42,
    workers 8 with the worker_init_fn fix.
TRAINING DATA is the only variable.

  control    : P22-LONG, 811K params, 5.04 h dev-clean-derived corpus  -> 15.00 dB val
  treatment  : same model, LibriSpeech train-clean-100, 100.59 h, 251 speakers

Motivation: Phases 16-23 eliminated mask shape, optimizer steps, supervision density and
capacity. Phase 23 was decisive on capacity - a 4x model reached the SAME final train loss
(-0.2370 vs -0.2383). Data volume is the only constraint still standing, and the previous
corpus was LibriSpeech dev-clean (the DEVELOPMENT split) at 5.04 h, ~9,073 distinct 2 s
segments each reused ~225 times.

This ALSO repairs two disclosed defects, because train-clean-100 is speaker-disjoint from
dev-clean by corpus construction:
  - speaker overlap train x test : 41 of 41  ->  0
  - content overlap train x test : 30.12%    ->  0.00%
val/test file lists are unchanged and guarded by the frozen manifest, so the result remains
directly comparable to every number in the ledger.

HONEST EXPECTATION: the result may land BELOW 15.26 dB. More data pushes up; removing
same-speaker and same-content overlap removes an optimistic bias and pushes down. Which
dominates is not knowable in advance, and either outcome is informative.

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

import data.real_dataset_loader as L
from data.real_dataset_loader import get_hybrid_dataloaders, verify_eval_splits_frozen
from model.causal_anc_net import CausalANCNet
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
# The control (P22-LONG) used 8. Two attempts at 8 died here with
#   RuntimeError: Couldn't open shared file mapping ... error code: <1455>   (ERROR_COMMITMENT_LIMIT)
# raised from the DataLoader collate path -- Windows backs torch shared memory with the paging
# file, and 8 persistent workers x prefetch 4 against the 28,539-file corpus exceeds the commit
# limit on this 16.5 GB machine. The second crash happened before epoch 1, so it is not
# contention from other processes.
#
# 4 workers runs clean, and measured steady-state throughput is 25 ms/batch (~0.8 min/epoch),
# so this costs nothing in wall-clock. Worker count changes WHICH samples each worker draws
# (each has its own RNG via worker_init_fn) but NOT the distribution they are drawn from, so
# the comparison against the control remains fair. Disclosed on the ledger row.
WORKERS = 4
SAVE_DIR = "checkpoints/phase24_disjoint_100h"
CONTROL_VAL = 15.00   # P22-LONG, same recipe, 5.04 h corpus


def lr_at(step, total):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    p = min(1.0, max(0.0, (step - WARMUP) / max(1, total - WARMUP)))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + np.cos(np.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    epochs = 2 if a.smoke else EPOCHS
    train_samples = 1600 if a.smoke else TRAIN_SAMPLES
    save_dir = SAVE_DIR + ("_smoke" if a.smoke else "")

    # --- REGIME: explicit, logged, and verified before a single step is taken -------------
    cfg = L.set_speaker_disjoint_training(True)
    verify_eval_splits_frozen()
    m = L.RealAudioManifest()
    n_train = len(m.get_split_files("clean_speech", "train"))

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = CausalANCNet(hidden_dim=128, num_gru_layers=2, norm_type="batch",
                         mask_bound=1.0, mask_mode="component_tanh").to(dev)
    n_par = sum(p.numel() for p in model.parameters())

    print("=" * 92)
    print("PHASE 24 - 100 h SPEAKER-DISJOINT RETRAIN")
    print("=" * 92)
    print(f"[P24] REGIME speaker_disjoint=True  excluded={cfg['excluded_from_train']}")
    print(f"[P24] clean-speech sources: {cfg['clean_speech_sources']}")
    print(f"[P24] train clean-speech files: {n_train:,}  (control used 2,546)")
    print("[P24] frozen eval manifest: VERIFIED - val/test unchanged")
    print(f"[P24] device={dev} fp32 workers={WORKERS}  params={n_par:,}")
    print(f"[P24] {epochs} ep x {train_samples} = {epochs*train_samples//BATCH:,} steps")
    print(f"[P24] CONTROL = P22-LONG @ {CONTROL_VAL:.2f} dB val (identical recipe, 5.04 h)")
    print("=" * 92, flush=True)

    train_loader, val_loader, _ = get_hybrid_dataloaders(
        mode="hybrid", batch_size=BATCH, train_epoch_size=train_samples,
        val_epoch_size=VAL_SAMPLES, hybrid_ratio=0.85,
        snr_sampler_mode="rebalanced", seed=SEED, num_workers=WORKERS)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    hyb = HybridANCLoss(sisnr_mode="clamped_alpha").to(dev)

    total_steps = epochs * max(1, len(train_loader))
    gstep, best, hist = 0, -999.0, []
    start_epoch = 1

    # ---- resume ----------------------------------------------------------------------
    # This machine has 16.5 GB of RAM and the run uses 8 dataloader workers against a 28,539
    # file corpus; an earlier attempt died at epoch 9 with ERROR_COMMITMENT_LIMIT when other
    # processes competed for memory. Losing two hours to that is avoidable, so full optimizer
    # + RNG state is written every epoch and picked up automatically on restart.
    resume_path = os.path.join(save_dir, "last_state.pt")
    if os.path.isfile(resume_path):
        # weights_only=False is REQUIRED here and is safe: this file is written by this script,
        # a few lines below, and never comes from anywhere else. PyTorch 2.6 changed the default
        # to True, which refuses the numpy RNG state (numpy._core.multiarray.scalar is not an
        # allowlisted global) and makes resume fail with an UnpicklingError that reads like file
        # corruption. Do not "fix" this by dropping the RNG state - losing it would silently
        # change the data stream on resume.
        ck = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
        np.random.set_state(ck["numpy_rng"])
        start_epoch = ck["epoch"] + 1
        gstep = ck["gstep"]
        best = ck["best"]
        hist = ck["hist"]
        print(f"[P24] RESUMED from epoch {ck['epoch']} (gstep {gstep}, best score {best:.2f})",
              flush=True)

    last_good = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tot = 0.0
        nb = 0
        skipped = 0
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

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"[P24] !! non-finite weights ep{epoch}; restoring")
            model.load_state_dict(last_good)
            continue

        model.eval()
        vs, vt = [], []
        with torch.no_grad():
            for b in val_loader:
                e = model(b["noisy"].to(dev))[0].float().cpu().numpy()
                cn, nn_ = b["clean"].numpy(), b["noisy"].numpy()
                for i in range(len(e)):
                    if not np.all(np.isfinite(e[i])):
                        continue
                    r = evaluate_batch_metrics(cn[i], nn_[i], e[i])
                    vs.append(r["enh_snr"])
                    vt.append(r["enh_stoi"])
        val_snr = float(np.mean(vs))
        val_stoi = float(np.mean(vt))
        score = val_snr + 10.0 * val_stoi
        is_best = np.isfinite(score) and score > best
        if is_best:
            best = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "hidden_dim": 128, "norm_type": "batch", "mask_bound": 1.0,
                        "mask_mode": "component_tanh", "val_snr": val_snr,
                        "val_stoi": val_stoi, "op_score": score,
                        "regime": "speaker_disjoint_100h",
                        "train_clean_files": n_train},
                       os.path.join(save_dir, "best_model.pt"))
            last_good = copy.deepcopy(model.state_dict())

        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb), "val_snr": val_snr,
                     "val_stoi": val_stoi, "op_score": score, "skipped": skipped})
        json.dump(hist, open(os.path.join(save_dir, "train_history.json"), "w"), indent=2)

        # Full resumable state, written every epoch (see the resume block above).
        torch.save({"epoch": epoch, "gstep": gstep, "best": best, "hist": hist,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state()},
                   os.path.join(save_dir, "last_state.pt"))
        print(f"[P24] Epoch [{epoch:02d}/{epochs}] ({time.time()-t0:.0f}s) "
              f"| Train {tot/max(1,nb):.4f} | Val SNR {val_snr:+.2f}dB | STOI {val_stoi:.4f} "
              f"| Score {score:.2f} | skip {skipped}"
              + (" >>> [BEST]" if is_best else ""), flush=True)

    b = max(hist, key=lambda x: x["op_score"])
    d = b["val_snr"] - CONTROL_VAL
    print("\n" + "=" * 92)
    print("PHASE 24 RESULT - 100.59 h speaker-disjoint vs 5.04 h overlapping, identical recipe")
    print("=" * 92)
    print(f"  100 h disjoint : {b['val_snr']:+7.2f} dB  STOI {b['val_stoi']:.4f}  (ep {b['epoch']})")
    print(f"  P22 control    : {CONTROL_VAL:+7.2f} dB  [5.04 h, same speakers+content as test]")
    print(f"  delta          : {d:+7.2f} dB")
    print("  NOTE: the control was measured with speaker and content overlap; this run has")
    print("        neither, so a smaller delta here can still be the better model.")
    print("=" * 92)


if __name__ == "__main__":
    main()
