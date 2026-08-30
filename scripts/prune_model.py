"""
Model pruning for edge deployment - PS 26052 item D3.

The problem statement lists "quantization, pruning, and ONNX / TensorRT conversion" among the
optimization techniques to be applied. Quantization (OpenVINO INT8) and ONNX export were
already implemented; pruning was not.

This sweeps unstructured magnitude pruning across sparsity levels and reports, for each, the
measured cost on the VALIDATION split. No claim is made that pruning helps: it is an
optimization that trades accuracy for size/compute, and the sweep exists to quantify that
trade honestly.

Two things this deliberately does NOT do:

  - It does not report a compressed file size for unstructured pruning as if it were a
    deployment win. Zeroed weights still occupy dense storage; the saving is only realised by
    a sparse runtime or by structured pruning. Both dense and sparse-aware sizes are shown.
  - It does not fine-tune after pruning. Fine-tuning would recover accuracy but costs a
    training run, and would need its own control to be interpretable.

    python scripts/prune_model.py                  # sweep, no write
    python scripts/prune_model.py --save 0.5       # also write the 50%-sparse checkpoint
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from eval.metrics_common import evaluate_batch_metrics

CKPT = "checkpoints/phase22_LONG150K/best_model.pt"
SPARSITIES = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
BUCKETS = [
    ("Op [0,5)", 0.0, 5.0, 92000),
    ("Op [5,10)", 5.0, 10.0, 92500),
    ("Op [10,15]", 10.0, 15.0, 93000),
]
PER_BUCKET = 40


def prunable(model):
    """
    Conv and Linear weights only. Norm layers and biases are excluded - pruning them is
    disproportionately damaging for negligible savings.

    GRU IS DELIBERATELY EXCLUDED. An earlier version of this script included it and the result
    was wrong twice over: prune.remove() raises for GRU (its parameters are renamed to
    <name>_orig once a mask is attached, so the second lookup no longer matches a pruned
    parameter), and the sparsity measurement then read the UNMASKED _orig tensors and reported
    0% zeros for those layers - understating true sparsity and making "requested" and "actual"
    disagree for the wrong reason. Separately, PyTorch's fused RNN kernels do not exploit
    unstructured sparsity, so pruning GRU weights costs accuracy without buying speed.
    """
    out = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            out.append((m, "weight"))
    return out


def apply_global_pruning(model, amount):
    if amount <= 0:
        return model
    prune.global_unstructured(prunable(model), pruning_method=prune.L1Unstructured,
                              amount=amount)
    for mod, name in prunable(model):
        prune.remove(mod, name)          # bake the mask in; must not fail silently
    return model


def measured_sparsity(model):
    tot = nz = 0
    for mod, name in prunable(model):
        w = getattr(mod, name)
        tot += w.numel()
        nz += int((w != 0).sum())
    return 1.0 - nz / max(tot, 1)


def sizes(model):
    """Dense bytes vs a sparse-aware estimate (CSR-ish: value + int32 index per nonzero)."""
    dense = sum(p.numel() * 4 for p in model.parameters())
    nz = 0
    pruned_total = 0
    for mod, name in prunable(model):
        w = getattr(mod, name)
        nz += int((w != 0).sum())
        pruned_total += w.numel()
    other = sum(p.numel() for p in model.parameters()) - pruned_total
    sparse = other * 4 + nz * 8
    return dense / 1e6, sparse / 1e6


def evaluate(model, dev="cpu"):
    model.eval().to(dev)
    snrs, stois = [], []
    for _, lo, hi, seed in BUCKETS:
        ds = HybridRealDataset(split="val", mode="hybrid", epoch_size=PER_BUCKET,
                               snr_range=(lo, hi), seed=seed, hybrid_ratio=0.85)
        for i in range(len(ds)):
            s = ds[i]
            with torch.no_grad():
                e = model(s["noisy"].unsqueeze(0).to(dev))[0][0].cpu().numpy()
            if not np.all(np.isfinite(e)):
                continue
            r = evaluate_batch_metrics(s["clean"].numpy(), s["noisy"].numpy(), e)
            snrs.append(r["enh_snr"]); stois.append(r["enh_stoi"])
    return float(np.mean(snrs)), float(np.std(snrs, ddof=1)), float(np.mean(stois)), \
        100.0 * float(np.mean(np.array(snrs) > 15.0))


def latency(model, frames=100):
    x = torch.randn(1, frames * 256)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(5):
            model(x)
        dt = (time.perf_counter() - t0) / 5
    return 1000.0 * dt / frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=float, default=None,
                    help="also write the checkpoint at this sparsity")
    a = ap.parse_args()

    base_sd = torch.load(CKPT, map_location="cpu")["model_state_dict"]

    print("=" * 100)
    print("PRUNING SWEEP  -  global unstructured L1 magnitude pruning (PS item D3)")
    print("=" * 100)
    m0 = build_causal_anc_model(hidden_dim=128, mask_mode="component_tanh", mask_bound=1.0)
    n_all = sum(p.numel() for p in m0.parameters())
    n_pr = sum(getattr(mod, nm).numel() for mod, nm in prunable(m0))
    print(f"  checkpoint : {CKPT}")
    print(f"  scope      : conv + linear weights = {n_pr:,} of {n_all:,} params "
          f"({100*n_pr/n_all:.1f}%); GRU excluded (see prunable())")
    print(f"  split      : VALIDATION, N={PER_BUCKET*len(BUCKETS)}   (test set untouched)")
    print("  NOTE: unstructured pruning zeroes weights but keeps DENSE storage. The dense size")
    print("        column is what a standard runtime loads; sparse-aware is the best case with")
    print("        a sparse kernel. No fine-tuning is applied after pruning.\n")
    print(f"  {'sparsity':>9} {'actual':>7} {'SNR dB':>8} {'sd':>6} {'STOI':>7} {'pass%':>7} "
          f"{'dense MB':>9} {'sparse MB':>10} {'ms/frame':>9}")
    print("  " + "-" * 88)

    rows = []
    for s in SPARSITIES:
        m = build_causal_anc_model(hidden_dim=128, mask_mode="component_tanh", mask_bound=1.0)
        m.load_state_dict(copy.deepcopy(base_sd))
        apply_global_pruning(m, s)
        act = measured_sparsity(m)
        snr, sd, stoi, p = evaluate(m)
        d_mb, s_mb = sizes(m)
        ms = latency(m)
        rows.append((s, act, snr, stoi, p, d_mb, s_mb, ms))
        print(f"  {s:9.0%} {act:7.1%} {snr:8.2f} {sd:6.2f} {stoi:7.4f} {p:6.1f}% "
              f"{d_mb:9.2f} {s_mb:10.2f} {ms:9.3f}", flush=True)

        if a.save is not None and abs(s - a.save) < 1e-9:
            out = f"checkpoints/pruned_{int(s*100)}pct"
            os.makedirs(out, exist_ok=True)
            torch.save({"model_state_dict": m.state_dict(), "hidden_dim": 128,
                        "mask_mode": "component_tanh", "mask_bound": 1.0,
                        "sparsity_requested": s, "sparsity_measured": act,
                        "val_snr": snr, "val_stoi": stoi,
                        "source_checkpoint": CKPT, "finetuned_after_prune": False},
                       os.path.join(out, "best_model.pt"))
            print(f"  -> wrote {out}/best_model.pt")

    base = rows[0]
    print("\n" + "=" * 100)
    print("  Accuracy cost relative to the unpruned model:")
    for s, act, snr, stoi, p, d, sm, ms in rows[1:]:
        print(f"    {s:4.0%} sparsity : {snr-base[2]:+6.2f} dB   STOI {stoi-base[3]:+7.4f}   "
              f"pass {p-base[4]:+5.1f} pp   sparse-size {100*(1-sm/base[6]):5.1f}% smaller")


    print(f"\n  CAUTION on reading this table: N={PER_BUCKET*len(BUCKETS)} and the per-utterance")
    print(f"  spread is sd~6 dB, so the standard error on each SNR cell is ~0.6 dB. ADJACENT")
    print(f"  sparsity levels are NOT separable, and the sweep is non-monotonic for that reason")
    print(f"  (BatchNorm running statistics are frozen while pruning shifts activation scales,")
    print(f"  which makes the degradation erratic rather than smooth). The reliable reading is")
    print(f"  the TREND across the full range, not the ranking of neighbouring rows.")
    print("=" * 100)


if __name__ == "__main__":
    main()
