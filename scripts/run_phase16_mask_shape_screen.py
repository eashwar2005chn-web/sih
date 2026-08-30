"""
Phase 16 / Section C.1 — Screening: does a LINEAR-in-interior mask bound break the ceiling?

Three arms, identical budget (8 epochs x 900 steps, lr 1e-3, batch 16, seed 42):
  C1-S0  component_tanh        -- CONTROL, the current parameterisation
  C1-S1  component_clamp       -- hardtanh, linear inside the same unit bound
  C1-S2  component_leaky_clamp -- as S1 but with a 0.05 slope outside the bound

Budget was fixed in eval/experiments_planned.csv BEFORE this ran and is identical across
arms. Checkpoint selection and all promote decisions use the VALIDATION split only; the
N=500 test set is not touched here.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from scripts.train_phase10_full_runs import train_full_model

EPOCHS = 8
STEPS = 900
LR = 1e-3
SEED = 42

ARMS = [
    ("C1-S0_tanh_control", "checkpoints/phase16_screen_tanh_control", "component_tanh"),
    ("C1-S1_linear_clamp", "checkpoints/phase16_screen_linear_clamp", "component_clamp"),
    ("C1-S2_leaky_clamp", "checkpoints/phase16_screen_leaky_clamp", "component_leaky_clamp"),
]


def main():
    device = "cpu"  # CUDA is unusable on this machine (sm_120 GPU, sm_90-max torch build)
    torch.set_num_threads(os.cpu_count() or 8)
    print(f"[Phase16] device={device} threads={torch.get_num_threads()}")
    print(f"[Phase16] budget per arm: {EPOCHS} epochs x {STEPS} steps, lr={LR}, seed={SEED}")

    summary = {}
    for run_name, save_dir, mask_mode in ARMS:
        t0 = time.time()
        print(f"\n{'='*88}\n[Phase16] ARM {run_name}  mask_mode={mask_mode}\n{'='*88}")
        train_full_model(
            run_name=run_name,
            save_dir=save_dir,
            norm_type="batch",
            mask_mode=mask_mode,
            mask_bound=1.0,
            epochs=EPOCHS,
            lr=LR,
            batch_size=16,
            train_steps=STEPS,
            val_steps=60,
            seed=SEED,
            device=device,
        )
        hist_path = os.path.join(save_dir, "train_history.json")
        best = None
        if os.path.exists(hist_path):
            hist = json.load(open(hist_path))
            best = max(hist, key=lambda h: h["op_score"])
        summary[run_name] = {
            "mask_mode": mask_mode,
            "save_dir": save_dir,
            "minutes": round((time.time() - t0) / 60, 1),
            "best_epoch": best["epoch"] if best else None,
            "best_val_snr": round(best["val_snr"], 4) if best else None,
            "best_val_stoi": round(best["val_stoi"], 4) if best else None,
            "best_op_score": round(best["op_score"], 4) if best else None,
        }
        with open("eval/phase16_screen_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Phase16] {run_name} done in {summary[run_name]['minutes']} min -> {summary[run_name]}")

    print("\n" + "=" * 88)
    print("PHASE 16 SCREENING SUMMARY (validation split)")
    print("=" * 88)
    print(f"{'arm':26} {'mask_mode':24} {'val SNR':>9} {'val STOI':>9} {'score':>8}")
    for k, v in summary.items():
        print(f"{k:26} {v['mask_mode']:24} {v['best_val_snr']:>9.2f} {v['best_val_stoi']:>9.4f} {v['best_op_score']:>8.2f}")

    ctrl = summary.get("C1-S0_tanh_control", {}).get("best_val_snr")
    print("\nPromote rule: beat control by > 1.0 dB val Output SNR at identical budget.")
    if ctrl is not None:
        for k, v in summary.items():
            if k.startswith("C1-S0"):
                continue
            d = v["best_val_snr"] - ctrl
            verdict = "PROMOTE" if d > 1.0 else "do not promote"
            print(f"  {k:26} delta vs control = {d:+6.2f} dB  -> {verdict}")


if __name__ == "__main__":
    main()
