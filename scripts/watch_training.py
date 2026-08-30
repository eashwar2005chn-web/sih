"""
Live progress viewer for the Phase 17 long-training run.

    python scripts/watch_training.py          # snapshot, then exit
    python scripts/watch_training.py --follow # refresh every 30s until finished
"""
import argparse
import json
import os
import sys
import time

LOG = "eval/phase18_gpu_long.log"
HIST = "checkpoints/phase18_gpu_long/train_history.json"
TOTAL = 75
BASELINE_VAL_SNR = 10.39   # published baseline, measured on the same val split
TARGET = 15.0


def render():
    if not os.path.exists(HIST):
        started = os.path.exists(LOG)
        print("no completed epochs yet " + ("(training is starting up)" if started else "(log not found)"))
        return False
    hist = json.load(open(HIST))
    if not hist:
        print("no completed epochs yet (first epoch takes ~5 min)")
        return False

    done = len(hist)
    best = max(hist, key=lambda h: h["op_score"])
    last = hist[-1]
    per_ep = 80.0
    eta_min = (TOTAL - done) * per_ep / 60.0

    print("=" * 74)
    print(f"  PHASE 17 LONG TRAIN     epoch {done}/{TOTAL}     ETA ~{eta_min:.0f} min")
    print("=" * 74)
    print(f"  {'ep':>3} {'train':>9} {'val SNR':>9} {'val STOI':>9} {'score':>8} {'skip':>5}")
    for h in hist[-12:]:
        star = " *" if h["epoch"] == best["epoch"] else "  "
        print(f"  {h['epoch']:>3} {h['train_loss']:>9.4f} {h['val_snr']:>+9.2f} "
              f"{h['val_stoi']:>9.4f} {h['op_score']:>8.2f} {h.get('skipped',0):>5}{star}")

    cur = last["val_snr"]
    print("-" * 74)
    print(f"  best so far : {best['val_snr']:+.2f} dB val SNR / STOI {best['val_stoi']:.4f} (epoch {best['epoch']})")
    print(f"  vs published baseline ({BASELINE_VAL_SNR:.2f} dB val): {best['val_snr']-BASELINE_VAL_SNR:+.2f} dB")
    print(f"  vs {TARGET:.1f} dB target                  : {best['val_snr']-TARGET:+.2f} dB")

    span = 34
    filled = int(max(0.0, min(1.0, cur / 20.0)) * span)
    tick = int(TARGET / 20.0 * span)
    bar = "".join("#" if i < filled else ("|" if i == tick else ".") for i in range(span))
    print(f"  [{bar}]  0 ....... target={TARGET:.0f} ....... 20 dB")
    print("  NOTE: val split only. Test set is untouched until training completes.")
    return done >= TOTAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    while True:
        os.system("cls" if os.name == "nt" else "clear") if a.follow else None
        finished = render()
        if not a.follow or finished:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
