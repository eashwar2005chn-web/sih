# Phase 16 — Section C.1: Is the tanh ceiling removable?

**Date**: 2026-08-29
**Outcome**: **Negative.** Hypothesis tested, not supported. No full retrain escalated; the 8-retrain budget remains **0 of 8 used**.

---

## 1. The hypothesis, and why it was worth testing

Phase 8's oracle diagnostic is the strongest single finding in this project, and it had been read as "the mask *bound* is the ceiling." That reading drove Phases 9–10, which swept the bound (K = 1.5 → 3.0) and found monotonic degradation — widening the bound lets the network amplify noise-dominated bins.

Re-reading that table showed something different. The oracle compared `tanh` against a **linear clamp at the same [-1, 1] bound**. Same bound, wildly different ceiling. So the variable was never the bound — it was the *curvature inside* it. `tanh(1) = 0.76`, so a bin that ought to pass at unity gain never passes more than 76% of it, and speech energy is removed systematically regardless of how well the network estimates.

Independently re-derived on the **validation split** (`scripts/oracle_mask_shape_probe.py`, ideal cRM, no trained model involved):

| Mask shaping (ideal mask) | [-10,-5) | [-5,0) | [0,5) | [5,10) | [10,15] | **Mean** |
|---|---|---|---|---|---|---|
| ideal, no shaping | 93.54 | 98.48 | 99.68 | 101.02 | 101.22 | **98.79** |
| **tanh — current model** | 13.23 | 12.93 | 12.86 | 12.85 | 13.06 | **12.99** |
| **linear clamp [-1,1]** | 22.53 | 25.93 | 31.13 | 31.09 | 31.94 | **28.52** |
| `x/(1+|x|)` | 6.68 | 6.40 | 6.27 | 6.21 | 6.26 | 6.36 |
| clamp [-2,2] | 32.05 | 35.72 | 41.47 | 43.22 | 42.89 | 39.07 |
| tanh K=2 | 22.08 | 22.18 | 22.50 | 22.69 | 23.01 | 22.49 |

The tanh row (12.99 dB) sits essentially exactly where every trained model in Phases 1–11 landed (12.75 dB). Swapping the shape at the same bound showed **+15.54 dB** of oracle headroom — comfortably past the 15.0 dB target. That is a strong, falsifiable prediction, and it had never been tested.

## 2. What was run

Three arms at **identical** pre-committed budget (8 epochs × 900 samples, lr 1e-3, batch 16, seed 42), logged in `eval/experiments_planned.csv` **before** execution. Checkpoint selection and all decisions used the **validation split**; the N=500 test set was never consulted.

| Arm | Mask shaping | Best val SNR | Best val STOI | Δ vs control |
|---|---|---|---|---|
| **C1-S0 (control)** | `tanh` | **+6.31 dB** | 0.8317 | — |
| C1-S1 | linear clamp (`hardtanh`) | +4.83 dB | 0.8080 | **−1.49 dB** |
| C1-S2 | leaky clamp (slope 0.05) | +5.42 dB | 0.8253 | **−0.90 dB** |

Pre-committed promote rule: beat the control by **> +1.0 dB** val Output SNR at equal budget. **Neither arm qualified**, so per Section E no variant was escalated to a full retrain.

## 3. Why the oracle headroom did not transfer

The oracle is handed the *ideal* mask. A trained network has to **estimate** it, and the two shapes behave very differently under estimation error:

- **tanh** compresses smoothly. An over-confident estimate near the bound is pulled back gently, and the gradient never vanishes, so the network keeps learning everywhere.
- **A hard clamp** is linear inside the bound — so estimation error in the interior passes through at full scale instead of being damped — and has **exactly zero gradient outside** it, so any unit that saturates stops learning entirely.

That the leaky variant beat the hard clamp by +0.59 dB is consistent with the dead-unit half of that explanation. That both still lost to tanh points at the other half: **the compression is load-bearing regularisation, not a removable artifact.** The headroom is real for an oracle and unreachable for an estimator.

This is the same lesson the K-sweep taught, in a different coordinate: every attempt to give the mask more expressive range costs more than it returns, because the extra range is spent on estimation error rather than on signal.

## 4. Status against the compliance targets

Unchanged. **Not met.**

| Target | Required | Current | Status |
|---|---|---|---|
| Aggregate Output SNR (0–15 dB) | > 15.0 dB | 12.75 dB | **Not met** |
| Per-utterance pass rate | > 80% | 28.0% | **Not met** |

Phase 16 adds a **sixth** independently-tested and rejected hypothesis to the five from Phases 4–11 (data distribution, capacity, mask bound, normalisation, temporal receptive field). It is the first one that also explains *why* the ceiling resists removal.

## 5. Side finding: training-time mixer bug fixed

While instrumenting, the Phase 14 degenerate-noise warning fired repeatedly **during training** — confirming that batches contained samples where `noisy == clean` (a silent noise window defeats the SNR scaler), which actively teaches pass-through behaviour.

Fixed in `HybridRealDataset._sample_segment()` via bounded retry for silent noise windows, enabled on the **train split only**. Verified:

- **Test split remains bit-exact** vs the committed eval CSVs (max |ΔSNR| = `0.00e+00`), so the entire audit trail is preserved.
- **Train split**: 0 degenerate samples in 300 draws (previously ~1.4%).

All three screening arms ran with this fix, so the comparison above is unaffected by it.

## Reproducing

```bash
python scripts/oracle_mask_shape_probe.py
```

```bash
python scripts/run_phase16_mask_shape_screen.py
```

Artifacts: `eval/phase16_screen_summary.json`, `eval/phase16_screen.log`, `eval/experiments_planned.csv`.
