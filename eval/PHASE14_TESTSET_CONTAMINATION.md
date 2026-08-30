# Phase 14 — Test-Set Contamination: Root Cause, Impact, and Remediation

**Date**: 2026-08-27
**Trigger**: Phase 13's claim audit noted that ~20 of the 500 held-out test utterances had measured input SNR far outside their labeled bucket (one at +107 dB). Unlike the rest of Phase 13, this was a suspected *code* defect, not a documentation defect.

**Reproducibility note**: the test set was regenerated from scratch by `scripts/diagnose_mixer_degenerate.py` using the same buckets, seeds, and `epoch_size` as `scripts/evaluate_phase10_final.py`. The regenerated measured input SNRs match the committed eval CSVs **bit-exactly** (max absolute difference across all 500 rows: `0.0`), confirming the test set is deterministic and that this diagnosis applies to the exact data every phase was evaluated on.

---

## Step 1 — Root cause: two distinct mechanisms, not one

The initial hypothesis ("the mixer is sometimes wrong") turned out to conflate two unrelated effects with very different implications. Separating them was necessary to size the problem correctly.

### Mechanism A — Silent noise segments defeat the SNR scaler (**a genuine bug**, 7 utterances)

**What happens.** Several ESC-50-derived noise clips in `datasets/generic_noise/` and `datasets/tactical_noise/` consist mostly of *exact digital silence* with a short sound event embedded. Measured directly:

| File | Total samples | Exact-zero samples | Longest zero run |
|---|---|---|---|
| `musan_crowd_1-19118-A-24.wav` | 80,000 | 70,607 | **70,574 (4.41 s of a 5.0 s clip)** |
| `musan_crowd_5-220027-A-21.wav` | 80,000 | 50,765 | 47,555 (2.97 s) |
| `audioset_siren_3-62878-B-42.wav` | 80,000 | 41,553 | 41,553 (2.60 s) |

`HybridRealDataset._sample_segment()` ([data/real_dataset_loader.py:226](../data/real_dataset_loader.py:226)) picks a **uniformly random** 2-second (32,000-sample) window. When the window lands entirely inside the silent run — highly likely for these files — the returned "noise" array is **all zeros**.

**Why it fails silently.** In `mix_speech_and_noise()` ([data/defence_noise_generator.py:370](../data/defence_noise_generator.py:370)):

```python
noise_power  = np.mean(noise ** 2) + 1e-12          # = 1e-12 for all-zero noise
target_noise_power = speech_power / (10.0 ** (target_snr_db / 10.0))
scale_factor = np.sqrt(target_noise_power / noise_power)   # ~1e5, enormous
scaled_noise = noise * scale_factor                  # 0 × 1e5 = 0  <-- silently zero
noisy = speech + scaled_noise                        # noisy == speech exactly
```

**The `+ 1e-12` epsilon is precisely the culprit.** It was added to prevent a division-by-zero crash, and it succeeds at that — but it converts a condition that *should* be a loud failure (noise clip contains no signal) into a silent no-op that returns a "noisy" mixture identical to the clean reference. `compute_snr` then reports `10·log10(speech_power / 1e-12) ≈ +100 dB`.

**This is a bounded, detectable edge case, not a subtle bias.** The failure is binary: either the noise window is exactly zero (→ ~+100 dB, unmistakable) or it contains real audio and the mix is exact. There is no intermediate regime quietly skewing other utterances — verified below.

- All-zero noise segments: **7 / 500**
- Near-silent but nonzero (`rms < 1e-4`): **0 / 500**

### Mechanism B — Post-mix band-limiting shifts measured SNR (**not a bug**, 86 utterances)

For every utterance with healthy (non-silent) noise, the mixing stage hits its target **exactly**:

```
healthy-noise rows: |snr_at_mix − target_snr|  mean = 0.000 dB,  max = 0.000 dB
healthy-noise rows: |final_snr  − target_snr|  mean = 3.259 dB,  max = 31.564 dB
```

The mixer is correct. The deviation is introduced **after** mixing, by two deliberate, documented stages in `__getitem__`:

1. **Shared RIR reverberation** (applied to clean and noisy with the same filter)
2. **Tactical VHF transceiver bandpass, 300–3400 Hz** (a stated Task-1 invariant in [DATASET_CARD.md](../data/DATASET_CARD.md))

Band-limiting to 300–3400 Hz removes noise energy concentrated *outside* that band — helicopter rotor fundamentals (~24 Hz), armored-vehicle engine rumble, gunshot transient energy — while speech energy sits largely *inside* it. Effective SNR therefore rises, sometimes dramatically. The affected utterances are exactly the ones you'd predict: `synth_helicopter`, `audioset_helo_*`, `mad_engine_*`, `gunshot_crack_*`.

**This is a definitional mismatch, not corruption.** The bucket label is a *nominal, pre-channel, wideband* target SNR. `raw_snr` is a *post-channel* measurement. Both are valid; they are simply different quantities. The noise was mixed correctly and is genuinely present — it has just been band-limited by a channel model we intend to simulate.

**Implication**: these 86 utterances must **not** be discarded. Excluding them would mean throwing away legitimate data because a label convention doesn't match a measurement convention, which would itself bias the evaluation.

---

## Step 2 — Impact on reported numbers

Threshold used: an utterance is flagged **out-of-band** if its measured input SNR falls more than **3.0 dB** outside its labeled bucket's `[low, high)` boundary. (3 dB chosen as one power-doubling — comfortably wider than ordinary within-bucket variation, narrow enough to catch genuine outliers.)

- **CLEAN-A** — exclude only the 7 silent-noise utterances (Mechanism A, the real bug).
- **CLEAN-B** — exclude all 93 out-of-band utterances (Mechanisms A + B). Reported for completeness; **not recommended**, for the reason given above.

### 2.1 Baseline 811K — aggregate

| Filter | N | Output SNR | Pass >15 dB | 95% CI | STOI | PESQ | **Mean ΔSNR** |
|---|---|---|---|---|---|---|---|
| **ORIGINAL** (published) | 300 | 12.75 ± 5.35 dB | 28.0% | [23.2%, 33.3%] | 0.8990 | 3.69 | **+2.77 dB** |
| **CLEAN-A** (recommended) | 295 | **12.46 ± 4.87 dB** | **26.8%** | [22.0%, 32.1%] | **0.8973** | **3.68** | **+4.00 dB** |
| CLEAN-B | 249 | 11.89 ± 4.19 dB | 23.7% | — | 0.8940 | 3.64 | +4.27 dB |

Stress range (-10–0 dB):

| Filter | N | Output SNR | STOI | PESQ | **Mean ΔSNR** |
|---|---|---|---|---|---|
| **ORIGINAL** (published) | 200 | 4.59 ± 5.97 dB | 0.7310 | 2.64 | **+6.84 dB** |
| **CLEAN-A** (recommended) | 198 | **4.37 ± 5.59 dB** | **0.7284** | **2.62** | **+7.65 dB** |
| CLEAN-B | 158 | 3.31 ± 4.74 dB | 0.7154 | 2.51 | +8.22 dB |

### 2.2 Causal TCN 899K — aggregate

| Filter | N | Op Output SNR | Pass >15 dB | 95% CI | Op STOI | Stress Output SNR | **Stress ΔSNR** |
|---|---|---|---|---|---|---|---|
| **ORIGINAL** (published) | 300 / 200 | 12.61 ± 5.30 dB | 27.7% | [22.9%, 33.0%] | 0.8946 | 5.13 ± 5.42 dB | **+7.39 dB** |
| **CLEAN-A** (recommended) | 295 / 198 | **12.33 ± 4.86 dB** | **26.4%** | [21.7%, 31.8%] | **0.8928** | **4.98 ± 5.20 dB** | **+8.25 dB** |
| CLEAN-B | 249 / 158 | 11.78 ± 4.17 dB | 23.7% | — | 0.8897 | 4.01 ± 4.42 dB | +8.92 dB |

### 2.3 Per-bucket Output SNR (both models)

| Bucket | Baseline ORIGINAL | Baseline CLEAN-A | TCN ORIGINAL | TCN CLEAN-A |
|---|---|---|---|---|
| Stress [-10, -5) dB | 2.80 dB | **2.56 dB** (n=99) | 3.66 dB | **3.44 dB** (n=99) |
| Stress [-5, 0) dB | 6.38 dB | **6.18 dB** (n=99) | 6.60 dB | **6.51 dB** (n=99) |
| Operational [0, 5) dB | 9.21 dB | **8.78 dB** (n=98) | 9.18 dB | **8.75 dB** (n=98) |
| Operational [5, 10) dB | 13.43 dB | **12.93 dB** (n=97) | 13.18 dB | **12.70 dB** (n=97) |
| Operational [10, 15] dB | 15.60 dB | **15.60 dB** (n=100) | 15.47 dB | **15.47 dB** (n=100) |

Per-bucket STOI shifts by at most 0.003 under CLEAN-A and is not tabulated here.

### 2.4 Reading the impact

**Output SNR / pass rate / STOI / PESQ: low impact.** Under CLEAN-A every aggregate moves by ≤ 0.29 dB (SNR), ≤ 1.2 pp (pass rate), ≤ 0.002 (STOI), ≤ 0.01 (PESQ). Wilson confidence intervals for the pass rate overlap almost entirely (baseline: [23.2%, 33.3%] → [22.0%, 32.1%]). No conclusion drawn in any phase changes. The hardest bucket moves 2.80 → 2.56 dB — a real but immaterial shift, and notably it moves *down* only slightly, so the "hardest bucket inflated by near-clean utterances" concern does **not** materialise at the bucket level.

**ΔSNR / snr_gain: materially affected — and in the opposite direction from expectation.** The 7 silent utterances each carry a measured `snr_gain` of roughly **−70 dB** (raw_snr ≈ +100 dB against enh_snr ≈ +28 dB), because the model necessarily *reduces* the SNR of a signal that was already clean. Seven such outliers drag the operational mean down substantially:

| | ΔSNR operational | ΔSNR stress |
|---|---|---|
| ORIGINAL (contaminated) | +2.77 dB | +6.84 dB |
| **CLEAN-A (correct)** | **+4.00 dB** | **+7.65 dB** |

The previously reported ΔSNR figures were **understated**, not overstated. This matters because Phase 13 had just corrected ΔSNR from an invalid midpoint-derivation to the measured `snr_gain` column — that correction was directionally right, but it landed on a value contaminated by these seven outliers.

---

## Step 3 — Remediation decision

**Decision: split by metric, per the criteria set out for this phase.**

**(a) Output SNR, pass rate, STOI, PESQ, per-bucket breakdowns → Path 1 (document, do not re-run).**
CLEAN-A shifts are within the existing confidence intervals and change no conclusion in any phase. Regenerating the test set and re-evaluating eleven-plus phases would cost substantial time for changes of ≤ 0.3 dB. The originally published figures stand, now with this contamination disclosed. Presentation-facing documents continue to use the ORIGINAL numbers for these metrics, so they remain consistent with the full evaluation history.

**(b) ΔSNR / SNR-gain figures → Path 2 (use cleaned numbers going forward).**
Here the contamination is material (+2.77 → +4.00 dB operational; +6.84 → +7.65 dB stress) because the metric is a difference against a corrupted reference, so seven degenerate rows dominate the mean. **Submission-facing documents must use the CLEAN-A values, labeled as excluding 7 degenerate utterances.**

**Not in scope now**: regenerating the test set against a fixed mixer and re-deriving the project's evaluation history. Recorded as future work below.

**Root-cause fix for future work** (not applied to the training pipeline in this phase, per scope):
1. In `_sample_segment()`, reject and resample windows whose RMS is below a threshold (e.g. `1e-4`), with a bounded retry count.
2. In `mix_speech_and_noise()`, raise or return an explicit failure flag when `np.mean(noise**2)` is below a floor, instead of relying on the `1e-12` epsilon to mask it.
3. Consider recording *post-channel* measured SNR alongside the nominal target in each dataset item, so Mechanism B is visible in the data rather than needing to be re-derived.

---

## Documents requiring updates as a result

Only ΔSNR figures change. Exactly four values, in four files:

| File | Current (contaminated) | **Corrected (CLEAN-A)** |
|---|---|---|
| [README.md](../README.md) | "+2.77 dB across the operational range, +6.84 dB across the -10 to 0 dB stress range" | **"+4.00 dB operational, +7.65 dB stress (N=295/198, excluding 7 degenerate utterances)"** |
| [docs/JUDGE_PITCH_ONEPAGER.md](../docs/JUDGE_PITCH_ONEPAGER.md) | "+2.77 dB operational, +6.84 dB stress" | **"+4.00 dB operational, +7.65 dB stress"** |
| [docs/DEMO_VIDEO_SCRIPT.md](../docs/DEMO_VIDEO_SCRIPT.md) | "+6.84 dB measured improvement"; "nearly seven decibels" | **"+7.65 dB measured improvement"; "over seven and a half decibels"** |
| [docs/COMPETITION_NARRATIVE.md](../docs/COMPETITION_NARRATIVE.md) §4 | per-bucket ΔSNR table + "+2.77 dB operational, +6.84 dB stress" | **corrected table below** |

Corrected per-bucket ΔSNR (CLEAN-A, baseline 811K):

| Bucket | n | Mean measured input SNR | Output SNR | **Mean ΔSNR** |
|---|---|---|---|---|
| [-10, -5) dB | 99 | -5.65 dB | 2.56 dB | **+8.20 dB** |
| [-5, 0) dB | 99 | -0.91 dB | 6.18 dB | **+7.09 dB** |
| [0, 5) dB | 98 | +3.59 dB | 8.78 dB | **+5.18 dB** |
| [5, 10) dB | 97 | +8.44 dB | 12.93 dB | **+4.48 dB** |
| [10, 15] dB | 100 | +13.24 dB | 15.60 dB | **+2.37 dB** |

**No other number in any submission-facing document changes.** Output SNR (12.75 dB), pass rate (28.0%), STOI (0.899), PESQ proxy (3.69), all per-bucket Output SNR values, latency, RTF, causality, parameter counts, and model sizes are unaffected and remain as published.

## Reproducing this analysis

```bash
python scripts/diagnose_mixer_degenerate.py
```

```bash
python scripts/quantify_degenerate_impact.py
```

Artifacts: `eval/mixer_degenerate_diagnosis.csv` (per-utterance provenance for all 500 test items, including source noise file, noise RMS, target vs. at-mix vs. final SNR).
