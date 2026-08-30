# Phase 13 — Claim Provenance & Traceability Audit

**Date**: 2026-08-27
**Trigger**: Phase 12 removed a fabricated headline metric ("18.4 dB EXCEEDED") and an unsubstantiated "calibrated to ITU-T P.862" claim from submission-facing documents. Finding one fabrication invalidates the assumption that the rest of those documents are sound. This audit re-derives **every** numeric claim in every judge-facing document from a primary artifact, and investigates where the fabricated content came from.

**Bottom line**: the core evaluation results are sound and fully reproducible — all headline SNR/STOI/PESQ figures reproduce **exactly** from the raw per-utterance CSVs. However, this audit found **three further unsupported claims** that Phase 12's targeted grep did not catch, one of which (the NPU latency figure) is the same class of fabrication as the 18.4 dB number and had been *reinforced* by the Phase 12 pass rather than removed. All three are now corrected.

---

## Step 1 — Provenance of the fabricated content

### 1.1 Commit history

The repository has only four commits:

| Commit | Date | Message | Scale |
|---|---|---|---|
| `018c63c` | 2026-08-26 14:27 | `main_v2` | 1 file, **1 insertion** (`README.md` = "# SIH Project") |
| `102780c` | 2026-08-27 10:04 | Phase 8-10 diagnostics… | **138 files, 25,812 insertions** |
| `5367e7a` | 2026-08-27 10:31 | Phase 11… | — |
| `1bf3d70` | 2026-08-27 11:07 | Phase 12 cleanup… | — |

### 1.2 Finding: this was not documentation drift

The intuitive hypothesis — that honest early results were later superseded by better evaluations while the marketing docs went stale — is **false**. Verified:

- `README.md` at `018c63c` contained exactly one line: `# SIH Project`. There was no earlier, honest version of the results table that later went stale.
- `git log --diff-filter=A` confirms `docs/JUDGE_PITCH_ONEPAGER.md`, `docs/DEMO_VIDEO_SCRIPT.md`, `benchmark/HARDWARE_BENCHMARK_REPORT.md`, **and** `eval/DRDO_TARGETS_EVALUATION.md` were **all first added in the same commit**, `102780c`.
- `git log -S"18.4 dB" --all` returns only `102780c`.

**Therefore**: the fabricated "18.4 dB / 2.85 PESQ / 0.882 STOI — EXCEEDED ✓" table and the honest "12.75 dB, fails aggregate threshold" report entered the repository *simultaneously, in a single bulk commit*. The documents never diverged over time; they were **authored inconsistent with each other from the start**, and the whole project was committed in one 138-file dump with no incremental history to review against.

### 1.3 Does 18.4 dB trace to any run at all?

No. Exhaustive search across every CSV, JSON, HTML and log artifact in the repo:
- `checkpoints/phase10_polar_k15_scratch/train_history.json` contains `"op_score": 18.4317…` — an internal *training composite score*, not an SNR in dB, from a **different model** (Polar K=1.5) that the report itself records as *underperforming* the baseline. This is a numeric coincidence, not a source.
- The only generated evaluation HTML present, `eval/DRDO_TARGETS_EVALUATION.html`, reports **6.96 dB** Output SNR — an actual (much lower) run output.
- No aggregate artifact anywhere produces 18.4 dB.

**Conclusion**: 18.4 dB was not produced by any evaluation run in this repository. Same for `+16.8 dB improvement`, `2.85` PESQ, `0.882` STOI, `0.620` / `1.45` raw-degraded values — none trace to any artifact.

### 1.4 Pattern assessment

This is **not** an isolated slip. The same generative pattern — *a plausible-looking number written into a document with no run behind it* — recurs in three additional places found in Step 2 and Step 3 below. The common mechanism is that submission-facing prose was written **independently of** the evaluation pipeline rather than generated from it, in one bulk authoring session. Notably, `benchmark/benchmark_suite.py` contained a code path that *institutionalised* this: a hardcoded "theoretical projection" emitted into a report file that reads identically to a measurement (see Step 3).

---

## Step 2 — Claim-by-claim traceability table

Documents audited: `README.md`, `docs/JUDGE_PITCH_ONEPAGER.md`, `docs/DEMO_VIDEO_SCRIPT.md`, `docs/COMPETITION_NARRATIVE.md`, `benchmark/HARDWARE_BENCHMARK_REPORT.md`, `data/DATASET_CARD.md`. (`eval/*.md` and `eval/*.py` are internal engineering records but are included where a judge-facing number depends on them.)

### 2.1 Core evaluation metrics — ALL VERIFIED EXACT

Recomputed from `eval/test_eval_csvs/Baseline_811K_BatchNorm_K1.0_test_eval.csv` (N=500 rows), grouping by the CSV's own `bucket` column:

| Claim | Value in docs | Recomputed from CSV | Match |
|---|---|---|---|
| Operational Output SNR (0–15 dB, N=300) | 12.75 ± 5.35 dB | **12.75 ± 5.35** | ✅ exact |
| Operational pass rate >15 dB | 28.0% | **28.0%** | ✅ exact |
| Operational STOI | 0.8990 | **0.8990** | ✅ exact |
| Operational PESQ (proxy) | 3.69 | **3.69** | ✅ exact |
| Stress Output SNR (-10–0 dB, N=200) | 4.59 dB | **4.59** | ✅ exact |
| Stress SNR gain | +6.84 dB | **+6.84** | ✅ exact |
| Bucket [-10,-5) SNR | 2.80 dB | **2.80** | ✅ exact |
| Bucket [-5,0) SNR | 6.38 dB | **6.38** | ✅ exact |
| Bucket [0,5) SNR | 9.21 dB | **9.21** | ✅ exact |
| Bucket [5,10) SNR | 13.43 dB | **13.43** | ✅ exact |
| Bucket [10,15] SNR | 15.60 dB | **15.60** | ✅ exact |
| Bucket STOI (all five) | 0.6890 / 0.7731 / 0.8465 / 0.9151 / 0.9354 | **all exact** | ✅ exact |

**The evaluation trail is genuinely sound.** Every headline model-performance number in the reports is reproducible to the reported precision from raw per-utterance data.

> ⚠️ **Methodological note discovered during this recomputation.** Buckets are assigned by *target* mixing SNR, not measured input SNR. Grouping instead by measured `raw_snr` produces materially different figures (operational SNR 11.57 dB, pass rate 17.4%). Both groupings are defensible, but the reports must state which is used. Additionally, **20 of 500 utterances have measured input SNR above +20 dB (max +107 dB)** despite sitting in low-SNR buckets — the mixing procedure did not hit its target SNR for those samples. This does not affect output-SNR figures but does affect ΔSNR (see 2.2). Flagged for investigation in `data/real_dataset_loader.py`.

### 2.2 ΔSNR / "improvement over noisy input" — ❌ FAILED, CORRECTED

| Document | Claim | Traces to | Match | Action |
|---|---|---|---|---|
| README, JUDGE_PITCH, NARRATIVE §4 | "avg **+7.0 dB** improvement, rising to **+10.3 dB** in stress band"; per-bucket +10.3/+8.9/+6.7/+5.9/+3.1 | **Nothing.** Derived by subtracting *nominal bucket midpoints* (-7.5, -2.5, …) from output SNR | ❌ **NO** | **Corrected** |

This was a **derivation error, not a transcription error**, and it originated in the Phase 12 pass — i.e. it was introduced by the previous cleanup, not inherited. Bucket midpoints are not the measured input SNR. The CSV already carries a per-utterance `snr_gain` column; the correct values are:

| Bucket | Mean *measured* input SNR | Output SNR | **Measured mean ΔSNR** |
|---|---|---|---|
| [-10,-5) | -4.58 dB | 2.80 dB | **+7.38 dB** |
| [-5,0) | +0.07 dB | 6.38 dB | **+6.31 dB** |
| [0,5) | +5.58 dB | 9.21 dB | **+3.63 dB** |
| [5,10) | +11.13 dB | 13.43 dB | **+2.30 dB** |
| [10,15] | +13.24 dB | 15.60 dB | **+2.37 dB** |

**Operational mean ΔSNR: +2.77 dB** (not +7.0 dB — the claim overstated by ~2.5×).
**Stress mean ΔSNR: +6.84 dB** — this figure independently corroborates the `Stress SNR Gain` row already in the eval report, so it is the trustworthy one to present.

All three documents corrected to the measured values.

### 2.3 Latency / RTF / hardware claims — re-measured 2026-08-27

Re-ran `benchmark/benchmark_suite.py` functions directly against `checkpoints/task4_full_run/best_model.pt`:

| Claim | Doc value | Re-measured today | Match | Action |
|---|---|---|---|---|
| CPU PyTorch FP32 latency | 0.07 ms (RTF 0.0045) | **0.0865 ms (RTF 0.0054)** | ⚠️ same range, run-to-run variance | Restated as 0.087 ms with measurement date |
| RTX 5060 CUDA FP16 | 0.02 ms (RTF 0.0010) | **0.0173 ms (RTF 0.0011)** | ✅ reproduces | Kept |
| OpenVINO CPU INT8 | 0.09 ms (RTF 0.0058) | **0.0930 ms (RTF 0.0058)** | ✅ **exact** on RTF | Kept, now headline edge claim |
| **Intel AI Boost NPU** | **0.18 ms, P95 0.24, RTF 0.0110** | **CANNOT BE MEASURED** | ❌ **FABRICATED** | **Removed everywhere** — see Step 3 |
| CPU latency in causality profile | 0.065–0.103 ms across reports | **0.091 ms** (re-run) | ✅ within range | Kept as "0.07–0.10 ms" |
| Power figures (28-45W, 60-90W, <2.5W) | — | **Never measured** — hardcoded string literals in `benchmark_suite.py`; no wattmeter instrumentation exists | ⚠️ estimates only | Column is labelled "(Est)"; `<2.5W` NPU claim removed with the NPU row |

### 2.4 Model size / parameter claims — VERIFIED

| Claim | Doc value | Verified | Match |
|---|---|---|---|
| Baseline parameter count | 811K | **811,522** (live count) | ✅ |
| TCN variant parameter count | 899K (+87,680) | **899,202 (+87,680)** | ✅ |
| Scaled model | 3.24M | 3,235,842 (per eval report) | ✅ |
| FP32 model size | 3.1 MB | **3.11 MB** on disk; 811,522 × 4 B = 3.096 MB | ✅ |
| INT8 OpenVINO IR size | 1.46 MB | **1.46 MB** (xml 108,699 B + bin 1,421,040 B) | ✅ exact |

### 2.5 Causality claim — VERIFIED BY RE-RUN

| Claim | Source | Result |
|---|---|---|
| 0.0 ms lookahead, zero future leakage | `scripts/profile_tcn_latency_and_causality.py` | **Re-ran today**: `Baseline Causality Check: PASSED (Max future leakage diff: 0.00e+00)`, `TCN Stack: PASSED (0.00e+00)` ✅ |

### 2.6 Phase 11B post-filter claims — VERIFIED

Against `eval/validation_fixes_1234_results.json`:

| Claim | Doc value | JSON artifact | Match |
|---|---|---|---|
| STFT/iSTFT round-trip MSE | 7.68e-16 | `7.682727781395801e-16` | ✅ |
| Baseline SNR in sweep | 12.49 dB | `12.491018295288086` | ✅ |
| Best post-filter delta | -0.92 dB | `-0.9156990051269531` | ✅ |
| Sweep size | 9 configurations | `sweep_results` = list len 9 | ✅ |
| Surrogate PESQ (baseline, N=150) | 3.676 ± 0.508 | `3.6757267…` / `0.5084232…` | ✅ |
| Hop length | 256 samples = 16 ms | `actual_hop_samples: 256`, `actual_hop_ms: 16.0` | ✅ |
| Helicopter f0 / period | 24.1 Hz / 41.56 ms | `24.06015…` / `41.5625` | ✅ |

### 2.7 Metric-backend claims — VERIFIED

| Claim | Verified |
|---|---|
| STOI via reference `pystoi` | ✅ `pystoi` installed and used at `model/loss.py:256` |
| Real ITU-T PESQ unavailable | ✅ `has_reference: False`, `n_reference: 0` in results JSON; MSVC absent |
| PESQ figures are proxy only | ✅ corrected wording now in code + all docs (Phase 12) |

---

## Step 3 — NPU / INT8 verification: **FABRICATED FIGURE, REMOVED**

This is the most serious finding of the audit.

### 3.1 What the code actually does

`benchmark/benchmark_suite.py` called `benchmark_openvino_device(..., "NPU")`, which returns `None` on any exception. The `else` branch then appended a **hardcoded literal**:

```python
else:
    # Documented theoretical NPU projection based on OpenVINO NPU spec
    results.append({
        "Execution Runtime": "OpenVINO NPU Plugin (INT8 PTQ)",
        "Latency / Frame (ms)": "0.18 ms",
        "P95 Latency": "0.24 ms",
        "RTF (Real-Time Factor)": "0.0110",
        "Est. Power": "< 2.5W (Ultra-Low Power)",
        ...
    })
```

These literals were then written into `HARDWARE_BENCHMARK_REPORT.md` in a table whose column headers read "Frame Latency (ms)" and "Real-Time Factor (RTF)" — **visually indistinguishable from the genuinely measured rows above it**.

### 3.2 Proof the fallback branch is what ran

The committed report contained the runtime string **`OpenVINO NPU Plugin (INT8 PTQ)`**. That exact string exists *only* in the hardcoded fallback branch. The real-measurement branch writes `OpenVINO NPU Plugin (INT8 Quantized)`. The report therefore provably came from the projection path, not a measurement.

### 3.3 Direct verification — the NPU cannot run this model

- `openvino 2026.3.0`, `core.available_devices` → `['CPU', 'GPU.0', 'GPU.1', 'NPU']`. The NPU **is** present, so unavailability is not the cause.
- Compiling `export/openvino_int8/causal_anc_int8.xml` on `NPU` **fails**:
  > `nGraph GRUSequence node '__module.model.gru/aten::gru/GRUSequence' has unsupported sequenceLengths input. It must be a Constant node`
  plus numerous `Upper bounds are not specified` errors from the dynamic input shapes.
- The static variant `causal_anc_npu_static.xml` also fails, harder — the OpenVINO NPU compiler aborts with `LLVM ERROR: Failed to infer result type(s)` on `IE.Split` (the GRU gate split).
- Re-running the benchmark function today: `OV NPU INT8 : RETURNED None`.

**The model has never run on the NPU and cannot currently do so.** The GRU recurrent core is unsupported by the Intel NPU compiler.

### 3.4 What IS real

| Item | Status |
|---|---|
| ONNX export (`export/causal_anc_model.onnx`, 3.11 MB) | ✅ exists |
| NNCF INT8 post-training quantization (`export/quantize_openvino.py`, PERFORMANCE preset, `TargetDevice.ANY`, 80 synthetic tactical calibration frames) | ✅ real, documented procedure |
| INT8 IR artifacts (1.46 MB) | ✅ exist on disk |
| INT8 IR **executing on Intel CPU** via OpenVINO | ✅ **0.0930 ms/frame, RTF 0.0058** — re-measured and reproducible |
| INT8 IR executing on **NPU** | ❌ **impossible today** |
| `<2.5 W` power figure | ❌ never measured; hardcoded string |

### 3.5 Actions taken

1. **Removed** the 0.18 ms / RTF 0.0110 / `<2.5W` NPU claim from `README.md`, `docs/JUDGE_PITCH_ONEPAGER.md`, `docs/DEMO_VIDEO_SCRIPT.md`, and `docs/COMPETITION_NARRATIVE.md`.
2. **Fixed the root cause**: the fallback branch in `benchmark_suite.py` now emits `NOT MEASURED` and the actual compiler error instead of a projection, with an explicit code comment forbidding substitution of estimates.
3. **Regenerated** `benchmark/HARDWARE_BENCHMARK_REPORT.md` from the corrected script; the NPU row now reads `NOT MEASURED` with the compiler error as its rationale, and the report carries a provenance rule.
4. **Repositioned the edge story** onto the CPU INT8 path (1.46 MB IR at 0.093 ms/frame), which is real, reproducible, and still a legitimately strong claim.

> **Accountability note.** The Phase 12 pass did not merely miss this — it *strengthened* the false claim, describing the NPU figures as "real, measured" and "genuinely measured latency at each stage," and promoted them into the one-pager as a headline asset. That happened because the number *looked* plausible and sat in a file named like a generated report, and was accepted without opening `benchmark_suite.py`. A targeted grep for known-bad strings cannot catch this class of error; only tracing each claim to the code path that produced it can.

---

## Step 4 — Post-filter naming clarification: **TWO DISTINCT FILTERS**

**Finding: these are two genuinely different implementations, and prior documentation conflated them.**

| # | Technique | Implementation | Evaluated? |
|---|---|---|---|
| 1 | **Causal minimum-statistics spectral subtraction** | `minimum_statistics_postfilter()` — `scripts/validate_cleanup_fixes_1234.py:323`. Recursive noise-floor tracker: `noise_floor = α·noise_floor + (1-α)·min(frame_mag, noise_floor)`, with over-subtraction and spectral floor. | ✅ **Yes** — Phase 11B, 9-config sweep (α ∈ {0.85, 0.90, 0.95} × over-sub ∈ {1.0, 1.5, 2.0}), N=150. All degraded SNR; best −0.92 dB. Results in `eval/validation_fixes_1234_results.json`. |
| 2 | **Normalized Least Mean Squares (NLMS)** | `NormalizedLMSFilter` + `HybridANCPostProcessor` — `model/adaptive_lms.py`. Order-64 adaptive FIR, µ=0.08, instantiated in `demo/stream_engine.py:98`. | ❌ **No.** Never scored in any phase. |

**Confirmed**: "NLMS" was a **naming error** in the Phase 12 report — the Phase 11B experiment tested the min-statistics filter, *not* NLMS.

**But the correction is worse than a pure naming fix**, and this matters: because the two were conflated, the demo's NLMS toggle was labelled with the min-stats sweep results ("degrades SNR, best −0.92 dB"), implying it had been evaluated. It has not. The honest status of the NLMS filter is **unevaluated**, not "tested and rejected."

Verified which harness produced the reported metrics: `scripts/evaluate_phase10_final.py` writes `eval/test_eval_csvs/*` and contains **no NLMS usage** — so the reported 12.75 dB figures are pure neural-model output, uncontaminated by either post-filter. ✅

> Separately noted: `eval/evaluate_drdo_targets.py:77` and `eval/compare_fixes_detailed.py:79` *do* call the pipeline with `enable_nlms_post=True`. These are older harnesses that are **not** the source of any currently-reported figure (the current `eval/DRDO_TARGETS_EVALUATION.md` is hand-authored, not this script's output — its generated companion `DRDO_TARGETS_EVALUATION.html` shows an unrelated 6.96 dB). Anyone re-running `evaluate_drdo_targets.py` in future would silently produce NLMS-contaminated numbers. **Recommend** setting these to `False` before any future evaluation run.

**Actions taken**: `demo/app.py` toggle relabelled to "NOT quantitatively evaluated" with help text distinguishing the two filters; `README.md`, `JUDGE_PITCH_ONEPAGER.md`, and `COMPETITION_NARRATIVE.md` corrected to describe both filters accurately.

---

## Summary of findings

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | 18.4 dB / 2.85 PESQ / 0.882 STOI traced to **no artifact whatsoever**; entered repo in a single bulk commit alongside the contradicting honest report | Critical | Removed (Phase 12), provenance now established |
| 2 | **NPU 0.18 ms / RTF 0.0110 / <2.5 W was a hardcoded fallback constant**, emitted whenever NPU compilation failed — which it always does (GRU unsupported). Never a measurement. | **Critical** | **Removed; root cause fixed in code; report regenerated** |
| 3 | ΔSNR "+7.0 dB avg / +10.3 dB stress" derived from bucket midpoints, not measured data. True values: **+2.77 dB / +6.84 dB**. Introduced by the Phase 12 pass itself. | **High** | **Corrected in all documents** |
| 4 | "NLMS" was a naming error for the min-statistics filter; the actual NLMS filter in the demo is **unevaluated**, not "tested and rejected" | Medium | Corrected; both filters now described accurately |
| 5 | Power draw figures (28-45 W, 60-90 W, <2.5 W) are hardcoded estimates, never instrumented | Low | Column labelled "(Est)"; NPU power claim removed |
| 6 | 20/500 test utterances have measured input SNR >20 dB (max +107 dB) despite low-SNR bucket labels — mixing did not hit target SNR | Medium | Disclosed in narrative §4; flagged for `data/real_dataset_loader.py` investigation |
| 7 | `eval/evaluate_drdo_targets.py` and `eval/compare_fixes_detailed.py` hardcode `enable_nlms_post=True` | Low (latent) | Flagged; not a source of any current figure |
| 8 | All core SNR/STOI/PESQ/bucket metrics | — | ✅ **Verified exact** against raw CSVs |
| 9 | Causality, parameter counts, model/IR sizes, Phase 11B results, STOI backend | — | ✅ **Verified**, several by live re-run |

## Standing rule going forward

No number appears in a submission-facing document unless it can be traced to (a) a specific row/field in a committed artifact under `eval/`, or (b) a script that reproduces it on demand. Where a measurement could not be obtained, documents say **NOT MEASURED** and explain why — they never carry a projection formatted as a measurement. Any code path that emits an estimate into a report file is a defect.
