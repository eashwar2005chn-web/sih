# Phase 15 — Section A Verification Audit

**Date**: 2026-08-29
**Scope**: All nine Section A verification gaps, completed before any Section C experiment.
**Status**: Section A complete. **No Section C experiment has been started.** Per Section E, this is a hard pause point awaiting explicit go-ahead.

**Headline**: five previously-unknown defects were found, three of them material. One ("0.0 ms lookahead") was a **standing headline compliance claim that is false as stated**. One (stale demo checkpoint) meant the live demo was running a different, far weaker model than every reported number. The good news is equally concrete: the leakage that was found does **not** inflate any reported metric, and the two silent-substitution bugs in the metric stack never once fired on real data.

---

## A.1 — Train/test split leakage

The split is computed as `md5(file_path) % 100` in `RealAudioManifest.get_split_files()`. That guarantees **file-level** disjointness by construction but controls nothing else.

| Channel | Result |
|---|---|
| **File-path overlap** (train∩test) | **0** in all four categories — clean |
| **Speaker overlap** (LibriSpeech/VCTK) | **41 of 41 test speakers also appear in training.** **99.7%** of clean-speech test files have a speaker seen in training |
| **Content duplication** (SHA-256 of decoded PCM) | clean_speech **103/342 test files (30.1%)**, generic_noise **124/275 (45.1%)**, rirs **1/14**, tactical_noise **0/34** |

**Root cause of the content duplication**: the corpus directories contain *renamed copies of the same recordings*. `datasets/clean_speech/voicebank/p225_137_1462-170142-0036.wav` is byte-identical audio to `datasets/clean_speech/librispeech/1462-170142-0036.flac`; `generic_noise/musan/musan_crowd_3-149042-A-24.wav` is identical to `generic_noise/esc50/3-149042-A-24.wav`. Because the split hashes the *path*, the two copies of one recording land in different splits.

**Impact on the actual N=500 evaluated test set**: **164 / 500 items (32.8%)** drew a clean-speech or noise file that is bit-identical to a training file. Roughly uniform across buckets (30–36%).

### Did it inflate the reported numbers? No.

Splitting the evaluated set into LEAKED vs CLEAN (silent-noise rows already excluded):

| Model / scope | LEAKED | CLEAN | Difference | Significant? |
|---|---|---|---|---|
| Baseline, operational | 12.56 dB (n=94) | 12.41 dB (n=201) | **+0.15 dB** [−1.10, +1.41] | No |
| Baseline, stress | 4.10 dB (n=66) | 4.51 dB (n=132) | **−0.41 dB** [−2.13, +1.31] | No |
| TCN, operational | 12.36 dB (n=94) | 12.31 dB (n=201) | **+0.05 dB** [−1.17, +1.28] | No |
| TCN, stress | 4.56 dB (n=66) | 5.19 dB (n=132) | **−0.63 dB** [−2.23, +0.98] | No |

Every confidence interval spans zero, and in the stress buckets the model is *worse* on leaked items. **The model is not memorising.** The leakage is a real methodological defect that must be disclosed, but it is not inflating results, and no reported number requires restatement on its account.

**Note (not leakage, opposite direction)**: `gunshot_kaggle` is excluded from training by license policy but 2 of its files appear in the test pool — unseen-source distribution shift that makes the test *harder*.

**Recommendation (not applied)**: a corrected split should group by speaker ID and by content hash, not by path. That would change the test set and therefore invalidate comparability with all fourteen prior phases; it is a deliberate decision requiring sign-off, not something to do silently.

---

## A.2 — Streaming / online inference parity

Two distinct findings.

**(a) The demo does not actually stream.** `RealTimeANCEngine.process_audio()` processes the entire buffer in one call. Measured against the locked offline eval path on identical utterances: **stream-engine-minus-direct = −0.00 dB** (identical). So the demo is correct — but it is batch inference, and it does not exercise or demonstrate streaming.

**(b) The model cannot be correctly streamed frame-by-frame through its current API.** `forward_spec(noisy_real, noisy_imag, h_state)` exposes only the GRU hidden state. The encoder/decoder use causal 2-D convolutions with kernel size 3 along time; offline, frame *t* sees frames *t−2, t−1* through left-padding of the real sequence. Fed one frame at a time, that padding is **zeros** — the convolutional receptive field is silently destroyed while `h_state` is carried correctly.

Measured recovery as left context is restored (N=10 utterances):

| Configuration | Output SNR | vs offline |
|---|---|---|
| offline (whole utterance) | 12.61 dB | — |
| chunk=1, left_ctx=0 (naive frame-by-frame) | 8.86 dB | **−3.75 dB** |
| chunk=1, left_ctx=4 | 9.69 dB | −2.92 dB |
| chunk=1, left_ctx=8 | 10.40 dB | −2.21 dB |
| chunk=1, left_ctx=16 | 10.76 dB | −1.85 dB |
| chunk=1, left_ctx=32 | 11.66 dB | −0.95 dB |

A correct streaming implementation must cache convolution input history, not just the GRU state. Until it does, true frame-by-frame streaming costs **−3.75 dB**. Per-hop compute is not the obstacle: measured **1.33 ms per 16 ms hop (RTF 0.083)**.

Harness: `scripts/streaming_parity_harness.py`, `scripts/streaming_context_probe.py`.

---

## A.3 — Manual listening pass

Exported **45 clips** (15 utterances × noisy/enhanced/clean) spanning all five buckets, to `eval/listening_clips/`, with `manifest.csv` giving per-clip target SNR, measured input SNR, output SNR, and input/output STOI so listening impressions tie back to numbers. Clips are from the authoritative epoch-50 checkpoint. **A human still needs to listen** — this item is delivered as an artifact, not a conclusion. Suggested focus: musical noise/warbling in the stress clips, speech-onset clipping, pumping between words.

---

## A.4 — Checkpoint provenance — **DEFECT FOUND AND FIXED**

| File | SHA-256 | Epoch |
|---|---|---|
| `checkpoints/task4_full_run/best_model.pt` (authoritative, all reported numbers) | `c0fed3f445a4aa44835d4ca24025549a1e9fac8cbde66d1e8ca9e7407728273c` | **50** |
| `checkpoints/best_model.pt` (what the demo loaded) | `b80306d9d968511c210c923b830cf13a9ca45a780e29373d7040eb0254e46c8d` | **6** |

`demo/app.py` and `demo/stream_engine.py` both defaulted to `checkpoints/best_model.pt` — a **different, epoch-6 checkpoint**. Same architecture and parameter names, but **maximum absolute weight difference 3600.0**. Every live demonstration was therefore running a materially under-trained model that no reported metric describes.

**Fixed**: both files now pin `checkpoints/task4_full_run/best_model.pt`, with the SHA-256 recorded in a comment and an explicit warning not to revert the path.

---

## A.5 — Silent-substitution bug class — **TWO MORE FOUND, ALL HARDENED**

The NPU fallback branch was already replaced (Phase 13) with a `NOT MEASURED` row; it now **also prints a loud stderr banner**. NPU compile status re-confirmed today, actual error recorded verbatim:

```
nGraph GRUSequence node '__module.model.gru/aten::gru/GRUSequence'
has unsupported sequenceLengths input. It must be a Constant node
```

Auditing the rest of the codebase for the same pattern found **two further instances, both in the metric stack**:

1. **`compute_stoi()`** — if `pystoi.stoi()` raised, the handler was a bare `except: pass` that fell through to a **correlation proxy** `0.5·(r+1)`, returned as if it were STOI, with no log or flag. Indistinguishable downstream from a real STOI value.
2. **`compute_pesq()`** — for a too-short signal it returned the literal **`2.5`**, which is *exactly* the project's PESQ pass threshold (`> 2.50`).

**Both were instrumented and replayed over the full N=500 test set (1000 calls each):**

| Path | Activations | Verdict |
|---|---|---|
| STOI correlation-proxy fallback | **0 / 1000** | Reported STOI values are genuine `pystoi` — **CLEAN** |
| PESQ hardcoded `2.5` | **0 / 1000** | Reported PESQ values are genuine surrogate — **CLEAN** |

So no published number is contaminated. Both now raise `RuntimeError`/`ValueError` rather than substituting. The correlation proxy is retained as `_stoi_correlation_proxy()` for deliberate, labelled opt-in use.

---

## A.6 — NLMS toggle corrected in the app itself

`demo/app.py` label now reads *"experimental, NOT quantitatively evaluated"*, with help text distinguishing it from the minimum-statistics filter that Phase 11B actually ruled out. Default remains **off**. The footer caption was corrected to match.

**And it is no longer unevaluated** — see A.7: it costs **−6.22 dB**.

---

## A.7 — Reconciling 6.96 dB vs 12.75 dB — **RESOLVED**

**Run-to-run variance of the locked configuration** (3 repeats, fixed seeds, N=300):

```
repeat 1: 12.749958 dB
repeat 2: 12.749958 dB
repeat 3: 12.749958 dB
spread of means = 2.7e-08 dB;  max per-utterance |diff| = 3.8e-06
```

The 12.75 dB figure is stable to six decimal places. Residual 1e-6 jitter is float32 GPU non-determinism. Combined with the Phase 14 finding that the test data regenerates **bit-exactly**, the locked eval is effectively deterministic.

**Source of the discrepancy**, isolated on identical utterances:

| Path | Output SNR |
|---|---|
| Direct model forward (locked eval path) | **12.75 dB** |
| Stream engine, NLMS **off** | **12.75 dB** (pipeline delta −0.00 dB) |
| Stream engine, NLMS **on** | **6.53 dB** |
| **Cost attributable to the NLMS post-filter** | **−6.22 dB** |

The 6.96 dB in `eval/DRDO_TARGETS_EVALUATION.html` came from `eval/evaluate_drdo_targets.py`, which hardcodes `enable_nlms_post=True` (lines 77 / 79 in `compare_fixes_detailed.py`). Its residual difference from 6.53 dB is explained by that harness's smaller N and empirical-SNR filtering. Its per-class tables are empty, indicating the run also failed to match any `noise_type` key.

**Authoritative**: **12.75 dB**, from `scripts/evaluate_phase10_final.py` — direct model forward, no post-filter, N=100/bucket, fixed seeds. The 6.96 dB run measured a pipeline that includes an unevaluated post-filter which is not part of the recommended system.

**This closes a loop**: until Phase 13 the demo shipped with NLMS **on by default** *and* the epoch-6 checkpoint. A judge clicking "Play" was seeing roughly half the reported SNR from a fraction of the training.

---

## A.8 — Full re-audit of submission-facing claims — **ONE HEADLINE CLAIM IS FALSE**

### The "0.0 ms lookahead" claim does not survive verification

The claim rested on future-**frame** perturbation testing in the spectral domain, which does pass (`0.00e+00` leakage). But `stft_forward()` calls `torch.stft(..., center=True)`, which centres frame *k* on sample *k·hop* and therefore requires `n_fft/2 = 256` samples (16 ms) of future audio.

Waveform-level test — perturb only future **samples**, find the earliest changed output sample:

| Perturbation start | First changed output sample | Backward leakage |
|---|---|---|
| 4000 | 3585 | 415 samples (**25.9 ms**) |
| 8000 | 7681 | 319 samples (**19.9 ms**) |
| 12000 | 11530 | 470 samples (**29.4 ms**) |

**The end-to-end system has ~16 ms of algorithmic lookahead (measured 20–29 ms including overlap-add span), not 0.0 ms.** The network is frame-causal; the front-end is not. This is within a 32 ms-window real-time budget, so the system remains deployable — but the compliance claim as written was wrong.

**Corrected in**: `README.md`, `docs/JUDGE_PITCH_ONEPAGER.md`, `docs/DEMO_VIDEO_SCRIPT.md`, `docs/COMPETITION_NARRATIVE.md`, `eval/DRDO_TARGETS_EVALUATION.md`. The met-metric count drops from **five of seven to four of seven**, and every document has been updated to say so.

### Other claims re-verified

| Claim | Traces to | Status |
|---|---|---|
| Output SNR 12.75 ± 5.35 dB, pass 28.0% | `Baseline_811K_..._test_eval.csv`, recomputed | ✅ exact |
| STOI 0.899 | same CSV; backend confirmed real `pystoi`, 0 fallbacks | ✅ exact |
| PESQ 3.69 (proxy) | same CSV; 0 hardcoded values; surrogate caveat in place | ✅ exact |
| CPU latency 0.07–0.10 ms | re-measured 0.087 ms (PyTorch), 0.093 ms (OpenVINO INT8) | ✅ |
| RTF 0.006–0.010 | re-measured 0.0054 / 0.0058 | ✅ |
| 811K / 899K params, 3.1 MB, 1.46 MB IR | live count + byte sizes | ✅ exact |
| "calibrated to ITU-T P.862" language | grep across all `.py`/`.md` | ✅ **zero occurrences** outside the audit describing its removal |

---

## A.9 — Training-time mixer

The training-time online mixer is the **same function** as the test-set mixer (`mix_speech_and_noise`), so it carries the identical `+1e-12` epsilon defect: an all-zero noise window yields `scaled_noise = 0`, silently producing a "noisy" signal equal to clean.

**Applied**: a `RuntimeWarning` for all-zero noise and a second for near-silent noise (`power < 1e-10`). **Numerical behaviour deliberately unchanged** — altering the mix would change training dynamics and break comparability with all fourteen prior phases. The warning fires correctly and was observed live during A.2/A.3 runs.

**Not triggered**: any retrain. Per instruction, that is a separate explicit decision. The proper fix (resample the noise window on detection) belongs with a planned retrain, not as a silent behaviour change.

---

## Section B status — guardrails NOT yet established

Section B requires a validation split distinct from both training-gradient data and the N=500 test set, a pre-committed experiment log, and the disclosure statement. **None of this has been set up yet**, because Section A had to complete first and its findings bear directly on how Section B should be built:

- A `val` split already exists (`80 ≤ md5(path)%100 < 90`, 346 clean-speech / 269 generic-noise files) and is **unused by any phase**. But it inherits the *same* speaker and content-duplication defects as the test split (it is the same hashing scheme), so it is not currently a clean model-selection surface.
- Whether to rebuild the splits properly before running Section C is a decision with real consequences — it would invalidate comparability with fourteen prior phases — and per Section E I am not making it autonomously.

---

## Required disclosure — how the N=500 test set was used, Phases 1–14

Stated unconditionally, as required:

- **Phases 1–11** each evaluated the held-out N=500 set after training, and results across phases were compared against each other to decide which architectural direction to pursue next. Concretely: the Phase 4 rebalancing result, the Phase 5–7 capacity result, the Phase 8 oracle diagnostic, the Phase 9–10 K-sweep, and the Phase 11 TCN result were all read off this set, and each informed whether the next phase was attempted. **This is iterative test-set-informed model selection.** No separate validation split was used for these decisions.
- **Phase 10's K-screening** compared five variants against the same set; the "winner" (K=1.5) was selected from among them and then given a full retrain.
- **Phase 11B** swept 9 post-filter configurations against N=150 drawn from the same distribution and selected the best.
- **Phases 12–14** did not train; they re-read the existing CSVs for auditing.
- **Phase 15 (this audit)** re-ran inference over the test set several times — for fallback instrumentation (A.5), determinism checks (A.7), streaming parity (A.2), and clip export (A.3). These are verification passes that did not inform any modelling decision, but they are disclosed for completeness.

**Consequence to state plainly in any final report**: the reported figures carry an optimistic bias of unknown magnitude from repeated test-set-informed selection across eleven phases, *in addition to* the disclosed content leakage. The Section B discipline is designed to stop this going forward; it cannot retroactively repair the prior phases.

---

## Summary of Section A findings

| # | Finding | Severity | Status |
|---|---|---|---|
| A.8 | "0.0 ms lookahead" is false — system has ~16 ms (measured 20–29 ms) waveform-level lookahead | **Critical** | Corrected in all 5 documents; met-count now 4/7 |
| A.4 | Demo loaded an epoch-6 checkpoint, not the epoch-50 reported model (weight diff 3600.0) | **Critical** | Fixed, SHA-pinned |
| A.7 | NLMS post-filter costs −6.22 dB; explains the 6.96 vs 12.75 discrepancy | **High** | Reconciled; 12.75 dB confirmed authoritative |
| A.1 | 100% speaker overlap; 32.8% of evaluated test items are exact-duplicate training audio | **High** | Disclosed; **verified not to inflate metrics** |
| A.2 | Model not streamable via `h_state` alone; naive streaming costs −3.75 dB | **High** | Documented; conv-state caching required |
| A.5 | Two further silent-substitution bugs (STOI proxy, PESQ `2.5`) | Medium | Hardened; **0/1000 activations — no number affected** |
| A.9 | Training mixer shares the epsilon defect | Medium | Warning added; no behaviour change, no retrain |
| A.7 | Run-to-run variance | — | 2.7e-08 dB — effectively deterministic |
| A.3 | 45 listening clips exported | — | Awaiting human listening |
| A.6 | NLMS toggle label | — | Fixed in app code |

## Reproducing

```bash
python scripts/audit_split_leakage.py && python scripts/quantify_content_leakage.py && python scripts/leakage_metric_impact.py
```

```bash
python scripts/audit_silent_fallbacks.py && python scripts/reconcile_eval_runs.py && python scripts/streaming_context_probe.py && python scripts/export_listening_clips.py
```

Artifacts: `eval/split_leakage_audit.json`, `eval/content_leakage_audit.json`, `eval/leakage_metric_impact.json`, `eval/listening_clips/manifest.csv`.

---

## Addendum (2026-08-30) — CUDA fixed, and a correction to A.8

**The environment was the fault, not the number.** The RTX 5060 is sm_120; the installed
torch 2.5.1+cu121 shipped kernels only to sm_90, so every CUDA op raised *"no kernel image
is available for execution on the device"* while `torch.cuda.is_available()` still returned
True. Upgrading to **torch 2.11.0+cu128** (arch list now includes `sm_120`) fixes it:

```
torch 2.11.0+cu128 | cuda 12.8 | capability (12, 0) | matmul on CUDA: OK
```

Rollback instructions: `eval/CUDA_FIX_ROLLBACK.md`.

### Correction: the RTX 5060 latency figure was NOT fabricated

Phase 13 grouped the GPU row with the NPU row and reported both as unreproducible. That was
wrong for the GPU. With CUDA working, the benchmark re-measures the GPU at **0.02 ms/frame,
RTF 0.0011** — matching the originally-reported figure. It had been a genuine measurement
taken under a working environment; my re-test failed only because the environment had since
broken, and I attributed that failure to the number rather than to the toolchain.

The NPU finding is unaffected and stands: that 0.18 ms figure came from a hardcoded
`else` branch in `benchmark_suite.py` labelled "theoretical NPU projection", the NPU still
fails to compile the GRU, and its row still reads NOT MEASURED.

**Lesson, same shape as the rest of this audit:** "cannot reproduce today" is not the same
claim as "was never measured", and collapsing the two is its own reporting error. Documents
corrected: `README.md`, `docs/JUDGE_PITCH_ONEPAGER.md`, `docs/DEMO_VIDEO_SCRIPT.md`,
`web/server.py`, `benchmark/HARDWARE_BENCHMARK_REPORT.md`.

### Verified unbroken by the upgrade

| Check | Result |
|---|---|
| Test-set regeneration vs committed CSVs | **bit-exact**, max diff `0.00e+00` |
| Baseline + Phase 17 checkpoints load | PASS |
| CPU vs CUDA forward agreement | max diff `1.83e-03` (normal float32 device variance) |
| torchaudio / OpenVINO INT8 / web server | all PASS; server now selects `cuda` |

### Throughput unlocked

| | CPU | CUDA | speedup |
|---|---|---|---|
| per training step (batch 16) | 0.242 s | **0.018 s** | **13.2×** |
| 100,000 steps | 6.7 h | **0.51 h** | |
| 300,000 steps | 20.2 h | **1.53 h** | |

The 30,000-step Phase 17 run took ~3.5 h on CPU. The same budget is now ~9 minutes, and the
100k–1M step range standard for this model class becomes practical for the first time.
