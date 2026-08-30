# DRDO Problem Statement 26052 — Executive Pitch & Technical Brief

**Title**: AI/ML-Enabled Adaptive Noise Cancellation (ANC) for Tactical Defence Communication  
**Target Platform**: Heterogeneous Tactical Edge Compute (Intel Core Ultra 9 CPU + NVIDIA RTX 5060 GPU + Intel AI Boost NPU)

---

## 1. Executive Summary & Problem Context
Tactical military communications operate in severe acoustic combat environments: cockpit avionics whine, continuous helicopter blade-pass modulation, armored vehicle diesel rumble, artillery/gunfire impulse transients, and radio RF degradation. Traditional DSP filtering either fails against non-stationary burst noises or causes phase destruction and severe speech cancellation.

We present a **causal complex-domain neural network (C-CRN)** that achieves **real-time causal complex-domain noise suppression** with a **strictly frame-causal network** (0.00e+00 future-frame leakage) and ~16 ms of waveform-level system lookahead from the centred STFT front-end — measured, not assumed. The model runs at 10-14× inside the real-time CPU latency budget on an 811K-parameter (3.1 MB) footprint, and has been quantized to a 1.46 MB INT8 OpenVINO IR and profiled on real Intel CPU silicon. Four of seven internal target metrics are met with substantial margin (latency, RTF, STOI, PESQ-on-proxy); the output-SNR target is met on the easiest input slice and falls short on the full aggregate operational range — a gap we diagnosed down to its root cause across eleven structured experimental phases rather than papering over it. Section 4 gives the honest compliance picture; Section 5 explains why that's a stronger position than a headline number that doesn't hold up under questioning.

---

## 2. Heterogeneous Hardware-to-Pipeline Mapping

| Hardware Engine | Pipeline Role | Technical Justification for DRDO |
|---|---|---|
| **NVIDIA GeForce RTX 5060 (CUDA)** | Model Training (AMP FP16) & High-Throughput Inference | $0.02\text{ ms/frame}$, $\text{RTF} = 0.0011$ — verified working after upgrading to a CUDA build with sm_120 kernels; ~$13\times$ faster per training step than CPU |
| **Intel AI Boost NPU (OpenVINO INT8)** | ⚠️ **Attempted, not achieved** | The INT8 IR does not compile on the NPU — the GRU core hits `GRUSequence node has unsupported sequenceLengths input`. **No NPU latency figure is claimed.** Disclosed rather than projected. |
| **Intel Core Ultra 9 CPU** (PyTorch FP32) | Audio I/O & Causal Framing | Deterministic sub-millisecond framing ($0.087\text{ms/frame}$, $\text{RTF} = 0.0054$) |
| **Intel Core Ultra 9 CPU** (OpenVINO INT8) | Real quantized edge-inference path | $0.093\text{ms/frame}$, $\text{RTF} = 0.0058$, from a **1.46 MB** INT8 IR — the genuine, verified edge-deployment story |
| **16GB DDR5 RAM** | Streamed Memory-Mapped Acoustic Dataloading | Zero RAM footprint overhead while training on infinite on-the-fly acoustic mixtures |

---

## 3. Technical Innovations & Architecture

```
[ Tactical Audio 16kHz ] 
       │
       ▼ (Host CPU: Causal STFT 32ms win / 16ms hop)
[ Causal Complex Recurrent Network (C-CRN) ] ── (cRM Masking: S = Y * M)
       │
       ▼ (Host CPU: iSTFT Synthesis)
[ Enhanced Tactical Voice Out ]
```

1. **Frame-Level Causality (with an honest caveat)**: Left-padding on 2D convolutions and causal GRU state tracking give the network strict frame-to-frame causality — verified at 0.00e+00 future-frame leakage. However the STFT analysis front-end uses centred windows, so the *end-to-end system* has ~16 ms of waveform-level lookahead (measured 20-29 ms). We report this rather than the "0.0 ms" figure earlier drafts claimed, which measured only the network and not the front-end.
2. **Complex Ratio Masking (cRM)**: Estimates both real and imaginary mask components ($M_r + j M_i$) to preserve critical speech phase information, preventing speech intelligibility degradation.
3. **Quantized Edge Footprint**: Model compressed from 3.1 MB (811,522 FP32 params) to a **1.46 MB** INT8 OpenVINO IR via NNCF post-training quantization, running at $0.093\text{ms/frame}$ on the Intel Core Ultra 9 CPU through the OpenVINO runtime. NPU offload was attempted and currently fails to compile (GRU unsupported) — we report the CPU INT8 path, which works, rather than an NPU number we cannot reproduce.
4. **Tested and deliberately not included**: a causal **minimum-statistics spectral-subtraction** post-filter was built and swept across 9 configurations (Phase 11B). Every configuration *degraded* Output SNR relative to the neural model alone (best case -0.92 dB), so it is excluded. A **separate** adaptive NLMS filter exists in the demo GUI only; it has never been quantitatively evaluated and is off by default. These are two distinct techniques — earlier drafts of our own documentation conflated them, and we corrected that rather than leaving it ambiguous.

---

## 4. Compliance Matrix Against Internal PS 26052 Target Specifications

Measured on the held-out test split, full 0-15 dB input-SNR operational range (N=300) unless noted. Full methodology and per-bucket breakdown: [eval/DRDO_TARGETS_EVALUATION.md](../eval/DRDO_TARGETS_EVALUATION.md).

| Metric | Target Threshold | Prototype Achievement | Verdict |
|---|---|---|---|
| **Output SNR** | **> 15.0 dB** | **12.75 ± 5.35 dB** aggregate mean (28.0% per-utterance pass rate). Passes only in the easiest [10,15] dB input slice (15.60 dB, 52% pass). Measured per-utterance SNR improvement: **+4.00 dB** operational, **+7.65 dB** across the -10 to 0 dB stress range (N=295/198, excluding 7 degenerate utterances). | ⚠️ **SHORT on aggregate** — see §5 |
| **STOI Intelligibility** | **> 0.850** | **0.899** | **PASSED ✓** |
| **PESQ (internal proxy)** | **> 2.50** | **3.69** — uncalibrated Bark-psychoacoustic surrogate, not the ITU-T P.862 reference (MSVC build tools unavailable in dev environment); not directly comparable to published PESQ figures | **PASSED ✓** *(on the proxy only)* |
| **Algorithmic Latency (CPU)** | **< 1.0 ms/frame** | **0.07-0.10 ms** | **PASSED ✓**, 10-14× margin |
| **Algorithmic Lookahead** | **0.0 ms** | ⚠️ **~16 ms at the waveform level** — the network is strictly frame-causal (0.00e+00 future-*frame* leakage), but the STFT front-end uses centred analysis windows, so perturbing future *samples* changes output 319-470 samples (20-29 ms) earlier. See [PHASE15](../eval/PHASE15_VERIFICATION_AUDIT.md) | ⚠️ **NOT MET as stated** |
| **CPU INT8 Edge Inference** | — | **0.093 ms/frame**, RTF 0.0058, from a 1.46 MB INT8 IR (re-measured and reproducible) | Demonstrates edge deployability |

---

## 5. Why Our Solution Wins
- **Honest, diagnosed engineering, not a cherry-picked number**: five independent architectural hypotheses (data rebalancing, 4× capacity scaling, mask-bound widening, causal normalization, extended temporal receptive field) were each tested with a full 50-epoch retrain and proper statistics. All five converge on the same ~12.6-12.8 dB ceiling — strong evidence this is a characterized property of single-channel causal STFT masking at this compute budget, not an unfound bug.
- **Real & Tested on Actual Silicon**: Not a simulated benchmark — the reported inference figures run live on Intel Core Ultra 9 CPU, both PyTorch FP32 (0.087 ms/frame) and OpenVINO INT8 (0.093 ms/frame), re-measured and reproducible from `benchmark/benchmark_suite.py`. The RTX 5060 runs at 0.02 ms/frame (RTF 0.0011) after we upgraded to a CUDA build with sm_120 kernels. One target still does **not** work — the Intel NPU, whose compiler rejects our GRU — and we report it as NOT MEASURED rather than projecting a number.
- **Defence Domain Specificity**: Trained on a hybrid mix of real clean speech (LibriSpeech/VCTK), real tactical noise (Military Audio Dataset, filtered AudioSet, Kaggle gunshot/drone corpora), and synthetic room impulse responses.
- **Complete Interactive Cockpit GUI**: Interactive Streamlit application with live mic/file playback, real-time before/after spectrograms, live-computed (not hardcoded) DRDO target pass/fail meters, and instant hardware switching for live panel evaluation.
