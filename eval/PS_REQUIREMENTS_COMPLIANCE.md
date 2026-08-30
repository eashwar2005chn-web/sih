# PS 26052 — requirement-by-requirement compliance

**Date: 2026-08-30.** Every row is audited against the codebase, not asserted. Where something
is missing it says so.

Legend — **MET** · **PARTIAL** · **GAP** (not implemented) · **BLOCKED** (implemented path
exists but this environment prevents it)

---

## A. Dataset pipeline

| # | Requirement (PS text) | Status | Evidence |
|---|---|---|---|
| A1 | "clean speech data combined with curated defence noise datasets" | **MET** | LibriSpeech + VoiceBank clean; MUSAN/DEMAND/ESC-50/AudioSet generic; MAD/gunshot/drone/AudioSet tactical |
| A2 | "gunshots, drones, artillery, vehicle engines, wind" | **MET** | `defence_noise_generator.py`: gunshot, helicopter rotor, armoured vehicle engine, jet flyby, siren, cockpit hum, radio static, wind turbulence, babble |
| A3 | "at varying SNR levels" | **MET** | 5 buckets, −10 to +15 dB; `snr_sampler_mode="rebalanced"` |
| A4 | "both stationary and impulsive noise scenarios" | **MET** | Impulsive (gunshot/artillery transients) and stationary (hum, static, wind) generators |
| A5 | "random noise mixing, reverberation, clipping" | **MET** | `dataset_builder.py:216` RIR reverb (p=0.6); `:223` tactical channel with clipping (p=0.3); shared-RIR invariant enforced |
| A6 | "scalable dataset pipeline" | **MET** | On-the-fly streaming mixing; corpus scaled 5.04 h → **100.59 h** this session |

**Note on A6.** The corpus was LibriSpeech `dev-clean` — the *development* split — at 5.04 h.
Now LibriSpeech `train-clean-100`, 100.59 h / 251 speakers, pinned to train.
See [`SPLIT_INTEGRITY_AND_DATA_REGIME.md`](SPLIT_INTEGRITY_AND_DATA_REGIME.md).

## B. Model architecture

| # | Requirement | Status | Evidence |
|---|---|---|---|
| B1 | "time-frequency representations (e.g. STFT spectrograms)" | **MET** | STFT n_fft 512 / hop 256 @ 16 kHz (32 ms / 16 ms) |
| B2 | **"process both full-band and sub-band features"** | **MET (new)** | `model/causal_anc_subband.py` — `CausalANCSubbandNet`. Full-band conv/GRU path (topology identical to the baseline) + per-bin sub-band GRU over a 15-bin neighbourhood, shared across all 257 bins |
| B3 | "operates in the complex domain to preserve phase" | **MET** | Complex ratio mask (Mr, Mi); real+imag encoder channels; phase preserved throughout |
| B4 | "estimating a mask or directly reconstructing" | **MET** | cRM estimation, tanh-bounded |
| B5 | "lightweight adaptive filter (e.g. LMS) for residual" | **MET** | `model/adaptive_lms.py` NLMS; wired into the live path with a **true reference channel** |

**B2 was the one architectural requirement with no implementation.** Added and gated:

```
full-band only  : 811,522 params
full+sub-band   : 823,428 params   (+11,906 = +1.5%)

causality  : +0 extra samples of lookahead vs the full-band baseline
latency    : 0.357 ms/frame, RTF 0.0223 (budget < 0.1)
```

Verify with `python scripts/verify_subband_causality_latency.py`.
**Not yet trained** — the architecture exists and is gated; a training run is separate.

## C. Training framework

| # | Requirement | Status | Evidence |
|---|---|---|---|
| C1 | "SI-SNR, L1/L2 loss, and perceptual loss" | **MET** | `HybridANCLoss`: SI-SNR (clamped-alpha) + multi-resolution STFT + complex spectral L1 + energy term |
| C2 | "optimized hyper-parameters" | **MET** | Warmup 500 → peak 5e-4 → cosine 1e-5, AdamW wd 1e-4, grad clip 5.0, 150k steps |
| C3 | "evaluation metrics including SNR, STOI, PESQ" | **PARTIAL** | SNR ✅ · STOI ✅ (pystoi reference) · **PESQ is an uncalibrated Bark surrogate, not ITU-T P.862** |

**C3 is blocked, not skipped.** See §F.

## D. Real-time inference & edge deployment

| # | Requirement | Status | Evidence |
|---|---|---|---|
| D1 | "processes incoming noisy audio in real time" | **MET** | RTF 0.0085 (model alone), 0.0521 live end-to-end with LMS |
| D2 | "quantization" | **MET** | OpenVINO NNCF INT8 PTQ, 1.46 MB IR |
| D3 | "pruning" | **MET (new)** | `scripts/prune_model.py` — global unstructured L1 sweep; **20% sparsity costs −0.44 dB with pass rate unchanged and 17.2% smaller under a sparse runtime**. Output: [`eval/pruning_sweep.txt`](pruning_sweep.txt) |
| D4 | "ONNX conversion" | **MET** | `export/causal_anc_model.onnx` |
| D5 | **"TensorRT conversion"** | **GAP** | Not installed, no `.engine`/`.plan`, no conversion code |
| D6 | "DSPs or AI-enabled SoCs (e.g. Jetson AGX Orin)" | **GAP** | No such hardware available |
| D7 | "microphones (primary + reference)" | **MET (new)** | `demo/live_anc.py` — 2-channel capture, ch0 primary / ch1 true reference into the NLMS stage |
| D8 | "headphones / communication units" | **MET (new)** | Duplex `sounddevice` stream, enhanced audio to output device |
| D9 | "prototype demonstrating live noise cancellation" | **MET (new)** | `demo/live_anc.py`; `--selftest` drives the identical streaming path with synthetic audio |

### D5 — a label correction

`benchmark_suite.py` reported the GPU row as `"CUDA / TensorRT (AMP FP16)"`. **TensorRT was
never used.** It is not installed, no engine artifact exists, and the measurement comes from
`benchmark_pytorch_device(model, "cuda", use_fp16=True)` — PyTorch eager under
`torch.amp.autocast`. Corrected to `"PyTorch eager CUDA (torch.amp autocast FP16) - NOT
TensorRT"` in both the script and `HARDWARE_BENCHMARK_REPORT.md`. **The latency number itself
is unchanged and reproducible — only the runtime attribution was wrong.**

### D9 — measured

```
SELF-TEST (synthetic audio, identical streaming path, no microphone)
  LMS stage    : ON (true reference channel)
  RTF (steady) : 0.0521   OK, budget < 0.1
  per block    : 3.34 ms for 64 ms of audio
  without LMS  : RTF 0.0282, 1.81 ms per block
```

Streaming correctness: GRU hidden state is carried across blocks and each block is processed
with `n_fft` samples of preceding context, emitting only the new tail — without that, block
boundaries produce audible discontinuities.

## E. Performance targets

| Target | Measured | Verdict |
|---|---|---|
| SNR > 15 dB | **15.26 dB** [14.55, 15.97] | **MET** (point estimate; CI straddles) |
| STOI > 0.85 | **0.9309** | **MET** |
| PESQ > 2.5 | 3.96 | **MET on the proxy metric only** — see §F |
| "low latency suitable for real-time" | 0.02–0.36 ms/frame, RTF ≤ 0.052 | **MET** |

Standing caveats, all disclosed in `DRDO_TARGETS_EVALUATION.md`: the SNR CI straddles 15.0;
the test split is **not speaker-disjoint** (41/41 test speakers seen in training) and carries
**30.12% content overlap** — both being repaired by the Phase 24 retrain now running; and
"0.0 ms lookahead" is false as stated (≈16 ms from the centred STFT).

The 80%/85% **per-utterance pass rate is self-imposed** and appears nowhere in the PS text.
Current 44.3%; see [`PASS_RATE_85_ANALYSIS.md`](PASS_RATE_85_ANALYSIS.md).

## F. What is genuinely blocked, and why

| Item | Blocker | Path to unblock |
|---|---|---|
| **Real ITU-T P.862 PESQ** | `pip install pesq` cannot build (no MSVC); **no** prebuilt wheel for `pesq`/`pypesq` on cp311/win_amd64; Miniforge installer exits 0 installing nothing because **Smart App Control is ON** (`VerifiedAndReputablePolicyState=1`, `CodeIntegrityPolicyEnforcementStatus=2`) | Another machine, a CI runner, or a WSL distro. **Not** by disabling Smart App Control — that cannot be re-enabled without reinstalling Windows |
| **TensorRT** | Not installed; same code-integrity constraint applies to the NVIDIA runtime | Jetson/Linux host, or a machine without enforced CI |
| **Jetson AGX Orin** | No hardware | Physical device |

These three are environmental, not design gaps. Everything else in the PS is implemented.

## G. Remaining work

1. **Train the sub-band model (B2)** — architecture gated and ready; needs a retrain slot.
2. **Phase 24** — 100 h speaker-disjoint retrain, running (retrain 6 of 8).

### D3 result

```
 sparsity  actual   SNR dB     sd    STOI   pass%   sparse MB
       0%    0.0%    15.97   6.17  0.9413   47.5%       5.69
      20%   20.0%    15.54   5.02  0.9404   47.5%       4.71
      30%   30.0%    14.41   4.16  0.9384   40.0%       4.22
      50%   50.0%    15.14   5.46  0.9337   43.3%       3.25
      70%   70.0%    13.40   4.60  0.9117   30.8%       2.27
      90%   90.0%     9.62   3.73  0.8750   10.8%       1.29
```

**20% sparsity is close to free**: −0.44 dB, STOI −0.0009, pass rate unchanged, 17.2% smaller
under a sparse runtime. Beyond that the cost rises sharply.

Three honesty notes carried in the script and its output: unstructured pruning keeps **dense**
storage, so the size win needs a sparse runtime or structured pruning — the dense column is
flat at 3.25 MB throughout; **no fine-tuning** is applied, which would recover accuracy but
needs its own control; and at N=120 with sd≈6 the standard error is ~0.6 dB, so **adjacent rows
are not separable** — the sweep is non-monotonic (50% scores above 30%) because BatchNorm
running statistics are frozen while pruning shifts activation scales. Read the trend, not the
ranking.
