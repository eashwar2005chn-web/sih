# 🛡️ DRDO PS 26052: AI/ML-Enabled Adaptive Noise Cancellation (ANC) for Defence Communication

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.11+CUDA](https://img.shields.io/badge/PyTorch-2.11%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![OpenVINO INT8](https://img.shields.io/badge/OpenVINO-2026.3%20INT8-00549f.svg)](https://docs.openvino.ai/)
[![Hardware](https://img.shields.io/badge/Target-Intel%20Ultra%209%20%2B%20RTX%205060%20%2B%20NPU-00c853.svg)](https://intel.com)

A complete, production-grade, hackathon-winning prototype engineered for **DRDO Problem Statement 26052: AI/ML-enabled Adaptive Noise Cancellation (ANC) for defence communication**.

---

## 🏗️ System Architecture

Our solution is a **Causal Complex-Domain Neural Network (C-CRN)** — a causal STFT encoder, a 2-layer causal GRU temporal core, and a causal decoder predicting a complex ratio mask — trained on the RTX 5060 and deployable across CPU/NPU targets via OpenVINO INT8 quantization.

> **Two separate post-filters exist in this repo and must not be conflated:**
> 1. A **causal minimum-statistics spectral-subtraction** post-filter (`scripts/validate_cleanup_fixes_1234.py`) — this is what Phase 11B evaluated, sweeping 9 configurations; all degraded Output SNR (best: -0.92 dB). Ruled out. See [eval/DRDO_TARGETS_EVALUATION.md §9.5](eval/DRDO_TARGETS_EVALUATION.md).
> 2. An **adaptive NLMS filter** (`model/adaptive_lms.py`) — a different classical technique, wired only into the demo GUI. It has **never been quantitatively evaluated**; no metric in any project report was produced with it enabled. It is off by default in the demo.
>
> Neither is part of the recommended pipeline below.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 TACTICAL AUDIO INPUT (16 kHz)           │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │  Intel Core Ultra 9 CPU: Causal STFT (32ms win, 16ms hop│
                  └────────────────────────────┬────────────────────────────┘
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        │                                             │
                        ▼                                             ▼
       ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
       │   NVIDIA RTX 5060 Laptop GPU    │           │    Intel AI Boost NPU / CPU     │
       │   AMP Mixed Precision (FP16)    │           │   OpenVINO Quantized INT8 IR    │
       │  High Throughput / Training     │           │  Low-Power Edge Deployment      │
       └────────────────┬────────────────┘           └────────────────┬────────────────┘
                        │                                             │
                        └──────────────────────┬──────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ Complex Ratio Mask (cRM): S_real + j S_imag             │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │  Intel Core Ultra 9 CPU: iSTFT Waveform Synthesis       │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ ENHANCED TACTICAL AUDIO OUTPUT                          │
                  └─────────────────────────────────────────────────────────┘
```

---

## ⚡ Heterogeneous Hardware-to-Pipeline Mapping

| Hardware Engine | Role in Pipeline | Empirical Performance | Strategic Justification for DRDO |
|---|---|---|---|
| **NVIDIA GeForce RTX 5060 (CUDA)** | Model Training (AMP FP16) & High-Throughput Inference | **0.02 ms / frame** (RTF: 0.0011), re-measured 2026-08-30 after upgrading to torch 2.11.0+cu128 (sm_120 kernels) | Training acceleration; ~13x faster per training step than CPU |
| **Intel AI Boost NPU (OpenVINO)** | ⚠️ **Attempted, not achieved** | **No measurement available** — the INT8 IR fails to compile on the NPU (`GRUSequence node has unsupported sequenceLengths input`). The GRU core is not currently supported by the Intel NPU compiler. | Honest limitation; edge story rests on the CPU INT8 numbers below, not on NPU |
| **Intel Core Ultra 9 CPU** (PyTorch FP32) | Audio I/O, Causal STFT/iSTFT | **0.087 ms / frame** (RTF: 0.0054), re-measured 2026-08-27 | Zero-jitter causal framing |
| **Intel Core Ultra 9 CPU** (OpenVINO INT8) | Quantized edge inference on host silicon | **0.093 ms / frame** (RTF: 0.0058), re-measured 2026-08-27 | Real INT8 deployment path (1.46 MB IR), verified reproducible |
| **16GB DDR5 RAM** | On-the-Fly Streaming Data Pipeline | 0% RAM bloat | Eliminates memory leaks via dynamic parametric audio generation |

---

## 📊 Evaluation Results vs DRDO PS 26052 Targets

Figures below are the baseline 811K-parameter model (`checkpoints/task4_full_run/best_model.pt`), measured on the held-out test split across the full 0-15 dB input-SNR operational range (N=300). Full methodology, per-bucket breakdown, and the diagnostic program behind the SNR gap: [eval/DRDO_TARGETS_EVALUATION.md](eval/DRDO_TARGETS_EVALUATION.md).

| DRDO PS 26052 Metric | Required Target | Prototype Achieved (0-15 dB aggregate) | Compliance Verdict |
|---|---|---|---|
| **Output SNR** | **> 15.0 dB** | **12.75 ± 5.35 dB**. Measured mean SNR improvement over the noisy input: **+4.00 dB** across the operational range, **+7.65 dB** across the -10 to 0 dB stress range (per-utterance `snr_gain`, N=295/198 excluding 7 degenerate test utterances — see [PHASE14](eval/PHASE14_TESTSET_CONTAMINATION.md)) | ⚠️ **SHORT on aggregate mean** — passes only in the easiest [10,15] dB input slice (15.60 dB); 28.0% per-utterance pass rate at the strict >15 dB threshold |
| **Speech Intelligibility (STOI)** | **> 0.850** | **0.899** | **PASSED ✓** |
| **Speech Quality (PESQ, internal proxy)** | **> 2.50** | **3.69** *(uncalibrated Bark-psychoacoustic surrogate — no MSVC build tools available for the ITU-T P.862 reference implementation; not literature-comparable, see caveat below)* | **PASSED ✓** *(on the proxy metric only)* |
| **Algorithmic Latency (host CPU)** | **< 1.0 ms / frame** | **0.07-0.10 ms** | **PASSED ✓** (10-14× margin) |
| **Algorithmic Lookahead** | **0.0 ms** | ⚠️ **~16 ms at the waveform level** — the network is strictly frame-causal (0.00e+00 future-*frame* leakage), but the STFT front-end uses centred analysis windows, so perturbing future *samples* changes output 319-470 samples (20-29 ms) earlier. See [PHASE15](eval/PHASE15_VERIFICATION_AUDIT.md) | ⚠️ **NOT MET as stated** |

Five independent architectural hypotheses (data rebalancing, 4× capacity scaling, mask-bound widening, causal normalization, extended temporal receptive field) were tested via full-budget retrains to close the SNR gap; all five converge on the same ~12.6-12.8 dB ceiling, indicating this is a characterized property of single-channel causal STFT masking at this compute budget rather than an undertrained model. Full diagnostic trail in [eval/DRDO_TARGETS_EVALUATION.md](eval/DRDO_TARGETS_EVALUATION.md).

---

## 📂 Repository Structure

```
├── benchmark/
│   ├── benchmark_suite.py           # Cross-hardware latency/RTF benchmarking harness
│   └── HARDWARE_BENCHMARK_REPORT.md # Generated hardware comparison report
├── data/
│   ├── dataset_builder.py           # Streaming PyTorch dataset and DataLoader
│   ├── defence_noise_generator.py   # Parametric defence noise synthesizers
│   └── DATASET_CARD.md              # Dataset documentation & synthetic caveat notice
├── demo/
│   ├── app.py                       # Interactive Streamlit Tactical Cockpit GUI
│   └── stream_engine.py             # Multi-backend audio streaming processor
├── docs/
│   ├── COMPETITION_NARRATIVE.md     # Full presentation brief: methodology, honest gap framing, Q&A prep
│   ├── JUDGE_PITCH_ONEPAGER.md      # One-page executive brief for DRDO evaluation panel
│   └── DEMO_VIDEO_SCRIPT.md         # 90-second video demo storyboard & script
├── eval/
│   ├── evaluate_drdo_targets.py     # Evaluation harness against DRDO criteria
│   ├── DRDO_TARGETS_EVALUATION.md   # Markdown evaluation report
│   └── DRDO_TARGETS_EVALUATION.html # HTML evaluation report
├── export/
│   ├── export_onnx.py               # Causal streaming PyTorch -> ONNX exporter
│   ├── quantize_openvino.py         # OpenVINO NNCF INT8 post-training quantization
│   └── openvino_int8/               # Exported INT8/FP16 IR models (.xml + .bin)
├── model/
│   ├── adaptive_lms.py              # CPU Normalized Least Mean Squares (NLMS) filter
│   ├── causal_anc_net.py            # Causal Complex Recurrent Network (C-CRN)
│   └── loss.py                      # Multi-resolution STFT + SI-SNR + Complex L1 loss
├── train.py                         # Full training engine (Mixed Precision AMP)
├── train_quickstart.py              # Rapid training recipe for RTX 5060
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Launch Live Interactive Cockpit GUI
```bash
streamlit run demo/app.py
```
*Open your browser at `http://localhost:8501` to test live audio processing, toggle noise injection (helicopter, gunfire, jet, cockpit), and switch hardware backends in real time.*

### 2. Run Cross-Hardware Latency Benchmark
```bash
python benchmark/benchmark_suite.py
```

### 3. Run DRDO Evaluation Harness
```bash
python eval/evaluate_drdo_targets.py
```

### 4. Train Model on RTX 5060 (CUDA)
```bash
python train_quickstart.py
```

### 5. Export to ONNX & Quantize to OpenVINO INT8
```bash
python export/export_onnx.py
python export/quantize_openvino.py
```

---

## 🎖️ Key Highlights for DRDO Judges
- **Phase Preservation**: Uses Complex Ratio Masking (cRM) to prevent speech distortion.
- **Frame-causal network, ~16 ms system lookahead**: the network has 0.00e+00 future-*frame* leakage (reproducible via `scripts/profile_tcn_latency_and_causality.py`), but the centred STFT front-end gives the end-to-end system ~16 ms of waveform-level lookahead (measured 20-29 ms including overlap-add span). The earlier "0.0 ms" claim measured frame causality only and has been corrected.
- **Edge Deployable**: Quantized to 1.46 MB INT8 OpenVINO IR (verified on disk), running at 0.093 ms/frame on Intel Core Ultra 9 CPU via the OpenVINO runtime. NPU offload was attempted and does **not** currently work — see the hardware table above.
- **Every number traceable**: all reported metrics trace to raw per-utterance CSVs in `eval/test_eval_csvs/` and are re-derivable; see [eval/PHASE13_CLAIM_AUDIT.md](eval/PHASE13_CLAIM_AUDIT.md).
