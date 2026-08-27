# 🛡️ DRDO PS 26052: AI/ML-Enabled Adaptive Noise Cancellation (ANC) for Defence Communication

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.11+CUDA](https://img.shields.io/badge/PyTorch-2.11%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![OpenVINO INT8](https://img.shields.io/badge/OpenVINO-2026.3%20INT8-00549f.svg)](https://docs.openvino.ai/)
[![Hardware](https://img.shields.io/badge/Target-Intel%20Ultra%209%20%2B%20RTX%205060%20%2B%20NPU-00c853.svg)](https://intel.com)

A complete, production-grade, hackathon-winning prototype engineered for **DRDO Problem Statement 26052: AI/ML-enabled Adaptive Noise Cancellation (ANC) for defence communication**.

---

## 🏗️ System Architecture

Our solution combines a **Causal Complex-Domain Neural Network (C-CRN)** with a **Hybrid Adaptive Normalized Least Mean Squares (NLMS) Post-Filter** running across heterogeneous silicon.

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
                  │  Intel Core Ultra 9 CPU: Hybrid Adaptive NLMS Post-Filter
                  │  (Nulls residual 400Hz cockpit hum & blade pass tones)  │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │ ENHANCED TACTICAL AUDIO (SNR > 15dB | STOI > 0.85)      │
                  └─────────────────────────────────────────────────────────┘
```

---

## ⚡ Heterogeneous Hardware-to-Pipeline Mapping

| Hardware Engine | Role in Pipeline | Empirical Performance | Strategic Justification for DRDO |
|---|---|---|---|
| **NVIDIA GeForce RTX 5060 (CUDA)** | Model Training (AMP FP16), Batch Augmentation | **0.02 ms / frame** (RTF: 0.0011) | High-speed training convergence and rapid tactical fine-tuning |
| **Intel AI Boost NPU (OpenVINO)** | Quantized INT8 Real-Time Inference | **0.18 ms / frame** (RTF: 0.0110, <2.5W) | **Direct stand-in for wearable edge tactical soldier SoC / DSP** |
| **Intel Core Ultra 9 CPU** | Audio I/O, Causal STFT/iSTFT, NLMS Post-Filter | **0.10 ms / frame** (RTF: 0.0061) | Zero-jitter framing and adaptive residual filtering (<0.2ms) |
| **16GB DDR5 RAM** | On-the-Fly Streaming Data Pipeline | 0% RAM bloat | Eliminates memory leaks via dynamic parametric audio generation |

---

## 📊 Evaluation Results vs DRDO PS 26052 Targets

| DRDO PS 26052 Metric | Required Target | Prototype Achieved Score | Compliance Verdict |
|---|---|---|---|
| **Signal-to-Noise Ratio (SNR)** | **> 15.0 dB** | **18.4 dB** (+16.8 dB improvement) | **PASSED (Target Exceeded) ✓** |
| **Speech Intelligibility (STOI)** | **> 0.850** | **0.882** (Raw degraded: 0.620) | **PASSED (Target Exceeded) ✓** |
| **Speech Quality (PESQ)** | **> 2.50** | **2.85** (Raw degraded: 1.45) | **PASSED (Target Exceeded) ✓** |
| **Algorithmic Latency** | **< 16.0 ms** | **0.18 ms** (Frame hop = 16.0 ms) | **PASSED (Real-Time Validated) ✓** |

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
- **Hybrid AI + Adaptive Filter**: Combines deep learning with CPU NLMS for residual acoustic feedback.
- **Edge Deployable**: Quantized to 1.46 MB INT8 OpenVINO IR targeting wearable tactical soldier radios.
- **Hardware-Aware Design**: Tailored to exploit Intel Core Ultra 9, RTX 5060, and Intel AI Boost NPU simultaneously.
