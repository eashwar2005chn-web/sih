# DRDO PS 26052: Cross-Hardware Benchmark & Edge Optimization Report

## Hardware-to-Pipeline Empirical Mapping

| Hardware Target | Execution Runtime | Precision | Frame Latency (ms) | P95 Latency | Real-Time Factor (RTF) | Power Draw (Est) | Strategic Role in Defence ANC |
|---|---|---|---|---|---|---|---|
| **Intel Core Ultra 9 CPU** | PyTorch FP32 (Host Fallback) | `FP32` | **0.07 ms** | 0.08 ms | **0.0044** | `28W - 45W` | Audio I/O & Causal Framing |
| **NVIDIA RTX 5060 Laptop GPU** | PyTorch eager CUDA (AMP FP16) — **not** TensorRT | `FP16` | **0.02 ms** | 0.02 ms | **0.0011** | `60W - 90W` | Fast Model Training & High-Throughput Baseline |
| **Intel Core Ultra 9 CPU** | OpenVINO Runtime (Vector AVX2/VNNI) | `INT8 / FP16` | **0.09 ms** | 0.10 ms | **0.0057** | `15W - 25W` | Host Edge Acceleration |
| **Intel AI Boost NPU** | OpenVINO NPU Plugin — COMPILATION FAILED | `n/a` | **NOT MEASURED** | NOT MEASURED | **NOT MEASURED** | `n/a` | Unavailable: NPU compiler rejects the GRU core (GRUSequence sequenceLengths must be Constant) |

> **Runtime label correction (2026-08-30).** This row previously read "CUDA / TensorRT (AMP FP16)". TensorRT is **not** installed in this environment, no `.engine`/`.plan` artifact exists, and no TensorRT conversion is performed anywhere in the repository. The measurement is plain PyTorch eager execution under `torch.amp.autocast` (`benchmark_pytorch_device(model, "cuda", use_fp16=True)`). The latency figure itself is unchanged and reproducible — only the runtime attribution was wrong. TensorRT conversion remains an unimplemented item of the problem statement; see [`eval/PS_REQUIREMENTS_COMPLIANCE.md`](../eval/PS_REQUIREMENTS_COMPLIANCE.md).
---

## Key Hardware Findings for DRDO Judging Panel:
1. **Zero-Latency Jitter**: All *successfully benchmarked* targets achieve frame processing latency far below the 16 ms causal audio frame hop budget ($	ext{RTF} \ll 1.0$).
2. **Intel CPU INT8 as the edge path**: The 1.46 MB INT8 OpenVINO IR runs on the Intel Core Ultra 9 CPU via the OpenVINO runtime, which is the genuine low-footprint deployment story for PS 26052.
3. **RTX 5060 Training Acceleration**: The RTX 5060 GPU with mixed precision (AMP FP16) allows rapid model convergence.
4. **NPU status**: Intel AI Boost NPU offload is **not currently working** — the NPU compiler rejects the model's GRU core. Any row above marked `NOT MEASURED` is a real failure, not a placeholder to be filled in with an estimate.

> **Provenance rule for this report**: every number in the table above is emitted only from an actual timed run. If a target fails to compile or is unavailable, its row reads `NOT MEASURED`. Do not hand-edit projected values into this file.
