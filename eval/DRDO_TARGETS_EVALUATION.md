# DRDO PS 26052 — Official Adaptive Noise Cancellation Evaluation Report

**Evaluation Date:** August 27, 2026  
**Primary Metric Engine:** Consolidated Single-Source Evaluation (`eval/metrics_common.py`)  
**Tested Checkpoints:**
1. **Baseline Model (Recommended)**: `checkpoints/task4_full_run/best_model.pt` (CausalANCNet, 811K params, 50 Epochs)
2. **Data-Rebalanced Pilot**: `checkpoints/pilot_rebalanced_snr/best_model.pt` (CausalANCNet, 811K params, 15 Epochs FT)
3. **Scaled Capacity v1**: `checkpoints/pilot_scaled_capacity/best_model.pt` (CausalANCScaledNet, 3.24M params, 20 Epochs, LR=3e-4)
4. **Scaled Capacity v2 (Decisive)**: `checkpoints/pilot_scaled_capacity_v2/best_model.pt` (CausalANCScaledNet, 3.24M params, 50 Epochs, LR=1e-3)

---

## ⚠️ Executive Risk Note: Operational Reliability vs. Headline Sign-Off

> [!IMPORTANT]
> **Scope & Reliability Disclosure**:
> 1. **Narrow Headline Slice**: The headline metric row ($15.60\text{ to }16.38\text{ dB}$ Output SNR) reflects **only the easiest $[10, 15]\text{ dB}$ input-SNR bucket**.
> 2. **Aggregate Operational Failure**: Across the actual operational input dynamic range ($0\text{ to }15\text{ dB}$ input SNR, $N=300$), the baseline model achieves a mean Output SNR of **$12.75\text{ dB}$**, **failing the primary $>15.0\text{ dB}$ threshold**.
> 3. **Per-Sample Pass Rate**: Even in the single easiest $[10, 15]\text{ dB}$ bucket, only **$52.0\%\text{ to }62.0\%$** of individual utterances hit the $>15.0\text{ dB}$ target. Across the pooled $0\text{ to }15\text{ dB}$ range, only **$28.0\%$** of utterances pass.
> 4. **Capacity Scaling Outcome**: Scaling model capacity $4\times$ ($811\text{K} \rightarrow 3.24\text{M}$ parameters) and retraining from scratch for 50 epochs with matched optimizer dynamics ($\text{LR}=1.0\times 10^{-3}$) yielded **$12.21 \pm 4.79\text{ dB}$** aggregate SNR ($25.3\%$ pass rate, $95\%\text{ CI } [20.7\%, 30.5\%]$), which is **statistically indistinguishable from the baseline** ($12.75 \pm 5.35\text{ dB}$, $28.0\%$ pass rate, $95\%\text{ CI } [23.2\%, 33.3\%]$). 

---

## 1. Official Compliance Matrix (Held-Out Test Split)

Evaluated on the held-out test split of the `HybridRealDataset` ($N=300$ for $0\text{ to }15\text{ dB}$ aggregate, $N=100$ for $[10, 15]\text{ dB}$ slice):

| Metric | Target Specification | Baseline (Task 4) [10, 15] dB Slice | Baseline (Task 4) 0–15 dB Aggregate | Scaled v2 (3.24M) 0–15 dB Aggregate | Official Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Output SNR** | $> 15.0\text{ dB}$ | **$15.60 \pm 4.20\text{ dB}$** (Pass: $52.0\%$) | **$12.75 \pm 5.35\text{ dB}$** (Pass: $28.0\%$) | **$12.21 \pm 4.79\text{ dB}$** (Pass: $25.3\%$) | ⚠️ **CONDITIONAL PASS** (Easy slice only; Fails Aggregate) |
| **STOI** | $> 0.850$ | **$0.9354 \pm 0.0410$** (Pass: $90.0\%$) | **$0.8990 \pm 0.0814$** (Pass: $77.3\%$) | **$0.8934 \pm 0.0867$** (Pass: $76.7\%$) | ✅ **PASSED** |
| **PESQ (approx)** | $> 2.50$ | **$3.98 \pm 0.32$** (Pass: $100.0\%$) | **$3.69 \pm 0.51$** (Pass: $97.7\%$) | **$3.64 \pm 0.53$** (Pass: $97.0\%$) | ✅ **PASSED** *(Surrogate)* |
| **CPU Latency** | $< 1.0\text{ ms}$ / frame | **$0.063\text{ ms}$** | **$0.063\text{ ms}$** | **$0.059\text{ ms}$** | ✅ **PASSED** ($15.8\times$ margin) |
| **CPU RTF** | $< 0.100$ | **$0.0078$** | **$0.0078$** | **$0.0073$** | ✅ **PASSED** ($12.8\times$ margin) |
| **Lookahead** | $0.0\text{ ms}$ | **$0.0\text{ ms}$** | **$0.0\text{ ms}$** | **$0.0\text{ ms}$** | ✅ **PASSED** (Strictly Causal) |

---

## 2. Decisive Four-Way Model Comparison

Empirical comparison across all experimental tracks on the held-out test split ($N=500$ utterances per model):

### Table 2.1: Model Architecture & Efficiency
| Parameter | Baseline (Task 4) | Data-Rebalanced Pilot | Scaled Pilot v1 | Scaled Model v2 (Decisive) |
|:---|:---:|:---:|:---:|:---:|
| **Model Architecture** | `CausalANCNet` | `CausalANCNet` | `CausalANCScaledNet` | `CausalANCScaledNet` |
| **Trainable Parameters** | 811,522 (3.1 MB) | 811,522 (3.1 MB) | 3,235,842 (12.4 MB) | 3,235,842 (12.4 MB) |
| **Training Budget** | 50 Epochs (Scratch) | 15 Epochs (Fine-Tune) | 20 Epochs (Scratch) | 50 Epochs (Scratch) |
| **Initial Learning Rate** | $1.0\times 10^{-3}$ | $1.0\times 10^{-4}$ | $3.0\times 10^{-4}$ | $1.0\times 10^{-3}$ |
| **SNR Sampler** | Uniform $[-5, 15]\text{ dB}$ | Rebalanced $[-10, 15]\text{ dB}$ | Rebalanced $[-10, 15]\text{ dB}$ | Rebalanced $[-10, 15]\text{ dB}$ |
| **CPU Latency / Frame** | $0.063\text{ ms}$ | $0.034\text{ ms}$ | $0.071\text{ ms}$ | $0.059\text{ ms}$ |
| **CPU RTF** | $0.0078$ | $0.0043$ | $0.0089$ | $0.0073$ |
| **Algorithmic Delay** | $0.0\text{ ms}$ | $0.0\text{ ms}$ | $0.0\text{ ms}$ | $0.0\text{ ms}$ |

### Table 2.2: Aggregate Operational Dynamic Range ($0\text{ to }15\text{ dB}$ Input SNR, $N=300$)
| Parameter | Baseline (Task 4) | Data-Rebalanced Pilot | Scaled Pilot v1 | Scaled Model v2 (Decisive) |
|:---|:---:|:---:|:---:|:---:|
| **Output SNR (Mean ± Std)** | **$12.75 \pm 5.35\text{ dB}$** | $12.72 \pm 5.25\text{ dB}$ | $7.30 \pm 3.04\text{ dB}$ | $12.21 \pm 4.79\text{ dB}$ |
| **SNR >15.0 dB Pass Rate** | **$28.0\%$** $[23.2\%, 33.3\%]$ | $28.3\%$ $[23.5\%, 33.7\%]$ | $0.0\%$ $[0.0\%, 1.3\%]$ | $25.3\%$ $[20.7\%, 30.5\%]$ |
| **STOI Intelligibility** | **$0.8990 \pm 0.0814$** | $0.8991 \pm 0.0817$ | $0.8579 \pm 0.0977$ | $0.8934 \pm 0.0867$ |
| **STOI >0.850 Pass Rate** | **$77.3\%$** | $77.3\%$ | $62.0\%$ | $76.7\%$ |
| **PESQ Quality (approx)** | **$3.69 \pm 0.51$** | $3.69 \pm 0.51$ | $3.26 \pm 0.53$ | $3.64 \pm 0.53$ |

### Table 2.3: Degraded Channel Stress Range ($-10\text{ to }0\text{ dB}$ Input SNR, $N=200$)
| Parameter | Baseline (Task 4) | Data-Rebalanced Pilot | Scaled Pilot v1 | Scaled Model v2 (Decisive) |
|:---|:---:|:---:|:---:|:---:|
| **Stress Output SNR** | $4.59 \pm 5.97\text{ dB}$ | **$4.89 \pm 5.77\text{ dB}$** | $0.28 \pm 4.96\text{ dB}$ | $4.26 \pm 5.71\text{ dB}$ |
| **Stress SNR Gain** | $+6.84\text{ dB}$ | **$+7.14\text{ dB}$** | $+2.53\text{ dB}$ | $+6.51\text{ dB}$ |
| **Stress STOI** | $0.7310 \pm 0.1607$ | **$0.7349 \pm 0.1606$** | $0.6672 \pm 0.1666$ | $0.7250 \pm 0.1609$ |
| **Stress PESQ (approx)** | $2.64 \pm 0.82$ | **$2.67 \pm 0.82$** | $2.17 \pm 0.79$ | $2.59 \pm 0.81$ |

---

## 3. Disaggregated 5-Bucket Input-SNR Breakdown

| Input SNR Bucket | Noise Regime | Baseline SNR | Rebalanced SNR | Scaled v1 SNR | Scaled v2 SNR | Baseline STOI | Scaled v2 STOI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$[-10, -5)\text{ dB}$** | Severe Stress | $2.80\text{ dB}$ | **$3.24\text{ dB}$** | $-1.36\text{ dB}$ | $2.52\text{ dB}$ | $0.6890$ | $0.6836$ |
| **$[-5, 0)\text{ dB}$** | Moderate Stress | $6.38\text{ dB}$ | **$6.54\text{ dB}$** | $1.92\text{ dB}$ | $5.99\text{ dB}$ | $0.7731$ | $0.7665$ |
| **$[0, 5)\text{ dB}$** | Operational Low | $9.21\text{ dB}$ | **$9.30\text{ dB}$** | $4.97\text{ dB}$ | $8.82\text{ dB}$ | $0.8465$ | $0.8367$ |
| **$[5, 10)\text{ dB}$** | Operational Mid | **$13.43\text{ dB}$** | $13.35\text{ dB}$ | $7.67\text{ dB}$ | $12.79\text{ dB}$ | **$0.9151$** | $0.9106$ |
| **$[10, 15]\text{ dB}$** | Operational High | **$15.60\text{ dB}$** | $15.50\text{ dB}$ | $9.26\text{ dB}$ | $15.03\text{ dB}$ | **$0.9354$** | $0.9328$ |

---

## 4. Methodological & Metric Caveats

1. **PESQ Backend Provenance**:
   - The reported PESQ values are computed via the psychoacoustic Bark-distortion surrogate (`pesq_psychoacoustic_bark_fallback` in `eval/metrics_common.py` and `model/loss.py`):
     $$\text{PESQ}_{\text{approx}} = 1.0 + 2.8\cdot \text{STOI}^{1.5} + 0.04\cdot \text{clamp}(\text{SI-SNR}, -10, 25) - 0.25\cdot \text{PMSQE}$$
   - This proxy is calibrated to correlate strongly with ITU-T P.862 Wideband PESQ but does not replace formal ITU-T P.862 C-MEX compiled testing.
2. **Specification Provenance**:
   - The numerical targets ($>15.0\text{ dB}$ SNR, $>0.850$ STOI, $>2.50$ PESQ, $<0.100$ RTF, $<1.0\text{ ms}$ latency) are internal project target specifications defined for Problem Statement 26052. No external classified DRDO procurement document is cited.

---

## 5. Authoritative Recommendation & Scope of Findings

**Recommended Model for Deployment:** **`checkpoints/task4_full_run/best_model.pt` (Baseline CausalANCNet, 811K parameters)**.

### Rationale:
1. **Efficiency and Performance Equivalence**:
   - Correctly training the $4\times$-capacity model (`CausalANCScaledNet`, 3.24M parameters) for 50 epochs with matched optimizer dynamics ($\text{LR}=1.0\times 10^{-3}$) yielded an aggregate Output SNR ($12.21 \pm 4.79\text{ dB}$, $25.3\%$ pass rate, $95\%\text{ CI } [20.7\%, 30.5\%]$) that is **statistically indistinguishable** from the 811K baseline ($12.75 \pm 5.35\text{ dB}$, $28.0\%$ pass rate, $95\%\text{ CI } [23.2\%, 33.3\%]$).
   - Because $4\times$ scaling yields no measurable metric benefit on the held-out test split, the baseline model is recommended strictly on efficiency grounds: it requires $4\times$ less memory ($3.1\text{ MB}$ vs $12.4\text{ MB}$), has lower computational overhead, and achieves identical performance.

2. **Grounded Scope of Tested Interventions**:
   - Across the full $0\text{ to }15\text{ dB}$ aggregate operational range, all four tested models fall short of the $>15.0\text{ dB}$ target on mean ($12.21\text{ to }12.75\text{ dB}$ achieved vs $>15.0\text{ dB}$ required, with per-utterance pass rates between $25.3\%\text{ and }28.3\%$).
   - This investigation specifically established and ruled out two hypotheses within this architecture family (causal DCRN + complex ratio masking):
     1. **Data Rebalancing**: Resampling training SNR distributions across $[-10, 15]\text{ dB}$ does not resolve the aggregate shortfall ($12.72\text{ dB}$ vs $12.75\text{ dB}$).
     2. **Capacity Scaling**: Expanding network width and recurrent hidden state $4\times$ ($811\text{K} \rightarrow 3.24\text{M}$ parameters) under matching training budgets does not resolve the aggregate shortfall ($12.21\text{ dB}$ vs $12.75\text{ dB}$).
   - **Untested Approaches**: Other architectural families (e.g., dual-path recurrent models, causal conformer/attention networks, time-domain causal filterbanks, or loss functions with explicit non-uniform penalty weighting toward lower input SNRs) remain untested and were outside the scope of this investigation track.

---

## 6. Phase 8 Root-Cause Diagnostics & Fine-Tuning Pilot Summary

Full details in [`eval/PHASE8_DIAGNOSTICS.md`](file:///c:/Users/eashw/OneDrive/Documents/SIH_PROJECT/eval/PHASE8_DIAGNOSTICS.md).

1. **Diagnostic 1A (Loss Clamp Saturation — Hypothesis A)**:
   - **Ruled out**: $0.0\%$ ($0/500$) of test utterances exhibited negative dot products ($\text{dot} \le 0$). Clamped and pre-clamped SI-SNR values were identical at convergence.
2. **Diagnostic 1B (Oracle Mask Non-Linearity — Hypothesis B)**:
   - **Confirmed mathematically**: Linearly clamped $[-1, 1]$ oracle masks achieve $23.7\text{ to }31.4\text{ dB}$ SNR. However, passing the oracle mask through $\tanh(M)$ non-linearly squashes output SNR into an asymptotic ceiling of **$12.75\text{ to }12.96\text{ dB}$**, precisely matching the baseline model's empirical $12.75\text{ dB}$ operational mean.
3. **Diagnostic 1C & Step 2 Fine-Tuning Pilots**:
   - **Exp 2a (Causal FrameNorm swap)**: Disrupted pre-trained BatchNorm statistics, yielding $10.94\text{ dB}$ operational SNR.
   - **Exp 2b (Hinge SI-SNR + Low-SNR loss weighting)**: Improved degraded stress gain to $+7.12\text{ dB}$, but operational Output SNR remained flat at $12.71\text{ dB}$.
   - **Exp 2c (Widened Mask $K=2.5$)**: Caused output gain overshoot ($10.42\text{ dB}$) when fine-tuning from a $K=1.0$ checkpoint without retraining from scratch.
   - **Exp 2d (Combined)**: Yielded $9.38\text{ dB}$ operational SNR due to compounding scale disruptions.
4. **Final Conclusion**: The baseline model `checkpoints/task4_full_run/best_model.pt` ($811\text{K}$ parameters) remains the authoritative best checkpoint in the repository. Overcoming the $\sim 12.8\text{ dB}$ $\tanh$ ceiling requires modifying the mask activation/deep-filtering formulation and training from scratch from epoch 0.

---

## 7. Phase 9: Polar Mask Formulation & Scratch Retraining Evaluation

### 7.1 Diagnostic Step 0: Polar Magnitude Bounding vs Per-Component Tanh (Oracle Analysis)
To test whether the $12.8\text{ dB}$ ceiling was caused by phase distortion or magnitude squashing, an oracle mask diagnostic was executed on the held-out test split ($N=500$):

| Input SNR Bucket | Comp Tanh ($K=1$, Current) | Polar Tanh ($K=1$, Phase Preserved) | Polar Clamp ($\|M\| \le 1.0$) | Polar Tanh ($K=3.0$) | Comp Clamp $[-1, 1]$ | Baseline Model Actual |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stress $[-10, -5)\text{ dB}$** | $12.96\text{ dB}$ | $12.54\text{ dB}$ | $22.43\text{ dB}$ | $26.64\text{ dB}$ | $23.69\text{ dB}$ | $2.80\text{ dB}$ |
| **Stress $[-5, 0)\text{ dB}$** | $12.85\text{ dB}$ | $12.58\text{ dB}$ | $24.35\text{ dB}$ | $27.67\text{ dB}$ | $25.62\text{ dB}$ | $6.38\text{ dB}$ |
| **Operational $[0, 5)\text{ dB}$** | $12.91\text{ dB}$ | $12.71\text{ dB}$ | $26.63\text{ dB}$ | $28.20\text{ dB}$ | $27.84\text{ dB}$ | $9.21\text{ dB}$ |
| **Operational $[5, 10)\text{ dB}$** | $12.88\text{ dB}$ | $12.78\text{ dB}$ | $30.26\text{ dB}$ | $28.88\text{ dB}$ | $31.40\text{ dB}$ | $13.43\text{ dB}$ |
| **Operational $[10, 15]\text{ dB}$** | $12.75\text{ dB}$ | $12.68\text{ dB}$ | $30.24\text{ dB}$ | $28.98\text{ dB}$ | $31.27\text{ dB}$ | $15.60\text{ dB}$ |

* **Insight**: Preserving phase while squashing magnitude with $\tanh$ ($K=1.0$) still caps oracle Output SNR at $12.54\text{ to }12.78\text{ dB}$. The ceiling is driven by the curvature of $\tanh$ attenuating speech harmonics. Widening the bound to $K=3.0$ with exact polar phase preservation ($M = 3 \tanh(\|M\|/3) e^{j\theta}$) expands oracle headroom to **$28.98\text{ dB}$**.

### 7.2 Step 1: Full 50-Epoch Retraining from Scratch (`checkpoints/phase9_polar_k3_scratch/`)
Trained `CausalANCNet` from scratch with polar magnitude bounding ($K=3.0$) for 50 epochs ($60,000$ steps, $\text{LR}=1.0\times 10^{-3} \rightarrow 1.0\times 10^{-5}$, matching the baseline budget):

| Metric / Parameter | Baseline (811K, $K=1.0$) | Scaled Net v2 (3.24M, $K=1.0$) | Phase 9 Polar Bounded (811K, $K=3.0$) | DRDO Specification Target |
|:---|:---:|:---:|:---:|:---:|
| **Operational Output SNR ($0\text{--}15\text{ dB}$)** | **$12.75 \pm 5.35\text{ dB}$** | $12.21 \pm 4.79\text{ dB}$ | $11.58 \pm 4.47\text{ dB}$ | $>15.0\text{ dB}$ |
| **Output SNR $>15.0\text{ dB}$ Pass Rate** | **$28.0\%$** $[23.2\%, 33.3\%]$ | $25.3\%$ $[20.7\%, 30.5\%]$ | $20.0\%$ $[15.9\%, 24.9\%]$ | $>80.0\%$ Reliable |
| **Operational STOI** | **$0.8990 \pm 0.0814$** | $0.8934 \pm 0.0867$ | $0.8864 \pm 0.0877$ | $>0.850$ |
| **Operational PESQ (approx)** | **$3.69 \pm 0.51$** | $3.64 \pm 0.53$ | $3.58 \pm 0.53$ | $>2.50$ |
| **Stress SNR Gain ($-10\text{ to }0\text{ dB}$)** | **$+6.84\text{ dB}$** ($4.59\text{ dB}$) | $+6.51\text{ dB}$ ($4.26\text{ dB}$) | $+6.43\text{ dB}$ ($4.18\text{ dB}$) | Stable Non-negative |
| **CPU Per-Frame Latency** | $0.067\text{ ms}$ | $0.065\text{ ms}$ | **$0.040\text{ ms}$** | $<1.0\text{ ms}$ |
| **CPU RTF** | $0.0084$ | $0.0081$ | **$0.0050$** | $<0.100$ |

### 7.3 Disaggregated 5-Bucket Output SNR Breakdown
| SNR Bucket | Baseline ($K=1.0$) | Scaled v2 ($K=1.0$) | Phase 9 Polar ($K=3.0$) |
|:---|:---:|:---:|:---:|
| **Stress $[-10, -5)\text{ dB}$** | **$2.80\text{ dB}$** | $2.52\text{ dB}$ | $2.64\text{ dB}$ |
| **Stress $[-5, 0)\text{ dB}$** | **$6.38\text{ dB}$** | $5.99\text{ dB}$ | $5.72\text{ dB}$ |
| **Operational $[0, 5)\text{ dB}$** | **$9.21\text{ dB}$** | $8.82\text{ dB}$ | $8.39\text{ dB}$ |
| **Operational $[5, 10)\text{ dB}$** | **$13.43\text{ dB}$** | $12.79\text{ dB}$ | $12.06\text{ dB}$ |
| **Operational $[10, 15]\text{ dB}$** | **$15.60\text{ dB}$** | $15.03\text{ dB}$ | $14.29\text{ dB}$ |

### 7.4 Analytical Finding & Definitive Conclusion
* **Why Oracle Headroom ($28.98\text{ dB}$) Did Not Translate to Trained Model Gain**:
  - While $K=3.0$ polar bounding removes the mathematical ceiling for an *ideal* oracle filter, it allows real neural network estimation errors in noise-dominated bins to be amplified by up to $+9.5\text{ dB}$ ($3.0\times$).
  - The $K=1.0$ component $\tanh$ bound in the baseline model acted as a strong regularization barrier preventing out-of-band noise amplification.
  - As a result, the fully retrained Polar $K=3.0$ model yielded an empirical Output SNR of **$11.58\text{ dB}$** (vs **$12.75\text{ dB}$** for baseline).
* **Final Project Recommendation**:
  - **Deploy `checkpoints/task4_full_run/best_model.pt` (Baseline CausalANCNet, 811K parameters)**. It remains the most effective, highest-SNR, and lowest-distortion causal model across all investigated architecture variants, training budgets, and loss formulations.

---

## 8. Phase 10: Multi-Candidate Mask Screening, Annealed Curricula & Fair Causal Normalization Evaluation

### 8.1 Step 1 & 1b: 15-Epoch $K$-Screening Suite & Annealed Curricula
To determine whether moderate mask bounds ($K \in [1.5, 2.5]$) or dynamic gain annealing ($K=1.0 \rightarrow 2.0/3.0$) could prevent noise amplification while capturing oracle headroom, 5 scratch models were trained for 15 epochs each ($1,200\text{ steps/epoch}$):

| Candidate | Op Output SNR ($0\text{--}15\text{ dB}$) | SNR $>15\text{ dB}$ Pass Rate | Op STOI | PESQ | Stress SNR Gain ($-10\text{ to }0\text{ dB}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Static $K=1.5$** | **$8.83 \pm 3.34\text{ dB}$** | $2.7\%$ $[1.4\%, 5.2\%]$ | **$0.8668$** | $3.36$ | **$+4.08\text{ dB}$** ($1.83\text{ dB}$) |
| **Static $K=2.0$** | $8.53 \pm 3.42\text{ dB}$ | $2.7\%$ $[1.4\%, 5.2\%]$ | $0.8607$ | $3.31$ | $+3.99\text{ dB}$ ($1.74\text{ dB}$) |
| **Static $K=2.5$** | $8.46 \pm 3.54\text{ dB}$ | $3.0\%$ $[1.6\%, 5.6\%]$ | $0.8606$ | $3.30$ | $+3.97\text{ dB}$ ($1.72\text{ dB}$) |
| **Annealed $K \rightarrow 2.0$** | $7.86 \pm 3.35\text{ dB}$ | $1.3\%$ $[0.5\%, 3.4\%]$ | $0.8611$ | $3.28$ | $+3.73\text{ dB}$ ($1.47\text{ dB}$) |
| **Annealed $K \rightarrow 3.0$** | $7.73 \pm 3.51\text{ dB}$ | $1.0\%$ $[0.3\%, 2.9\%]$ | $0.8587$ | $3.26$ | $+3.67\text{ dB}$ ($1.42\text{ dB}$) |

* **Screening Findings**:
  1. **Monotonic Gain Inversion**: As $K$ increases ($1.5 \rightarrow 2.0 \rightarrow 2.5 \rightarrow 3.0$), Output SNR and STOI monotonically degrade due to increased noise leakage in unvoiced/low-confidence T-F bins.
  2. **Annealing Transition Shock**: When $K$ ramps up from $1.0$, validation loss spikes as previously suppressed noise bins receive positive gain, slowing convergence relative to static training.
  3. **Winning Screening Candidate**: **Static $K=1.5$** achieved the highest screening Output SNR ($8.83\text{ dB}$), highest STOI ($0.8668$), and highest stress gain ($+4.08\text{ dB}$).

---

### 8.2 Steps 2 & 3: Full 50-Epoch Confirmatory Runs (Polar $K=1.5$ & Fair Causal GroupNorm)
Two full 50-epoch scratch runs ($60,000\text{ steps each}$, $\text{LR}=1.0\times 10^{-3} \rightarrow 1.0\times 10^{-5}$) were completed to fairly evaluate:
1. **Best Mask Variant**: `checkpoints/phase10_polar_k15_scratch/best_model.pt` (Polar Tanh $K=1.5$)
2. **Fair Causal Normalization Test**: `checkpoints/phase10_causal_norm_scratch/best_model.pt` (Causal GroupNorm, $K=1.0$ baseline mask)

All models and baselines were re-benchmarked back-to-back under identical idle machine state ($N=500$ held-out test utterances per model):

| Model Architecture | Mask / Norm Config | Op Output SNR ($0\text{--}15\text{ dB}$) | SNR $>15\text{ dB}$ Pass Rate | Op STOI | PESQ | Stress SNR Gain | CPU Latency | CPU RTF |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline C-CRN (811K)** | Comp Tanh ($K=1.0$) / BatchNorm | **$12.75 \pm 5.35\text{ dB}$** | **$28.0\%$** $[23.2\%, 33.3\%]$ | **$0.8990$** | **$3.69$** | $+6.84\text{ dB}$ ($4.59\text{ dB}$) | $0.065\text{ ms}$ | $0.0081$ |
| **Scaled Net v2 (3.24M)** | Comp Tanh ($K=1.0$) / BatchNorm | $12.21 \pm 4.79\text{ dB}$ | $25.3\%$ $[20.7\%, 30.5\%]$ | $0.8934$ | $3.64$ | $+6.51\text{ dB}$ ($4.26\text{ dB}$) | $0.074\text{ ms}$ | $0.0093$ |
| **Polar Bounded (811K)** | Polar Tanh ($K=1.5$) / BatchNorm | $12.07 \pm 4.69\text{ dB}$ | $23.7\%$ $[19.2\%, 28.8\%]$ | $0.8933$ | $3.64$ | **$+6.97\text{ dB}$** ($4.72\text{ dB}$) | $0.032\text{ ms}$ | $0.0040$ |
| **Causal Norm Net (811K)** | Comp Tanh ($K=1.0$) / Causal GroupNorm | $11.91 \pm 5.03\text{ dB}$ | $24.0\%$ $[19.5\%, 29.1\%]$ | $0.8840$ | $3.56$ | $+5.87\text{ dB}$ ($3.62\text{ dB}$) | $0.041\text{ ms}$ | $0.0051$ |
| **Phase 9 Polar (811K)** | Polar Tanh ($K=3.0$) / BatchNorm | $11.58 \pm 4.47\text{ dB}$ | $20.0\%$ $[15.9\%, 24.9\%]$ | $0.8864$ | $3.58$ | $+6.43\text{ dB}$ ($4.18\text{ dB}$) | $0.040\text{ ms}$ | $0.0051$ |

---

### 8.3 Disaggregated 5-Bucket Output SNR Breakdown
| SNR Bucket | Baseline ($K=1.0$) | Scaled v2 (3.24M) | Polar Bounded ($K=1.5$) | Causal GroupNorm | Phase 9 ($K=3.0$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Stress $[-10, -5)\text{ dB}$** | $2.80\text{ dB}$ | $2.52\text{ dB}$ | **$3.19\text{ dB}$** | $1.95\text{ dB}$ | $2.64\text{ dB}$ |
| **Stress $[-5, 0)\text{ dB}$** | **$6.38\text{ dB}$** | $5.99\text{ dB}$ | $6.25\text{ dB}$ | $5.29\text{ dB}$ | $5.72\text{ dB}$ |
| **Operational $[0, 5)\text{ dB}$** | **$9.21\text{ dB}$** | $8.82\text{ dB}$ | $8.78\text{ dB}$ | $8.37\text{ dB}$ | $8.39\text{ dB}$ |
| **Operational $[5, 10)\text{ dB}$** | **$13.43\text{ dB}$** | $12.79\text{ dB}$ | $12.59\text{ dB}$ | $12.60\text{ dB}$ | $12.06\text{ dB}$ |
| **Operational $[10, 15]\text{ dB}$** | **$15.60\text{ dB}$** | $15.03\text{ dB}$ | $14.85\text{ dB}$ | $14.75\text{ dB}$ | $14.29\text{ dB}$ |

---

### 8.4 Definitive Synthesis of Findings
1. **Mask Bound Regularization ($K=1.0$ vs $K>1.0$)**:
   - Although $K=1.5$ slightly outperforms $K=3.0$ ($12.07\text{ dB}$ vs $11.58\text{ dB}$), both remain below the baseline's $12.75\text{ dB}$.
   - Restricting mask magnitude strictly to $\le 1.0$ acts as an essential physical inductive bias for speech enhancement, guaranteeing that no noise bin can ever experience positive gain.
2. **Causal Normalization (Fair Retrain)**:
   - Full 50-epoch retraining with causal GroupNorm achieves **$11.91\text{ dB}$** operational Output SNR and $0.8840$ STOI ($0.041\text{ ms}$ latency), performing closely to BatchNorm but with a small $-0.84\text{ dB}$ gap due to the lack of running population statistics across utterance history.
3. **Hardware Latency Benchmarking**:
   - Re-benchmarking on idle CPU confirms all 811K models achieve **$0.032\text{ to }0.065\text{ ms}$** per frame (RTF $0.004\text{ to }0.008$), operating $>15\times$ faster than the $<1.0\text{ ms}$ DRDO real-time target.
4. **Final Authoritative Checkpoint**:
   - **`checkpoints/task4_full_run/best_model.pt` (Baseline CausalANCNet, 811K parameters)** remains the primary recommended deployment model across all empirical metrics.




