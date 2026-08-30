# DRDO PS 26052 — 90-Second Live Demonstration Script & Storyboard

**Project**: Tactical AI/ML Adaptive Noise Cancellation System  
**Hardware Stack**: Intel Core Ultra 9 CPU (PyTorch FP32 + OpenVINO INT8) + NVIDIA RTX 5060 Laptop GPU (CUDA FP16). *One accelerator is NOT demonstrated: the Intel AI Boost NPU, whose compiler rejects the GRU core.*

---

## Storyboard Timeline

```
[00:00 - 00:15] Problem Hook: Extreme Combat Acoustic Distortion
[00:15 - 00:35] Architecture: Causal Complex-Domain AI (C-CRN)
[00:35 - 00:55] Live Interactive GUI Demo: Audio Injection & Spectrograms
[00:55 - 01:15] Silicon Story: RTX 5060 vs Intel CPU, FP32 vs quantized INT8
[01:15 - 01:30] Honest Metrics vs Targets & Closing Verdict
```

---

## Demonstration Script

### [00:00 - 00:15] Scene 1: The Tactical Problem
* **Visual**: Show degraded noisy waveform on the tactical dashboard playing loud helicopter rotor slap and gunfire.
* **Speaker**: 
  > *"In modern combat operations, clear voice communication is a matter of mission survival. Whether inside a Chinook helicopter, an armored tank, or under heavy perimeter gunfire, background noise destroys intelligibility. Traditional filters either fail against sudden transients or destroy speech phase, making words completely muffled."*

---

### [00:15 - 00:35] Scene 2: The Causal Complex-Domain Solution
* **Visual**: Display the architectural block diagram: causal STFT → C-CRN complex ratio mask → causal iSTFT.
* **Speaker**: 
  > *"To solve DRDO Problem Statement 26052, we developed a causal deep complex recurrent network. Unlike conventional magnitude-only suppression, our complex ratio masking preserves both amplitude and phase. The network itself is strictly frame-causal — we verified that empirically, not just architecturally — and the full system needs about 16 milliseconds of audio context from its analysis window, well inside a real-time budget."*

---

### [00:35 - 00:55] Scene 3: Live Cockpit GUI Demonstration
* **Visual**: Click "Play" on degraded audio with **Neural AI ANC ON** (NLMS post-filter toggle left at its default OFF). Show the real-time spectrogram clearing the noise floor while vocal formants remain intact.
* **Speaker**: 
  > *"Watch this live demonstration: here is a tactical voice transmission buried in helicopter blade-passing noise and radio static. When we engage the neural ANC model, the noise floor is measurably reduced and speech harmonics are preserved. Look at the real-time spectrogram, and the live SNR/STOI meters updating on real, computed metrics — not fixed numbers."*

---

### [00:55 - 01:15] Scene 4: Heterogeneous Hardware Acceleration
* **Visual**: Toggle the hardware switcher across `NVIDIA RTX 5060 (CUDA FP16)`, `Intel Core Ultra 9 CPU (PyTorch)` and `Intel Core Ultra 9 CPU (OpenVINO INT8)`. Show the live latency meter. **Do not select the NPU option on camera** — its compiler rejects the GRU core, so that path has no working measurement behind it.
* **Speaker**: 
  > *"We didn't just write code — we mapped every component of this laptop to its optimal role. We train on the RTX 5060 at two hundredths of a millisecond per frame, orchestrate low-jitter audio framing on the Intel Core Ultra 9 CPU, and quantized the model down to a 1.46-megabyte INT8 OpenVINO IR that runs on Intel silicon at under a tenth of a millisecond per frame. One accelerator we could not get working — the AI Boost NPU rejects our recurrent layer — and we'd rather tell you that than show you a number we can't reproduce."*

---

### [01:15 - 01:30] Scene 5: Honest Metrics & Wrap-up
* **Visual**: Show the live compliance badges on the dashboard for the current sample — some green, some not, because they're computed live rather than fixed. Cut to a summary slide: STOI 0.899 ✓, latency 10-14× inside budget ✓, network frame-causality verified ✓ (system lookahead ~16 ms, disclosed), Output SNR 12.75 dB aggregate (short of the 15 dB target; +7.65 dB measured improvement in the -10 to 0 dB stress range).
* **Speaker**: 
  > *"Five of our seven target metrics are met with substantial margin — intelligibility, latency, and network frame-causality, all verified live. Our output-SNR target isn't fully met across the whole input range yet, and we're not going to hide that: we ran a structured diagnostic program across eleven phases that pinpointed exactly why, and the system still delivers over seven and a half decibels of measured improvement in the harshest sub-zero-SNR conditions this problem statement targets. Every number we've shown you traces back to a per-utterance CSV in the repository. This is a complete, field-ready prototype, and we can tell you exactly what it does and doesn't do yet."*
