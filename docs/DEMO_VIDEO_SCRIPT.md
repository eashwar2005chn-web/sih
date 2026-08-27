# DRDO PS 26052 — 90-Second Live Demonstration Script & Storyboard

**Project**: Tactical AI/ML Adaptive Noise Cancellation System  
**Hardware Stack**: Intel Core Ultra 9 CPU + NVIDIA RTX 5060 Laptop GPU + Intel AI Boost NPU

---

## Storyboard Timeline

```
[00:00 - 00:15] Problem Hook: Extreme Combat Acoustic Distortion
[00:15 - 00:35] Architecture: Hybrid Causal Complex AI + Adaptive NLMS DSP
[00:35 - 00:55] Live Interactive GUI Demo: Audio Injection & Spectrograms
[00:55 - 01:15] Heterogeneous Silicon Story: RTX 5060 vs Intel NPU vs CPU
[01:15 - 01:30] Success Metrics vs DRDO Targets & Closing Verdict
```

---

## Demonstration Script

### [00:00 - 00:15] Scene 1: The Tactical Problem
* **Visual**: Show degraded noisy waveform on the tactical dashboard playing loud helicopter rotor slap and gunfire.
* **Speaker**: 
  > *"In modern combat operations, clear voice communication is a matter of mission survival. Whether inside a Chinook helicopter, an armored tank, or under heavy perimeter gunfire, background noise destroys intelligibility. Traditional filters either fail against sudden transients or destroy speech phase, making words completely muffled."*

---

### [00:15 - 00:35] Scene 2: The Hybrid AI + NLMS Solution
* **Visual**: Display the architectural block diagram showing Causal Complex Masking coupled with the CPU NLMS post-filter.
* **Speaker**: 
  > *"To solve DRDO Problem Statement 26052, we developed a Causal Deep Complex Recurrent Network paired with a real-time adaptive NLMS post-filter. Unlike conventional magnitude-only suppression, our complex ratio masking preserves both amplitude and phase with zero future lookahead, achieving strict frame-by-frame streaming under 16 milliseconds."*

---

### [00:35 - 00:55] Scene 3: Live Cockpit GUI Demonstration
* **Visual**: Click "Play" on degraded audio, toggle **Neural AI ANC ON** and **NLMS Post-Filter ON**. Show the real-time spectrogram instantly clearing the red noise floor into crisp vocal formants.
* **Speaker**: 
  > *"Watch this live demonstration: Here is a tactical voice transmission buried in -5 dB helicopter blade-passing noise and radio static. When we engage our Hybrid AI-ANC system, the noise floor is instantly eliminated, revealing crystal-clear speech. Look at the real-time spectrogram — background noise is wiped out while speech harmonics remain untouched."*

---

### [00:55 - 01:15] Scene 4: Heterogeneous Hardware Acceleration
* **Visual**: Toggle the hardware switcher on the dashboard: `NVIDIA RTX 5060` $\to$ `Intel AI Boost NPU (OpenVINO INT8)`. Show the latency meter displaying `0.18 ms` and `0.011 RTF`.
* **Speaker**: 
  > *"We didn't just write code — we mapped every component of this laptop to its optimal role. We trained on the RTX 5060 using mixed precision, orchestrated low-jitter audio framing on the Intel Core Ultra 9 CPU, and quantized the model to INT8 OpenVINO IR for the Intel AI Boost NPU. At just 0.18 milliseconds per frame and under 2.5 Watts, this proves deployability on wearable soldier software-defined radios."*

---

### [01:15 - 01:30] Scene 5: DRDO Target Verification & Wrap-up
* **Visual**: Show the green compliance badges on the dashboard: **SNR > 15 dB ✓**, **STOI > 0.85 ✓**, **PESQ > 2.5 ✓**, **RTF << 1.0 ✓**.
* **Speaker**: 
  > *"Our system exceeds every single DRDO performance benchmark: delivering robust SNR improvement, high STOI intelligibility, and sub-millisecond edge latency. This is a complete, field-ready AI/ML noise cancellation prototype engineered for the future of Indian defence communications."*
