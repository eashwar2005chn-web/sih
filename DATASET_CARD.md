# DATASET_CARD.md — DRDO PS 26052 Hybrid Training Set (Task 3)

## Overview
Task 3 replaces the fully-synthetic tactical noise generator (used through Task 2G) with a **hybrid** mix of real clean speech, real environmental/tactical noise, and real room impulse responses, layered on top of the existing synthetic pipeline. Goal: close the remaining Output-SNR gap (6.86 dB vs. >15 dB target) by giving the model natural formant/phonetic variance the synthetic speech generator can't provide.

---

## Sources

### Clean speech
| Dataset | Role | Approx. size | Notes |
|---|---|---|---|
| **LibriSpeech** (OpenSLR/12) | Primary clean speech | ~6 GB (train-clean-100) | Formant, phonetic, and speaker vocal tract variance |
| **VoiceBank+DEMAND** | Secondary clean speech + paired reference | ~2.5–3 GB | Clean speech recordings across diverse accents |

### Generic / non-tactical noise
| Dataset | Role | Approx. size | Notes |
|---|---|---|---|
| **MUSAN** (OpenSLR/17) | Broadband technical noise, music, speech babble | ~1.5 GB | Background ambient sounds |
| **DEMAND** (via VoiceBank+DEMAND) | Everyday environmental real-world noise | (bundled above) | Multi-channel recorded noise environments |
| **ESC-50** | 50-class environmental sound events | ~600 MB | Download in full — small, well-labeled |
| **AudioSet** (filtered, non-tactical) | Fills background ambient gaps not covered by MUSAN/ESC-50 | ~0.5–1 GB filtered | See filtering procedure below |

### Tactical / defence noise
| Dataset | Role | Approx. size | Confirmed License | Licensing Verification & Usage Flags |
|---|---|---|---|---|
| **MAD** (Military Audio Dataset) | Core defence-noise source (~12 hrs) | ~1–2 GB | **CC BY 4.0** (Kim et al., 2024) | ✅ Approved for research & training with attribution (Kim et al.) |
| **Kaggle: Gunshot Audio Dataset** | Impulsive gunshot noise | ~500 MB–1 GB | **Unknown / Unspecified** (Aydemir et al.) | ⚠️ **RESTRICTION FLAG**: License is marked Unknown on Kaggle. Cite Tuncer et al. (2021). **Restricted to non-commercial evaluation; do not redistribute.** |
| **Kaggle: Drone Sound Audio** | Drone acoustic signatures | ~200–300 MB | **MIT License** | ✅ Fully permissive open-source license |
| **AudioSet** (filtered, tactical labels)| Gunfire, artillery fire, explosion, helicopter | ~0.5–1 GB filtered | **YouTube ToS / Research** | ⚠️ Research fair use proxy (extracted via `scripts/filter_audioset.py`) |

### Room impulse responses (reverb augmentation)
| Dataset | Role | Approx. size | Notes |
|---|---|---|---|
| **OpenSLR26** (simulated RIRs) | Synthetic physical RIR bank | ~2 GB | Varied room dimensions and absorption profiles |
| **OpenSLR28** (REVERB Challenge RIRs) | Real-recorded RIRs + ambient noise | ~2 GB | Real measured impulse responses |

**Total footprint:** ~9–11 GB for everything except AudioSet, +~0.5–1 GB for a filtered AudioSet pull $\rightarrow$ **~10–12 GB total**.

---

## Directory Layout Hierarchy

```text
SIH_PROJECT/
├── datasets/
│   ├── clean_speech/
│   │   ├── librispeech/         # OpenSLR/12 (Clean speech .flac/.wav)
│   │   └── voicebank/           # VoiceBank+DEMAND (Clean speech .wav)
│   │
│   ├── generic_noise/
│   │   ├── musan/               # OpenSLR/17 (Music, speech, technical noise)
│   │   ├── demand/              # DEMAND environmental noise
│   │   ├── esc50/               # ESC-50 Environmental Audio
│   │   └── audioset/            # Filtered AudioSet (Generic background ambient)
│   │
│   ├── tactical_noise/
│   │   ├── mad/                 # Military Audio Dataset (Combat gunfire, weapons)
│   │   ├── gunshot/             # Kaggle Gunshot Audio Dataset (Impulsive cracks/blasts)
│   │   ├── drone/               # Kaggle Drone Sound Audio Dataset (UAV rotor noise)
│   │   └── audioset/            # Filtered AudioSet (Gunfire, Artillery, Helicopter, Explosion)
│   │
│   └── rirs/
│       ├── openslr26/           # OpenSLR/26 (Simulated Small/Medium/Large Room RIRs)
│       └── openslr28/           # OpenSLR/28 (REVERB Challenge Real Measured Room RIRs)
```

---

## AudioSet Filtering Procedure
AudioSet is not downloaded as a massive multi-terabyte zip. Instead:
1. Pull the ontology and segment CSVs (`class_labels_indices.csv`, `balanced_train_segments.csv`) from [Google AudioSet](https://research.google.com/audioset/download.html).
2. Run `scripts/filter_audioset.py` against the label IDs of interest:
   - **Tactical set**: Gunshot/gunfire (`/m/03m5k`), Machine gun (`/m/02z550`), Artillery fire (`/m/012ffv`), Explosion (`/m/012n7d`), Helicopter (`/m/09ct_`), Aircraft (`/m/0k4j`), Siren (`/m/03q5t`), Engine knock (`/m/07p55fl`).
   - **Generic set**: Background crowd babble, machinery, ambient traffic.
3. Use `yt-dlp` to download each filtered `(YTID, start_sec, end_sec)` row, capped at ~150 clips per label ($\approx 300\text{--}500$ clips total, a few hundred MB / 1–2 hrs of audio).

---

## Hybrid Mixing Pipeline & Non-Negotiable Invariants

For each training sample in `HybridRealDataset` ([`data/real_dataset_loader.py`](file:///c:/Users/eashw/OneDrive/Documents/SIH_PROJECT/data/real_dataset_loader.py)):
1. **Clean Speech**: Drawn from real clean speech (LibriSpeech/VoiceBank) or synthetic speech generator based on `--hybrid_ratio`.
2. **Noise Mixture**: Drawn from real tactical noise (MAD/Gunshot/Drone/AudioSet), generic noise (MUSAN/DEMAND/ESC-50), or synthetic defence noise generator. Tactical noise is sampled with higher probability (40%) to keep defence scenarios well represented.
3. **Shared RIR Reverb (Task 2B Invariant)**: The **exact same RIR draw** (from OpenSLR26/28 or physical acoustic simulator) is convolved into both the clean reference and the noisy mix:
   $$s_{\text{clean\_rev}} = s_{\text{clean}} * h_{\text{RIR}}$$
   $$s_{\text{noisy\_rev}} = (s_{\text{clean}} + n) * h_{\text{RIR}}$$
4. **Tactical VHF Bandpass (Task 1 Invariant)**: The tactical 300–3400 Hz bandpass filter is applied to both the noisy mix and the clean reference signal prior to loss computation or evaluation metrics:
   $$\hat{s}_{\text{clean\_ref}} = \text{Bandpass}_{300-3400\text{Hz}}(s_{\text{clean}})$$
5. **Polyphase Resampling**: High-quality polyphase resampler (`scipy.signal.resample_poly` / `load_and_resample_audio`) automatically normalizes all incoming audio sources to the project standard **`16,000 Hz mono float32`**.

---

## Known Open Items & Verification Status
- **Resampling Implementation**: ✅ Implemented and unit-tested in [`data/real_dataset_loader.py`](file:///c:/Users/eashw/OneDrive/Documents/SIH_PROJECT/data/real_dataset_loader.py) (`load_and_resample_audio` handles 24kHz, 44.1kHz, 48kHz, mono/stereo conversion).
- **Acoustic Invariants**: ✅ Unit test [`tests/test_hybrid_dataset.py`](file:///c:/Users/eashw/OneDrive/Documents/SIH_PROJECT/tests/test_hybrid_dataset.py) verifies zero error on shared-RIR and >20 dB out-of-band rejection on transceiver bandpass reference.
- **License Terms Notice**: Exact license terms for Kaggle datasets and MAD have been documented for non-commercial research use; verify before commercial redistribution.
- **PESQ Implementation**: Native ITU-T P.862 C-extension pending Visual Studio C++ Build Tools; reports continue to use wideband psychoacoustic model labeled as `PESQ (approx.)`.
