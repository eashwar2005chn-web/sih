"""
DRDO PS 26052: Real & Hybrid Dataset Loader
Integrates academic and tactical open-access datasets alongside the synthetic defence generator:
- Clean Speech: LibriSpeech (OpenSLR/12), VoiceBank+DEMAND (Clean)
- Generic Noise: MUSAN, DEMAND (Noise), ESC-50, Filtered AudioSet (Generic)
- Tactical Noise: MAD (Military Audio Dataset), Kaggle Gunshot, Kaggle Drone, Filtered AudioSet (Tactical)
- RIRs: OpenSLR26, OpenSLR28 (REVERB Challenge)

Enforces strict acoustic invariants:
1. Shared RIR: Exact same RIR filter convolved with both clean speech and noisy mixture.
2. Transceiver Bandpass: Tactical 300-3400 Hz bandpass applied to clean reference whenever channel distortion is active.
"""

import os
import glob
import json
import math
import hashlib
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union

from data.defence_noise_generator import DefenceNoiseGenerator, AcousticChannelSimulator, mix_speech_and_noise
from data.dataset_builder import SyntheticSpeechGenerator


DEFAULT_DATASET_DIRS = {
    "clean_speech": {
        "librispeech": "datasets/clean_speech/librispeech",
        "voicebank_clean": "datasets/clean_speech/voicebank"
    },
    "generic_noise": {
        "musan": "datasets/generic_noise/musan",
        "demand": "datasets/generic_noise/demand",
        "esc50": "datasets/generic_noise/esc50",
        "audioset_generic": "datasets/generic_noise/audioset"
    },
    "tactical_noise": {
        "mad": "datasets/tactical_noise/mad",
        "gunshot_kaggle": "datasets/tactical_noise/gunshot",
        "drone_kaggle": "datasets/tactical_noise/drone",
        "audioset_tactical": "datasets/tactical_noise/audioset"
    },
    "rirs": {
        "openslr26": "datasets/rirs/openslr26",
        "openslr28_reverb": "datasets/rirs/openslr28"
    }
}

# --------------------------------------------------------------------------------------
# Split pinning
# --------------------------------------------------------------------------------------
# Sources listed here bypass the hash splitter entirely and are assigned wholesale to one
# split. This exists so a large corpus can be added to TRAINING without the path-hash
# scattering ~20% of it into val/test, which would silently redefine the evaluation sets and
# make every previously published number incomparable.
#
# A pinned source contributes ALL of its files to its pinned split and NOTHING to any other.
#
#   librispeech_train100 : LibriSpeech train-clean-100 (100.6 h, 251 speakers). Speaker-disjoint
#                          from dev-clean by corpus construction, so pinning it to train makes
#                          evaluation genuinely speaker-disjoint while leaving val/test
#                          byte-identical to every prior run.
PINNED_SOURCES = {
    "librispeech_train100": "train",
}

# Sources that never contribute to the TRAIN split. They still hash-split between val and test,
# so excluding one changes training data only and leaves the evaluation sets untouched.
#
#   gunshot_kaggle : unspecified license -- excluded from training as a matter of policy.
#                    (Previously hardcoded in get_split_files; generalised here, same behaviour.)
EXCLUDED_FROM_TRAIN = {"gunshot_kaggle"}

# The evaluation corpus is LibriSpeech dev-clean. Two sources carry it:
#   librispeech      : dev-clean itself
#   voicebank_clean  : NOT VoiceBank -- verified byte-equivalent relabelled dev-clean audio
#                      (filenames embed the source utterance id, e.g.
#                      p225_000_1272-128104-0000.wav; waveform correlation 1.0000)
# While either is in the train split, training and evaluation share both SPEAKERS and CONTENT.
DEV_CLEAN_SOURCES = {"librispeech", "voicebank_clean"}


def set_speaker_disjoint_training(enabled: bool) -> Dict:
    """
    Switch the TRAINING corpus between two mutually exclusive regimes. val/test are identical
    in both -- only what the model is allowed to train on changes.

        False (DEFAULT)  historical regime. Train on dev-clean-derived audio (5.04 h).
                         Training shares SPEAKERS and 30.12% of CONTENT with the test split.
                         This is what every published number was measured under, so it is the
                         default and every prior run reproduces byte-for-byte.

        True             disjoint regime. Train on LibriSpeech train-clean-100 (100.59 h,
                         251 speakers) and exclude both dev-clean-derived sources from train.
                         Speaker overlap 0, content overlap 0.00%.

    Enabling this requires the corpus at LIBRISPEECH_TRAIN100_DIR; a missing directory raises
    rather than silently training on almost nothing.

    Any script that enables this MUST say so in its log -- the regime determines whether its
    result is comparable to the ledger.

    Returns a dict describing the active configuration.
    """
    if enabled:
        if not os.path.isdir(LIBRISPEECH_TRAIN100_DIR):
            raise RuntimeError(
                f"Speaker-disjoint training requires LibriSpeech train-clean-100 at "
                f"'{LIBRISPEECH_TRAIN100_DIR}', which does not exist. Set the "
                f"LIBRISPEECH_TRAIN100_DIR environment variable, or fetch it from "
                f"https://www.openslr.org/resources/12/train-clean-100.tar.gz . "
                f"Refusing to enable: excluding dev-clean without a replacement would leave "
                f"the training split nearly empty and silently produce a meaningless run."
            )
        DEFAULT_DATASET_DIRS["clean_speech"]["librispeech_train100"] = LIBRISPEECH_TRAIN100_DIR
        EXCLUDED_FROM_TRAIN.update(DEV_CLEAN_SOURCES)
    else:
        DEFAULT_DATASET_DIRS["clean_speech"].pop("librispeech_train100", None)
        EXCLUDED_FROM_TRAIN.difference_update(DEV_CLEAN_SOURCES)
    return {
        "speaker_disjoint": bool(enabled),
        "excluded_from_train": sorted(EXCLUDED_FROM_TRAIN),
        "clean_speech_sources": sorted(DEFAULT_DATASET_DIRS["clean_speech"]),
    }

# Location of the pinned LibriSpeech training corpus. Kept OUT of the repo (and out of any
# synced folder) because it is ~6 GB of immutable third-party audio. Override with the
# LIBRISPEECH_TRAIN100_DIR environment variable. A missing directory is not an error: the
# source simply contributes zero files, and the loader behaves exactly as it did before.
LIBRISPEECH_TRAIN100_DIR = os.environ.get(
    "LIBRISPEECH_TRAIN100_DIR",
    os.path.join(os.path.expanduser("~"), "datasets_local", "LibriSpeech", "train-clean-100"),
)
# NOT registered at import time. It is added only by set_speaker_disjoint_training(True), so the
# default configuration reproduces every previously published run byte-for-byte.

# Frozen evaluation manifest. Written by scripts/freeze_eval_splits.py; checked by
# verify_eval_splits_frozen(). Guards against ANY future change to the val/test file lists.
FROZEN_EVAL_MANIFEST = "eval/frozen_eval_manifest.json"

AUDIO_EXTENSIONS = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.WAV", "*.FLAC")


def find_audio_files(directory: str) -> List[str]:
    """Recursively discover all audio files in a directory."""
    if not os.path.isdir(directory):
        return []
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(list(set(files)))


def load_and_resample_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file, convert to mono, and resample to target_sr using polyphase filtering.
    """
    try:
        audio, sr = sf.read(file_path, dtype="float32")
    except Exception as e:
        # Fallback to torchaudio if soundfile fails on specific codecs
        waveform, sr = torchaudio.load(file_path)
        audio = waveform.numpy()
        if audio.ndim == 2:
            audio = audio.T

    # Convert multi-channel to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if sample rate differs
    if sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)

    # Normalize amplitude range
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


class RealAudioManifest:
    """
    Scans and indexes configured real dataset directories into deterministic train/val/test splits.
    """
    def __init__(self, dataset_dirs: Optional[Dict] = None, data_root: str = "."):
        self.dataset_dirs = dataset_dirs or DEFAULT_DATASET_DIRS
        self.data_root = data_root
        self.manifest = {
            "clean_speech": {},
            "generic_noise": {},
            "tactical_noise": {},
            "rirs": {}
        }
        self.scan()

    def scan(self):
        """Scan all categories and populate file paths."""
        for category, sources in self.dataset_dirs.items():
            for source_name, rel_path in sources.items():
                full_path = os.path.join(self.data_root, rel_path) if not os.path.isabs(rel_path) else rel_path
                files = find_audio_files(full_path)
                self.manifest[category][source_name] = files

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """Return counts of audio files per category and source."""
        summary = {}
        for category, sources in self.manifest.items():
            summary[category] = {src: len(files) for src, files in sources.items()}
        return summary

    def get_split_files(self, category: str, split: str = "train") -> List[str]:
        """
        Get files for a specific category and split using deterministic hashing.
        Train: 80% (hash % 100 < 80)
        Val:   10% (80 <= hash % 100 < 90)
        Test:  10% (hash % 100 >= 90)
        
        NOTE: gunshot_kaggle has an unspecified license and is excluded from training splits.

        PINNED sources (see PINNED_SOURCES) bypass the hash entirely: all of their files go to
        the pinned split and none to any other. This keeps a newly added training corpus from
        leaking ~20% of itself into val/test and silently redefining the evaluation sets.
        """
        all_files = []
        pinned_files = []
        for src, files in self.manifest.get(category, {}).items():
            if src in PINNED_SOURCES:
                if PINNED_SOURCES[src] == split:
                    pinned_files.extend(files)
                continue
            if split == "train" and src in EXCLUDED_FROM_TRAIN:
                # Never trained on: unspecified license, or (when speaker-disjoint training is
                # enabled) a source whose content also appears in val/test.
                continue
            all_files.extend(files)

        split_files = list(pinned_files)
        for f in all_files:
            h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16) % 100
            if split == "train" and h < 80:
                split_files.append(f)
            elif split == "val" and 80 <= h < 90:
                split_files.append(f)
            elif split == "test" and h >= 90:
                split_files.append(f)

        return split_files


def _rel_key(path: str) -> str:
    """Stable, OS-independent identity for a file in the manifest."""
    return os.path.normpath(path).replace(os.sep, "/")


def eval_split_fingerprint(manifest: Optional["RealAudioManifest"] = None) -> Dict:
    """
    Fingerprint the val and test file lists across every category.

    Returns per-split, per-category file counts and a sha256 over the sorted relative paths.
    Any change to which files land in val/test -- from a new source, a renamed directory, or a
    change to the split function -- changes the digest.
    """
    m = manifest or RealAudioManifest()
    out = {}
    for split in ("val", "test"):
        out[split] = {}
        for category in m.manifest:
            files = sorted(_rel_key(f) for f in m.get_split_files(category, split))
            h = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()
            out[split][category] = {"n_files": len(files), "sha256": h}
    return out


def verify_eval_splits_frozen(manifest_path: str = FROZEN_EVAL_MANIFEST,
                              strict: bool = True) -> bool:
    """
    Assert that val/test are byte-identical to the frozen manifest.

    Call this before any evaluation whose numbers will be compared against previously published
    results. Raises RuntimeError on drift when `strict` (the default) -- deliberately loud,
    because a silent change to the evaluation set is exactly the failure this guards against.

    Returns True if the splits match, False if they drift and `strict` is False.
    A missing manifest is a hard error: unverified is not the same as verified.
    """
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            f"Frozen evaluation manifest not found at '{manifest_path}'. "
            f"Generate it with `python scripts/freeze_eval_splits.py` BEFORE adding any new "
            f"data source, so the current val/test sets are captured as the reference."
        )

    with open(manifest_path, "r", encoding="utf-8") as fh:
        frozen = json.load(fh)

    current = eval_split_fingerprint()
    drift = []
    for split in ("val", "test"):
        for category, ref in frozen.get("fingerprint", {}).get(split, {}).items():
            cur = current.get(split, {}).get(category)
            if cur is None:
                drift.append(f"{split}/{category}: category missing entirely")
            elif cur["sha256"] != ref["sha256"]:
                drift.append(
                    f"{split}/{category}: {ref['n_files']} files -> {cur['n_files']} files "
                    f"(sha256 {ref['sha256'][:12]} -> {cur['sha256'][:12]})"
                )

    if not drift:
        return True

    msg = ("EVALUATION SPLIT DRIFT DETECTED -- val/test no longer match the frozen manifest.\n"
           + "\n".join(f"    {d}" for d in drift)
           + "\n  Every published metric was measured on the frozen sets. Comparing new numbers "
             "against them is invalid until this is resolved.\n"
             "  If a new source was added, pin it to 'train' in PINNED_SOURCES. Only re-freeze "
             "deliberately, and treat every prior number as incomparable if you do.")
    if strict:
        raise RuntimeError(msg)
    import warnings
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return False


class HybridRealDataset(Dataset):
    """
    Production PyTorch Dataset for DRDO PS 26052:
    Supports synthetic, real, and hybrid modes:
    - 'synthetic': Uses parametric speech + defence noise generators (fallback / stand-in).
    - 'real': Samples exclusively from scanned real dataset directories (LibriSpeech, MUSAN, MAD, OpenSLR).
    - 'hybrid': Seamlessly blends real speech with tactical/synthetic noises according to hybrid_ratio.
    """
    def __init__(
        self,
        mode: str = "hybrid",
        split: str = "train",
        sample_rate: int = 16000,
        segment_len_sec: float = 2.0,
        epoch_size: int = 1200,
        hybrid_ratio: float = 0.5,
        dataset_dirs: Optional[Dict] = None,
        data_root: str = ".",
        snr_range: Tuple[float, float] = (-5.0, 15.0),
        snr_sampler_mode: str = "uniform",
        add_reverb: bool = True,
        add_channel_distortion: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.mode = mode
        self.split = split
        self.sr = sample_rate
        self.segment_samples = int(segment_len_sec * sample_rate)
        self.epoch_size = epoch_size
        self.hybrid_ratio = hybrid_ratio
        self.snr_range = snr_range
        self.snr_sampler_mode = snr_sampler_mode
        self.add_reverb = add_reverb
        self.add_channel_distortion = add_channel_distortion

        # Generators
        self.noise_gen = DefenceNoiseGenerator(sample_rate=sample_rate)
        self.channel_sim = AcousticChannelSimulator(sample_rate=sample_rate)
        self.synth_speech_gen = SyntheticSpeechGenerator(sample_rate=sample_rate)

        # Manifest scanner
        self.manifest = RealAudioManifest(dataset_dirs=dataset_dirs, data_root=data_root)
        self.clean_files = self.manifest.get_split_files("clean_speech", split=split)
        self.generic_noise_files = self.manifest.get_split_files("generic_noise", split=split)
        self.tactical_noise_files = self.manifest.get_split_files("tactical_noise", split=split)
        self.rir_files = self.manifest.get_split_files("rirs", split=split)

        # Deterministic seed partition
        # Only the TRAIN split resamples silent noise windows; val/test keep the exact
        # original RNG draw sequence so they remain bit-identical to committed results.
        self.resample_silent_noise = (split == "train")
        self.split_seed = seed + (0 if split == "train" else (1000 if split == "val" else 2000))
        self.rng = np.random.RandomState(self.split_seed)

        # Synthetic speaker partition
        if split == "train":
            self.synth_speaker_ids = list(range(0, 80))
            self.synth_noise_types = ["gunshot", "helicopter", "armored_vehicle", "jet_flyby", "siren", "cockpit", "radio_static", "babble"]
        elif split == "val":
            self.synth_speaker_ids = list(range(80, 90))
            self.synth_noise_types = ["gunshot", "helicopter", "armored_vehicle", "cockpit", "radio_static"]
        else:
            self.synth_speaker_ids = list(range(90, 100))
            self.synth_noise_types = ["gunshot", "helicopter", "armored_vehicle", "jet_flyby", "siren", "cockpit", "radio_static", "babble"]

    def __len__(self) -> int:
        return self.epoch_size

    def _sample_segment(self, audio: np.ndarray, rng: Optional[np.random.RandomState] = None,
                        require_energy: bool = False) -> np.ndarray:
        """
        Extract a random fixed-length chunk or zero-pad.

        `require_energy` (used for NOISE on the training split only): several ESC-50-derived
        noise clips are mostly exact digital silence with a short event embedded -- one is
        4.41 s of silence in a 5 s file. A uniformly random window frequently lands entirely
        inside that silence, and mix_speech_and_noise() then returns a "noisy" signal that is
        bit-identical to the clean reference, teaching the model to pass audio through
        unchanged. When set, retry a bounded number of offsets and keep the best-energy
        window found. See eval/PHASE14_TESTSET_CONTAMINATION.md.

        IMPORTANT: this defaults to False so val/test generation consumes exactly the same
        number of RNG draws as before and stays bit-identical to the committed eval CSVs.
        """
        r = rng if rng is not None else self.rng
        if len(audio) >= self.segment_samples:
            start = r.randint(0, len(audio) - self.segment_samples + 1)
            seg = audio[start:start + self.segment_samples]
            if require_energy:
                best, best_rms = seg, float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
                tries = 0
                while best_rms < 1e-4 and tries < 8:
                    start = r.randint(0, len(audio) - self.segment_samples + 1)
                    cand = audio[start:start + self.segment_samples]
                    rms = float(np.sqrt(np.mean(cand.astype(np.float64) ** 2)))
                    if rms > best_rms:
                        best, best_rms = cand, rms
                    tries += 1
                return best
            return seg
        else:
            padded = np.zeros(self.segment_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            return padded

    def _get_clean_speech(self, idx: int, rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, str]:
        """Fetch real or synthetic clean speech according to mode."""
        r = rng if rng is not None else self.rng
        use_real = (self.mode == "real") or (self.mode == "hybrid" and r.rand() < self.hybrid_ratio)
        
        if use_real and len(self.clean_files) > 0:
            f_path = self.clean_files[r.randint(0, len(self.clean_files))]
            audio = load_and_resample_audio(f_path, target_sr=self.sr)
            return self._sample_segment(audio, rng=r), f"real:{os.path.basename(f_path)}"
        
        # Synthetic fallback
        spk_id = self.synth_speaker_ids[idx % len(self.synth_speaker_ids)]
        duration = self.segment_samples / self.sr
        synth_audio = self.synth_speech_gen.generate_utterance(duration_sec=duration, speaker_id=spk_id, rng=r)
        return synth_audio, f"synth_spk_{spk_id}"

    def _get_noise(self, rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, str]:
        """Fetch real tactical, generic, or synthetic noise."""
        r = rng if rng is not None else self.rng
        use_real_tactical = (self.mode in ["real", "hybrid"]) and (len(self.tactical_noise_files) > 0) and (r.rand() < 0.4)
        use_real_generic = (self.mode in ["real", "hybrid"]) and (len(self.generic_noise_files) > 0) and (r.rand() < 0.3)

        if use_real_tactical:
            f_path = self.tactical_noise_files[r.randint(0, len(self.tactical_noise_files))]
            audio = load_and_resample_audio(f_path, target_sr=self.sr)
            return self._sample_segment(audio, rng=r, require_energy=self.resample_silent_noise), f"real_tactical:{os.path.basename(f_path)}"
        elif use_real_generic:
            f_path = self.generic_noise_files[r.randint(0, len(self.generic_noise_files))]
            audio = load_and_resample_audio(f_path, target_sr=self.sr)
            return self._sample_segment(audio, rng=r, require_energy=self.resample_silent_noise), f"real_generic:{os.path.basename(f_path)}"

        # Synthetic tactical noise generator
        noise_type = r.choice(self.synth_noise_types)
        duration = self.segment_samples / self.sr
        synth_noise = self.noise_gen.get_noise_by_type(noise_type, duration_sec=duration, rng=r)
        return self._sample_segment(synth_noise, rng=r, require_energy=self.resample_silent_noise), f"synth_{noise_type}"

    def _apply_shared_rir(self, clean: np.ndarray, noisy: np.ndarray, rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the exact same Room Impulse Response (RIR) to both clean speech and noisy mixture.
        Supports real RIR audio files (OpenSLR26/28) or synthetic physics RIR simulator.
        """
        r = rng if rng is not None else self.rng
        if len(self.rir_files) > 0 and r.rand() < 0.5:
            # Real RIR filter
            rir_path = self.rir_files[r.randint(0, len(self.rir_files))]
            rir = load_and_resample_audio(rir_path, target_sr=self.sr)
            # Truncate RIR to max 0.5s for efficiency
            max_rir_len = int(0.5 * self.sr)
            rir = rir[:max_rir_len]
            # Convolve both with the SAME filter
            clean_rev = scipy.signal.convolve(clean, rir, mode="same").astype(np.float32)
            noisy_rev = scipy.signal.convolve(noisy, rir, mode="same").astype(np.float32)
            return clean_rev, noisy_rev
        else:
            # Synthetic physical RIR
            rt60 = float(r.uniform(0.1, 0.35))
            rir = self.channel_sim.generate_rir(rt60_sec=rt60, rng=r)
            clean_rev = self.channel_sim.simulate_rir(clean, rir=rir)
            noisy_rev = self.channel_sim.simulate_rir(noisy, rir=rir)
            return clean_rev, noisy_rev

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        # For validation and test, derive deterministic index-specific RNG
        r = np.random.RandomState(self.split_seed + idx) if self.split in ("val", "test") else self.rng

        # 1. Fetch clean speech
        clean, speech_meta = self._get_clean_speech(idx, rng=r)

        # 2. Fetch noise
        noise, noise_meta = self._get_noise(rng=r)

        # 3. Target SNR selection
        if self.snr_sampler_mode == "rebalanced":
            # 50% density on [-10, 5) dB (low/stress), 35% on [5, 10) dB, 15% on [10, 15] dB
            p = r.rand()
            if p < 0.50:
                snr_db = float(r.uniform(-10.0, 5.0))
            elif p < 0.85:
                snr_db = float(r.uniform(5.0, 10.0))
            else:
                snr_db = float(r.uniform(10.0, 15.0))
        else:
            snr_db = float(r.uniform(self.snr_range[0], self.snr_range[1]))

        # 4. Mix speech and noise
        noisy, clean, _ = mix_speech_and_noise(clean, noise, target_snr_db=snr_db)

        # 5. Shared RIR Reverberation (Invariant: single RIR applied to both channels)
        if self.add_reverb and r.rand() > 0.35:
            clean, noisy = self._apply_shared_rir(clean, noisy, rng=r)

        # 6. Transceiver Bandpass & Non-linear mic clipping (Invariant: reference alignment)
        if self.add_channel_distortion and r.rand() > 0.3:
            noisy = self.channel_sim.apply_tactical_channel(noisy, add_clipping=bool(r.rand() > 0.7))
            clean = self.channel_sim.apply_tactical_channel(clean, add_clipping=False)

        return {
            "noisy": torch.from_numpy(noisy).float(),
            "clean": torch.from_numpy(clean).float(),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "noise_type": noise_meta,
            "speech_source": speech_meta
        }


def _worker_init_fn(worker_id: int):
    """
    Give every DataLoader worker its OWN random stream.

    CRITICAL: for the train split, __getitem__ draws from `self.rng`, a RandomState built
    once in __init__. Every worker process receives an identical COPY of that state, so all
    workers generate the SAME sequence of samples. Measured before this fix: at
    num_workers=8, 640 drawn samples contained only 151 unique ones -- 76.4% duplicates,
    i.e. roughly a 4x reduction in effective training data.

    val/test are unaffected either way: they seed per-index from split_seed + idx, which is
    what keeps them bit-reproducible.
    """
    import numpy as _np
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "rng") and hasattr(ds, "split_seed"):
        ds.rng = _np.random.RandomState((ds.split_seed + 9973 * (worker_id + 1)) % (2 ** 31 - 1))


def get_hybrid_dataloaders(
    mode: str = "hybrid",
    batch_size: int = 16,
    num_workers: int = 0,
    segment_len_sec: float = 2.0,
    train_epoch_size: int = 1200,
    val_epoch_size: int = 200,
    test_epoch_size: int = 100,
    hybrid_ratio: float = 0.5,
    dataset_dirs: Optional[Dict] = None,
    data_root: str = ".",
    snr_sampler_mode: str = "uniform",
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory helper to instantiate train, val, and test DataLoaders for HybridRealDataset.
    """
    train_ds = HybridRealDataset(
        mode=mode,
        split="train",
        segment_len_sec=segment_len_sec,
        epoch_size=train_epoch_size,
        hybrid_ratio=hybrid_ratio,
        dataset_dirs=dataset_dirs,
        data_root=data_root,
        snr_sampler_mode=snr_sampler_mode,
        seed=seed
    )
    val_ds = HybridRealDataset(
        mode=mode,
        split="val",
        segment_len_sec=segment_len_sec,
        epoch_size=val_epoch_size,
        hybrid_ratio=hybrid_ratio,
        dataset_dirs=dataset_dirs,
        data_root=data_root,
        seed=seed
    )
    test_ds = HybridRealDataset(
        mode=mode,
        split="test",
        segment_len_sec=segment_len_sec,
        epoch_size=test_epoch_size,
        hybrid_ratio=hybrid_ratio,
        dataset_dirs=dataset_dirs,
        data_root=data_root,
        seed=seed
    )

    # persistent_workers + prefetch matter a lot here: samples are SYNTHESISED per item
    # (file load, resample, RIR convolution, bandpass, mix) at ~6 ms each, i.e. ~96 ms per
    # batch of 16 against ~18 ms of GPU compute. Without workers the GPU idles ~84% of the
    # time; without persistence, Windows respawns the worker pool every epoch and the
    # spawn cost dominates short epochs. Only applies when num_workers > 0.
    _persist = num_workers > 0
    _extra = {"persistent_workers": True, "prefetch_factor": 4,
              "worker_init_fn": _worker_init_fn} if _persist else {}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), **_extra
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), **_extra
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), **_extra
    )

    return train_loader, val_loader, test_loader
