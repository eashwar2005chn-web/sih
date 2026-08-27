"""
Unit Tests for Real & Hybrid Dataset Loader (DRDO PS 26052)
Verifies:
1. Audio manifest scanning & multi-source indexing.
2. Automatic resampling to 16,000 Hz mono from arbitrary sample rates (24kHz, 44.1kHz, 48kHz).
3. Shared-RIR acoustic invariant: Exact same RIR filter applied to both clean and noisy channels.
4. Tactical VHF transceiver bandpass invariant: Both clean reference and noisy mix are filtered identically.
5. PyTorch DataLoader batching & tensor shape integrity.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import torch
import unittest

from data.real_dataset_loader import (
    RealAudioManifest,
    HybridRealDataset,
    load_and_resample_audio,
    find_audio_files
)


class TestHybridDataset(unittest.TestCase):

    def setUp(self):
        """Create temporary fixture directory with multi-rate synthetic audio files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

        # Structure fixture subdirectories
        self.dirs = {
            "clean_speech": {
                "librispeech": os.path.join(self.root, "clean_speech", "librispeech"),
                "voicebank_clean": os.path.join(self.root, "clean_speech", "voicebank")
            },
            "generic_noise": {
                "musan": os.path.join(self.root, "generic_noise", "musan"),
                "demand": os.path.join(self.root, "generic_noise", "demand"),
                "esc50": os.path.join(self.root, "generic_noise", "esc50"),
                "audioset_generic": os.path.join(self.root, "generic_noise", "audioset")
            },
            "tactical_noise": {
                "mad": os.path.join(self.root, "tactical_noise", "mad"),
                "gunshot_kaggle": os.path.join(self.root, "tactical_noise", "gunshot"),
                "drone_kaggle": os.path.join(self.root, "tactical_noise", "drone"),
                "audioset_tactical": os.path.join(self.root, "tactical_noise", "audioset")
            },
            "rirs": {
                "openslr26": os.path.join(self.root, "rirs", "openslr26"),
                "openslr28_reverb": os.path.join(self.root, "rirs", "openslr28")
            }
        }

        for cat, sources in self.dirs.items():
            for src, path in sources.items():
                os.makedirs(path, exist_ok=True)

        # Generate test audio clips at different sample rates and channels
        # 1. Clean speech at 24 kHz (mono)
        t24 = np.linspace(0, 2.0, 48000, endpoint=False)
        speech_24k = 0.5 * np.sin(2 * np.pi * 440 * t24).astype(np.float32)
        sf.write(os.path.join(self.dirs["clean_speech"]["librispeech"], "spk01_clean.wav"), speech_24k, 24000)

        # 2. Generic noise at 44.1 kHz (stereo)
        t44 = np.linspace(0, 2.0, 88200, endpoint=False)
        noise_44k = np.stack([
            0.3 * np.random.randn(88200).astype(np.float32),
            0.3 * np.random.randn(88200).astype(np.float32)
        ], axis=1)
        sf.write(os.path.join(self.dirs["generic_noise"]["musan"], "ambient_stereo.wav"), noise_44k, 44100)

        # 3. Tactical gunshot noise at 48 kHz (mono)
        t48 = np.linspace(0, 1.5, 72000, endpoint=False)
        gunshot_48k = (np.exp(-t48 * 20.0) * np.sin(2 * np.pi * 120 * t48)).astype(np.float32)
        sf.write(os.path.join(self.dirs["tactical_noise"]["gunshot_kaggle"], "gunshot_01.wav"), gunshot_48k, 48000)

        # 4. RIR impulse response at 16 kHz (mono)
        rir_16k = np.zeros(1600, dtype=np.float32)
        rir_16k[0] = 1.0
        rir_16k[200] = 0.4
        rir_16k[500] = 0.2
        sf.write(os.path.join(self.dirs["rirs"]["openslr26"], "room_rir.wav"), rir_16k, 16000)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_manifest_discovery(self):
        """Verify manifest discovers audio files across categories."""
        manifest = RealAudioManifest(dataset_dirs=self.dirs, data_root=self.root)
        summary = manifest.get_summary()

        self.assertEqual(summary["clean_speech"]["librispeech"], 1)
        self.assertEqual(summary["generic_noise"]["musan"], 1)
        self.assertEqual(summary["tactical_noise"]["gunshot_kaggle"], 1)
        self.assertEqual(summary["rirs"]["openslr26"], 1)

    def test_resampling_and_mono_conversion(self):
        """Verify 24kHz, 44.1kHz, and 48kHz audio resample accurately to 16kHz mono."""
        # Clean speech 24kHz -> 16kHz
        speech_path = os.path.join(self.dirs["clean_speech"]["librispeech"], "spk01_clean.wav")
        resampled_speech = load_and_resample_audio(speech_path, target_sr=16000)
        self.assertEqual(len(resampled_speech), 32000)  # 2.0s * 16000
        self.assertEqual(resampled_speech.ndim, 1)

        # Generic noise 44.1kHz stereo -> 16kHz mono
        noise_path = os.path.join(self.dirs["generic_noise"]["musan"], "ambient_stereo.wav")
        resampled_noise = load_and_resample_audio(noise_path, target_sr=16000)
        self.assertEqual(len(resampled_noise), 32000)
        self.assertEqual(resampled_noise.ndim, 1)

    def test_shared_rir_invariant(self):
        """
        Verify Shared-RIR Invariant:
        When identical speech is passed through the RIR stage with zero noise,
        clean_rev and noisy_rev MUST be identical to machine precision.
        """
        dataset = HybridRealDataset(
            mode="real",
            split="train",
            sample_rate=16000,
            segment_len_sec=1.0,
            epoch_size=10,
            dataset_dirs=self.dirs,
            data_root=self.root,
            add_reverb=True,
            add_channel_distortion=False,
            seed=1234
        )

        sample = dataset._sample_segment(np.ones(16000, dtype=np.float32))
        clean_rev, noisy_rev = dataset._apply_shared_rir(sample, sample)

        # Clean and noisy must match identically since the same RIR is applied
        max_diff = np.max(np.abs(clean_rev - noisy_rev))
        self.assertAlmostEqual(max_diff, 0.0, places=6, msg="Shared RIR invariant violated: Clean and Noisy RIR filters diverged!")

    def test_vhf_bandpass_reference_alignment_invariant(self):
        """
        Verify VHF Transceiver Reference Alignment Invariant:
        When tactical transceiver channel distortion is active,
        both clean reference and noisy mix must be bandpass filtered (300-3400 Hz).
        """
        from data.defence_noise_generator import AcousticChannelSimulator
        sim = AcousticChannelSimulator(sample_rate=16000)

        # 100 Hz out-of-band tone + 1000 Hz in-band tone + 6000 Hz out-of-band tone
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        test_signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 6000 * t)
        test_signal = test_signal.astype(np.float32)

        filtered_clean = sim.apply_tactical_channel(test_signal, add_clipping=False)
        filtered_noisy = sim.apply_tactical_channel(test_signal, add_clipping=False)

        # Both must match and reject out-of-band 100Hz and 6000Hz components
        self.assertTrue(np.allclose(filtered_clean, filtered_noisy, atol=1e-5))

        # Check FFT attenuation of 100Hz and 6000Hz vs 1000Hz in-band
        fft_orig = np.abs(np.fft.rfft(test_signal))
        fft_filt = np.abs(np.fft.rfft(filtered_clean))
        freqs = np.fft.rfftfreq(len(test_signal), 1.0 / 16000)

        idx_100 = np.argmin(np.abs(freqs - 100))
        idx_1000 = np.argmin(np.abs(freqs - 1000))
        idx_6000 = np.argmin(np.abs(freqs - 6000))

        # 100Hz and 6000Hz should be attenuated by >20 dB relative to 1000Hz
        ratio_100 = fft_filt[idx_100] / fft_filt[idx_1000]
        ratio_6000 = fft_filt[idx_6000] / fft_filt[idx_1000]

        self.assertLess(ratio_100, 0.1, "100 Hz low-cut bandpass filter failed!")
        self.assertLess(ratio_6000, 0.1, "6000 Hz high-cut bandpass filter failed!")

    def test_hybrid_dataset_dataloader_batch(self):
        """Verify DataLoader batching with HybridRealDataset."""
        dataset = HybridRealDataset(
            mode="hybrid",
            split="train",
            sample_rate=16000,
            segment_len_sec=2.0,
            epoch_size=16,
            hybrid_ratio=0.5,
            dataset_dirs=self.dirs,
            data_root=self.root,
            seed=42
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        self.assertEqual(batch["noisy"].shape, (4, 32000))
        self.assertEqual(batch["clean"].shape, (4, 32000))
        self.assertEqual(batch["snr_db"].shape, (4,))
        self.assertFalse(torch.isnan(batch["noisy"]).any())
        self.assertFalse(torch.isnan(batch["clean"]).any())


if __name__ == "__main__":
    unittest.main()
