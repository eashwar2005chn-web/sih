"""
Unit Test: Verify Tactical Bandpass Reference Matching for STOI, PESQ, and SNR.
Verifies that evaluating bandpass-filtered audio against a matching bandpass-filtered
clean reference gives higher, more accurate scores than evaluating against an
unfiltered (full-bandwidth) reference.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_builder import SyntheticSpeechGenerator
from data.defence_noise_generator import AcousticChannelSimulator, DefenceNoiseGenerator, mix_speech_and_noise
from model.loss import compute_snr, compute_stoi, compute_pesq


class TestEvaluationBandpass(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.speech_gen = SyntheticSpeechGenerator(sample_rate=self.sr)
        self.channel_sim = AcousticChannelSimulator(sample_rate=self.sr)
        self.noise_gen = DefenceNoiseGenerator(sample_rate=self.sr)

    def test_bandpass_reference_alignment_metrics(self):
        """
        Verify filtered_clean vs filtered_enhanced > unfiltered_clean vs filtered_enhanced.
        """
        # Generate clean utterance
        clean_unfiltered = self.speech_gen.generate_utterance(duration_sec=3.0, speaker_id=1)
        noise = self.noise_gen.get_noise_by_type("helicopter", duration_sec=3.0)
        
        # Mix at 5 dB
        noisy, clean_ref, _ = mix_speech_and_noise(clean_unfiltered, noise, target_snr_db=5.0)

        # Apply tactical radio bandpass filter (300 - 3400 Hz) to simulate transceiver channel
        noisy_filtered = self.channel_sim.apply_tactical_channel(noisy)
        clean_filtered = self.channel_sim.apply_tactical_channel(clean_ref, add_clipping=False)

        # Assume enhanced signal produced by the model operating on bandpass channel
        # For direct testing, even the clean_filtered or a mildly denoised version serves as enhanced
        enhanced_audio = clean_filtered + 0.1 * (noisy_filtered - clean_filtered)

        # Compute metrics against UNFILTERED clean reference
        stoi_unfiltered = compute_stoi(clean_unfiltered, enhanced_audio, sr=self.sr)
        pesq_unfiltered = compute_pesq(clean_unfiltered, enhanced_audio, sr=self.sr)
        snr_unfiltered = compute_snr(clean_unfiltered, enhanced_audio)

        # Compute metrics against FILTERED clean reference (aligned)
        stoi_filtered = compute_stoi(clean_filtered, enhanced_audio, sr=self.sr)
        pesq_filtered = compute_pesq(clean_filtered, enhanced_audio, sr=self.sr)
        snr_filtered = compute_snr(clean_filtered, enhanced_audio)

        print(f"\n[TestBandpassAlignment]")
        print(f"STOI  - Unfiltered Ref: {stoi_unfiltered:.4f} | Filtered Ref: {stoi_filtered:.4f}")
        print(f"PESQ  - Unfiltered Ref: {pesq_unfiltered:.4f} | Filtered Ref: {pesq_filtered:.4f}")
        print(f"SNR   - Unfiltered Ref: {snr_unfiltered:.2f} dB | Filtered Ref: {snr_filtered:.2f} dB")

        # Assertions: filtered reference must yield higher (or equal) fidelity scores
        self.assertGreater(stoi_filtered, stoi_unfiltered, "Filtered STOI must be higher than unfiltered STOI")
        self.assertGreater(pesq_filtered, pesq_unfiltered, "Filtered PESQ must be higher than unfiltered PESQ")
        self.assertGreater(snr_filtered, snr_unfiltered, "Filtered SNR must be higher than unfiltered SNR")


if __name__ == "__main__":
    unittest.main()
