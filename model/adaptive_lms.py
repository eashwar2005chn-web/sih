"""
DRDO PS 26052: Hybrid Adaptive NLMS (Normalized Least Mean Squares) Post-Filter
Implements real-time DSP adaptive residual filtering on the Intel Core Ultra 9 CPU.
Acts as a post-filter to the deep learning stage to eliminate residual stationary noise,
acoustic feedback, and cockpit hum with deterministic <0.2ms latency.
"""

import numpy as np
from typing import Tuple, Optional


class NormalizedLMSFilter:
    """
    Real-time Normalized Least Mean Squares (NLMS) adaptive filter with VAD gating
    and leaky coefficient integration.
    """

    def __init__(
        self,
        filter_order: int = 64,
        mu: float = 0.05,
        eps: float = 1e-6,
        leakage: float = 0.9999,
        vad_threshold: float = 0.03
    ):
        self.order = filter_order
        self.mu = mu
        self.eps = eps
        self.leakage = leakage
        self.vad_threshold = vad_threshold

        # Filter tap weights
        self.weights = np.zeros(filter_order, dtype=np.float32)
        # Input history buffer (delay line)
        self.buffer = np.zeros(filter_order, dtype=np.float32)

    def reset(self):
        """Reset internal filter states."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)

    def process_sample(self, primary_signal: float, ref_signal: float) -> Tuple[float, float]:
        """
        Process a single audio sample.
        Args:
            primary_signal (d): The neural network enhanced speech + residual noise.
            ref_signal (x): The estimated noise reference (e.g. noisy - enhanced).
        Returns:
            error_signal (e): The refined clean speech output.
            filter_out (y): The estimated residual noise.
        """
        # 1. Update delay line with reference noise sample
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = ref_signal

        # 2. Compute filter output: y(n) = w^T * x(n)
        y = float(np.dot(self.weights, self.buffer))

        # 3. Compute error: e(n) = d(n) - y(n)
        e = primary_signal - y

        # 4. Energy-normalized step adaptation: mu_norm = mu / (eps + ||x||^2)
        energy = float(np.dot(self.buffer, self.buffer))
        norm_mu = self.mu / (self.eps + energy)

        # 5. Speech VAD gating: adapt only when residual noise is dominant
        # To avoid speech distortion, slow adaptation if error signal has high speech energy
        if abs(e) < self.vad_threshold:
            # Leaky LMS update
            self.weights = self.leakage * self.weights + norm_mu * e * self.buffer

        return e, y

    def process_frame(self, primary_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        """
        Process an audio frame in-place or return output array.
        Inputs are 1D numpy arrays of equal length.
        """
        out_frame = np.zeros_like(primary_frame, dtype=np.float32)
        for i in range(len(primary_frame)):
            e, _ = self.process_sample(float(primary_frame[i]), float(ref_frame[i]))
            out_frame[i] = e
        return out_frame

    def filter_batch(
        self,
        primary_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Batch filtering function for complete audio clips.
        If reference_audio is None, uses a delayed version of primary_audio
        or a linear predictive self-reference model.
        """
        N = len(primary_audio)
        if reference_audio is None:
            # Generate self-reference via decorrelation delay (D=128 samples / 8ms)
            delay = 128
            reference_audio = np.zeros_like(primary_audio)
            reference_audio[delay:] = primary_audio[:-delay]

        enhanced = np.zeros(N, dtype=np.float32)
        for i in range(N):
            e, _ = self.process_sample(float(primary_audio[i]), float(reference_audio[i]))
            enhanced[i] = e

        return enhanced


class HybridANCPostProcessor:
    """
    Combines deep complex masking output with CPU-optimized adaptive NLMS post-filtering.
    """

    def __init__(self, sample_rate: int = 16000, filter_order: int = 48, mu: float = 0.08):
        self.sr = sample_rate
        self.nlms = NormalizedLMSFilter(filter_order=filter_order, mu=mu)

    def enhance(self, raw_noisy: np.ndarray, neural_enhanced: np.ndarray) -> np.ndarray:
        """
        Takes raw noisy audio and neural network enhanced audio, computes the estimated
        residual noise reference, and applies the adaptive NLMS post-filter.
        """
        # Noise reference is the portion removed by the neural net
        noise_ref = raw_noisy - neural_enhanced
        
        # Apply NLMS
        self.nlms.reset()
        final_enhanced = self.nlms.process_frame(neural_enhanced, noise_ref)
        
        # Peak normalization safety
        peak = np.max(np.abs(final_enhanced)) + 1e-8
        if peak > 0.98:
            final_enhanced = final_enhanced / peak * 0.95

        return final_enhanced.astype(np.float32)
