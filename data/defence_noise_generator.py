"""
DRDO PS 26052: Tactical Defence Noise Generator & Acoustic Environment Simulator
Generates high-fidelity defence noise profiles and acoustic channel effects:
- Impulsive: Gunshots, artillery bursts, transient impacts (with synthetic caveat)
- Non-Stationary: Helicopter rotor (blade passing frequency), armored vehicle engine, jet flyby, sirens
- Stationary: Cockpit background hum, tactical radio channel static, wind turbulence, multi-talker babble
- Acoustic Channels: Room Impulse Response (RIR) reverberation, military bandpass filtering, clipping
"""

import warnings

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, Any


class DefenceNoiseGenerator:
    """
    Parametric synthetic defence noise generator for tactical audio enhancement.
    All outputs are normalized 1D numpy arrays at the specified sample rate (default 16kHz).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def generate_impulsive_gunshot(
        self,
        duration_sec: float = 3.0,
        num_shots: int = 3,
        calibre: str = "medium",
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize impulsive high-energy gunfire bursts with ballistic crack and muzzle blast.
        Note: Documented synthetic stand-in for defence impulsive transients.
        """
        r = rng if rng is not None else np.random
        total_samples = int(duration_sec * self.sr)
        audio = np.zeros(total_samples, dtype=np.float32)

        # Distribute shots randomly
        shot_positions = np.sort(
            r.randint(int(0.2 * self.sr), int((duration_sec - 0.5) * self.sr), size=num_shots)
        )

        for pos in shot_positions:
            # 1. Shockwave / ballistic crack (rapid transient, < 5ms)
            crack_len = int(0.005 * self.sr)
            t_crack = np.linspace(0, 0.005, crack_len)
            crack = r.randn(crack_len) * np.exp(-t_crack / 0.001)

            # 2. Muzzle blast (low frequency thud, 50-200ms)
            blast_len = int(0.15 * self.sr)
            t_blast = np.linspace(0, 0.15, blast_len)
            blast_freq = 60 if calibre == "heavy" else 120
            blast = np.sin(2 * np.pi * blast_freq * t_blast) * np.exp(-t_blast / 0.04)
            blast += r.randn(blast_len) * np.exp(-t_blast / 0.02) * 0.5

            # Combine and insert
            shot_len = max(crack_len, blast_len)
            shot = np.zeros(shot_len, dtype=np.float32)
            shot[:crack_len] += crack * 1.5
            shot[:blast_len] += blast

            end_idx = min(pos + shot_len, total_samples)
            actual_len = end_idx - pos
            audio[pos:end_idx] += shot[:actual_len]

        return audio

    def generate_helicopter_rotor(
        self,
        duration_sec: float = 3.0,
        bpf_hz: float = 24.0,  # Blade Passing Frequency (e.g. 4-blade at 360 RPM = 24 Hz)
        rotor_harmonics: int = 6,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize non-stationary helicopter acoustic signature with periodic blade slap & turbine whine.
        """
        r = rng if rng is not None else np.random
        n_samples = int(duration_sec * self.sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Blade slap (sum of low harmonics modulated by narrow pulse train)
        blade_slap = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, rotor_harmonics + 1):
            blade_slap += (1.0 / h) * np.sin(2 * np.pi * bpf_hz * h * t + r.uniform(0, 2*np.pi))

        # Pulse modulation (sharp periodic thumps)
        pulse_mod = np.clip(np.sin(2 * np.pi * bpf_hz * t), 0, 1) ** 8
        blade_slap *= (1.0 + 2.0 * pulse_mod)

        # Turbine whine (high-frequency gear/compressor tones: ~1.5 kHz - 3 kHz)
        turbine = (
            0.15 * np.sin(2 * np.pi * 1850 * t + 0.05 * np.sin(2 * np.pi * 5 * t)) +
            0.10 * np.sin(2 * np.pi * 2700 * t)
        )

        # Broadband rotor turbulence / vortex noise
        broadband = r.randn(n_samples).astype(np.float32)
        b, a = signal.butter(4, [80 / (self.sr / 2), 1200 / (self.sr / 2)], btype='band')
        broadband = signal.lfilter(b, a, broadband)

        audio = 0.5 * blade_slap + 0.3 * turbine + 0.4 * broadband
        return audio.astype(np.float32)

    def generate_armored_vehicle_engine(
        self,
        duration_sec: float = 3.0,
        rpm: float = 2200.0,
        cylinders: int = 8,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize heavy armored diesel engine rumble and track rattle.
        """
        r = rng if rng is not None else np.random
        n_samples = int(duration_sec * self.sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Fundamental firing frequency
        firing_freq = (rpm / 60.0) * (cylinders / 2.0)

        # Multi-harmonic engine core
        engine = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, 10):
            amp = 1.0 / (h ** 0.8)
            jitter = np.sin(2 * np.pi * 0.5 * t) * 2.0  # slight RPM variation
            engine += amp * np.sin(2 * np.pi * (firing_freq * h + jitter) * t)

        # Track squeal / metal rattle (narrowband high-pitched resonances)
        track_noise = r.randn(n_samples).astype(np.float32)
        b, a = signal.butter(2, [800 / (self.sr / 2), 2400 / (self.sr / 2)], btype='band')
        track_noise = signal.lfilter(b, a, track_noise) * 0.35

        # Sub-bass chassis vibration (<100 Hz)
        sub_bass = np.sin(2 * np.pi * 45 * t) * 0.4

        audio = 0.5 * engine + track_noise + sub_bass
        return audio.astype(np.float32)

    def generate_jet_flyby(
        self,
        duration_sec: float = 3.0,
        center_time: float = 1.5,
        doppler_shift: float = 1.3,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize fighter jet high-speed flyby with Doppler shift and aerodynamic roar.
        """
        r = rng if rng is not None else np.random
        n_samples = int(duration_sec * self.sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Gaussian amplitude envelope centered at flyby moment
        envelope = np.exp(-((t - center_time) ** 2) / (2 * (0.6 ** 2)))

        # Doppler-shifted jet engine whine
        f_approach = 1400.0 * doppler_shift
        f_recede = 1400.0 / doppler_shift
        freq_profile = f_recede + (f_approach - f_recede) / (1.0 + np.exp(6.0 * (t - center_time)))
        phase = 2 * np.pi * np.cumsum(freq_profile) / self.sr
        whine = np.sin(phase) + 0.5 * np.sin(2 * phase)

        # Turbulent jet exhaust roar (colored pink noise)
        white = r.randn(n_samples).astype(np.float32)
        b, a = signal.butter(3, [120 / (self.sr / 2), 3200 / (self.sr / 2)], btype='band')
        roar = signal.lfilter(b, a, white)

        audio = envelope * (0.4 * whine + 0.6 * roar)
        return audio.astype(np.float32)

    def generate_tactical_siren(
        self,
        duration_sec: float = 3.0,
        f_low: float = 400.0,
        f_high: float = 900.0,
        cycle_sec: float = 1.5,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize tactical emergency / air-raid / alarm wailing siren.
        """
        n_samples = int(duration_sec * self.sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        tri_wave = 0.5 * (1.0 + signal.sawtooth(2 * np.pi * (1.0 / cycle_sec) * t, width=0.5))
        freqs = f_low + (f_high - f_low) * tri_wave
        phase = 2 * np.pi * np.cumsum(freqs) / self.sr
        siren = np.sin(phase) + 0.3 * np.sin(3 * phase) + 0.15 * np.sin(5 * phase)
        return siren.astype(np.float32)

    def generate_cockpit_hum(
        self,
        duration_sec: float = 3.0,
        fan_freq: float = 400.0,  # 400 Hz military standard avionics power
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize stationary cockpit & cabin background noise (400 Hz avionics hum + HVAC airflow).
        """
        r = rng if rng is not None else np.random
        n_samples = int(duration_sec * self.sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # 400 Hz avionics ground-loop + harmonics
        hum = (
            0.5 * np.sin(2 * np.pi * fan_freq * t) +
            0.25 * np.sin(2 * np.pi * 2 * fan_freq * t) +
            0.15 * np.sin(2 * np.pi * 3 * fan_freq * t)
        )

        # Airflow hiss (low-passed pink noise)
        pink = np.cumsum(r.randn(n_samples)).astype(np.float32)
        pink = pink - np.mean(pink)
        b, a = signal.butter(3, 800 / (self.sr / 2), btype='lowpass')
        airflow = signal.lfilter(b, a, pink)
        airflow = airflow / (np.max(np.abs(airflow)) + 1e-8)

        audio = 0.4 * hum + 0.6 * airflow
        return audio.astype(np.float32)

    def generate_tactical_radio_static(
        self,
        duration_sec: float = 3.0,
        burstiness: float = 0.3,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize VHF/UHF tactical radio channel background hiss, static bursts, and squelch noise.
        """
        r = rng if rng is not None else np.random
        n_samples = int(duration_sec * self.sr)

        # Broadband channel white noise
        static = r.randn(n_samples).astype(np.float32)
        b, a = signal.butter(4, [300 / (self.sr / 2), 3400 / (self.sr / 2)], btype='band')
        static = signal.lfilter(b, a, static)

        # Intermittent crackle / atmospheric bursts
        crackle_mask = (r.rand(n_samples) < (burstiness * 0.005)).astype(np.float32)
        b_c, a_c = signal.butter(2, 2000 / (self.sr / 2), btype='highpass')
        crackle = signal.lfilter(b_c, a_c, crackle_mask * r.randn(n_samples)) * 3.0

        audio = static + crackle
        return audio.astype(np.float32)

    def generate_babble_noise(
        self,
        duration_sec: float = 3.0,
        num_speakers: int = 8,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Synthesize command post / tactical operations center multi-talker speech babble.
        """
        r = rng if rng is not None else np.random
        n_samples = int(duration_sec * self.sr)
        babble = np.zeros(n_samples, dtype=np.float32)

        for s in range(num_speakers):
            f0 = float(r.uniform(100.0, 240.0))
            t = np.linspace(0, duration_sec, n_samples, endpoint=False)
            formant1 = float(r.uniform(400.0, 800.0))
            formant2 = float(r.uniform(1200.0, 2400.0))

            carrier = np.sin(2 * np.pi * f0 * t)
            mod = 0.5 * np.sin(2 * np.pi * formant1 * t) + 0.3 * np.sin(2 * np.pi * formant2 * t)
            envelope = np.clip(np.sin(2 * np.pi * r.uniform(1.0, 4.0) * t), 0, 1)

            speaker = carrier * mod * envelope
            babble += speaker

        b, a = signal.butter(3, [200 / (self.sr / 2), 3600 / (self.sr / 2)], btype='band')
        babble = signal.lfilter(b, a, babble)
        return babble.astype(np.float32)

    def get_noise_by_type(self, noise_type: str, duration_sec: float = 3.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Fetch synthetic tactical noise by category string."""
        r = rng if rng is not None else np.random
        noise_map = {
            "gunshot": lambda: self.generate_impulsive_gunshot(duration_sec=duration_sec, rng=r),
            "helicopter": lambda: self.generate_helicopter_rotor(duration_sec=duration_sec, rng=r),
            "armored_vehicle": lambda: self.generate_armored_vehicle_engine(duration_sec=duration_sec, rng=r),
            "jet_flyby": lambda: self.generate_jet_flyby(duration_sec=duration_sec, rng=r),
            "siren": lambda: self.generate_tactical_siren(duration_sec=duration_sec, rng=r),
            "cockpit": lambda: self.generate_cockpit_hum(duration_sec=duration_sec, rng=r),
            "radio_static": lambda: self.generate_tactical_radio_static(duration_sec=duration_sec, rng=r),
            "babble": lambda: self.generate_babble_noise(duration_sec=duration_sec, rng=r),
        }
        gen_fn = noise_map.get(noise_type.lower(), lambda: self.generate_cockpit_hum(duration_sec=duration_sec, rng=r))
        noise = gen_fn()
        max_val = np.max(np.abs(noise)) + 1e-8
        return (noise / max_val * 0.8).astype(np.float32)


class AcousticChannelSimulator:
    """
    Simulates defence acoustic environments:
    - RIR Reverberation
    - Tactical Radio Channel Bandpass (300 Hz - 3400 Hz)
    - Non-linear mic clipping & ADC quantization
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def generate_rir(self, rt60_sec: float = 0.25, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate a synthetic Room Impulse Response (RIR) filter."""
        rir_length = max(int(rt60_sec * self.sr), 64)
        t = np.linspace(0, rt60_sec, rir_length)
        decay_constant = 6.91 / max(rt60_sec, 0.05)
        envelope = np.exp(-decay_constant * t)
        r = rng if rng is not None else np.random
        rir = r.randn(rir_length) * envelope
        rir[0] = 1.0
        rir /= np.sqrt(np.sum(rir ** 2) + 1e-8)
        return rir.astype(np.float32)

    def simulate_rir(
        self,
        audio: np.ndarray,
        rt60_sec: float = 0.25,
        rir: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Convolve audio with simulated Room Impulse Response (RIR).
        If `rir` is passed, applies that exact acoustic filter.
        """
        if rir is None:
            rir = self.generate_rir(rt60_sec, rng=rng)
        
        reverberant = signal.fftconvolve(audio, rir, mode='same')
        return reverberant.astype(np.float32)

    def apply_tactical_channel(
        self,
        audio: np.ndarray,
        low_cut: float = 300.0,
        high_cut: float = 3400.0,
        add_clipping: bool = False,
        clip_threshold: float = 0.85
    ) -> np.ndarray:
        """
        Simulate tactical military radio transceiver frequency response and non-linear distortion.
        """
        nyq = self.sr / 2.0
        b, a = signal.butter(3, [low_cut / nyq, high_cut / nyq], btype='band')
        filtered = signal.lfilter(b, a, audio)

        if add_clipping:
            filtered = np.clip(filtered, -clip_threshold, clip_threshold) / clip_threshold

        return filtered.astype(np.float32)


def mix_speech_and_noise(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix clean speech and noise at a specified target SNR (dB).
    Returns (noisy_audio, clean_speech, scaled_noise).
    """
    min_len = min(len(speech), len(noise))
    speech = speech[:min_len].astype(np.float32)
    noise = noise[:min_len].astype(np.float32)

    speech_power = np.mean(speech ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    # BUG-CLASS GUARD (Phase 15 / Section A.9). The `+ 1e-12` epsilons below prevent a
    # divide-by-zero crash, but for an all-zero (digitally silent) noise segment they
    # silently produce scaled_noise == 0, i.e. NO noise is mixed at all and the returned
    # "noisy" signal is bit-identical to the clean reference (measured SNR ~ +100 dB).
    # This is the defect Phase 14 found in the held-out test set (7/500 utterances).
    # The SAME function is used by the training-time online mixer, so training batches are
    # affected at an unmeasured rate. We warn loudly rather than silently no-op. Behaviour
    # is deliberately NOT changed here: altering the mix would change training dynamics and
    # invalidate comparability with all fourteen prior phases. Documented as a known
    # limitation; a proper fix (resample the noise window) belongs with a planned retrain.
    if not np.any(noise):
        warnings.warn(
            f"mix_speech_and_noise: noise segment is all-zero; target_snr_db={target_snr_db:.2f} "
            "CANNOT be achieved and the returned mixture will equal the clean signal. "
            "This sample is degenerate and should be excluded or resampled.",
            RuntimeWarning, stacklevel=2,
        )
    elif np.mean(noise ** 2) < 1e-10:
        warnings.warn(
            f"mix_speech_and_noise: noise segment is near-silent (power={np.mean(noise**2):.3e}); "
            f"the requested SNR of {target_snr_db:.2f} dB may not be achieved accurately.",
            RuntimeWarning, stacklevel=2,
        )

    target_noise_power = speech_power / (10.0 ** (target_snr_db / 10.0))
    scale_factor = np.sqrt(target_noise_power / noise_power)

    scaled_noise = noise * scale_factor
    noisy = speech + scaled_noise

    peak = max(np.max(np.abs(noisy)), np.max(np.abs(speech)), 1e-6)
    if peak > 0.95:
        norm = 0.95 / peak
        noisy *= norm
        speech *= norm
        scaled_noise *= norm

    return noisy.astype(np.float32), speech.astype(np.float32), scaled_noise.astype(np.float32)
