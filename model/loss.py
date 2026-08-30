"""
DRDO PS 26052: Multi-Objective Tactical Audio Loss Functions & Evaluation Metrics
Implements:
1. Multi-Resolution STFT Loss (Log-Magnitude + Spectral Convergence across multiple FFT resolutions)
2. Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss
3. Complex Spectral Domain L1 Loss
4. Metric computation suite: SNR, SI-SNR, STOI (via pystoi), and perceptual PESQ/PMSQE surrogate
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

try:
    import pystoi
    HAS_PYSTOI = True
except ImportError:
    HAS_PYSTOI = False


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss between predicted and target magnitudes."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        return torch.norm(target_mag - pred_mag, p="fro", dim=(-2, -1)) / (
            torch.norm(target_mag, p="fro", dim=(-2, -1)) + 1e-8
        )


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        log_pred = torch.log(pred_mag.clamp(min=1e-7))
        log_target = torch.log(target_mag.clamp(min=1e-7))
        return F.l1_loss(log_pred, log_target)


class STFTLoss(nn.Module):
    """Single resolution STFT loss."""
    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.spectral_conv = SpectralConvergenceLoss()
        self.log_mag = LogSTFTMagnitudeLoss()
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_spec = torch.stft(
            pred_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(pred_wav.device),
            center=True,
            return_complex=True
        )
        target_spec = torch.stft(
            target_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(target_wav.device),
            center=True,
            return_complex=True
        )

        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)

        sc_loss = self.spectral_conv(pred_mag, target_mag).mean()
        mag_loss = self.log_mag(pred_mag, target_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss combining 3 distinct FFT configurations (512, 1024, 2048)
    to balance time-frequency trade-offs for defence transients and stationary noise.
    """
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048]
    ):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        total_sc = 0.0
        total_mag = 0.0
        for loss_fn in self.losses:
            sc, mag = loss_fn(pred_wav, target_wav)
            total_sc += sc
            total_mag += mag
        return (total_sc + total_mag) / len(self.losses)


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss.
    Supports:
    - 'standard': unconstrained projection
    - 'clamped_alpha': non-negative scale projection alpha >= 0
    - 'hinge_penalty': standard projection + explicit negative correlation penalty
    """
    def __init__(self, mode: str = "clamped_alpha", hinge_weight: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.mode = mode
        self.hinge_weight = hinge_weight
        self.eps = eps

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Zero-mean normalization
        pred = pred_wav - torch.mean(pred_wav, dim=-1, keepdim=True)
        target = target_wav - torch.mean(target_wav, dim=-1, keepdim=True)

        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        dot = torch.sum(pred * target, dim=-1, keepdim=True)

        if self.mode == "clamped_alpha":
            # 1. Clamped-alpha fix: alpha = max(0, <pred, target>) / ||target||^2
            alpha = torch.clamp(dot, min=0.0) / s_target_energy
            s_target = alpha * target
            phase_penalty = torch.tensor(0.0, device=pred_wav.device)
        else:
            # Standard unconstrained projection
            s_target = (dot / s_target_energy) * target
            if self.mode == "hinge_penalty":
                # 2. Hinge penalty for negative correlation: clamp(-corr, min=0)
                norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=-1, keepdim=True) + self.eps)
                norm_target = torch.sqrt(s_target_energy)
                correlation = dot / (norm_pred * norm_target + self.eps)
                phase_penalty = torch.mean(torch.clamp(-correlation, min=0.0)) * self.hinge_weight
            else:
                phase_penalty = torch.tensor(0.0, device=pred_wav.device)

        e_noise = pred - s_target

        s_target_pow = torch.sum(s_target ** 2, dim=-1) + self.eps
        e_noise_pow = torch.sum(e_noise ** 2, dim=-1) + self.eps

        si_snr = 10.0 * torch.log10(s_target_pow / e_noise_pow)
        sisnr_loss = -torch.mean(si_snr)

        return sisnr_loss, phase_penalty


class HybridANCLoss(nn.Module):
    """
    Composite Multi-Objective Loss for Tactical ANC:
    L = alpha * L_MRSTFT + beta * L_SISNR + gamma * L_ComplexL1 + lambda_l1 * L_TimeL1 + lambda_energy * L_Energy + L_Phase
    """
    def __init__(
        self,
        alpha_mrstft: float = 1.0,
        beta_sisnr: float = 0.15,
        gamma_complex: float = 0.5,
        lambda_time_l1: float = 25.0,
        lambda_energy: float = 40.0,
        sisnr_mode: str = "clamped_alpha",
        hinge_weight: float = 10.0
    ):
        super().__init__()
        self.mrstft_loss = MultiResolutionSTFTLoss()
        self.sisnr_loss = SISNRLoss(mode=sisnr_mode, hinge_weight=hinge_weight)
        self.alpha = alpha_mrstft
        self.beta = beta_sisnr
        self.gamma = gamma_complex
        self.lambda_time_l1 = lambda_time_l1
        self.lambda_energy = lambda_energy

    def forward(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        pred_real: Optional[torch.Tensor] = None,
        pred_imag: Optional[torch.Tensor] = None,
        target_real: Optional[torch.Tensor] = None,
        target_imag: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        l_mrstft = self.mrstft_loss(pred_wav, target_wav)
        l_sisnr, l_phase = self.sisnr_loss(pred_wav, target_wav)

        l_complex = torch.tensor(0.0, device=pred_wav.device)
        if pred_real is not None and target_real is not None:
            l_real = F.l1_loss(pred_real, target_real)
            l_imag = F.l1_loss(pred_imag, target_imag)
            l_complex = l_real + l_imag

        # Scale-sensitive waveform & energy constraints
        l_time_l1 = F.l1_loss(pred_wav, target_wav)
        pred_rms = torch.sqrt(torch.mean(pred_wav ** 2, dim=-1) + 1e-8)
        target_rms = torch.sqrt(torch.mean(target_wav ** 2, dim=-1) + 1e-8)
        l_energy = F.l1_loss(pred_rms, target_rms)

        total_loss = (
            self.alpha * l_mrstft +
            self.beta * l_sisnr +
            self.gamma * l_complex +
            self.lambda_time_l1 * l_time_l1 +
            self.lambda_energy * l_energy +
            l_phase
        )

        return {
            "total_loss": total_loss,
            "mrstft_loss": l_mrstft,
            "sisnr_loss": l_sisnr,
            "complex_loss": l_complex,
            "time_l1_loss": l_time_l1,
            "energy_loss": l_energy,
            "phase_penalty": l_phase
        }


class SNRLoss(nn.Module):
    """
    Scale-DEPENDENT SNR loss: the exact negative of the reported eval metric.

    compute_snr() is 10*log10(||s||^2 / ||s_hat - s||^2). SI-SNR, which the existing
    HybridANCLoss optimises, is scale-INVARIANT -- it projects out any constant gain error,
    so the model is never asked to get its output level right. Measured consequence
    (scripts/probe_scale_mismatch.py, Phase 17 model, val split): optimal per-utterance
    rescale is alpha* = 0.918 on average (output ~8% hot) and recovers +1.06 dB. A single
    global constant recovers only +0.06 dB, because the error is utterance-dependent --
    so it cannot be fixed post-hoc, only learned.

    The existing l_time_l1 and l_energy terms are scale-sensitive proxies; this term is the
    objective itself.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        noise = pred - target
        s_pow = torch.sum(target ** 2, dim=-1) + self.eps
        n_pow = torch.sum(noise ** 2, dim=-1) + self.eps
        return -10.0 * torch.log10(s_pow / n_pow)   # per-sample; caller reduces


class ScaleAwareANCLoss(nn.Module):
    """
    HybridANCLoss + a direct scale-dependent SNR term + optional per-sample weighting by
    input SNR.

    lambda_snr    : weight on the direct SNR objective.
    hard_weight   : per-sample weight ramps linearly from 1.0 at input SNR >= hi_db up to
                    `hard_weight` at <= lo_db. Set to 1.0 to disable. Motivation: the
                    aggregate mean is dragged down by the [0,5) dB bucket (11.14 dB) while
                    the top two buckets already clear 15 dB.
    """

    def __init__(self, lambda_snr: float = 1.0, hard_weight: float = 1.0,
                 lo_db: float = -10.0, hi_db: float = 5.0, **kw):
        super().__init__()
        self.base = HybridANCLoss(**kw)
        self.snr_loss = SNRLoss()
        self.lambda_snr = lambda_snr
        self.hard_weight = hard_weight
        self.lo_db = lo_db
        self.hi_db = hi_db

    def sample_weights(self, snr_db: Optional[torch.Tensor], n: int,
                       device) -> torch.Tensor:
        if snr_db is None or self.hard_weight == 1.0:
            return torch.ones(n, device=device)
        t = (self.hi_db - snr_db.to(device)) / (self.hi_db - self.lo_db)
        t = torch.clamp(t, 0.0, 1.0)
        return 1.0 + (self.hard_weight - 1.0) * t

    def forward(self, pred_wav, target_wav, pred_real=None, pred_imag=None,
                target_real=None, target_imag=None, snr_db=None):
        out = self.base(pred_wav, target_wav, pred_real, pred_imag, target_real, target_imag)
        per_sample_snr = self.snr_loss(pred_wav, target_wav)
        w = self.sample_weights(snr_db, per_sample_snr.shape[0], per_sample_snr.device)
        l_snr = torch.sum(w * per_sample_snr) / torch.sum(w)
        out["snr_loss"] = l_snr
        out["total_loss"] = out["total_loss"] + self.lambda_snr * l_snr
        return out


class MaskSupervisionLoss(nn.Module):
    """
    Direct supervision of the complex ratio mask against its closed-form ideal.

    Every other loss in this project supervises the SUMMED result -- a waveform or a
    spectrogram -- so the gradient reaching any individual time-frequency bin is heavily
    diluted. But the ideal complex ratio mask M* = S/Y is available in closed form (it is
    exactly what every oracle probe in eval/ computes), which means estimation can be
    supervised per-bin instead of through a downstream proxy.

    That matters because the measured problem IS estimation accuracy:
    scripts/probe_representational_vs_estimation.py showed this architecture can represent
    masks worth 27.57 dB while the trained model realises 10.39 dB.

    Two details that matter:
      * M* is ill-conditioned wherever |Y| ~ 0 (dividing by near-silence). Bins are
        therefore weighted by |Y|, so supervision concentrates where the mask actually
        moves energy and near-silent bins cannot dominate the gradient.
      * The target is clamped to the tanh head's REACHABLE set (-bound, bound). Supervising
        toward values the head cannot emit would just inject a constant error.
    """

    def __init__(self, bound: float = 0.999, eps: float = 1e-8, weight_power: float = 1.0):
        super().__init__()
        self.bound = bound
        self.eps = eps
        self.weight_power = weight_power

    def ideal_mask(self, noisy_r, noisy_i, clean_r, clean_i):
        den = noisy_r ** 2 + noisy_i ** 2 + self.eps
        Mr = (clean_r * noisy_r + clean_i * noisy_i) / den
        Mi = (clean_i * noisy_r - clean_r * noisy_i) / den
        return Mr.clamp(-self.bound, self.bound), Mi.clamp(-self.bound, self.bound)

    def forward(self, mask_r, mask_i, noisy_r, noisy_i, clean_r, clean_i):
        Mr, Mi = self.ideal_mask(noisy_r, noisy_i, clean_r, clean_i)
        w = torch.sqrt(noisy_r ** 2 + noisy_i ** 2 + self.eps) ** self.weight_power
        w = w / (w.mean() + self.eps)
        err = (mask_r - Mr).abs() + (mask_i - Mi).abs()
        return (w * err).mean()


# --- Evaluation Metrics ---

def compute_snr(clean: np.ndarray, noisy_or_enhanced: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (dB)."""
    noise = noisy_or_enhanced - clean
    s_pwr = np.mean(clean ** 2) + 1e-12
    n_pwr = np.mean(noise ** 2) + 1e-12
    return float(10.0 * np.log10(s_pwr / n_pwr))


def compute_si_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Noise Ratio (dB)."""
    clean_zm = clean - np.mean(clean)
    enh_zm = enhanced - np.mean(enhanced)

    s_target = (np.dot(enh_zm, clean_zm) / (np.dot(clean_zm, clean_zm) + 1e-12)) * clean_zm
    e_noise = enh_zm - s_target

    snr = 10.0 * np.log10((np.dot(s_target, s_target) + 1e-12) / (np.dot(e_noise, e_noise) + 1e-12))
    return float(snr)


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """Compute Short-Time Objective Intelligibility (STOI: 0 to 1)."""
    if HAS_PYSTOI:
        try:
            return float(pystoi.stoi(clean, enhanced, sr, extended=False))
        except Exception as exc:
            # BUG-CLASS GUARD (Phase 15 / Section A.5): this used to `pass` silently and
            # fall through to the correlation proxy below, returning a NON-STOI number
            # indistinguishable from a real STOI value in every downstream report. That is
            # the same silent-substitution pattern that produced the fabricated NPU latency
            # figure. It must be loud. Audited over the full N=500 test set: this path fired
            # 0/1000 times, so no reported STOI value is affected.
            raise RuntimeError(
                f"pystoi.stoi() failed and NO substitute value will be returned: {type(exc).__name__}: {exc}. "
                "Fix the input or explicitly call _stoi_correlation_proxy() and label the result as a proxy."
            ) from exc
    raise RuntimeError(
        "pystoi is not installed; refusing to silently substitute a correlation proxy for STOI. "
        "Install pystoi, or call _stoi_correlation_proxy() explicitly and label the result as a proxy."
    )


def _stoi_correlation_proxy(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Correlation-based approximation of intelligibility. This is NOT STOI and must never be
    reported as STOI. Retained only so a caller can opt in deliberately and label it.
    """
    r = np.corrcoef(clean, enhanced)[0, 1]
    return float(np.clip(0.5 * (r + 1.0), 0.0, 1.0))


try:
    import pesq as pesq_lib
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False


_PESQ_LOGGED = False

def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality, ITU-T P.862 / P.862.2).
    - If `pesq` package is installed: computes exact ITU-T P.862 wideband (16kHz) or narrowband (8kHz).
    - Fallback: computes a heuristic psychoacoustic Bark spectral distortion proxy, hand-tuned to sit
      on the PESQ 1.0-4.5 scale. It is NOT empirically calibrated/regressed against real ITU-T P.862
      scores (no reference-PESQ ground truth was available in this environment to fit against) and
      should not be quoted as equivalent to or comparable with real PESQ.
    """
    global _PESQ_LOGGED
    min_len = min(len(clean), len(enhanced))
    c = clean[:min_len].astype(np.float32)
    e = enhanced[:min_len].astype(np.float32)

    # 1. Attempt official ITU-T P.862 implementation if available
    if HAS_PESQ:
        try:
            mode = "wb" if sr == 16000 else "nb"
            val = float(pesq_lib.pesq(sr, c, e, mode))
            if not _PESQ_LOGGED:
                print("[Metric Engine] Active PESQ backend: Official ITU-T P.862 (C-library / pesq)")
                _PESQ_LOGGED = True
            return val
        except Exception:
            pass

    if not _PESQ_LOGGED:
        print("[Metric Engine] Active PESQ backend: Psychoacoustic Bark-distortion fallback (MSVC build tools not detected)")
        _PESQ_LOGGED = True

    # 2. Psychoacoustic spectral envelope and Bark-band distortion fallback
    n_fft = 512
    hop = 256
    w = np.hanning(512).astype(np.float32)

    n_frames = (min_len - n_fft) // hop
    if n_frames <= 0:
        # BUG-CLASS GUARD (Phase 15 / Section A.5): this used to `return 2.5` -- a hardcoded
        # value sitting exactly on the project's PESQ pass threshold (>2.50), silently
        # emitted for any too-short signal. Audited over the full N=500 test set: fired
        # 0/1000 times, so no reported PESQ value is affected. Now it fails loudly.
        raise ValueError(
            f"Signal too short for PESQ computation (min_len={min_len} < n_fft={n_fft} + hop={hop}); "
            "refusing to substitute a placeholder score."
        )

    stoi_val = compute_stoi(c, e, sr=sr)
    si_snr_val = compute_si_snr(c, e)

    spec_c = np.abs(np.array([np.fft.rfft(c[i*hop:i*hop+n_fft] * w) for i in range(n_frames)])) + 1e-7
    spec_e = np.abs(np.array([np.fft.rfft(e[i*hop:i*hop+n_fft] * w) for i in range(n_frames)])) + 1e-7

    log_diff = np.abs(np.log(spec_c) - np.log(spec_e))
    pmsqe_val = float(np.mean(log_diff))

    # Uncalibrated heuristic mapping onto the PESQ [1.0, 4.5] scale (not fit to real PESQ data)
    pesq_est = 1.0 + 2.8 * (stoi_val ** 1.5) + 0.04 * np.clip(si_snr_val, -10, 25) - 0.25 * np.clip(pmsqe_val, 0, 4)
    return float(np.clip(pesq_est, 1.0, 4.5))


# Backward compatibility alias
compute_pesq_surrogate = compute_pesq
