"""
Causal full-band + sub-band complex ANC network (FullSubNet-style), DRDO PS 26052.

The problem statement requires that models "process both full-band and sub-band features to
capture global and local dependencies". The existing CausalANCNet is full-band only: its
encoder strides across the whole frequency axis, so every output bin is produced from a
globally-pooled representation and no path in the network looks at a single frequency bin's
own local neighbourhood over time.

This module adds the missing sub-band path and keeps the full-band path byte-identical to
CausalANCNet, so the two can be compared with capacity as the only meaningful difference.

    FULL-BAND PATH   STFT -> complex conv encoder -> causal GRU -> decoder -> per-bin PRIOR
                     (global spectral-temporal context; identical topology to CausalANCNet)

    SUB-BAND PATH    for each frequency bin f, gather the local neighbourhood
                     [f-K .. f+K] of the noisy complex spectrum, concatenate the full-band
                     prior at f, and run a SHARED causal GRU along time.
                     One GRU, applied independently to all 257 bins -> local detail at low
                     parameter cost.

    OUTPUT           complex ratio mask (Mr, Mi) bounded by tanh, applied to the noisy
                     spectrum, then iSTFT.

Causality: the full-band path uses the project's CausalConv2d/CausalConvTranspose2d (no
future frames) and a unidirectional GRU. The sub-band GRU is likewise unidirectional over
time. Frequency-axis context is NOT a causality violation -- the whole frame is available at
frame time; only the time axis must be causal.

Interface (stft_forward / istft_forward / forward_spec / forward) matches CausalANCNet
exactly, so existing training and evaluation scripts work unchanged.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.causal_anc_net import CausalConv2d, CausalConvTranspose2d, CausalFrameNorm


class CausalANCSubbandNet(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        norm_type: str = "batch",
        mask_bound: float = 1.0,
        mask_mode: str = "component_tanh",
        subband_k: int = 7,
        subband_hidden: int = 48,
        subband_layers: int = 1,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.norm_type = norm_type
        self.mask_bound = mask_bound
        self.mask_mode = mask_mode
        self.subband_k = subband_k

        self.register_buffer("window", torch.hann_window(win_length))

        def make_norm(channels):
            if norm_type == "group":
                return CausalFrameNorm(channels, norm_type="group", num_groups=4)
            return nn.BatchNorm2d(channels)

        # ---------------- FULL-BAND PATH (topology identical to CausalANCNet) -------------
        self.enc_conv1 = CausalConv2d(2, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn1 = make_norm(16)
        self.enc_act1 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_conv2 = CausalConv2d(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn2 = make_norm(32)
        self.enc_act2 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_conv3 = CausalConv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn3 = make_norm(64)
        self.enc_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_freq_dim = 33
        self.rnn_in_dim = 64 * self.enc_freq_dim

        self.rnn_proj_in = nn.Linear(self.rnn_in_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=num_gru_layers, batch_first=True)
        self.rnn_proj_out = nn.Linear(hidden_dim, self.rnn_in_dim)
        self.rnn_act = nn.LeakyReLU(0.1, inplace=True)

        self.dec_deconv3 = CausalConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn3 = make_norm(32)
        self.dec_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.dec_deconv2 = CausalConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn2 = make_norm(16)
        self.dec_act2 = nn.LeakyReLU(0.1, inplace=True)

        # Full-band head now emits a PRIOR rather than the final mask.
        self.dec_deconv1 = CausalConvTranspose2d(32, 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))

        # ---------------- SUB-BAND PATH ---------------------------------------------------
        # Per-bin input: the (2K+1)-bin complex neighbourhood (real+imag) + the full-band
        # prior at that bin (2 values). One GRU shared across all 257 bins.
        self.subband_in_dim = 2 * (2 * subband_k + 1) + 2
        self.subband_gru = nn.GRU(input_size=self.subband_in_dim, hidden_size=subband_hidden,
                                  num_layers=subband_layers, batch_first=True)
        self.subband_out = nn.Linear(subband_hidden, 2)

        self.mask_act = nn.Tanh()

    # ------------------------------------------------------------------ STFT / iSTFT ----
    def stft_forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length,
                          self.window.to(waveform.device), center=True, return_complex=True)
        return spec.real, spec.imag, torch.abs(spec)

    def istft_forward(self, real: torch.Tensor, imag: torch.Tensor, original_len: int) -> torch.Tensor:
        return torch.istft(torch.complex(real, imag), self.n_fft, self.hop_length,
                           self.win_length, self.window.to(real.device),
                           center=True, length=original_len)

    # ------------------------------------------------------------------ sub-band --------
    def _subband_features(self, n_r: torch.Tensor, n_i: torch.Tensor,
                          prior: torch.Tensor) -> torch.Tensor:
        """
        n_r, n_i : (B, F, T) noisy complex spectrum
        prior    : (B, 2, F, T) full-band per-bin prior
        returns  : (B*F, T, subband_in_dim)
        """
        B, Fr, T = n_r.shape
        K = self.subband_k

        x = torch.stack([n_r, n_i], dim=1)                       # (B, 2, F, T)

        # Causal per-bin magnitude normalisation: divide by the CUMULATIVE mean magnitude
        # along time. Uses only past and present frames, so it is safe for streaming.
        mag = torch.sqrt(n_r ** 2 + n_i ** 2 + 1e-10)            # (B, F, T)
        cum = torch.cumsum(mag, dim=-1)
        cnt = torch.arange(1, T + 1, device=mag.device, dtype=mag.dtype).view(1, 1, T)
        scale = (cum / cnt).clamp_min(1e-6).unsqueeze(1)         # (B, 1, F, T)
        x = x / scale

        # Gather the [f-K .. f+K] neighbourhood for every bin.
        xp = F.pad(x, (0, 0, K, K), mode="reflect")              # pad freq axis -> (B,2,F+2K,T)
        nb = xp.unfold(2, 2 * K + 1, 1)                          # (B, 2, F, T, 2K+1)
        nb = nb.permute(0, 2, 3, 1, 4).reshape(B, Fr, T, 2 * (2 * K + 1))

        pr = prior.permute(0, 2, 3, 1)                           # (B, F, T, 2)
        feat = torch.cat([nb, pr], dim=-1)                       # (B, F, T, in_dim)
        return feat.reshape(B * Fr, T, self.subband_in_dim)

    # ------------------------------------------------------------------ forward ---------
    def forward_spec(self, real: torch.Tensor, imag: torch.Tensor,
                     hidden: Optional[torch.Tensor] = None):
        B, Fr, T = real.shape
        x = torch.stack([real, imag], dim=1)                     # (B, 2, F, T)

        # ---- full-band path
        e1 = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        e2 = self.enc_act2(self.enc_bn2(self.enc_conv2(e1)))
        e3 = self.enc_act3(self.enc_bn3(self.enc_conv3(e2)))

        r = e3.permute(0, 3, 1, 2).reshape(B, T, self.rnn_in_dim)
        r = self.rnn_proj_in(r)
        r, hidden = self.gru(r, hidden)
        r = self.rnn_act(self.rnn_proj_out(r))
        r = r.reshape(B, T, 64, self.enc_freq_dim).permute(0, 2, 3, 1)

        d3 = self.dec_act3(self.dec_bn3(self.dec_deconv3(torch.cat([r, e3], dim=1))))
        d2 = self.dec_act2(self.dec_bn2(self.dec_deconv2(torch.cat([d3, e2], dim=1))))
        prior = self.dec_deconv1(torch.cat([d2, e1], dim=1))     # (B, 2, F, T)

        # ---- sub-band path
        feat = self._subband_features(real, imag, prior)         # (B*F, T, D)
        s, _ = self.subband_gru(feat)
        m = self.subband_out(s).reshape(B, Fr, T, 2).permute(0, 3, 1, 2)   # (B, 2, F, T)

        m = self.mask_act(m) * self.mask_bound
        mask_r, mask_i = m[:, 0], m[:, 1]

        enh_r = real * mask_r - imag * mask_i
        enh_i = real * mask_i + imag * mask_r
        return enh_r, enh_i, mask_r, mask_i, hidden

    def forward(self, waveform: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        n = waveform.shape[-1]
        r, i, _ = self.stft_forward(waveform)
        enh_r, enh_i, _, _, hidden = self.forward_spec(r, i, hidden)
        return self.istft_forward(enh_r, enh_i, n), enh_r, enh_i, hidden


def build_subband_model(**kw) -> CausalANCSubbandNet:
    return CausalANCSubbandNet(**kw)
