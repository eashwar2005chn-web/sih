"""
DRDO PS 26052: Causal Deep Complex Recurrent Network (Causal-CRN) for Real-Time ANC
Architecture Features:
1. Strict Temporal Causality in Convolutions (zero lookahead into future frames)
2. Complex-Domain Processing with Complex Ratio Masking (cRM) for optimal phase preservation
3. Causal GRU recurrent core for temporal modeling with lightweight parameter count
4. Direct streaming support (stateful frame-by-frame inference for sub-15ms latency)
5. INT8-quantization friendly architecture (LeakyReLU/Tanh, standard convolutions, linear ops)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class CausalConv2d(nn.Module):
    """
    2D Convolution with strict time-causality (padding ONLY on past time frames).
    Input format: (Batch, Channels, Frequency, Time)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),  # (freq_kernel, time_kernel)
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 0)       # (freq_pad, time_pad=0)
    ):
        super().__init__()
        self.freq_kernel, self.time_kernel = kernel_size
        self.stride = stride
        self.freq_pad = padding[0]
        self.time_pad = self.time_kernel - 1  # Causal left-pad amount

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.freq_pad, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, F, T)
        if self.time_pad > 0:
            x = F.pad(x, (self.time_pad, 0, 0, 0))
        return self.conv(x)


class CausalConvTranspose2d(nn.Module):
    """
    2D Transposed Convolution with strict time-causality.
    Input format: (Batch, Channels, Frequency, Time)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 0)
    ):
        super().__init__()
        self.freq_kernel, self.time_kernel = kernel_size
        self.stride = stride
        self.freq_pad = padding[0]
        self.time_pad = self.time_kernel - 1

        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.freq_pad, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, F, T)
        out = self.deconv(x)
        if self.time_pad > 0:
            out = out[:, :, :, :-self.time_pad]
        return out


class CausalFrameNorm(nn.Module):
    """
    Per-frame Group/Layer Normalization that normalizes strictly within each time frame t.
    Guarantees 0.0 ms lookahead and zero future-frame statistics leak during both train & eval.
    Input: (B, C, F, T)
    """
    def __init__(self, num_channels: int, norm_type: str = "group", num_groups: int = 4):
        super().__init__()
        self.norm_type = norm_type
        self.num_channels = num_channels
        groups = min(num_groups, num_channels)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B, C, F_dim, T = x.shape
        # Permute/reshape to (B*T, C, F_dim)
        x_perm = x.permute(0, 3, 1, 2).contiguous().view(B * T, C, F_dim)
        out = self.norm(x_perm)
        # Reshape back to (B, C, F_dim, T)
        return out.view(B, T, C, F_dim).permute(0, 2, 3, 1).contiguous()


class CausalANCNet(nn.Module):
    """
    Real-Time AI/ML Causal Complex ANC Model for Tactical Defence Audio.
    Uses STFT -> Complex Encoder -> Causal GRU -> Complex Decoder -> cRM Mask -> iSTFT.
    Supports configurable causal frame normalization ('batch', 'group'), mask bounds, and mask modes ('component_tanh', 'polar_tanh').
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        norm_type: str = "batch",
        mask_bound: float = 1.0,
        mask_mode: str = "component_tanh"
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1  # 257 frequency bins for 512 FFT
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.norm_type = norm_type
        self.mask_bound = mask_bound
        self.mask_mode = mask_mode

        # Window buffer
        self.register_buffer("window", torch.hann_window(win_length))

        # Helper for norm creation
        def make_norm(channels):
            if norm_type == "group":
                return CausalFrameNorm(channels, norm_type="group", num_groups=4)
            return nn.BatchNorm2d(channels)

        # --- Encoder (Processes Real & Imaginary spectral components: 2 input channels) ---
        # Layer 1: 2 -> 16 (Freq: 257 -> 129)
        self.enc_conv1 = CausalConv2d(2, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn1 = make_norm(16)
        self.enc_act1 = nn.LeakyReLU(0.1, inplace=True)

        # Layer 2: 16 -> 32 (Freq: 129 -> 65)
        self.enc_conv2 = CausalConv2d(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn2 = make_norm(32)
        self.enc_act2 = nn.LeakyReLU(0.1, inplace=True)

        # Layer 3: 32 -> 64 (Freq: 65 -> 33)
        self.enc_conv3 = CausalConv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn3 = make_norm(64)
        self.enc_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_freq_dim = 33
        self.rnn_in_dim = 64 * self.enc_freq_dim  # 64 * 33 = 2112

        # --- Recurrent Core (Causal GRU for low latency & fast edge compute) ---
        self.rnn_proj_in = nn.Linear(self.rnn_in_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True
        )
        self.rnn_proj_out = nn.Linear(hidden_dim, self.rnn_in_dim)
        self.rnn_act = nn.LeakyReLU(0.1, inplace=True)

        # --- Decoder (Mirror of Encoder with Skip Connections) ---
        # Layer 3 Deconv: (64 + 64 skip) = 128 -> 32 (Freq: 33 -> 65)
        self.dec_deconv3 = CausalConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn3 = make_norm(32)
        self.dec_act3 = nn.LeakyReLU(0.1, inplace=True)

        # Layer 2 Deconv: (32 + 32 skip) = 64 -> 16 (Freq: 65 -> 129)
        self.dec_deconv2 = CausalConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn2 = make_norm(16)
        self.dec_act2 = nn.LeakyReLU(0.1, inplace=True)

        # Layer 1 Deconv: (16 + 16 skip) = 32 -> 2 (Real & Imaginary Mask components: Mr, Mi) (Freq: 129 -> 257)
        self.dec_deconv1 = CausalConvTranspose2d(32, 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        
        # Mask activation: Bounded complex mask
        self.mask_act = nn.Tanh()

    def stft_forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute STFT.
        Input: (B, TimeSamples)
        Returns:
            real: (B, Freq, TimeFrames)
            imag: (B, Freq, TimeFrames)
            complex_spec: (B, Freq, TimeFrames) complex64
        """
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True
        )
        return spec.real, spec.imag, spec

    def istft_forward(self, real: torch.Tensor, imag: torch.Tensor, original_len: int) -> torch.Tensor:
        """
        Synthesize waveform via inverse STFT.
        """
        spec = torch.complex(real, imag)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(real.device),
            center=True,
            length=original_len
        )
        return wav

    def forward_spec(
        self,
        noisy_real: torch.Tensor,
        noisy_imag: torch.Tensor,
        h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Core neural network forward pass operating directly on Real & Imaginary spectral tensors.
        Inputs:
            noisy_real: (B, F=257, T)
            noisy_imag: (B, F=257, T)
            h_state: (num_layers, B, hidden_dim) optional GRU hidden state for streaming
        Returns:
            enhanced_real: (B, F=257, T)
            enhanced_imag: (B, F=257, T)
            mask_real: (B, F=257, T)
            mask_imag: (B, F=257, T)
            next_h_state: updated GRU state
        """
        B, F_dim, T_dim = noisy_real.shape

        # Stack into 2-channel 2D tensor: (B, 2, F, T)
        x = torch.stack([noisy_real, noisy_imag], dim=1)

        # Encoder forward pass with skip connections
        e1 = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))          # (B, 16, 129, T)
        e2 = self.enc_act2(self.enc_bn2(self.enc_conv2(e1)))         # (B, 32, 65, T)
        e3 = self.enc_act3(self.enc_bn3(self.enc_conv3(e2)))         # (B, 64, 33, T)

        # Reshape for GRU: (B, 64 * 33, T) -> (B, T, 2112)
        gru_in = e3.permute(0, 3, 1, 2).contiguous().view(B, T_dim, -1)
        gru_proj = self.rnn_proj_in(gru_in)                          # (B, T, hidden_dim)

        if h_state is not None:
            gru_out, next_h = self.gru(gru_proj, h_state)
        else:
            gru_out, next_h = self.gru(gru_proj)

        gru_rec = self.rnn_act(self.rnn_proj_out(gru_out))           # (B, T, 2112)
        d3_in = gru_rec.view(B, T_dim, 64, self.enc_freq_dim).permute(0, 2, 3, 1).contiguous()  # (B, 64, 33, T)

        # Decoder forward pass with concatenation skips
        d3_cat = torch.cat([d3_in, e3], dim=1)                       # (B, 128, 33, T)
        d3 = self.dec_act3(self.dec_bn3(self.dec_deconv3(d3_cat)))   # (B, 32, 65, T)
        if d3.shape[2] != e2.shape[2]:
            d3 = d3[:, :, :e2.shape[2], :]

        d2_cat = torch.cat([d3, e2], dim=1)                          # (B, 64, 65, T)
        d2 = self.dec_act2(self.dec_bn2(self.dec_deconv2(d2_cat)))   # (B, 16, 129, T)
        if d2.shape[2] != e1.shape[2]:
            d2 = d2[:, :, :e1.shape[2], :]

        d1_cat = torch.cat([d2, e1], dim=1)                          # (B, 32, 129, T)
        mask = self.dec_deconv1(d1_cat)                              # (B, 2, 257, T)
        if mask.shape[2] != F_dim:
            mask = mask[:, :, :F_dim, :]
        if self.mask_mode == "component_clamp":
            # Phase 16: LINEAR-in-interior bounding at the same |M| <= K limit.
            #
            # Motivation (scripts/oracle_mask_shape_probe.py, validation split): pushing the
            # IDEAL complex ratio mask through tanh caps oracle output SNR at 12.99 dB -- which
            # is essentially exactly where every trained model in Phases 1-11 landed (12.75 dB).
            # Clamping that same ideal mask to the SAME [-1, 1] bound instead reaches 28.52 dB.
            # The ceiling is therefore the CURVATURE of tanh inside the bound, not the bound:
            # tanh(1) = 0.76, so a bin that ought to pass at unity gain never passes more than
            # 76% of it, and speech energy is systematically removed.
            #
            # Phases 9-10 swept the BOUND (K = 1.5 .. 3.0) and it monotonically hurt, because a
            # wider bound lets the network amplify noise-dominated bins. This is the orthogonal
            # change: keep the unit bound that provably regularises, drop the interior squashing.
            # hardtanh is exactly clamp() with gradient 1 inside the bound and 0 outside.
            K = self.mask_bound
            mask = torch.nn.functional.hardtanh(mask, -K, K)
            mask_r = mask[:, 0, :, :]
            mask_i = mask[:, 1, :, :]
        elif self.mask_mode == "component_leaky_clamp":
            # As above, but with a small slope retained outside the bound so a saturated unit
            # still receives gradient and can recover. Guards against the dead-unit failure mode
            # that a hard clamp can introduce early in training.
            K = self.mask_bound
            slope = 0.05
            mask = torch.nn.functional.hardtanh(mask, -K, K) + slope * (mask - mask.clamp(-K, K))
            mask_r = mask[:, 0, :, :]
            mask_i = mask[:, 1, :, :]
        elif self.mask_mode == "polar_tanh":
            # Polar magnitude bounding with exact phase preservation
            raw_r = mask[:, 0, :, :]
            raw_i = mask[:, 1, :, :]
            raw_mag = torch.sqrt(raw_r ** 2 + raw_i ** 2 + 1e-12)
            phase_cos = raw_r / raw_mag
            phase_sin = raw_i / raw_mag

            K = self.mask_bound
            bounded_mag = K * torch.tanh(raw_mag / K)
            mask_r = bounded_mag * phase_cos
            mask_i = bounded_mag * phase_sin
        else:
            if self.mask_bound != 1.0:
                mask = self.mask_bound * self.mask_act(mask / self.mask_bound)
            else:
                mask = self.mask_act(mask)                                   # Bounded in [-1, 1]
            mask_r = mask[:, 0, :, :]
            mask_i = mask[:, 1, :, :]

        # Complex Ratio Mask (cRM) multiplication:
        # S_r = Y_r * M_r - Y_i * M_i
        # S_i = Y_r * M_i + Y_i * M_r
        enh_r = noisy_real * mask_r - noisy_imag * mask_i
        enh_i = noisy_real * mask_i + noisy_imag * mask_r

        return enh_r, enh_i, mask_r, mask_i, next_h

    def forward(
        self,
        noisy_wav: torch.Tensor,
        h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        End-to-end forward pass from noisy waveform to enhanced waveform.
        Input:
            noisy_wav: (B, TimeSamples)
            h_state: optional recurrent state
        Returns:
            enhanced_wav: (B, TimeSamples)
            enh_real: (B, F, T)
            enh_imag: (B, F, T)
            next_h_state: updated recurrent state
        """
        orig_len = noisy_wav.shape[-1]
        n_real, n_imag, _ = self.stft_forward(noisy_wav)
        enh_r, enh_i, mask_r, mask_i, next_h = self.forward_spec(n_real, n_imag, h_state=h_state)
        enh_wav = self.istft_forward(enh_r, enh_i, original_len=orig_len)
        return enh_wav, enh_r, enh_i, next_h


def build_causal_anc_model(
    hidden_dim: int = 128,
    num_gru_layers: int = 2,
    norm_type: str = "batch",
    mask_bound: float = 1.0,
    mask_mode: str = "component_tanh"
) -> CausalANCNet:
    """Factory helper to build baseline model."""
    model = CausalANCNet(
        n_fft=512,
        hop_length=256,
        win_length=512,
        hidden_dim=hidden_dim,
        num_gru_layers=num_gru_layers,
        norm_type=norm_type,
        mask_bound=mask_bound,
        mask_mode=mask_mode
    )
    return model


class CausalANCScaledNet(nn.Module):
    """
    Scaled Capacity Variant of Causal Complex ANC Model (Wide C-CRN, ~3.35M parameters).
    Features wider complex convolutions [32, 64, 128] + 256-dim Causal GRU core.
    Preserves strict temporal causality (0.0 ms lookahead).
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        hidden_dim: int = 256,
        num_gru_layers: int = 2
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1  # 257 frequency bins
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        # Window buffer
        self.register_buffer("window", torch.hann_window(win_length))

        # --- Scaled Complex Encoder (2 -> 32 -> 64 -> 128) ---
        self.enc_conv1 = CausalConv2d(2, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_act1 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_conv2 = CausalConv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_act2 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_conv3 = CausalConv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_freq_dim = 33
        self.rnn_in_dim = 128 * self.enc_freq_dim  # 128 * 33 = 4224

        # --- Scaled Recurrent Core (256-dim Causal GRU) ---
        self.rnn_proj_in = nn.Linear(self.rnn_in_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True
        )
        self.rnn_proj_out = nn.Linear(hidden_dim, self.rnn_in_dim)
        self.rnn_act = nn.LeakyReLU(0.1, inplace=True)

        # --- Scaled Complex Decoder with Skip Connections ---
        self.dec_deconv3 = CausalConvTranspose2d(256, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.dec_deconv2 = CausalConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_act2 = nn.LeakyReLU(0.1, inplace=True)

        self.dec_deconv1 = CausalConvTranspose2d(64, 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.mask_act = nn.Tanh()

    def stft_forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True
        )
        return spec.real, spec.imag, spec

    def istft_forward(self, real: torch.Tensor, imag: torch.Tensor, original_len: int) -> torch.Tensor:
        spec = torch.complex(real, imag)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(real.device),
            center=True,
            length=original_len
        )
        return wav

    def forward_spec(
        self,
        noisy_real: torch.Tensor,
        noisy_imag: torch.Tensor,
        h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, F_dim, T = noisy_real.shape
        x = torch.stack([noisy_real, noisy_imag], dim=1)

        # Encoder
        e1 = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        e2 = self.enc_act2(self.enc_bn2(self.enc_conv2(e1)))
        e3 = self.enc_act3(self.enc_bn3(self.enc_conv3(e2)))

        # Reshape for GRU
        e3_flat = e3.permute(0, 3, 1, 2).contiguous().view(B, T, self.rnn_in_dim)
        rnn_in = self.rnn_proj_in(e3_flat)
        rnn_out, next_h = self.gru(rnn_in, h_state)
        rnn_out = self.rnn_act(self.rnn_proj_out(rnn_out))
        d3_in = rnn_out.view(B, T, 128, self.enc_freq_dim).permute(0, 2, 3, 1).contiguous()

        # Decoder with skip connections
        d3_cat = torch.cat([d3_in, e3], dim=1)
        d3 = self.dec_act3(self.dec_bn3(self.dec_deconv3(d3_cat)))
        if d3.shape[2] != e2.shape[2]:
            d3 = d3[:, :, :e2.shape[2], :]

        d2_cat = torch.cat([d3, e2], dim=1)
        d2 = self.dec_act2(self.dec_bn2(self.dec_deconv2(d2_cat)))
        if d2.shape[2] != e1.shape[2]:
            d2 = d2[:, :, :e1.shape[2], :]

        d1_cat = torch.cat([d2, e1], dim=1)
        mask = self.dec_deconv1(d1_cat)
        if mask.shape[2] != F_dim:
            mask = mask[:, :, :F_dim, :]

        mask = self.mask_act(mask)
        mask_r = mask[:, 0, :, :]
        mask_i = mask[:, 1, :, :]

        enh_r = noisy_real * mask_r - noisy_imag * mask_i
        enh_i = noisy_real * mask_i + noisy_imag * mask_r

        return enh_r, enh_i, mask_r, mask_i, next_h

    def forward(
        self,
        noisy_wav: torch.Tensor,
        h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        orig_len = noisy_wav.shape[-1]
        n_real, n_imag, _ = self.stft_forward(noisy_wav)
        enh_r, enh_i, mask_r, mask_i, next_h = self.forward_spec(n_real, n_imag, h_state=h_state)
        enh_wav = self.istft_forward(enh_r, enh_i, original_len=orig_len)
        return enh_wav, enh_r, enh_i, next_h


def build_scaled_causal_anc_model(
    hidden_dim: int = 256,
    num_gru_layers: int = 2
) -> CausalANCScaledNet:
    """Factory helper to build scaled capacity model."""
    model = CausalANCScaledNet(
        n_fft=512,
        hop_length=256,
        win_length=512,
        hidden_dim=hidden_dim,
        num_gru_layers=num_gru_layers
    )
    return model


class CausalTCNBlock(nn.Module):
    """
    Causal Dilated Depthwise-Separable Temporal Convolution Block.
    Guarantees strict left-context-only causality via explicit left-padding:
    Pad: ( (kernel_size - 1) * dilation, 0 ) along time.
    Structure:
      Left-padded Depthwise Conv1D (groups=C) -> Norm -> PReLU -> Pointwise Conv1D (1x1) -> Norm -> PReLU -> Residual
    """
    def __init__(
        self,
        channels: int = 128,
        dilation: int = 1,
        kernel_size: int = 3,
        norm_type: str = "batch"
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.pad_len = (kernel_size - 1) * dilation

        # Depthwise 1D conv along time axis
        self.dw_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels) if norm_type == "batch" else nn.GroupNorm(4, channels)
        self.act1 = nn.PReLU(channels)

        # Pointwise 1D conv
        self.pw_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels) if norm_type == "batch" else nn.GroupNorm(4, channels)
        self.act2 = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        x_pad = F.pad(x, (self.pad_len, 0))
        out = self.act1(self.bn1(self.dw_conv(x_pad)))
        out = self.act2(self.bn2(self.pw_conv(out)))
        return out + residual


class CausalANCTCNNet(nn.Module):
    """
    Real-Time AI/ML Causal Complex ANC Model with Dilated Temporal Convolutional Network (TCN) Stack.
    Pipeline: STFT -> Complex 2D Encoder -> Linear Proj -> Causal Dilated TCN Stack -> Causal GRU -> Linear Proj -> Complex 2D Decoder -> cRM Mask.
    Expands the feedforward temporal receptive field across dilations [1, 2, 4, 8, 16] (~62 frames / 496 ms context)
    before feeding into the recurrent memory core.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        norm_type: str = "batch",
        mask_bound: float = 1.0,
        mask_mode: str = "component_tanh"
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1  # 257
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.tcn_dilations = tcn_dilations
        self.norm_type = norm_type
        self.mask_bound = mask_bound
        self.mask_mode = mask_mode

        self.register_buffer("window", torch.hann_window(win_length))

        def make_norm2d(channels):
            if norm_type == "group":
                return CausalFrameNorm(channels, norm_type="group", num_groups=4)
            return nn.BatchNorm2d(channels)

        # Encoder (2 -> 16 -> 32 -> 64)
        self.enc_conv1 = CausalConv2d(2, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn1 = make_norm2d(16)
        self.enc_act1 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_conv2 = CausalConv2d(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn2 = make_norm2d(32)
        self.enc_act2 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_conv3 = CausalConv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.enc_bn3 = make_norm2d(64)
        self.enc_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.enc_freq_dim = 33
        self.rnn_in_dim = 64 * self.enc_freq_dim  # 2112

        # Linear Projection to hidden_dim
        self.rnn_proj_in = nn.Linear(self.rnn_in_dim, hidden_dim)

        # Causal Dilated TCN Stack (Temporal receptive field expansion)
        tcn_blocks = []
        for d in tcn_dilations:
            tcn_blocks.append(CausalTCNBlock(channels=hidden_dim, dilation=d, kernel_size=3, norm_type=norm_type))
        self.tcn_stack = nn.Sequential(*tcn_blocks)

        # Causal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True
        )

        # Linear Projection back to frequency feature maps
        self.rnn_proj_out = nn.Linear(hidden_dim, self.rnn_in_dim)
        self.rnn_act = nn.LeakyReLU(0.1, inplace=True)

        # Decoder (Mirror with Skips)
        self.dec_deconv3 = CausalConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn3 = make_norm2d(32)
        self.dec_act3 = nn.LeakyReLU(0.1, inplace=True)

        self.dec_deconv2 = CausalConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.dec_bn2 = make_norm2d(16)
        self.dec_act2 = nn.LeakyReLU(0.1, inplace=True)

        self.dec_deconv1 = CausalConvTranspose2d(32, 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))
        self.mask_act = nn.Tanh()

    def stft_forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True
        )
        return spec.real, spec.imag, spec

    def istft_forward(self, real: torch.Tensor, imag: torch.Tensor, original_len: int) -> torch.Tensor:
        spec = torch.complex(real, imag)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(real.device),
            center=True,
            length=original_len
        )
        return wav

    def forward_spec(
        self,
        noisy_real: torch.Tensor,
        noisy_imag: torch.Tensor,
        h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, F_dim, T = noisy_real.shape
        x = torch.stack([noisy_real, noisy_imag], dim=1)

        # 1. Encoder
        e1 = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        e2 = self.enc_act2(self.enc_bn2(self.enc_conv2(e1)))
        e3 = self.enc_act3(self.enc_bn3(self.enc_conv3(e2)))

        # 2. Reshape & Projection
        e3_flat = e3.permute(0, 3, 1, 2).contiguous().view(B, T, self.rnn_in_dim)
        proj_in = self.rnn_proj_in(e3_flat)  # (B, T, H)

        # 3. Causal Dilated TCN Stack
        # Permute to (B, H, T) for 1D Temporal Convolutions
        tcn_in = proj_in.permute(0, 2, 1).contiguous()
        tcn_out = self.tcn_stack(tcn_in)  # (B, H, T)
        tcn_out_t = tcn_out.permute(0, 2, 1).contiguous()  # (B, T, H)

        # 4. Causal GRU
        rnn_out, next_h = self.gru(tcn_out_t, h_state)
        rnn_out = self.rnn_act(self.rnn_proj_out(rnn_out))
        d3_in = rnn_out.view(B, T, 64, self.enc_freq_dim).permute(0, 2, 3, 1).contiguous()

        # 5. Decoder with Skip Connections
        d3_cat = torch.cat([d3_in, e3], dim=1)
        d3 = self.dec_act3(self.dec_bn3(self.dec_deconv3(d3_cat)))
        if d3.shape[2] != e2.shape[2]:
            d3 = d3[:, :, :e2.shape[2], :]

        d2_cat = torch.cat([d3, e2], dim=1)
        d2 = self.dec_act2(self.dec_bn2(self.dec_deconv2(d2_cat)))
        if d2.shape[2] != e1.shape[2]:
            d2 = d2[:, :, :e1.shape[2], :]

        d1_cat = torch.cat([d2, e1], dim=1)
        mask = self.dec_deconv1(d1_cat)
        if mask.shape[2] != F_dim:
            mask = mask[:, :, :F_dim, :]

        # 6. Bounded Mask
        if self.mask_mode == "polar_tanh":
            z_r = mask[:, 0, :, :]
            z_i = mask[:, 1, :, :]
            mag = torch.sqrt(z_r**2 + z_i**2 + 1e-12)
            u_r = z_r / mag
            u_i = z_i / mag
            g = self.mask_bound * torch.tanh(mag / self.mask_bound)
            mask_r = g * u_r
            mask_i = g * u_i
        else:
            mask = self.mask_bound * self.mask_act(mask)
            mask_r = mask[:, 0, :, :]
            mask_i = mask[:, 1, :, :]

        enh_r = noisy_real * mask_r - noisy_imag * mask_i
        enh_i = noisy_real * mask_i + noisy_imag * mask_r

        return enh_r, enh_i, mask_r, mask_i, next_h

    def forward(
        self,
        noisy_wav: torch.Tensor,
        h_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        orig_len = noisy_wav.shape[-1]
        n_real, n_imag, _ = self.stft_forward(noisy_wav)
        enh_r, enh_i, mask_r, mask_i, next_h = self.forward_spec(n_real, n_imag, h_state=h_state)
        enh_wav = self.istft_forward(enh_r, enh_i, original_len=orig_len)
        return enh_wav, enh_r, enh_i, next_h


def build_tcn_causal_anc_model(
    hidden_dim: int = 128,
    num_gru_layers: int = 2,
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)
) -> CausalANCTCNNet:
    """Factory helper to build Causal ANC model with Dilated TCN stack."""
    model = CausalANCTCNNet(
        n_fft=512,
        hop_length=256,
        win_length=512,
        hidden_dim=hidden_dim,
        num_gru_layers=num_gru_layers,
        tcn_dilations=tcn_dilations
    )
    return model


