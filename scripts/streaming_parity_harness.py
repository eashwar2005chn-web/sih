"""
Section A.2 — True frame-by-frame stateful streaming inference harness, and parity
check against the offline batched evaluation path.

Why this matters beyond "does the demo work": the offline path calls torch.stft with
center=True, which pads by n_fft//2 and centres frame k on sample k*hop. Producing
frame k therefore requires samples up to k*hop + n_fft/2, i.e. 256 samples = 16 ms of
FUTURE audio. The model's GRU/conv stack is strictly causal frame-to-frame (verified
0.00e+00 future-frame leakage), but the STFT framing in front of it is not causal at
the waveform level.

This harness implements genuinely causal streaming (no future samples: each output hop
uses only already-received audio) and quantifies the difference against the offline path.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr

CKPT = "checkpoints/task4_full_run/best_model.pt"
N_FFT, HOP = 512, 256


class CausalStreamingANC:
    """
    Stateful, frame-by-frame ANC. Consumes HOP samples at a time and emits HOP samples.
    Uses ONLY past/current samples -- no centring, no future lookahead.
    """

    def __init__(self, model, device="cpu"):
        self.m = model.eval().to(device)
        self.dev = device
        self.win = torch.hann_window(N_FFT).to(device)
        self.reset()

    def reset(self):
        self.inbuf = np.zeros(N_FFT, dtype=np.float32)   # sliding analysis buffer
        self.ola = np.zeros(N_FFT, dtype=np.float32)     # overlap-add accumulator
        self.h = None                                    # persistent GRU state
        self.wsum = np.zeros(N_FFT, dtype=np.float32)    # window-square normaliser

    @torch.no_grad()
    def push(self, hop_samples: np.ndarray) -> np.ndarray:
        assert len(hop_samples) == HOP
        # slide analysis buffer: newest HOP samples at the end
        self.inbuf = np.concatenate([self.inbuf[HOP:], hop_samples.astype(np.float32)])

        frame = torch.from_numpy(self.inbuf).to(self.dev) * self.win
        spec = torch.fft.rfft(frame, n=N_FFT)
        re = spec.real.view(1, -1, 1)
        im = spec.imag.view(1, -1, 1)

        out_re, out_im, self.h = self._step(re, im)

        cspec = torch.complex(out_re.view(-1), out_im.view(-1))
        rec = torch.fft.irfft(cspec, n=N_FFT).cpu().numpy() * self.win.cpu().numpy()

        self.ola = np.concatenate([self.ola[HOP:], np.zeros(HOP, dtype=np.float32)])
        self.ola += rec
        self.wsum = np.concatenate([self.wsum[HOP:], np.zeros(HOP, dtype=np.float32)])
        self.wsum += (self.win.cpu().numpy() ** 2)

        out = self.ola[:HOP] / np.maximum(self.wsum[:HOP], 1e-8)
        return out.astype(np.float32)

    def _step(self, re, im):
        """One spectral frame through the network, carrying GRU state across calls."""
        # forward_spec -> (enhanced_real, enhanced_imag, mask_real, mask_imag, next_h)
        enh_r, enh_i, _mr, _mi, next_h = self.m.forward_spec(re, im, self.h)
        return enh_r, enh_i, next_h


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)

    print("=" * 88)
    print("A.2 STREAMING PARITY HARNESS")
    print("=" * 88)
    print(f"  device={dev}  n_fft={N_FFT} hop={HOP}")

    # --- structural lookahead check on the offline path ---
    print("\n--- Offline STFT framing: centred or causal? ---")
    probe = torch.zeros(1, 4096, device=dev)
    probe[0, 2048] = 1.0     # impulse at sample 2048
    with torch.no_grad():
        r, i, _ = m.stft_forward(probe)
    energy = (r[0] ** 2 + i[0] ** 2).sum(dim=0).cpu().numpy()
    first = int(np.argmax(energy > 1e-6))
    print(f"  impulse at sample 2048 -> first frame with energy: {first} (frame start sample {first*HOP})")
    if first * HOP < 2048:
        la = 2048 - first * HOP
        print(f"  ==> offline STFT uses CENTERED framing: frame {first} begins {la} samples "
              f"({1000*la/16000:.1f} ms) BEFORE the impulse.")
        print(f"  ==> {la} samples = {1000*la/16000:.1f} ms of algorithmic LOOKAHEAD at the waveform level.")
    else:
        print("  ==> offline STFT framing appears causal.")

    # --- parity on real test utterances ---
    print("\n--- Streaming vs offline on held-out test utterances ---")
    ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000, segment_len_sec=2.0,
                           epoch_size=12, snr_range=(0.0, 15.0), hybrid_ratio=0.85, seed=5000)
    streamer = CausalStreamingANC(m, dev)

    off_snr, str_snr, maxdiffs, per_hop_ms = [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        clean, noisy = it["clean"].numpy(), it["noisy"].numpy()
        with torch.no_grad():
            enh_off, _, _, _ = m(it["noisy"].unsqueeze(0).to(dev))
        enh_off = enh_off.squeeze(0).cpu().numpy()

        streamer.reset()
        n_hops = len(noisy) // HOP
        out = np.zeros(n_hops * HOP, dtype=np.float32)
        t0 = time.perf_counter()
        for k in range(n_hops):
            out[k * HOP:(k + 1) * HOP] = streamer.push(noisy[k * HOP:(k + 1) * HOP])
        per_hop_ms.append((time.perf_counter() - t0) * 1000.0 / n_hops)

        L = min(len(out), len(enh_off), len(clean))
        off_snr.append(compute_snr(clean[:L], enh_off[:L]))
        str_snr.append(compute_snr(clean[:L], out[:L]))
        maxdiffs.append(float(np.max(np.abs(out[:L] - enh_off[:L]))))

    print(f"  N={len(off_snr)} utterances")
    print(f"  offline   mean Output SNR : {np.mean(off_snr):7.3f} dB")
    print(f"  streaming mean Output SNR : {np.mean(str_snr):7.3f} dB")
    print(f"  delta (streaming-offline) : {np.mean(str_snr)-np.mean(off_snr):+7.3f} dB")
    print(f"  max |sample difference|   : {np.max(maxdiffs):.4e}")
    print(f"  streaming per-hop latency : {np.mean(per_hop_ms):.4f} ms "
          f"(budget {1000*HOP/16000:.1f} ms/hop, RTF {np.mean(per_hop_ms)/(1000*HOP/16000):.4f})")


if __name__ == "__main__":
    main()
