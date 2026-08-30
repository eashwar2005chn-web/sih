"""
Live real-time ANC prototype - primary + reference microphone into headphones.

Covers the PS 26052 requirement for "a prototype system demonstrating live noise cancellation
using microphones / headset integration", with "microphones (primary + reference)".

    primary mic  (ch 0) --> STFT -> causal model -> cRM -> iSTFT --.
                                                                    +--> optional LMS --> out
    reference mic (ch 1) ------------------------- noise reference -'

The reference channel feeds the adaptive LMS stage as a TRUE second observation. When only a
mono device is available the engine falls back to the derived pseudo-reference
(noisy - enhanced), which is what the offline path uses; the mode in use is printed at start
and is never silently substituted.

Streaming correctness: the model is a causal GRU network, so hidden state is carried across
blocks. The STFT is centre-padded, so each block is processed with `n_fft` samples of
preceding context and only the new tail is emitted - without this the block boundaries produce
audible discontinuities.

    python demo/live_anc.py --list-devices
    python demo/live_anc.py --selftest            # no microphone; synthetic audio
    python demo/live_anc.py --duration 30         # live capture -> headphones

NOTE: live mode records from your microphone for the requested duration. It writes nothing to
disk unless you pass --record.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.causal_anc_net import build_causal_anc_model
from model.adaptive_lms import NormalizedLMSFilter

SR = 16000
N_FFT, HOP = 512, 256
DEFAULT_CKPT = "checkpoints/phase22_LONG150K/best_model.pt"


class LiveANCProcessor:
    """Block-wise causal enhancement with carried recurrent state."""

    def __init__(self, ckpt=DEFAULT_CKPT, hidden_dim=128, use_lms=True, lms_taps=64, lms_mu=0.01):
        self.model = build_causal_anc_model(hidden_dim=hidden_dim,
                                            mask_mode="component_tanh", mask_bound=1.0)
        sd = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(sd["model_state_dict"])
        self.model.eval()
        self.hidden_dim = hidden_dim
        self.ckpt = ckpt
        self.ckpt_epoch = sd.get("epoch")
        self.ckpt_val = sd.get("val_snr")

        self.use_lms = use_lms
        self.lms = NormalizedLMSFilter(filter_order=lms_taps, mu=lms_mu) if use_lms else None

        self.reset()

    def reset(self):
        self.h = torch.zeros(2, 1, self.hidden_dim, dtype=torch.float32)
        self.ctx = np.zeros(N_FFT, dtype=np.float32)   # preceding-sample context for the STFT
        self.n_blocks = 0
        self.proc_time = 0.0

    def process(self, primary: np.ndarray, reference=None) -> np.ndarray:
        """primary/reference: 1-D float32 block. Returns an enhanced block of the same length."""
        t0 = time.perf_counter()
        n = len(primary)
        buf = np.concatenate([self.ctx, primary]).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(buf).unsqueeze(0)
            r, i, _ = self.model.stft_forward(x)
            enh_r, enh_i, _, _, self.h = self.model.forward_spec(r, i, self.h)
            y = self.model.istft_forward(enh_r, enh_i, len(buf))[0].numpy()

        out = y[-n:].astype(np.float32)          # emit only the new tail
        self.ctx = buf[-N_FFT:]                  # carry context forward

        if self.use_lms and self.lms is not None:
            ref = reference if reference is not None else (primary - out)
            out = self.lms.process_frame(out.astype(np.float64),
                                         np.asarray(ref, dtype=np.float64)[:n]).astype(np.float32)

        self.proc_time += time.perf_counter() - t0
        self.n_blocks += 1
        return out

    def reset_timing(self):
        """Zero the timing accumulators, keeping model/LMS state. Used to exclude warm-up."""
        self.n_blocks = 0
        self.proc_time = 0.0

    def rtf(self, total_audio_sec):
        return self.proc_time / max(total_audio_sec, 1e-9)


def db(x):
    p = float(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return 10.0 * np.log10(p + 1e-12)


def list_devices():
    import sounddevice as sd
    print("=" * 84)
    print("AUDIO DEVICES")
    print("=" * 84)
    for i, d in enumerate(sd.query_devices()):
        io = []
        if d["max_input_channels"]:
            io.append(f"in:{d['max_input_channels']}")
        if d["max_output_channels"]:
            io.append(f"out:{d['max_output_channels']}")
        print(f"  [{i:2d}] {d['name'][:50]:50s} {' '.join(io)}")
    print(f"\n  default (in, out) = {sd.default.device}")
    print("=" * 84)


def selftest(proc):
    """Drive the exact streaming path with synthetic audio - no microphone involved."""
    from data.defence_noise_generator import DefenceNoiseGenerator, mix_speech_and_noise
    rng = np.random.RandomState(0)
    t = np.arange(3 * SR) / SR
    speech = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    noise = DefenceNoiseGenerator(sample_rate=SR).get_noise_by_type("helicopter", 3.0, rng)
    noisy, clean, _ = mix_speech_and_noise(speech, noise, target_snr_db=0.0)

    block = 1024
    out = np.zeros_like(noisy, dtype=np.float32)
    starts = list(range(0, len(noisy) - block + 1, block))
    WARMUP = 8   # first calls pay lazy-init and allocator costs that never recur in a live run

    for k, s in enumerate(starts):
        if k == WARMUP:
            proc.reset_timing()          # steady-state timing only
        blk = noisy[s:s + block].astype(np.float32)
        out[s:s + block] = proc.process(blk, reference=(noisy[s:s + block] - clean[s:s + block]))

    used = len(starts) * block
    steady_sec = (len(starts) - WARMUP) * block / SR
    print("=" * 84)
    print("SELF-TEST  (synthetic audio through the live streaming path; no microphone)")
    print("=" * 84)
    print(f"  checkpoint     : {proc.ckpt} (epoch {proc.ckpt_epoch}, val {proc.ckpt_val:.2f} dB)")
    print(f"  blocks         : {proc.n_blocks} x {block} samples ({block/SR*1000:.0f} ms each)")
    print(f"  LMS stage      : {'ON (true reference channel)' if proc.use_lms else 'OFF'}")
    print(f"  input level    : {db(noisy[:used]):+7.2f} dBFS")
    print(f"  output level   : {db(out[:used]):+7.2f} dBFS")
    print(f"  finite output  : {bool(np.all(np.isfinite(out)))}")
    r = proc.rtf(steady_sec)
    print(f"  RTF (steady)   : {r:.4f}  "
          f"({'OK' if r < 0.1 else 'OVER BUDGET'}, budget < 0.1; first {WARMUP} blocks excluded)")
    print(f"  per block      : {1000.0*proc.proc_time/max(proc.n_blocks,1):.2f} ms "
          f"for {1000.0*block/SR:.0f} ms of audio")
    print(f"  block boundary : max |discontinuity| = "
          f"{np.max(np.abs(np.diff(out[:used]))):.4f}")
    print("=" * 84)
    return np.all(np.isfinite(out))


def live(proc, duration, block, in_dev, out_dev, record):
    import sounddevice as sd
    dev_info = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])
    in_ch = min(2, int(dev_info["max_input_channels"]))
    dual = in_ch >= 2

    print("=" * 84)
    print("LIVE ANC  -  press Ctrl+C to stop")
    print("=" * 84)
    print(f"  input device   : {dev_info['name']}  ({in_ch} ch)")
    print(f"  reference      : {'TRUE second microphone (ch 1)' if dual else 'DERIVED (noisy - enhanced), mono device'}")
    print(f"  checkpoint     : {proc.ckpt} (epoch {proc.ckpt_epoch})")
    print(f"  LMS stage      : {'ON' if proc.use_lms else 'OFF'}")
    print(f"  block          : {block} samples ({1000*block/SR:.0f} ms)")
    print("=" * 84, flush=True)

    captured, enhanced = [], []
    state = {"in_db": -99.0, "out_db": -99.0}

    def cb(indata, outdata, frames, tinfo, status):
        if status:
            print(f"  [stream] {status}", file=sys.stderr)
        prim = indata[:, 0].copy()
        ref = indata[:, 1].copy() if dual else None
        out = proc.process(prim, reference=ref)
        outdata[:, 0] = out
        if outdata.shape[1] > 1:
            outdata[:, 1] = out
        state["in_db"] = db(prim)
        state["out_db"] = db(out)
        if record:
            captured.append(prim)
            enhanced.append(out)

    with sd.Stream(samplerate=SR, blocksize=block, dtype="float32",
                   channels=(in_ch, 2), device=(in_dev, out_dev), callback=cb):
        t0 = time.time()
        try:
            while duration <= 0 or time.time() - t0 < duration:
                time.sleep(0.25)
                el = time.time() - t0
                print(f"\r  t={el:6.1f}s   in {state['in_db']:+7.2f} dBFS   "
                      f"out {state['out_db']:+7.2f} dBFS   "
                      f"suppression {state['in_db']-state['out_db']:+6.2f} dB   "
                      f"RTF {proc.rtf(el):.4f}   ", end="", flush=True)
        except KeyboardInterrupt:
            print("\n  stopped by user")

    print(f"\n  blocks processed: {proc.n_blocks}   final RTF: {proc.rtf(max(time.time()-t0,1e-9)):.4f}")
    if record and captured:
        import soundfile as sf
        os.makedirs("eval/live_captures", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        a = f"eval/live_captures/{ts}_noisy.wav"
        b = f"eval/live_captures/{ts}_enhanced.wav"
        sf.write(a, np.concatenate(captured), SR)
        sf.write(b, np.concatenate(enhanced), SR)
        print(f"  wrote {a}\n  wrote {b}")


def main():
    ap = argparse.ArgumentParser(description="Live primary+reference microphone ANC prototype")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="synthetic audio, no microphone")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds; <=0 runs until Ctrl+C")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--in-device", type=int, default=None)
    ap.add_argument("--out-device", type=int, default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--no-lms", action="store_true")
    ap.add_argument("--record", action="store_true", help="save noisy/enhanced wavs")
    a = ap.parse_args()

    if a.list_devices:
        list_devices()
        return 0

    proc = LiveANCProcessor(ckpt=a.ckpt, use_lms=not a.no_lms)

    if a.selftest:
        return 0 if selftest(proc) else 1

    live(proc, a.duration, a.block, a.in_device, a.out_device, a.record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
