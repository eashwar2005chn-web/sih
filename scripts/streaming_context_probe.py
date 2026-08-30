"""
Section A.2 (continued) — Isolate WHY naive frame-by-frame streaming degrades.

Hypothesis: forward_spec(re, im, h_state) exposes only the GRU hidden state, but the
encoder/decoder use CAUSAL 2D convolutions with kernel size 3 along time. Offline, frame t
sees frames t-2, t-1 via left-padding of the real sequence. Fed one frame at a time, that
left-padding is ZEROS -- so the conv receptive field is destroyed even though h_state is
carried correctly. If true, output quality should recover monotonically as we feed larger
chunks with carried-over left context.

This distinguishes "the model is not streamable as written" from "my OLA is broken".
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.real_dataset_loader import HybridRealDataset
from model.causal_anc_net import build_causal_anc_model
from model.loss import compute_snr

CKPT = "checkpoints/task4_full_run/best_model.pt"


@torch.no_grad()
def chunked_spec_infer(m, re, im, chunk, left_ctx, dev):
    """
    Run forward_spec over the spectrogram in chunks of `chunk` frames, prepending
    `left_ctx` frames of real history to each chunk (discarding their outputs).
    left_ctx=0 with chunk=1 reproduces naive frame-by-frame streaming.
    """
    T = re.shape[-1]
    outs_r, outs_i = [], []
    h = None
    t = 0
    while t < T:
        end = min(t + chunk, T)
        s = max(0, t - left_ctx)
        r_in = re[..., s:end]
        i_in = im[..., s:end]
        er, ei, _, _, h = m.forward_spec(r_in, i_in, None if left_ctx > 0 else h)
        keep = end - t
        outs_r.append(er[..., -keep:])
        outs_i.append(ei[..., -keep:])
        t = end
    return torch.cat(outs_r, dim=-1), torch.cat(outs_i, dim=-1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu")
    m = build_causal_anc_model(hidden_dim=128)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(dev)

    ds = HybridRealDataset(mode="hybrid", split="test", sample_rate=16000, segment_len_sec=2.0,
                           epoch_size=10, snr_range=(0.0, 15.0), hybrid_ratio=0.85, seed=5000)

    print("=" * 84)
    print("A.2 PROBE: does carried left-context recover streaming quality?")
    print("=" * 84)

    configs = [
        ("offline (whole utterance)", None, None),
        ("chunk=1  left_ctx=0  (naive frame-by-frame)", 1, 0),
        ("chunk=1  left_ctx=2", 1, 2),
        ("chunk=1  left_ctx=4", 1, 4),
        ("chunk=1  left_ctx=8", 1, 8),
        ("chunk=1  left_ctx=16", 1, 16),
        ("chunk=1  left_ctx=32", 1, 32),
        ("chunk=16 left_ctx=8", 16, 8),
    ]

    results = {lbl: [] for lbl, _, _ in configs}
    for idx in range(len(ds)):
        it = ds[idx]
        clean = it["clean"].numpy()
        wav = it["noisy"].unsqueeze(0).to(dev)
        with torch.no_grad():
            re, im, _ = m.stft_forward(wav)
            for lbl, chunk, lc in configs:
                if chunk is None:
                    enh, _, _, _ = m(wav)
                    out = enh.squeeze(0).cpu().numpy()
                else:
                    er, ei = chunked_spec_infer(m, re, im, chunk, lc, dev)
                    out = m.istft_forward(er, ei, wav.shape[-1]).squeeze(0).cpu().numpy()
                L = min(len(out), len(clean))
                results[lbl].append(compute_snr(clean[:L], out[:L]))

    print(f"\n  N={len(ds)} utterances\n")
    base = np.mean(results["offline (whole utterance)"])
    print(f"  {'configuration':46} {'Output SNR':>11} {'vs offline':>12}")
    for lbl, _, _ in configs:
        mu = np.mean(results[lbl])
        print(f"  {lbl:46} {mu:>8.2f} dB {mu-base:>+10.2f} dB")

    print("\n  Interpretation:")
    naive = np.mean(results["chunk=1  left_ctx=0  (naive frame-by-frame)"])
    ctx32 = np.mean(results["chunk=1  left_ctx=32"])
    if ctx32 - naive > 1.0:
        print("   -> Carrying left context recovers quality: the causal CONV layers hold temporal")
        print("      state that forward_spec()'s h_state argument does NOT expose. A correct")
        print("      streaming implementation must cache conv input history, not just the GRU state.")
    else:
        print("   -> Left context does not explain the gap; investigate elsewhere.")


if __name__ == "__main__":
    main()
