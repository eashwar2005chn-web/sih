import os, sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.causal_anc_net import build_causal_anc_model

def measure_rtf(checkpoint_path: str, n_frames: int = 1000):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_causal_anc_model(hidden_dim=ckpt.get("hidden_dim", 128))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Frame parameters: 16 kHz sample rate, 512 FFT size (32 ms), 128 hop size (8 ms)
    sr = 16000
    hop_size = 128  # 8 ms
    audio_duration_sec = (n_frames * hop_size) / sr  # 8.0 seconds of audio

    dummy_audio = torch.randn(1, n_frames * hop_size)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_audio[:, :1024])

    # CPU Inference Timing
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_audio)
    end = time.perf_counter()
    cpu_wall_time = end - start
    cpu_rtf = cpu_wall_time / audio_duration_sec
    cpu_chunk_latency_ms = (cpu_wall_time / n_frames) * 1000.0

    print(f"Total Processed Audio : {audio_duration_sec:.2f} seconds ({n_frames} streaming frames)")
    print(f"Hop / Frame Size      : {hop_size} samples ({hop_size / sr * 1000.0:.1f} ms audio)")
    print(f"CPU Compute Time      : {cpu_wall_time * 1000.0:.2f} ms")
    print(f"CPU Compute / Frame   : {cpu_chunk_latency_ms:.3f} ms")
    print(f"Real-Time Factor (RTF): {cpu_rtf:.4f} (Target < 0.100)")

    if torch.cuda.is_available():
        model_cuda = model.cuda()
        dummy_cuda = dummy_audio.cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model_cuda(dummy_cuda)
        torch.cuda.synchronize()
        end = time.perf_counter()
        gpu_wall_time = end - start
        gpu_rtf = gpu_wall_time / audio_duration_sec
        gpu_chunk_latency_ms = (gpu_wall_time / n_frames) * 1000.0
        print(f"GPU (RTX 5060) RTF   : {gpu_rtf:.4f} | Compute / Frame: {gpu_chunk_latency_ms:.3f} ms")

if __name__ == "__main__":
    measure_rtf("checkpoints/task4_full_run/best_model.pt")
