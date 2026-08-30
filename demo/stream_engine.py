"""
DRDO PS 26052: Real-Time Audio Streaming Engine & Multi-Backend Processor
Handles continuous streaming audio frames (32ms window / 16ms hop) with low-latency buffering.
Supports runtime switching between:
- NVIDIA RTX 5060 (CUDA)
- Intel AI Boost NPU (OpenVINO INT8)
- Intel Core Ultra 9 CPU (OpenVINO / PyTorch)
- Hybrid Adaptive NLMS Post-Filter
- Raw Passthrough
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any

# Ensure parent directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_anc_net import build_causal_anc_model, CausalANCNet
from model.adaptive_lms import HybridANCPostProcessor, NormalizedLMSFilter
from data.defence_noise_generator import DefenceNoiseGenerator, mix_speech_and_noise


class RealTimeANCEngine:
    """
    Streaming Inference Engine with Heterogeneous Hardware Dispatch.
    """

    def __init__(
        self,
        # Authoritative epoch-50 checkpoint (sha256 c0fed3f4...). Do NOT change to
        # "checkpoints/best_model.pt" — that path holds a stale epoch-6 model.
        checkpoint_path: str = "checkpoints/task4_full_run/best_model.pt",
        openvino_model_path: str = "export/openvino_int8/causal_anc_int8.xml",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        hidden_dim: int = 128
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.hidden_dim = hidden_dim

        self.window = torch.hann_window(self.win_length)
        self.window_np = np.hanning(self.win_length).astype(np.float32)

        # 1. Initialize PyTorch CUDA & CPU Models
        self.pytorch_model = build_causal_anc_model(hidden_dim=hidden_dim).eval()
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.pytorch_model.load_state_dict(state)
                print(f"[Engine] Loaded PyTorch checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"[Engine] Checkpoint load warning: {e}")

        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.cuda_model = build_causal_anc_model(hidden_dim=hidden_dim).eval()
            self.cuda_model.load_state_dict(self.pytorch_model.state_dict())
            self.cuda_model = self.cuda_model.to("cuda")

        # 2. Initialize OpenVINO Runtime & Devices
        self.ov_core = None
        self.ov_compiled_npu = None
        self.ov_compiled_cpu = None
        self.available_ov_devices = []

        try:
            import openvino as ov
            self.ov_core = ov.Core()
            self.available_ov_devices = self.ov_core.available_devices
            print(f"[Engine] OpenVINO Devices Detected: {self.available_ov_devices}")

            target_xml = openvino_model_path
            if not os.path.exists(target_xml):
                target_xml = "export/openvino_int8/causal_anc_fp16.xml"

            if os.path.exists(target_xml):
                if "NPU" in self.available_ov_devices:
                    try:
                        self.ov_compiled_npu = self.ov_core.compile_model(target_xml, "NPU")
                        print("[Engine] OpenVINO NPU Engine initialized successfully!")
                    except Exception as e:
                        print(f"[Engine] NPU Init notice: {e}")

                self.ov_compiled_cpu = self.ov_core.compile_model(target_xml, "CPU")
                print("[Engine] OpenVINO CPU Engine initialized successfully!")
        except Exception as e:
            print(f"[Engine] OpenVINO runtime not available: {e}")

        # 3. Hybrid NLMS Post-Filter
        self.hybrid_post = HybridANCPostProcessor(sample_rate=sample_rate)
        self.nlms_filter = NormalizedLMSFilter(filter_order=64, mu=0.08)

        # 4. Noise generator for synthetic tactical injection
        self.noise_gen = DefenceNoiseGenerator(sample_rate=sample_rate)

        # Recurrent state buffers for streaming
        self.reset_states()

    def reset_states(self):
        """Reset internal streaming recurrent hidden states and delay lines."""
        self.pt_h_state = torch.zeros(2, 1, self.hidden_dim, dtype=torch.float32)
        if self.cuda_available:
            self.cuda_h_state = torch.zeros(2, 1, self.hidden_dim, dtype=torch.float32, device="cuda")
        self.ov_h_state = np.zeros((2, 1, self.hidden_dim), dtype=np.float32)
        self.nlms_filter.reset()

    def process_audio(
        self,
        audio_input: np.ndarray,
        backend: str = "CUDA (RTX 5060)",
        enable_neural: bool = True,
        enable_nlms_post: bool = True
    ) -> Dict[str, Any]:
        """
        Process a complete audio buffer through the selected backend.
        Returns:
            enhanced_audio: 1D numpy array
            latency_ms: Execution time in milliseconds
            rtf: Real-time factor (execution_time / audio_duration)
            backend_used: Actual backend utilized
        """
        t_start = time.perf_counter()
        audio_len_sec = len(audio_input) / self.sr
        backend_used = backend

        if not enable_neural:
            enhanced = audio_input.copy()
            if enable_nlms_post:
                enhanced = self.nlms_filter.filter_batch(audio_input)
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0
            return {
                "enhanced": enhanced,
                "latency_ms": elapsed_ms,
                "rtf": elapsed_ms / (audio_len_sec * 1000.0 + 1e-8),
                "backend_used": "Raw + NLMS Filter" if enable_nlms_post else "Raw Passthrough"
            }

        # Select neural backend
        if "CUDA" in backend and self.cuda_available:
            enhanced = self._infer_pytorch_cuda(audio_input)
            backend_used = "NVIDIA RTX 5060 (CUDA FP16)"
        elif "NPU" in backend and self.ov_compiled_npu is not None:
            enhanced = self._infer_openvino(audio_input, self.ov_compiled_npu)
            backend_used = "Intel AI Boost NPU (OpenVINO INT8)"
        elif "OpenVINO" in backend and self.ov_compiled_cpu is not None:
            enhanced = self._infer_openvino(audio_input, self.ov_compiled_cpu)
            backend_used = "Intel Core Ultra 9 CPU (OpenVINO INT8)"
        else:
            enhanced = self._infer_pytorch_cpu(audio_input)
            backend_used = "Intel Core Ultra 9 CPU (PyTorch)"

        # Apply hybrid adaptive NLMS post-filter if enabled
        if enable_nlms_post:
            enhanced = self.hybrid_post.enhance(audio_input, enhanced)

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0
        rtf = elapsed_ms / (audio_len_sec * 1000.0 + 1e-8)

        return {
            "enhanced": enhanced.astype(np.float32),
            "latency_ms": elapsed_ms,
            "rtf": rtf,
            "backend_used": backend_used
        }

    def _infer_pytorch_cuda(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out, _, _, _ = self.cuda_model(tensor)
            return out.squeeze(0).cpu().numpy()

    def _infer_pytorch_cpu(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0)
            out, _, _, _ = self.pytorch_model(tensor)
            return out.squeeze(0).numpy()

    def _infer_openvino(self, audio: np.ndarray, compiled_ov_model) -> np.ndarray:
        # Convert audio to STFT complex components
        tensor = torch.from_numpy(audio).unsqueeze(0)
        spec = torch.stft(
            tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True
        )
        real_np = spec.real.numpy()
        imag_np = spec.imag.numpy()
        h_np = np.zeros((2, 1, self.hidden_dim), dtype=np.float32)

        infer_request = compiled_ov_model.create_infer_request()
        res = infer_request.infer({
            "noisy_real": real_np,
            "noisy_imag": imag_np,
            "h_state": h_np
        })

        # Fetch enhanced spectral outputs
        enh_r = res[0]
        enh_i = res[1]

        # iSTFT synthesis
        enh_spec = torch.complex(torch.from_numpy(enh_r), torch.from_numpy(enh_i))
        enh_wav = torch.istft(
            enh_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=len(audio)
        )
        return enh_wav.squeeze(0).numpy()
