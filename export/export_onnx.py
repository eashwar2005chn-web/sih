"""
DRDO PS 26052: Causal ANC Model ONNX Exporter
Exports the trained Causal-CRN spectral core to ONNX format.
Supports stateful streaming inference (frame-by-frame) and chunk processing.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import torch
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_anc_net import build_causal_anc_model, CausalANCNet


class StreamingCausalCore(torch.nn.Module):
    """
    Export wrapper for the neural network core that operates on STFT complex components
    and preserves GRU hidden state across streaming frames.
    """
    def __init__(self, model: CausalANCNet):
        super().__init__()
        self.model = model

    def forward(
        self,
        noisy_real: torch.Tensor,
        noisy_imag: torch.Tensor,
        h_state: torch.Tensor
    ):
        enh_r, enh_i, mask_r, mask_i, next_h = self.model.forward_spec(
            noisy_real=noisy_real,
            noisy_imag=noisy_imag,
            h_state=h_state
        )
        return enh_r, enh_i, mask_r, mask_i, next_h


def export_to_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_path: str = "export/causal_anc_model.onnx",
    hidden_dim: int = 128,
    num_gru_layers: int = 2
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = build_causal_anc_model(hidden_dim=hidden_dim, num_gru_layers=num_gru_layers)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model.eval()
    core = StreamingCausalCore(model).eval()

    dummy_real = torch.randn(1, 257, 125, dtype=torch.float32)
    dummy_imag = torch.randn(1, 257, 125, dtype=torch.float32)
    dummy_h = torch.zeros(num_gru_layers, 1, hidden_dim, dtype=torch.float32)

    dynamic_axes = {
        "noisy_real": {0: "batch", 2: "time_frames"},
        "noisy_imag": {0: "batch", 2: "time_frames"},
        "h_state": {1: "batch"},
        "enh_real": {0: "batch", 2: "time_frames"},
        "enh_imag": {0: "batch", 2: "time_frames"},
        "mask_real": {0: "batch", 2: "time_frames"},
        "mask_imag": {0: "batch", 2: "time_frames"},
        "next_h_state": {1: "batch"}
    }

    print(f"Exporting ONNX model to {output_path}...")
    torch.onnx.export(
        core,
        (dummy_real, dummy_imag, dummy_h),
        output_path,
        input_names=["noisy_real", "noisy_imag", "h_state"],
        output_names=["enh_real", "enh_imag", "mask_real", "mask_imag", "next_h_state"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX export successful! File size: {file_size_mb:.2f} MB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch ANC Model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="export/causal_anc_model.onnx")
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        hidden_dim=args.hidden_dim
    )
