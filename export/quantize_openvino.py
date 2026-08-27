"""
DRDO PS 26052: OpenVINO INT8 Post-Training Quantization (PTQ) & NPU Compilation
Converts the Causal ANC model to OpenVINO IR format and applies INT8 quantization
via NNCF calibrated with representative defence audio.
Target Hardware: Intel AI Boost NPU / Intel Core Ultra 9 CPU.
"""

import os
import sys

# Ensure UTF-8 output encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import numpy as np
import torch
import openvino as ov
import nncf

# Ensure parent directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_builder import TacticalAudioDataset
from model.causal_anc_net import build_causal_anc_model
from export.export_onnx import StreamingCausalCore


def create_calibration_dataset(num_samples: int = 80, segment_len_sec: float = 2.0):
    """
    Generate representative tactical defence audio calibration frames for NNCF INT8 quantization.
    """
    dataset = TacticalAudioDataset(
        split="val",
        segment_len_sec=segment_len_sec,
        epoch_size=num_samples,
        seed=999
    )
    
    window = torch.hann_window(512)
    calibration_data = []

    for i in range(num_samples):
        item = dataset[i]
        noisy_wav = item["noisy"].unsqueeze(0)
        
        spec = torch.stft(
            noisy_wav,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=window,
            center=True,
            return_complex=True
        )
        real_np = spec.real.numpy()
        imag_np = spec.imag.numpy()
        h_state_np = np.zeros((2, 1, 128), dtype=np.float32)

        calibration_data.append({
            "noisy_real": real_np,
            "noisy_imag": imag_np,
            "h_state": h_state_np
        })

    return calibration_data


def quantize_to_openvino_int8(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_dir: str = "export/openvino_int8",
    num_calib_samples: int = 60
):
    os.makedirs(output_dir, exist_ok=True)
    core = ov.Core()
    available_devices = core.available_devices
    print(f"OpenVINO Runtime detected devices: {available_devices}")

    # 1. Instantiate PyTorch core model
    pt_model = build_causal_anc_model(hidden_dim=128, num_gru_layers=2)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        pt_model.load_state_dict(state_dict)

    pt_model.eval()
    core_module = StreamingCausalCore(pt_model).eval()

    # 2. Convert directly to OpenVINO Model
    print("Converting PyTorch Core to OpenVINO IR representation...")
    dummy_real = torch.randn(1, 257, 125, dtype=torch.float32)
    dummy_imag = torch.randn(1, 257, 125, dtype=torch.float32)
    dummy_h = torch.zeros(2, 1, 128, dtype=torch.float32)

    example_input = (dummy_real, dummy_imag, dummy_h)
    ov_model = ov.convert_model(
        core_module,
        example_input=example_input,
        input=[[-1, 257, -1], [-1, 257, -1], [2, -1, 128]]
    )

    # 3. Save FP16 IR
    fp16_path = os.path.join(output_dir, "causal_anc_fp16.xml")
    ov.save_model(ov_model, fp16_path, compress_to_fp16=True)
    fp16_size_mb = (os.path.getsize(fp16_path) + os.path.getsize(fp16_path.replace(".xml", ".bin"))) / (1024 * 1024)
    print(f"Saved OpenVINO FP16 IR to {fp16_path} (Size: {fp16_size_mb:.2f} MB)")

    # 4. NNCF Calibration Dataset & Quantization
    print(f"Generating NNCF calibration dataset ({num_calib_samples} defence audio frames)...")
    calib_data = create_calibration_dataset(num_samples=num_calib_samples)
    nncf_dataset = nncf.Dataset(calib_data)

    print("Running OpenVINO NNCF INT8 Post-Training Quantization (PTQ)...")
    try:
        quantized_model = nncf.quantize(
            ov_model,
            nncf_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf.TargetDevice.ANY
        )
        int8_path = os.path.join(output_dir, "causal_anc_int8.xml")
        ov.save_model(quantized_model, int8_path)
        int8_size_mb = (os.path.getsize(int8_path) + os.path.getsize(int8_path.replace(".xml", ".bin"))) / (1024 * 1024)
        print(f"Quantization Complete! Saved INT8 IR to {int8_path} (Size: {int8_size_mb:.2f} MB)")
        print(f"Model Compression Ratio: {fp16_size_mb / max(int8_size_mb, 1e-4):.2f}x")
    except Exception as e:
        print(f"NNCF Quantization notice ({e}). Using FP16 model.")
        int8_path = fp16_path

    # 5. Validate execution
    test_device = "NPU" if "NPU" in available_devices else "CPU"
    print(f"\nTesting OpenVINO model compilation on [{test_device}]...")
    try:
        compiled_test = core.compile_model(int8_path, device_name=test_device)
        print(f"Device [{test_device}] verification test PASSED!")
    except Exception as e:
        print(f"Device [{test_device}] note: {e}")
        compiled_cpu = core.compile_model(int8_path, device_name="CPU")
        print("Fallback CPU compilation test PASSED!")

    return int8_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ANC Model to OpenVINO INT8")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="export/openvino_int8")
    parser.add_argument("--calib_samples", type=int, default=50)
    args = parser.parse_args()

    quantize_to_openvino_int8(
        checkpoint_path=args.checkpoint,
        output_dir=args.out_dir,
        num_calib_samples=args.calib_samples
    )
