"""
DRDO PS 26052: Fast Model Training Recipe for RTX 5060
Rapidly trains a high-quality Causal ANC model on NVIDIA GeForce RTX 5060
and saves checkpoints for ONNX/OpenVINO NPU edge deployment.
"""

import os
import sys
import torch

from train import main

if __name__ == "__main__":
    # Configure fast training recipe
    sys.argv = [
        "train.py",
        "--epochs", "15",
        "--batch_size", "16",
        "--lr", "1.5e-3",
        "--hidden_dim", "128",
        "--train_steps", "600",
        "--val_steps", "100",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--save_dir", "checkpoints"
    ]
    print("Launching fast RTX 5060 training recipe...")
    main()
