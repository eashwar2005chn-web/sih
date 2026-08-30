# CUDA upgrade — rollback instructions

Performed 2026-08-30 to make the RTX 5060 (sm_120) usable. Prior working state:

    torch       2.5.1+cu121
    torchaudio  2.5.1+cu121
    torchvision 0.20.1+cu121
    openvino    2026.3.0
    driver      610.88   (supports CUDA 12.8+, not the limiter)

Problem: torch 2.5.1+cu121 ships kernels only up to sm_90. The RTX 5060 is sm_120, so
`torch.cuda.is_available()` returns True but every CUDA op raises
"no kernel image is available for execution on the device".

## To roll back exactly

```
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

Nothing else was changed. Checkpoints are plain state_dicts and load under either version.
