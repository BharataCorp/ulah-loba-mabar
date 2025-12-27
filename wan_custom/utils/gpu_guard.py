# wan/utils/gpu_guard.py

import torch

def assert_cuda_available():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WAN pipelines")

def get_vram_gb():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / 1e9
