# wan/utils/gpu_profile.py
from __future__ import annotations

import torch


def detect_gpu_profile() -> dict:
    """
    Detect GPU architecture and decide safe optimizations.
    """

    if not torch.cuda.is_available():
        return {
            "gpu": "cpu",
            "tf32": False,
            "compile": False,
        }

    name = torch.cuda.get_device_name().lower()
    major, minor = torch.cuda.get_device_capability()

    # Hopper / Blackwell (H100, H200, B200)
    if major >= 9:
        return {
            "gpu": name,
            "tf32": True,
            "compile": True,
        }

    # Ampere / Ada (A100, L40, RTX)
    if major == 8:
        return {
            "gpu": name,
            "tf32": True,
            "compile": True,
        }

    # Older GPUs
    return {
        "gpu": name,
        "tf32": False,
        "compile": False,
    }


def apply_global_optimizations(profile: dict):
    """
    Apply safe global optimizations ONCE per process.
    """
    if not torch.cuda.is_available():
        return

    if profile.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
