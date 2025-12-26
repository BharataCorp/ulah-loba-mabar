"""
WAN Module
==========

This package provides a clean, maintainable wrapper
around WAN 2.2 pipelines (T2V, I2V, TI2V, Animate, S2V).

Design goals:
- Zero side effects on import
- No model loading at package import time
- Safe for RunPod stop/start
- KISS & production-ready
"""

__all__ = [
    "config",
    "logger",
    "pipelines",
    "services",
    "utils",
]

__version__ = "0.1.0"


def get_version() -> str:
    """Return WAN wrapper version."""
    return __version__


def check_runtime():
    """
    Optional runtime check.
    Does NOT load models or GPU.
    Safe to call manually.
    """
    try:
        import torch
        import diffusers
        import transformers
        import accelerate
        import peft
    except Exception as e:
        raise RuntimeError(f"[WAN] Dependency check failed: {e}")

    return {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "peft": peft.__version__,
    }
