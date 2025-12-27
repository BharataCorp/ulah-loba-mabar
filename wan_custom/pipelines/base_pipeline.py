"""
Base WAN Pipeline
=================

Shared base class for all WAN 2.2 pipelines.
- Ensures model is loaded once
- GPU-safe
- No duplicate memory allocation
"""

import os
import threading
import torch
from typing import Optional

from wan.logger import get_logger
from wan import config

_logger = get_logger("WAN.BasePipeline")


class BasePipeline:
    """
    Base pipeline with singleton-style loading.
    """

    _lock = threading.Lock()
    _pipeline = None
    _loaded = False

    def __init__(self):
        if not self._loaded:
            raise RuntimeError(
                "Pipeline not loaded. Use `load()` before instantiation."
            )

    @classmethod
    def load(cls):
        """
        Load pipeline once into memory.
        Safe to call multiple times.
        """
        if cls._loaded:
            _logger.info("Pipeline already loaded, skipping.")
            return cls

        with cls._lock:
            if cls._loaded:
                return cls

            _logger.info("Loading WAN pipeline base...")

            cls._setup_torch_env()
            cls._pipeline = cls._load_pipeline()
            cls._loaded = True

            _logger.info("WAN pipeline loaded successfully.")
            return cls

    @classmethod
    def _setup_torch_env(cls):
        """
        Torch & CUDA safety setup.
        """
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if torch.cuda.is_available():
            _logger.info(
                f"CUDA detected: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
            )
        else:
            _logger.warning("CUDA not available, using CPU (VERY SLOW).")

    @classmethod
    def _load_pipeline(cls):
        """
        To be implemented by subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement _load_pipeline()"
        )

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def clear_cuda(cls):
        """
        Clear CUDA cache between jobs.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            _logger.info("CUDA cache cleared.")
