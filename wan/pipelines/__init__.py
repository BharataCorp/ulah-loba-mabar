"""
WAN Pipelines Package
=====================

This package contains all WAN 2.2 pipelines:
- BasePipeline (shared logic)
- T2V / I2V / TI2V / Animate / S2V pipelines

IMPORTANT:
- No model loading here
- No GPU allocation
"""

from .base_pipeline import BasePipeline

__all__ = [
    "BasePipeline",
]
