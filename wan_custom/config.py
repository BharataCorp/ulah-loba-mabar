"""
WAN Global Configuration
========================

Central configuration for WAN 2.2 pipelines.
- No model loading here
- No torch.cuda calls
- Safe to import anywhere
"""

import os
from dataclasses import dataclass

BASE_API_URL_MABAR = os.environ.get("BASE_API_URL_MABAR", "")
if BASE_API_URL_MABAR.endswith("/"):
    BASE_API_URL_MABAR = BASE_API_URL_MABAR[:-1]

# =========================
# BASIC PATH CONFIG
# =========================

WAN_ROOT = os.environ.get("WAN_ROOT", "/workspace/Wan2.2")

MODEL_DIRS = {
    "t2v": os.path.join(WAN_ROOT, "Wan2.2-T2V-A14B"),
    "i2v": os.path.join(WAN_ROOT, "Wan2.2-I2V-A14B"),
    "ti2v": os.path.join(WAN_ROOT, "Wan2.2-TI2V-5B"),
    "animate": os.path.join(WAN_ROOT, "Wan2.2-Animate-14B"),
    "s2v": os.path.join(WAN_ROOT, "Wan2.2-S2V-14B"),
}

OUTPUT_DIR = os.environ.get("WAN_OUTPUT_DIR", os.path.join(WAN_ROOT, "output_videos"))
LOG_DIR = os.environ.get("WAN_LOG_DIR", os.path.join(WAN_ROOT, "logs"))

RIFE_BIN = "/workspace/rife/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "/workspace/rife/model/rife-v4.25"



# =========================
# GENERATION LIMITS
# =========================

FPS = 16

# WAN 2.2 rule: frame_num = 4N + 1
def seconds_to_frames(seconds: int) -> int:
    base = seconds * FPS
    return 4 * (base // 4) + 1


MAX_DURATION_SECONDS = int(os.environ.get("WAN_MAX_DURATION", "30"))


# =========================
# PERFORMANCE DEFAULTS
# =========================

DEFAULT_SAMPLE_STEPS = int(os.environ.get("WAN_SAMPLE_STEPS", "16"))
DEFAULT_SAMPLE_SHIFT = float(os.environ.get("WAN_SAMPLE_SHIFT", "10"))
DEFAULT_GUIDE_SCALE = (3.0, 4.0)

DEFAULT_SIZE = os.environ.get("WAN_DEFAULT_SIZE", "832*480")

CACHE_DIR = "/workspace/cache"

# =========================
# GPU / MEMORY BEHAVIOR
# =========================

USE_OFFLOAD = os.environ.get("WAN_OFFLOAD_MODEL", "true").lower() == "true"
USE_CONVERT_DTYPE = os.environ.get("WAN_CONVERT_DTYPE", "true").lower() == "true"
USE_T5_CPU = os.environ.get("WAN_T5_CPU", "false").lower() == "true"
MABAR_POD_ID = os.environ.get("MABAR_POD_ID", "0")
KEY_MANAGEMENT_ID = os.environ.get("KEY_MANAGEMENT_ID", "0")


# =========================
# VALIDATION
# =========================

VALID_SIZES = {
    "720*1280",
    "1280*720",
    "480*832",
    "832*480",
    "704*1280",
    "1280*704",
    "1024*704",
    "704*1024",
}


def validate_size(size: str):
    if size not in VALID_SIZES:
        raise ValueError(f"[WAN] Invalid size '{size}', allowed: {sorted(VALID_SIZES)}")
