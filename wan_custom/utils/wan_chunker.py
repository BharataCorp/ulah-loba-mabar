# wan/utils/wan_chunker.py
"""
WAN Chunking Utilities
======================

Rules:
- WAN 2.2 requires frame_num = 4n + 1
- Safe chunk size = 5 seconds (81 frames @ 16fps)
"""

from typing import List


WAN_FPS = 16
WAN_CHUNK_SECONDS = 5  # SAFE DEFAULT


def seconds_to_wan_frames(seconds: int, fps: int = WAN_FPS) -> int:
    """
    Convert seconds to WAN-compatible frame_num (4n + 1).
    """
    raw = seconds * fps
    n = raw // 4
    return (n * 4) + 1


def split_duration_to_chunks(
    total_seconds: int,
    chunk_seconds: int = WAN_CHUNK_SECONDS,
) -> List[int]:
    """
    Split duration into WAN-safe chunks.
    """
    chunks: List[int] = []
    remain = total_seconds

    while remain > 0:
        if remain >= chunk_seconds:
            chunks.append(chunk_seconds)
            remain -= chunk_seconds
        else:
            chunks.append(remain)
            break

    return chunks
