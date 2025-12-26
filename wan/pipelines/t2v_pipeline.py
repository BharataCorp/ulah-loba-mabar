"""
WAN 2.2 Text-to-Video Pipeline (NATIVE CLI)
==========================================

- Calls WAN 2.2 via generate.py (OFFICIAL ENTRYPOINT)
- No Diffusers
- No fake Python imports
- Safe for RunPod / daemon usage
"""

from __future__ import annotations

import os
import subprocess
from typing import Union, Dict, Any

from wan.logger import get_logger
from wan import config
from wan.utils.duration import seconds_to_frames
from wan.utils.prompt_parser import normalize_prompt

_logger = get_logger("WAN.T2V")


class T2VPipeline:
    """
    WAN 2.2 Native Text-to-Video Pipeline (CLI-based)
    """

    @classmethod
    def generate(
        cls,
        prompt: Union[str, Dict[str, Any]],
        target_duration: int,
        size: str | None = None,
        sample_steps: int | None = None,
        output_path: str | None = None,
    ) -> str:
        # ----------------------------
        # Validate & normalize
        # ----------------------------
        prompt_text = normalize_prompt(prompt)
        if not prompt_text:
            raise ValueError("Prompt is empty")

        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        if target_duration > config.MAX_DURATION_SECONDS:
            raise ValueError(
                f"target_duration exceeds limit {config.MAX_DURATION_SECONDS}s"
            )

        frame_num = seconds_to_frames(target_duration)

        size = size or config.DEFAULT_SIZE
        config.validate_size(size)

        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS

        output_path = (
            output_path
            or cls._default_output_path(prompt_text, size)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ----------------------------
        # Build WAN generate.py command
        # ----------------------------
        cmd = [
            "python3",
            "generate.py",
            "--task", "t2v-A14B",
            "--ckpt_dir", config.MODEL_DIRS["t2v"],
            "--prompt", prompt_text,
            "--size", size,
            "--frame_num", str(frame_num),
            "--sample_steps", str(sample_steps),
            "--save_file", output_path,
        ]

        _logger.info("Executing WAN T2V:")
        _logger.info(" ".join(cmd))

        subprocess.run(
            cmd,
            cwd=config.WAN_ROOT,   # /workspace/Wan2.2
            check=True,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("WAN did not produce output video")

        _logger.info(f"T2V video generated: {output_path}")
        return output_path

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        safe = (
            prompt[:60]
            .replace(" ", "_")
            .replace("/", "")
            .replace("\\", "")
            .replace('"', "")
            .replace("'", "")
        )
        filename = f"t2v_{size}_{safe}.mp4"
        return os.path.join(config.OUTPUT_DIR, filename)
