# wan/pipelines/t2v_pipeline.py
"""
WAN 2.2 Text-to-Video Pipeline (FINAL - Chunked)
===============================================

Design:
- Native WAN generate.py execution
- VRAM-safe chunking (n × 4 + 1 frames)
- No Diffusers misuse
- Suitable for RunPod / daemon workers
"""

from __future__ import annotations

import os
import subprocess
from typing import Union, Dict, Any

from wan.logger import get_logger
from wan import config
from wan.utils.wan_chunker import (
    split_duration_to_chunks,
    seconds_to_wan_frames,
)

_logger = get_logger("WAN.T2V")


class T2VPipeline:
    """
    WAN 2.2 Text-to-Video Pipeline (chunked execution)
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
        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------
        prompt_text = cls._normalize_prompt(prompt)
        if not prompt_text:
            raise ValueError("Prompt is empty")

        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        if target_duration > config.MAX_DURATION_SECONDS:
            raise ValueError(
                f"target_duration exceeds limit "
                f"{config.MAX_DURATION_SECONDS}s"
            )

        size = size or config.DEFAULT_SIZE
        config.validate_size(size)

        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS

        output_path = (
            output_path
            or cls._default_output_path(prompt_text, size)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --------------------------------------------------
        # Split into WAN-safe chunks
        # --------------------------------------------------
        chunks = split_duration_to_chunks(target_duration)

        _logger.info(
            f"T2V chunking: total={target_duration}s -> {chunks}"
        )

        part_videos: list[str] = []

        # --------------------------------------------------
        # Execute WAN generate.py per chunk
        # --------------------------------------------------
        for idx, sec in enumerate(chunks):
            frame_num = seconds_to_wan_frames(sec)

            part_path = output_path.replace(
                ".mp4", f"_part{idx + 1}.mp4"
            )

            cmd = [
                "python3", "generate.py",
                "--task", "t2v-A14B",
                "--ckpt_dir", config.MODEL_DIRS["t2v"],
                "--prompt", prompt_text,
                "--offload_model", "True",
                "--convert_model_dtype",
                "--size", size,
                "--t5_cpu"
                "--frame_num", str(frame_num),
                "--sample_steps", str(sample_steps),
                "--sample_shift", "10",
                "--save_file", part_path,
            ]

            _logger.info(f"T2V chunk {idx + 1}: {sec}s ({frame_num} frames)")
            _logger.info(" ".join(cmd))

            subprocess.run(
                cmd,
                cwd=config.WAN_ROOT,  # /workspace/Wan2.2
                check=True,
            )

            if not os.path.exists(part_path):
                raise RuntimeError(
                    f"WAN failed to generate chunk {idx + 1}"
                )

            part_videos.append(part_path)

        # --------------------------------------------------
        # Concat chunks (lossless)
        # --------------------------------------------------
        if len(part_videos) == 1:
            _logger.info("Single chunk, no concat needed")
            return part_videos[0]

        cls._concat_videos(part_videos, output_path)

        _logger.info(f"T2V FINAL video saved: {output_path}")
        return output_path

    # ==================================================
    # Helpers
    # ==================================================

    @staticmethod
    def _normalize_prompt(
        prompt: Union[str, Dict[str, Any]]
    ) -> str:
        if isinstance(prompt, str):
            return prompt.strip()

        if isinstance(prompt, dict):
            if "prompt_for_wan_one_t2v" in prompt:
                return str(prompt["prompt_for_wan_one_t2v"]).strip()

            scenes = prompt.get("scenes", [])
            texts = []
            for s in scenes:
                actions = ", ".join(s.get("actions", []))
                mood = s.get("mood", "")
                texts.append(f"{actions}. Mood: {mood}")
            return " ".join(texts).strip()

        raise TypeError("prompt must be str or dict")

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        safe = (
            prompt[:60]
            .replace(" ", "_")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .replace('"', "")
        )
        return os.path.join(
            config.OUTPUT_DIR,
            f"t2v_{size}_{safe}.mp4",
        )

    @staticmethod
    def _concat_videos(parts: list[str], output: str):
        txt = output.replace(".mp4", "_concat.txt")

        with open(txt, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", txt,
                "-c", "copy",
                output,
            ],
            check=True,
        )
