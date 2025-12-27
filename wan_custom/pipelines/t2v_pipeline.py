# wan/pipelines/t2v_pipeline.py
"""
WAN 2.2 Text-to-Video Pipeline (FINAL - Cached Execution)
========================================================

Design:
- Single WAN generate.py invocation
- Model cached inside generate.py
- No subprocess loop
- Compatible with RunPod
"""

from __future__ import annotations

import os
import subprocess
from typing import Union, Dict, Any

from wan_custom.logger import get_logger
from wan_custom import config
from wan_custom.utils.wan_chunker import seconds_to_wan_frames
from wan_custom.utils.gpu_profile import (
    detect_gpu_profile,
    apply_global_optimizations,
)
from generate import generate_in_process

_logger = get_logger("wan_custom.T2V")


class T2VPipeline:

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
        # GPU optimization (process-level)
        # --------------------------------------------------
        profile = detect_gpu_profile()
        apply_global_optimizations(profile)
        _logger.info(f"GPU profile: {profile}")

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
        # 🔥 SINGLE WAN INVOCATION (CACHE FRIENDLY)
        # --------------------------------------------------
        frame_num = seconds_to_wan_frames(target_duration)

        cmd = [
            "python3", "generate.py",
            "--task", "t2v-A14B",
            "--ckpt_dir", config.MODEL_DIRS["t2v"],
            "--prompt", prompt_text,
            "--convert_model_dtype",
            "--size", size,
            "--t5_cpu",
            "--frame_num", str(frame_num),
            "--sample_steps", str(sample_steps),
            "--sample_shift", "10",
            "--save_file", output_path,
        ]

        _logger.info("Executing WAN generate.py (single run)")
        _logger.info(" ".join(cmd))

        # subprocess.run(
        #     cmd,
        #     cwd=config.WAN_ROOT,  # /workspace/Wan2.2
        #     check=True,
        # )

        generate_in_process(
            task="t2v-A14B",
            prompt=prompt_text,
            ckpt_dir=config.MODEL_DIRS["t2v"],
            size=size,
            frame_num=str(frame_num),
            save_file=output_path,
            sample_steps=sample_steps,
            sample_shift=10,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("WAN T2V generation failed")

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
