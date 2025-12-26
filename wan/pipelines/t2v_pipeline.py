"""
WAN 2.2 Text-to-Video Pipeline (NATIVE)
======================================

- Native WAN 2.2 (NO Diffusers)
- Model loaded ONCE (persistent)
- Safe for RunPod long-running worker
- Supports string or JSON prompt
- target_duration (seconds → frame_num)
"""

from __future__ import annotations

import os
from typing import Union, Dict, Any

import torch

from wan.logger import get_logger
from wan import config
from wan.pipelines.base_pipeline import BasePipeline

# WAN native imports (OFFICIAL)
from wan.text2video import WanT2V
from wan.configs import load_config

_logger = get_logger("WAN.T2V")


class T2VPipeline(BasePipeline):
    """
    WAN 2.2 Native Text-to-Video Pipeline
    """

    # ==========================================================
    # INTERNAL: LOAD PIPELINE (ONCE)
    # ==========================================================

    @classmethod
    def _load_pipeline(cls) -> WanT2V:
        model_dir = config.MODEL_DIRS.get("t2v")
        if not model_dir:
            raise RuntimeError("MODEL_DIRS['t2v'] not configured")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"T2V model dir not found: {model_dir}")

        _logger.info(f"Loading WAN 2.2 T2V from {model_dir}")

        # Load WAN config (OFFICIAL METHOD)
        cfg = load_config(
            task="t2v-A14B",
            size=config.DEFAULT_SIZE,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = WanT2V(
            config=cfg,
            ckpt_dir=model_dir,
            device=device,
            offload=config.USE_OFFLOAD,
        )

        _logger.info("WAN 2.2 T2V pipeline loaded successfully")
        return pipe

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    @classmethod
    def generate(
        cls,
        prompt: Union[str, Dict[str, Any]],
        target_duration: int,
        size: str | None = None,
        sample_steps: int | None = None,
        output_path: str | None = None,
    ) -> str:
        """
        Generate T2V video using WAN native engine.

        Returns:
            output mp4 path
        """

        cls.load()

        prompt_text = cls._normalize_prompt(prompt)
        if not prompt_text:
            raise ValueError("Prompt is empty")

        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        if target_duration > config.MAX_DURATION_SECONDS:
            raise ValueError(
                f"target_duration exceeds limit ({config.MAX_DURATION_SECONDS}s)"
            )

        frame_num = config.seconds_to_frames(target_duration)

        size = size or config.DEFAULT_SIZE
        config.validate_size(size)

        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS

        output_path = (
            output_path
            or cls._default_output_path(prompt_text, size)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        _logger.info(
            f"T2V generate | duration={target_duration}s "
            f"frames={frame_num} size={size} steps={sample_steps}"
        )

        # ==========================
        # WAN NATIVE GENERATION
        # ==========================
        video_path = cls._pipeline.generate(
            prompt=prompt_text,
            frame_num=frame_num,
            sample_steps=sample_steps,
            save_file=output_path,
        )

        _logger.info(f"T2V video saved: {video_path}")
        return video_path

    # ==========================================================
    # HELPERS
    # ==========================================================

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
            parts = []
            for s in scenes:
                actions = ", ".join(s.get("actions", []))
                mood = s.get("mood", "")
                setting = s.get("setting", "")
                parts.append(
                    f"{actions}. Setting: {setting}. Mood: {mood}."
                )
            return " ".join(parts).strip()

        raise TypeError("prompt must be string or dict")

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
