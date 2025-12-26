"""
WAN 2.2 Text-to-Video Pipeline (FINAL)
=====================================

Design goals:
- Load WAN 2.2 T2V model ONCE (persistent pipeline)
- Safe for long-running daemon (RunPod / GPU worker)
- Supports string prompt OR structured JSON prompt
- Supports target_duration (seconds → frames)
- Strict Diffusers API usage (no hidden / invalid args)
"""

from __future__ import annotations

import os
import json
from typing import Union, Dict, Any

import torch
import imageio.v3 as iio
from diffusers import DiffusionPipeline

from wan.pipelines.base_pipeline import BasePipeline
from wan.logger import get_logger
from wan import config

_logger = get_logger("WAN.T2V")


class T2VPipeline(BasePipeline):
    """
    WAN 2.2 Text-to-Video Pipeline (Diffusers-based)
    """

    # ==========================================================
    # INTERNAL: PIPELINE LOADER
    # ==========================================================

    @classmethod
    def _load_pipeline(cls) -> DiffusionPipeline:
        """
        Load WAN 2.2 T2V pipeline from local directory.

        This method is called ONCE and cached by BasePipeline.
        """
        model_dir = config.MODEL_DIRS.get("t2v")

        if not model_dir:
            raise RuntimeError("MODEL_DIRS['t2v'] is not configured")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"[WAN T2V] Model directory not found: {model_dir}"
            )

        _logger.info(f"Loading WAN 2.2 T2V model from: {model_dir}")

        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # ---- device placement ----
        if config.USE_OFFLOAD:
            _logger.info("Enabling CPU offload for WAN T2V")
            pipe.enable_model_cpu_offload()
        else:
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

        pipe.set_progress_bar_config(disable=False)

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
        Generate video from text prompt.

        Args:
            prompt:
                - string prompt
                - OR structured JSON prompt
            target_duration:
                target video duration in seconds
            size:
                resolution string, e.g. "832*480"
            sample_steps:
                diffusion inference steps
            output_path:
                final mp4 path

        Returns:
            Absolute path to generated video file
        """

        # ---- ensure pipeline is loaded ----
        cls.load()

        # ---- normalize prompt ----
        prompt_text = cls._normalize_prompt(prompt)

        if not prompt_text:
            raise ValueError("Prompt text is empty after normalization")

        # ---- duration validation ----
        if target_duration <= 0:
            raise ValueError("target_duration must be > 0 seconds")

        if target_duration > config.MAX_DURATION_SECONDS:
            raise ValueError(
                f"target_duration exceeds limit "
                f"({config.MAX_DURATION_SECONDS}s)"
            )

        frame_num = config.seconds_to_frames(target_duration)

        # ---- resolution handling ----
        size = size or config.DEFAULT_SIZE
        config.validate_size(size)

        width, height = map(int, size.split("*"))

        # ---- inference params ----
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

        # ---- inference ----
        with torch.no_grad():
            result = cls._pipeline(
                prompt=prompt_text,
                num_frames=frame_num,
                height=height,
                width=width,
                num_inference_steps=sample_steps,
            )

        if not hasattr(result, "frames"):
            raise RuntimeError("Pipeline output does not contain frames")

        frames = result.frames

        # ---- save video ----
        cls._save_video(
            frames=frames,
            output_path=output_path,
            fps=config.DEFAULT_FPS,
        )

        _logger.info(f"T2V video saved: {output_path}")

        return output_path

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    @staticmethod
    def _normalize_prompt(
        prompt: Union[str, Dict[str, Any]]
    ) -> str:
        """
        Convert structured JSON prompt into WAN-friendly text.
        """
        if isinstance(prompt, str):
            return prompt.strip()

        if isinstance(prompt, dict):
            # preferred explicit field
            if "prompt_for_wan_one_t2v" in prompt:
                return str(prompt["prompt_for_wan_one_t2v"]).strip()

            # fallback: scene-based description
            scenes = prompt.get("scenes", [])
            texts: list[str] = []

            for scene in scenes:
                actions = ", ".join(scene.get("actions", []))
                mood = scene.get("mood", "")
                setting = scene.get("setting", "")
                texts.append(
                    f"{actions}. Setting: {setting}. Mood: {mood}."
                )

            return " ".join(texts).strip()

        raise TypeError("prompt must be str or dict")

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        """
        Generate safe default output filename.
        """
        safe_prompt = (
            prompt[:60]
            .replace(" ", "_")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .replace('"', "")
        )

        filename = f"t2v_{size}_{safe_prompt}.mp4"
        return os.path.join(config.OUTPUT_DIR, filename)

    @staticmethod
    def _save_video(
        frames,
        output_path: str,
        fps: int,
    ) -> None:
        """
        Save frames to mp4 using imageio.
        """
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
        )
