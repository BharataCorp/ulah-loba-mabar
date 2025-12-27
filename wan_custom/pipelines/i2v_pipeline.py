"""
WAN 2.2 Image-to-Video Pipeline (FINAL)
======================================

Design goals:
- Persistent WAN 2.2 I2V pipeline
- Supports image + text prompt
- Supports structured JSON prompt
- Supports target_duration (seconds → frames)
- Diffusers-compliant (2025-safe)
"""

from __future__ import annotations

import os
from typing import Union, Dict, Any

import torch
import imageio.v3 as iio
from PIL import Image
from diffusers import DiffusionPipeline

from wan_custom.pipelines.base_pipeline import BasePipeline
from wan_custom.logger import get_logger
from wan_custom import config

_logger = get_logger("wan_custom.I2V")


class I2VPipeline(BasePipeline):
    """
    WAN 2.2 Image-to-Video Pipeline
    """

    # ==========================================================
    # INTERNAL: PIPELINE LOADER
    # ==========================================================

    @classmethod
    def _load_pipeline(cls) -> DiffusionPipeline:
        """
        Load WAN 2.2 I2V pipeline from local directory.
        This method is called ONLY ONCE.
        """
        model_dir = config.MODEL_DIRS.get("i2v")

        if not model_dir:
            raise RuntimeError("MODEL_DIRS['i2v'] is not configured")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"[WAN I2V] Model directory not found: {model_dir}"
            )

        _logger.info(f"Loading WAN 2.2 I2V model from: {model_dir}")

        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # ---- device placement ----
        if config.USE_OFFLOAD:
            _logger.info("Enabling CPU offload for WAN I2V")
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
        image_path: str,
        prompt: Union[str, Dict[str, Any]],
        target_duration: int,
        size: str | None = None,
        sample_steps: int | None = None,
        output_path: str | None = None,
    ) -> str:
        """
        Generate video from input image + prompt.

        Args:
            image_path:
                path to input image (jpg / png)
            prompt:
                string or structured JSON
            target_duration:
                target duration in seconds
            size:
                resolution string, e.g. "832*480"
            sample_steps:
                diffusion inference steps
            output_path:
                final mp4 path

        Returns:
            Absolute path to generated video
        """

        # ---- ensure pipeline loaded ----
        cls.load()

        # ---- image validation ----
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # ---- prompt handling ----
        prompt_text = cls._normalize_prompt(prompt)

        if not prompt_text:
            raise ValueError("Prompt text is empty after normalization")

        # ---- duration validation ----
        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

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

        # Resize image to match video resolution
        image = image.resize((width, height), Image.BICUBIC)

        # ---- inference params ----
        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS

        output_path = (
            output_path
            or cls._default_output_path(prompt_text, size)
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        _logger.info(
            f"I2V generate | frames={frame_num} "
            f"size={size} steps={sample_steps}"
        )

        # ---- inference ----
        with torch.no_grad():
            result = cls._pipeline(
                image=image,
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

        _logger.info(f"I2V video saved: {output_path}")

        return output_path

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    @staticmethod
    def _normalize_prompt(
        prompt: Union[str, Dict[str, Any]]
    ) -> str:
        """
        Convert JSON prompt into WAN-friendly string.
        """
        if isinstance(prompt, str):
            return prompt.strip()

        if isinstance(prompt, dict):
            if "prompt_for_wan_one_i2v" in prompt:
                return str(prompt["prompt_for_wan_one_i2v"]).strip()

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

        filename = f"i2v_{size}_{safe_prompt}.mp4"
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
