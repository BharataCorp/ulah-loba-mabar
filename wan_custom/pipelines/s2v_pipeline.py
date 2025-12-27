"""
WAN 2.2 Speech-to-Video Pipeline
================================

- One-time model loading
- Supports:
  - prompt (string / JSON)
  - image (reference image)
  - audio (wav / mp3)
- Safe for long-running daemon
- KISS & production-ready
"""

import os
from typing import Union, Dict, Any

import torch

from wan.pipelines.base_pipeline import BasePipeline
from wan.logger import get_logger
from wan import config
from wan.utils.duration import seconds_to_frames
from wan.utils.prompt_parser import normalize_prompt

import wan  # official WAN package (from Wan2.2 repo)

_logger = get_logger("WAN.S2V")


class S2VPipeline(BasePipeline):
    """
    WAN 2.2 Speech-to-Video Pipeline
    """

    # =====================================================
    # INTERNAL: LOAD PIPELINE (ONCE)
    # =====================================================

    @classmethod
    def _load_pipeline(cls):
        """
        Load WAN2.2 S2V model from local directory.
        This method is called ONLY ONCE.
        """
        model_dir = config.MODEL_DIRS["s2v"]

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"[WAN S2V] Model directory not found: {model_dir}"
            )

        _logger.info(f"Loading WAN2.2 S2V from {model_dir}")

        # WAN S2V uses WanS2V wrapper (NOT DiffusionPipeline directly)
        pipeline = wan.WanS2V(
            ckpt_dir=model_dir,
            offload_model=config.USE_OFFLOAD,
            convert_model_dtype=config.USE_CONVERT_DTYPE,
            t5_cpu=config.USE_T5_CPU,
        )

        _logger.info("WAN2.2 S2V pipeline loaded")
        return pipeline

    # =====================================================
    # PUBLIC API
    # =====================================================

    @classmethod
    def generate(
        cls,
        prompt: Union[str, Dict[str, Any]],
        audio_path: str,
        image_path: str,
        target_duration: int,
        size: str = None,
        sample_steps: int = None,
        sample_shift: float = None,
        output_path: str = None,
    ) -> str:
        """
        Generate video from speech + reference image.

        Args:
            prompt: string or structured JSON
            audio_path: wav / mp3 file
            image_path: reference image
            target_duration: seconds
            size: e.g. "832*480"
            sample_steps: diffusion steps
            sample_shift: motion shift
            output_path: final mp4 path

        Returns:
            output_path
        """
        # Ensure model loaded
        cls.load()

        # -------------------------------
        # Validate inputs
        # -------------------------------
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if target_duration > config.MAX_DURATION_SECONDS:
            raise ValueError(
                f"target_duration exceeds limit ({config.MAX_DURATION_SECONDS}s)"
            )

        # -------------------------------
        # Prompt
        # -------------------------------
        prompt_str = normalize_prompt(prompt)

        # -------------------------------
        # Duration → frames
        # -------------------------------
        frame_num = seconds_to_frames(
            target_duration,
            fps=config.DEFAULT_FPS,
        )

        # -------------------------------
        # Size
        # -------------------------------
        size = size or config.DEFAULT_SIZE
        config.validate_size(size)
        width, height = map(int, size.split("*"))

        # -------------------------------
        # Sampling params
        # -------------------------------
        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS
        sample_shift = sample_shift or config.DEFAULT_SAMPLE_SHIFT

        # -------------------------------
        # Output path
        # -------------------------------
        if output_path is None:
            output_path = cls._default_output_path(prompt_str, size)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        _logger.info(
            f"S2V generate | duration={target_duration}s "
            f"frames={frame_num} size={size}"
        )

        # -------------------------------
        # Inference
        # -------------------------------
        with torch.no_grad():
            cls._pipeline.generate(
                prompt=prompt_str,
                image=image_path,
                audio=audio_path,
                frame_num=frame_num,
                size=size,
                sample_steps=sample_steps,
                sample_shift=sample_shift,
                save_file=output_path,
            )

        _logger.info(f"S2V video saved to {output_path}")

        # -------------------------------
        # Cleanup
        # -------------------------------
        cls.clear_cuda()
        return output_path

    # =====================================================
    # INTERNAL HELPERS
    # =====================================================

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        safe_prompt = (
            prompt[:60]
            .replace(" ", "_")
            .replace("/", "")
            .replace("'", "")
        )
        filename = f"s2v_{size}_{safe_prompt}.mp4"
        return os.path.join(config.OUTPUT_DIR, filename)
