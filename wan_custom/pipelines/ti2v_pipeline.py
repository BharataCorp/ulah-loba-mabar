"""
WAN 2.2 Text-Image-to-Video (TI2V) Pipeline
==========================================

- Loads WAN2.2 TI2V pipeline once
- Supports string or structured JSON prompt
- Requires a single input image
- Supports target_duration (seconds)
- Safe for long-running daemon usage
"""

import os
from typing import Union, Dict, Any

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from wan_custom.pipelines.base_pipeline import BasePipeline
from wan_custom.logger import get_logger
from wan_custom import config

_logger = get_logger("wan_custom.TI2V")


class TI2VPipeline(BasePipeline):
    """
    WAN 2.2 Text-Image-to-Video Pipeline
    """

    @classmethod
    def _load_pipeline(cls):
        """
        Load WAN2.2 TI2V model from local directory.
        Called only once per process.
        """
        model_dir = config.MODEL_DIRS["ti2v"]

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"[WAN TI2V] Model directory not found: {model_dir}"
            )

        _logger.info(f"Loading WAN2.2 TI2V from {model_dir}")

        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        if config.USE_OFFLOAD:
            _logger.info("Enabling CPU offload for TI2V pipeline")
            pipe.enable_model_cpu_offload()

        if config.USE_CONVERT_DTYPE:
            pipe.to(torch.bfloat16)

        pipe.set_progress_bar_config(disable=False)
        return pipe

    # =============================
    # PUBLIC API
    # =============================

    @classmethod
    def generate(
        cls,
        image_path: str,
        prompt: Union[str, Dict[str, Any]],
        target_duration: int,
        size: str = None,
        sample_steps: int = None,
        sample_shift: float = None,
        output_path: str = None,
    ) -> str:
        """
        Generate video from text + image.

        Args:
            image_path: path to single input image
            prompt: string or structured JSON
            target_duration: seconds (converted to frame_num)
            size: e.g. "832*480"
            sample_steps: diffusion steps
            sample_shift: motion shift
            output_path: final mp4 path

        Returns:
            output_path
        """
        cls.load()

        # -------- image validation --------
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # -------- prompt handling --------
        prompt_str = cls._normalize_prompt(prompt)

        # -------- duration handling --------
        if target_duration > config.MAX_DURATION_SECONDS:
            raise ValueError(
                f"target_duration exceeds limit ({config.MAX_DURATION_SECONDS}s)"
            )

        frame_num = config.seconds_to_frames(target_duration)

        # -------- size handling --------
        size = size or config.DEFAULT_SIZE
        config.validate_size(size)
        width, height = map(int, size.split("*"))

        # -------- inference params --------
        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS
        sample_shift = sample_shift or config.DEFAULT_SAMPLE_SHIFT

        output_path = output_path or cls._default_output_path(
            prompt_str, size
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        _logger.info(
            f"Generating TI2V | duration={target_duration}s "
            f"frames={frame_num} size={size}"
        )

        # -------- inference --------
        with torch.no_grad():
            result = cls._pipeline(
                prompt=prompt_str,
                image=image,
                num_frames=frame_num,
                height=height,
                width=width,
                num_inference_steps=sample_steps,
                sample_shift=sample_shift,
            )

        video_frames = result.frames

        # -------- save video --------
        cls._pipeline.save_video(video_frames, output_path)

        _logger.info(f"TI2V video saved to {output_path}")

        cls.clear_cuda()
        return output_path

    # =============================
    # INTERNAL HELPERS
    # =============================

    @staticmethod
    def _normalize_prompt(prompt: Union[str, Dict[str, Any]]) -> str:
        """
        Convert JSON prompt to WAN-friendly string.
        """
        if isinstance(prompt, str):
            return prompt.strip()

        if isinstance(prompt, dict):
            if "prompt_for_wan_one_ti2v" in prompt:
                return prompt["prompt_for_wan_one_ti2v"].strip()

            if "prompt_for_wan_one_t2v" in prompt:
                return prompt["prompt_for_wan_one_t2v"].strip()

            scenes = prompt.get("scenes", [])
            scene_texts = []
            for s in scenes:
                actions = ", ".join(s.get("actions", []))
                mood = s.get("mood", "")
                scene_texts.append(f"{actions}. Mood: {mood}")

            return " ".join(scene_texts)

        raise TypeError("Prompt must be string or dict")

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        safe_prompt = (
            prompt[:60]
            .replace(" ", "_")
            .replace("/", "")
            .replace("'", "")
        )
        filename = f"ti2v_{size}_{safe_prompt}.mp4"
        return os.path.join(config.OUTPUT_DIR, filename)
