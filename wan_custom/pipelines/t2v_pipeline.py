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

        profile = detect_gpu_profile()
        apply_global_optimizations(profile)
        _logger.info(f"GPU profile: {profile}")

        prompt_text = cls._normalize_prompt(prompt)
        if not prompt_text:
            raise ValueError("Prompt is empty")

        size = size or config.DEFAULT_SIZE
        sample_steps = sample_steps or config.DEFAULT_SAMPLE_STEPS

        output_path = output_path or cls._default_output_path(prompt_text, size)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        frame_num = seconds_to_wan_frames(target_duration)

        cmd = [
            "python3", "generate.py",
            "--task", "t2v-A14B",
            "--ckpt_dir", config.MODEL_DIRS["t2v"],
            "--prompt", prompt_text,
            "--convert_model_dtype",
            "--size", size,
            ""
            "--frame_num", str(frame_num),
            "--sample_steps", str(sample_steps),
            "--sample_shift", "10",
            "--save_file", output_path,
        ]

        _logger.info("Executing WAN generate.py (subprocess mode)")
        _logger.info(" ".join(cmd))

        subprocess.run(
            cmd,
            cwd=config.WAN_ROOT,
            check=True,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("WAN T2V generation failed")

        _logger.info(f"T2V video saved: {output_path}")
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
