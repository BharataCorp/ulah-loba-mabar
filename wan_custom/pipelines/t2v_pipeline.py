# wan_custom/pipelines/t2v_pipeline.py

from __future__ import annotations
import os
import subprocess
from typing import Union, Dict, Any, List

from wan_custom import config
from wan_custom.logger import get_logger
from wan_custom.utils.wan_chunker import (
    split_duration_to_chunks,
    seconds_to_wan_frames,
)

_logger = get_logger("wan_custom.T2V")


class T2VPipeline:
    """
    FINAL DYNAMIC T2V PIPELINE (NO FREEZE, NO RIFE)

    Strategy:
    - Chunk 1  : cinematic (user steps)
    - Chunk 2+ : fast motion (low steps)
    """

    SAFE_WAN_SECONDS = 5

    FAST_SAMPLE_STEPS = 6
    FAST_SAMPLE_SHIFT = 8

    @classmethod
    def generate(
        cls,
        prompt: Union[str, Dict[str, Any]],
        target_duration: int,
        size: str = "832*480",
        sample_steps: int = 10,
        sample_shift: int = 10,
        output_path: str | None = None,
    ) -> str:

        prompt_text = cls._normalize_prompt(prompt)
        if not prompt_text:
            raise ValueError("Prompt is empty")

        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        if output_path is None:
            output_path = cls._default_output_path(prompt_text, size)

        chunks = split_duration_to_chunks(target_duration)
        _logger.info(f"Splitting {target_duration}s into chunks: {chunks}")

        chunk_outputs: List[str] = []
        temp_dir = config.OUTPUT_DIR

        for idx, chunk_sec in enumerate(chunks):
            frame_num = seconds_to_wan_frames(chunk_sec)

            # 🔥 FAST MODE for chunk > 1
            if idx == 0:
                steps = sample_steps
                shift = sample_shift
                _logger.info("Chunk 1 → cinematic mode")
            else:
                steps = cls.FAST_SAMPLE_STEPS
                shift = cls.FAST_SAMPLE_SHIFT
                _logger.info("Chunk tail → FAST mode")

            chunk_out = os.path.join(
                temp_dir,
                os.path.basename(output_path).replace(
                    ".mp4", f"_chunk{idx + 1}.mp4"
                )
            )

            cmd = [
                "python3", "generate.py",
                "--task", "t2v-A14B",
                "--ckpt_dir", config.MODEL_DIRS["t2v"],
                "--prompt", prompt_text,
                "--size", size,
                "--frame_num", str(frame_num),
                "--sample_steps", str(steps),
                "--sample_shift", str(shift),
                "--offload_model", "True",
                "--t5_cpu",
                "--convert_model_dtype",
                "--save_file", chunk_out,
            ]

            _logger.info(f"[Chunk {idx + 1}/{len(chunks)}] steps={steps} shift={shift}")
            subprocess.run(cmd, cwd=config.WAN_ROOT, check=True)

            chunk_outputs.append(chunk_out)

        # ===== FINAL OUTPUT =====
        if len(chunk_outputs) == 1:
            os.replace(chunk_outputs[0], output_path)
            return output_path

        cls._concat_videos(chunk_outputs, output_path)
        return output_path

    # ==================================================
    # HELPERS
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
                setting = s.get("setting", "")
                mood = s.get("mood", "")
                texts.append(
                    f"{actions}. Setting: {setting}. Mood: {mood}"
                )
            return " ".join(texts).strip()

        raise TypeError("prompt must be str or dict")

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        safe = (
            prompt[:50]
            .replace(" ", "_")
            .replace("/", "")
            .replace("\\", "")
            .replace('"', "")
            .replace("'", "")
        )
        return os.path.join(
            config.OUTPUT_DIR,
            f"t2v_{size}_{safe}.mp4",
        )

    @staticmethod
    def _concat_videos(chunks: List[str], output: str):
        list_file = output.replace(".mp4", ".txt")
        with open(list_file, "w") as f:
            for c in chunks:
                f.write(f"file '{os.path.abspath(c)}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output,
            ],
            check=True,
        )
