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

    # =========================
    # SMART EXTEND CONSTANTS
    # =========================
    WAN_BASE_SECONDS = 5
    FFMPEG_MAX_EXTEND = 3
    MAX_SMART_EXTEND_SECONDS = 15

    @classmethod
    def generate(
        cls,
        prompt: Union[str, Dict[str, Any]],
        target_duration: int,
        size: str = "832*480",
        sample_steps: int = 16,
        sample_shift: int = 10,
        output_path: str | None = None,
        *,
        smart_extend: bool = True,
    ) -> str:

        prompt_text = cls._normalize_prompt(prompt)
        if not prompt_text:
            raise ValueError("Prompt is empty")

        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        if output_path is None:
            output_path = cls._default_output_path(prompt_text, size)

        # ==================================================
        # SMART EXTEND MODE
        # ==================================================
        if smart_extend:

            # --- Case 1: WAN only ---
            if target_duration <= cls.WAN_BASE_SECONDS:
                _logger.info("smart_extend: WAN only")
                return cls._generate_with_chunks(
                    prompt_text,
                    target_duration,
                    size,
                    sample_steps,
                    sample_shift,
                    output_path,
                )

            base_output = output_path.replace(".mp4", "_base.mp4")

            cls._generate_with_chunks(
                prompt_text,
                cls.WAN_BASE_SECONDS,
                size,
                sample_steps,
                sample_shift,
                base_output,
            )

            remain = target_duration - cls.WAN_BASE_SECONDS

            # --- Case 2: FFmpeg extend ---
            if remain <= cls.FFMPEG_MAX_EXTEND:
                _logger.info(f"smart_extend: FFmpeg extend {remain}s")
                cls._extend_ffmpeg(
                    base_output,
                    output_path,
                    target_duration,
                )
                return output_path

            # --- Case 3: RIFE extend ---
            if target_duration <= cls.MAX_SMART_EXTEND_SECONDS:
                _logger.info(f"smart_extend: RIFE extend to {target_duration}s")
                cls._extend_rife(
                    base_output,
                    output_path,
                )
                return output_path

            # --- Case 4: Fallback chunk ---
            _logger.warning("smart_extend: fallback to chunk")

            mid_output = output_path.replace(".mp4", "_extended.mp4")
            cls._extend_rife(base_output, mid_output)

            tail_seconds = target_duration - cls.MAX_SMART_EXTEND_SECONDS
            tail_output = output_path.replace(".mp4", "_tail.mp4")

            cls._generate_with_chunks(
                prompt_text,
                tail_seconds,
                size,
                sample_steps,
                sample_shift,
                tail_output,
            )

            cls._concat_videos([mid_output, tail_output], output_path)
            return output_path

        # ==================================================
        # LEGACY MODE
        # ==================================================
        return cls._generate_with_chunks(
            prompt_text,
            target_duration,
            size,
            sample_steps,
            sample_shift,
            output_path,
        )

    # ==================================================
    # WAN GENERATION (UNCHANGED)
    # ==================================================
    @classmethod
    def _generate_with_chunks(
        cls,
        prompt_text: str,
        target_duration: int,
        size: str,
        sample_steps: int,
        sample_shift: int,
        output_path: str,
    ) -> str:

        chunks = split_duration_to_chunks(target_duration)
        chunk_outputs: List[str] = []

        for idx, sec in enumerate(chunks):
            frame_num = seconds_to_wan_frames(sec)
            chunk_out = output_path.replace(".mp4", f"_chunk{idx+1}.mp4")

            cmd = [
                "python3", "generate.py",
                "--task", "t2v-A14B",
                "--ckpt_dir", config.MODEL_DIRS["t2v"],
                "--prompt", prompt_text,
                "--size", size,
                "--frame_num", str(frame_num),
                "--sample_steps", str(sample_steps),
                "--sample_shift", str(sample_shift),
                "--offload_model", "True",
                "--t5_cpu",
                "--convert_model_dtype",
                "--save_file", chunk_out,
            ]

            subprocess.run(cmd, cwd=config.WAN_ROOT, check=True)
            chunk_outputs.append(chunk_out)

        if len(chunk_outputs) == 1:
            os.replace(chunk_outputs[0], output_path)
            return output_path

        cls._concat_videos(chunk_outputs, output_path)
        return output_path

    # ==================================================
    # EXTEND METHODS
    # ==================================================
    @staticmethod
    def _extend_ffmpeg(input_video: str, output_video: str, duration: int):
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf",
            "minterpolate=fps=16:mi_mode=mci:mc_mode=aobmc:me_mode=bidir",
            "-t", str(duration),
            output_video,
        ]
        subprocess.run(cmd, check=True)

    @staticmethod
    def _extend_rife(input_video: str, output_video: str):
        cmd = [
            config.RIFE_BIN,
            "-i", input_video,
            "-o", output_video,
            "-m", config.RIFE_MODEL_DIR,
            "-n", "2",
        ]
        subprocess.run(cmd, check=True)

    # ==================================================
    # HELPERS
    # ==================================================
    @staticmethod
    def _normalize_prompt(prompt: Union[str, Dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            return prompt.strip()

        if isinstance(prompt, dict):
            if "prompt_for_wan_one_t2v" in prompt:
                return str(prompt["prompt_for_wan_one_t2v"]).strip()
            texts = []
            for s in prompt.get("scenes", []):
                texts.append(
                    f"{', '.join(s.get('actions', []))}. "
                    f"Setting: {s.get('setting', '')}. "
                    f"Mood: {s.get('mood', '')}"
                )
            return " ".join(texts).strip()

        raise TypeError("prompt must be str or dict")

    @staticmethod
    def _default_output_path(prompt: str, size: str) -> str:
        safe = prompt[:50].replace(" ", "_").replace("/", "")
        return os.path.join(config.OUTPUT_DIR, f"t2v_{size}_{safe}.mp4")

    @staticmethod
    def _concat_videos(chunks: List[str], output: str):
        list_file = output.replace(".mp4", ".txt")
        with open(list_file, "w") as f:
            for c in chunks:
                f.write(f"file '{os.path.abspath(c)}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output],
            check=True,
        )
