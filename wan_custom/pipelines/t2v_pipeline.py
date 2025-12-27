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
    FINAL T2V PIPELINE (RUNPOD SAFE)

    Strategy:
    - <= 5s  : WAN only
    - 5–15s  : WAN 5s + ffmpeg smart extend (NO RIFE)
    - > 15s  : WAN chunk (legacy, safe)
    """

    SAFE_WAN_SECONDS = 5
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
        # SMART EXTEND MODE (DEFAULT)
        # ==================================================
        if smart_extend:
            # --- case 1: fully safe ---
            if target_duration <= cls.SAFE_WAN_SECONDS:
                _logger.info("Duration within safe WAN range")
                return cls._generate_with_chunks(
                    prompt_text,
                    target_duration,
                    size,
                    sample_steps,
                    sample_shift,
                    output_path,
                )

            # --- base WAN clip (5s) ---
            _logger.info(
                f"Smart extend: WAN base {cls.SAFE_WAN_SECONDS}s"
            )
            base_output = output_path.replace(".mp4", "_base.mp4")

            cls._generate_with_chunks(
                prompt_text,
                cls.SAFE_WAN_SECONDS,
                size,
                sample_steps,
                sample_shift,
                base_output,
            )

            # --- extend with ffmpeg ---
            if target_duration <= cls.MAX_SMART_EXTEND_SECONDS:
                _logger.info(
                    f"Smart extend via ffmpeg to {target_duration}s"
                )
                cls._extend_with_ffmpeg(
                    base_output,
                    output_path,
                    target_duration,
                )
                return output_path

            # --- fallback: chunk ---
            _logger.warning(
                "Duration too long, fallback to WAN chunk mode"
            )
            return cls._generate_with_chunks(
                prompt_text,
                target_duration,
                size,
                sample_steps,
                sample_shift,
                output_path,
            )

        # ==================================================
        # PURE CHUNK MODE (LEGACY)
        # ==================================================
        _logger.info("smart_extend disabled → pure WAN chunk")
        return cls._generate_with_chunks(
            prompt_text,
            target_duration,
            size,
            sample_steps,
            sample_shift,
            output_path,
        )

    # ==================================================
    # WAN CHUNK GENERATION (UNCHANGED CORE FLOW)
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
        _logger.info(f"Splitting {target_duration}s into chunks: {chunks}")

        chunk_outputs: List[str] = []
        temp_dir = config.OUTPUT_DIR

        for idx, chunk_sec in enumerate(chunks):
            frame_num = seconds_to_wan_frames(chunk_sec)

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
                "--sample_steps", str(sample_steps),
                "--sample_shift", str(sample_shift),
                "--offload_model", "True",
                "--t5_cpu",
                "--convert_model_dtype",
                "--save_file", chunk_out,
            ]

            _logger.info(f"[Chunk {idx + 1}/{len(chunks)}]")
            subprocess.run(
                cmd,
                cwd=config.WAN_ROOT,
                check=True,
            )

            chunk_outputs.append(chunk_out)

        if len(chunk_outputs) == 1:
            os.replace(chunk_outputs[0], output_path)
            return output_path

        cls._concat_videos(chunk_outputs, output_path)
        return output_path

    # ==================================================
    # FFmpeg SMART EXTEND (ANTI-GHOSTING, ANTI-BLUR)
    # ==================================================
    @staticmethod
    def _extend_with_ffmpeg(
            input_video: str,
            output_video: str,
            target_duration: int,
    ):
        """
        Correct smart extend:
        - Extend duration properly
        - No ghosting
        - No blur
        - Affiliate-safe
        """

        vf = (
            # 1. Freeze last frame to extend duration
            f"tpad=stop_mode=clone:stop_duration={target_duration},"
            # 2. Smooth motion (no new hallucination)
            "minterpolate="
            "fps=16:"
            "mi_mode=mci:"
            "mc_mode=aobmc:"
            "me_mode=bidir:"
            "vsbmc=1"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", vf,
            "-t", str(target_duration),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_video,
        ]

        subprocess.run(cmd, check=True)

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
