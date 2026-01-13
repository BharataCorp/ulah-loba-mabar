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
        scenes: List[Dict[str, Any]],
        target_duration: int,
        size: str = "832*480",
        sample_steps: int = 10,
        sample_shift: int = 10,
        output_path: str | None = None,
    ) -> str:
        if target_duration <= 0:
            raise ValueError("target_duration must be > 0")

        # create directory config.OUTPUT_DIR if not exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        scene_outputs: List[str] = []
        temp_dir = config.OUTPUT_DIR

        # loop over scenes to create prompt text
        for scene in scenes:
            duration = scene.get("duration",5)
            # change duration if greather than 5, maximum safe seconds
            if duration > cls.SAFE_WAN_SECONDS:
                scene["duration"] = cls.SAFE_WAN_SECONDS

            prompt = scene.get("prompt", "")

            scene_out = os.path.join(
                temp_dir,
                f"t2v_scene_{len(scene_outputs) + 1}.mp4"
            )

            cmd = [
                "python3", "generate.py",
                "--task", "t2v-A14B",
                "--ckpt_dir", config.MODEL_DIRS["t2v"],
                "--prompt", prompt,
                "--size", size,
                "--frame_num", str(seconds_to_wan_frames(scene["duration"])),
                "--sample_steps", str(sample_steps),
                "--sample_shift", str(sample_shift),
                "--offload_model", "True",
                # "--t5_cpu",
                "--convert_model_dtype",
                "--save_file", scene_out,
            ]

            _logger.info(f"Generating scene with prompt: {prompt}")
            subprocess.run(cmd, cwd=config.WAN_ROOT, check=True)

            scene_outputs.append(scene_out)

        # concatenate all scenes into final output
        if output_path is None:
            output_path = cls._default_output_path("combined_scenes", size)
        cls._concat_videos(scene_outputs, output_path)
        return output_path

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
