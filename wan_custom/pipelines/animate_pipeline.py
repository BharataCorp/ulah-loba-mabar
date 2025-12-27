# wan/pipelines/mass_animate_pipeline.py

import os
import hashlib
import subprocess
from typing import List, Dict

from wan_custom.logger import get_logger
from wan_custom import config

_logger = get_logger("wan_custom.MassAnimate")


class MassAnimatePipeline:
    """
    WAN 2.2 Mass Animate / Replace Pipeline
    - One source video
    - Multiple character images
    - Shared preprocess cache
    """

    @staticmethod
    def _video_hash(video_path: str) -> str:
        stat = os.stat(video_path)
        key = f"{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key.encode()).hexdigest()

    @classmethod
    def preprocess_once(
        cls,
        video_path: str,
        cache_dir: str,
        replace: bool = False,
    ) -> str:
        os.makedirs(cache_dir, exist_ok=True)

        preprocess_dir = os.path.join(cache_dir, "preprocess")
        if os.path.exists(preprocess_dir):
            _logger.info("Preprocess cache found, skipping.")
            return preprocess_dir

        _logger.info("Running WAN animate preprocess (ONCE)")

        cmd = [
            "python",
            "wan/modules/animate/preprocess/preprocess_data.py",
            "--ckpt_path", config.ANIMATE_PROCESS_CKPT,
            "--video_path", video_path,
            "--save_path", preprocess_dir,
            "--resolution_area", "832", "480",
        ]

        if replace:
            cmd.append("--replace_flag")
        else:
            cmd.extend(["--retarget_flag", "--use_flux"])

        subprocess.run(cmd, check=True)
        return preprocess_dir

    @classmethod
    def generate_for_characters(
        cls,
        video_path: str,
        character_images: Dict[str, str],
        output_dir: str,
        replace: bool = False,
    ) -> List[str]:
        video_hash = cls._video_hash(video_path)
        cache_root = os.path.join(config.CACHE_DIR, "animate", video_hash)

        preprocess_dir = cls.preprocess_once(
            video_path,
            cache_root,
            replace=replace,
        )

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for char_id, image_path in character_images.items():
            out_path = os.path.join(output_dir, f"{char_id}.mp4")
            if os.path.exists(out_path):
                _logger.info(f"Skip existing output: {out_path}")
                results.append(out_path)
                continue

            _logger.info(f"Generating animate for character: {char_id}")

            cmd = [
                "python",
                "generate.py",
                "--task", "animate-14B",
                "--ckpt_dir", config.MODEL_DIRS["animate"],
                "--process_dir", preprocess_dir,
                "--image", image_path,
                "--save_file", out_path,
            ]

            subprocess.run(cmd, check=True)
            results.append(out_path)

        return results
