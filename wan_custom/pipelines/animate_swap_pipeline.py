"""
WAN 2.2 Animate Replacement (Face Swap / Character Swap)
========================================================

- Replace character in existing video
- Uses official preprocess_data.py
- Not generative T2V
- CPU + GPU hybrid pipeline
"""

import os
import subprocess
from typing import Optional

from wan_custom.logger import get_logger
from wan_custom import config

_logger = get_logger("wan_custom.AnimateSwap")


class AnimateSwapPipeline:
    """
    WAN Animate Replacement Pipeline
    """

    @staticmethod
    def generate(
        source_video: str,
        reference_image: str,
        output_path: str,
        resolution: str = "1280*720",
        iterations: int = 3,
        k: int = 7,
        w_len: int = 1,
        h_len: int = 1,
    ) -> str:
        """
        Replace character in video with reference image.

        Args:
            source_video: original video path
            reference_image: character image
            output_path: final mp4
            resolution: target resolution
            iterations, k, w_len, h_len: replacement params

        Returns:
            output_path
        """
        if not os.path.isfile(source_video):
            raise FileNotFoundError(f"Video not found: {source_video}")

        if not os.path.isfile(reference_image):
            raise FileNotFoundError(f"Reference image not found: {reference_image}")

        width, height = map(int, resolution.split("*"))

        work_dir = os.path.join(config.TMP_DIR, "animate_swap")
        os.makedirs(work_dir, exist_ok=True)

        preprocess_script = os.path.join(
            config.WAN_ROOT,
            "wan/modules/animate/preprocess/preprocess_data.py"
        )

        if not os.path.isfile(preprocess_script):
            raise FileNotFoundError(
                "preprocess_data.py not found. "
                "Ensure WAN repo structure is intact."
            )

        _logger.info("Running Animate Replacement preprocess")

        cmd = [
            "python3",
            preprocess_script,
            "--ckpt_path", config.MODEL_DIRS["animate"] + "/process_checkpoint",
            "--video_path", source_video,
            "--refer_path", reference_image,
            "--save_path", work_dir,
            "--resolution_area", str(width), str(height),
            "--iterations", str(iterations),
            "--k", str(k),
            "--w_len", str(w_len),
            "--h_len", str(h_len),
            "--replace_flag",
        ]

        subprocess.run(cmd, check=True)

        # Output video location defined by WAN preprocess convention
        generated_video = os.path.join(work_dir, "output.mp4")

        if not os.path.isfile(generated_video):
            raise RuntimeError("Animate replacement failed: output video not found")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.rename(generated_video, output_path)

        _logger.info(f"Animate replacement saved to {output_path}")
        return output_path
