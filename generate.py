# Copyright 2024-2025 The Alibaba Wan Team Authors.
# MODIFIED: DAEMON-ONLY CACHED VERSION (NO SUBPROCESS)

import argparse
import logging
import sys
import warnings
from datetime import datetime
import random
import gc

warnings.filterwarnings("ignore")

import torch
from PIL import Image

from wan_custom.text2video import WanT2V
from wan_custom.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan_custom.utils.utils import save_video, str2bool

# ==========================================================
# GLOBAL STATE (PROCESS-LOCAL CACHE)
# ==========================================================
WAN_MODEL = None


# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser("WAN Cached Generator (DAEMON MODE)")

    p.add_argument("--task", default="t2v-A14B")
    p.add_argument("--prompt", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--size", default="832*480")
    p.add_argument("--frame_num", type=int, required=True)
    p.add_argument("--save_file", required=True)

    p.add_argument("--sample_steps", type=int, default=8)
    p.add_argument("--sample_shift", type=float, default=8)
    p.add_argument("--sample_guide_scale", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=42)

    p.add_argument("--t5_cpu", action="store_true")
    p.add_argument("--convert_model_dtype", action="store_true")

    return p.parse_args()


# ==========================================================
# LOGGING
# ==========================================================
def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ==========================================================
# MODEL LOADER (CACHED)
# ==========================================================
def get_model(args):
    global WAN_MODEL

    if WAN_MODEL is not None:
        logging.info("🔥 Reusing cached WAN T2V model")
        return WAN_MODEL

    logging.info("🚀 Loading WAN T2V model (ONCE)")

    torch.cuda.set_device(0)

    cfg = WAN_CONFIGS[args.task]

    WAN_MODEL = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    return WAN_MODEL


# ==========================================================
# GENERATE
# ==========================================================
def generate(args):
    init_logging()

    model = get_model(args)

    logging.info("🎬 Generating video")

    video = model.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=False,
    )

    logging.info(f"💾 Saving → {args.save_file}")

    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    torch.cuda.synchronize()
    logging.info("✅ Done (model still cached)")


# ==========================================================
# ENTRY
# ==========================================================
if __name__ == "__main__":
    args = parse_args()
    generate(args)
