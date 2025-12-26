# Copyright 2024-2025 The Alibaba Wan Team Authors.
# MODIFIED: Cached Model Version for Production Inference

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import random

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool

# ==========================================================
# 🔥 GLOBAL MODEL CACHE (KEY CHANGE)
# ==========================================================
_WAN_MODEL_CACHE = {}


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    if "s2v" not in args.task:
        assert args.size in SUPPORTED_SIZES[args.task]


def _parse_args():
    parser = argparse.ArgumentParser("WAN Cached Generator")

    parser.add_argument("--task", type=str, default="t2v-A14B")
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)

    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")

    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--sample_solver", type=str, default="unipc")
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--convert_model_dtype", action="store_true")

    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None)

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _get_cached_model(args, cfg, device):
    """
    🔥 Core caching logic
    """
    cache_key = (
        args.task,
        args.ckpt_dir,
        device,
        args.t5_fsdp,
        args.dit_fsdp,
        args.ulysses_size,
        args.t5_cpu,
        args.convert_model_dtype,
    )

    if cache_key in _WAN_MODEL_CACHE:
        logging.info("🔥 Reusing cached WAN model")
        return _WAN_MODEL_CACHE[cache_key]

    logging.info("🚀 Loading WAN model (first time)")

    if "t2v" in args.task:
        model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=0,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
    else:
        raise NotImplementedError("Only T2V cached in this version")

    _WAN_MODEL_CACHE[cache_key] = model
    return model


def generate(args):
    _init_logging()

    device = 0
    torch.cuda.set_device(device)

    cfg = WAN_CONFIGS[args.task]

    logging.info(f"Task: {args.task}")
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Frames: {args.frame_num}")

    img = None
    if args.image:
        img = Image.open(args.image).convert("RGB")

    model = _get_cached_model(args, cfg, device)

    logging.info("🎬 Generating video ...")

    video = model.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )

    if args.save_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"{args.task}_{args.size}_{ts}.mp4"

    logging.info(f"💾 Saving video → {args.save_file}")

    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    torch.cuda.synchronize()
    logging.info("✅ Finished (model cached, process alive)")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
