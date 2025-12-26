#!/usr/bin/env python3
"""
Manual CLI runner for WAN 2.2 T2V Pipeline
==========================================

Example:
python3 generate_t2v.py \
  --prompt "A cinematic cat dancing on stage" \
  --target_duration 5 \
  --size 832*480
"""

import argparse
import os

from wan.pipelines.t2v_pipeline import T2VPipeline


def main():
    parser = argparse.ArgumentParser(
        description="WAN 2.2 Text-to-Video Generator (CLI)"
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt or JSON string"
    )

    parser.add_argument(
        "--target_duration",
        type=int,
        required=True,
        help="Target video duration in seconds"
    )

    parser.add_argument(
        "--size",
        default=None,
        help="Resolution, e.g. 832*480"
    )

    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Diffusion steps (optional)"
    )

    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Motion shift (optional)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path (optional)"
    )

    args = parser.parse_args()

    output_path = T2VPipeline.generate(
        prompt=args.prompt,
        target_duration=args.target_duration,
        size=args.size,
        sample_steps=args.sample_steps,
        output_path=args.output,
    )

    print("\n✅ Video generated successfully")
    print("📁 Output:", output_path)


if __name__ == "__main__":
    main()
