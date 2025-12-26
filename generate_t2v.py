#!/usr/bin/env python3
"""
CLI Wrapper for WAN 2.2 T2V (Manual Test)
========================================
"""

import argparse
from wan.pipelines.t2v_pipeline import T2VPipeline


def main():
    parser = argparse.ArgumentParser("WAN 2.2 T2V Generator")

    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt or JSON string",
    )

    parser.add_argument(
        "--target_duration",
        type=int,
        required=True,
        help="Target duration in seconds",
    )

    parser.add_argument(
        "--size",
        default=None,
        help="Resolution, e.g. 832*480",
    )

    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Diffusion steps (optional)",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path",
    )

    args = parser.parse_args()

    output = T2VPipeline.generate(
        prompt=args.prompt,
        target_duration=args.target_duration,
        size=args.size,
        sample_steps=args.sample_steps,
        output_path=args.output,
    )

    print("\n=== GENERATION COMPLETE ===")
    print("Output:", output)


if __name__ == "__main__":
    main()
