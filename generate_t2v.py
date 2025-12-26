# generate_t2v.py
from __future__ import annotations

import argparse
from wan.pipelines.t2v_pipeline import T2VPipeline


def main():
    parser = argparse.ArgumentParser(
        description="WAN 2.2 Text-to-Video Generator (Optimized)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation",
    )

    parser.add_argument(
        "--target_duration",
        type=int,
        required=True,
        help="Target duration in seconds",
    )

    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Video resolution (e.g. 832*480)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path",
    )

    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Sampling steps override",
    )

    args = parser.parse_args()

    out = T2VPipeline.generate(
        prompt=args.prompt,
        target_duration=args.target_duration,
        size=args.size,
        sample_steps=args.sample_steps,
        output_path=args.output,
    )

    print(f"\n=== T2V GENERATION COMPLETE ===")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
