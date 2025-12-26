#!/usr/bin/env python3

import argparse
from wan.pipelines.t2v_pipeline import T2VPipeline


def main():
    parser = argparse.ArgumentParser("WAN 2.2 T2V CLI")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target_duration", type=int, required=True)
    parser.add_argument("--size", default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    output = T2VPipeline.generate(
        prompt=args.prompt,
        target_duration=args.target_duration,
        size=args.size,
        sample_steps=args.sample_steps,
        output_path=args.output,
    )

    print("\n✅ DONE")
    print("Output:", output)


if __name__ == "__main__":
    main()
