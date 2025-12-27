# generate_t2v.py

import argparse
from wan_custom.pipelines.t2v_pipeline import T2VPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target_duration", type=int, required=True)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--sample_shift", type=int, default=10)

    args = parser.parse_args()

    out = T2VPipeline.generate(
        prompt=args.prompt,
        target_duration=args.target_duration,
        size=args.size,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        output_path=args.output,
    )

    print("Video saved to:", out)


if __name__ == "__main__":
    main()
