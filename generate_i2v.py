# generate_i2v.py

import argparse
from wan_custom.pipelines.i2v_pipeline import I2VPipeline


def main():
    parser = argparse.ArgumentParser("WAN I2V Generator")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target_duration", type=int, required=True)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--sample_shift", type=int, default=10)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    output = I2VPipeline.generate(
        image_path=args.image,
        prompt=args.prompt,
        target_duration=args.target_duration,
        size=args.size,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        output_path=args.output,
    )

    print("✅ Video saved to:", output)


if __name__ == "__main__":
    main()
