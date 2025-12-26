# submit_job.py
import argparse
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = "/workspace/Wan2.2"
QUEUE_DIR = f"{BASE_DIR}/job_queue/pending"

def main():
    parser = argparse.ArgumentParser("Submit WAN T2V Job")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target_duration", type=int, required=True)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    fps = 16
    frame_num = (args.target_duration * fps // 4) * 4 + 1

    job = {
        "prompt": args.prompt,
        "size": args.size,
        "frame_num": frame_num,
        "sample_steps": 25,
        "output": args.output,
        "ckpt_dir": f"{BASE_DIR}/Wan2.2-T2V-A14B",
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file = Path(QUEUE_DIR) / f"job_{ts}.json"

    with open(job_file, "w") as f:
        json.dump(job, f, indent=2)

    print(f"Job submitted: {job_file.name}")

if __name__ == "__main__":
    main()
