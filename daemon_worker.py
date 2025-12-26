# daemon_worker.py
import os
import time
import json
import logging
from pathlib import Path

from generate import generate, _parse_args

BASE_DIR = "/workspace/Wan2.2"
QUEUE_DIR = f"{BASE_DIR}/job_queue"
PENDING = f"{QUEUE_DIR}/pending"
RUNNING = f"{QUEUE_DIR}/running"
DONE = f"{QUEUE_DIR}/done"

os.makedirs(PENDING, exist_ok=True)
os.makedirs(RUNNING, exist_ok=True)
os.makedirs(DONE, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[DAEMON] %(message)s"
)

def load_job(path):
    with open(path) as f:
        return json.load(f)

def main():
    logging.info("Starting WAN daemon worker")
    logging.info("Waiting for jobs...")

    while True:
        jobs = sorted(Path(PENDING).glob("*.json"))

        if not jobs:
            time.sleep(2)
            continue

        job_file = jobs[0]
        job_name = job_file.name
        running_path = f"{RUNNING}/{job_name}"

        os.rename(job_file, running_path)
        logging.info(f"Picked job {job_name}")

        try:
            job = load_job(running_path)

            args = _parse_args()
            args.task = "t2v-A14B"
            args.ckpt_dir = job["ckpt_dir"]
            args.prompt = job["prompt"]
            args.size = job["size"]
            args.frame_num = job["frame_num"]
            args.sample_steps = job["sample_steps"]
            args.sample_shift = 10
            args.save_file = job["output"]
            args.offload_model = True
            args.convert_model_dtype = True
            args.t5_cpu = True

            generate(args)

            done_path = f"{DONE}/{job_name}"
            os.rename(running_path, done_path)
            logging.info(f"Job done → {job['output']}")

        except Exception as e:
            logging.error(f"Job failed: {e}")
            os.rename(running_path, f"{DONE}/{job_name}.failed")

if __name__ == "__main__":
    main()
