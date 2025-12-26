# daemon_worker.py
import os
import time
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from generate import generate

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


def build_args_from_job(job: dict):
    """
    Build args object manually (NO argparse)
    """
    return SimpleNamespace(
        # REQUIRED
        task="t2v-A14B",
        ckpt_dir=job["ckpt_dir"],
        prompt=job["prompt"],
        size=job["size"],
        frame_num=job["frame_num"],
        save_file=job["output"],

        # SAMPLING
        sample_steps=job.get("sample_steps", 25),
        sample_shift=10,
        sample_solver="unipc",
        sample_guide_scale=None,

        # MODEL FLAGS
        offload_model=True,
        convert_model_dtype=True,
        t5_cpu=True,
        t5_fsdp=False,
        dit_fsdp=False,
        ulysses_size=1,

        # OTHER
        base_seed=-1,
        image=None,
    )


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

            args = build_args_from_job(job)

            logging.info("Generating video...")
            generate(args)

            done_path = f"{DONE}/{job_name}"
            os.rename(running_path, done_path)
            logging.info(f"Job done → {job['output']}")

        except Exception as e:
            logging.exception("Job failed")
            os.rename(running_path, f"{DONE}/{job_name}.failed")


if __name__ == "__main__":
    main()
