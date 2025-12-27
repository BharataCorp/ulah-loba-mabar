import os
import json
import time
import logging
import psutil
import subprocess
from datetime import datetime
from pathlib import Path

# =====================================================
# BASIC LOGGER (NO WAN DEPENDENCY)
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="[DAEMON] %(message)s",
)
log = logging.info

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).parent
JOB_DIR = BASE_DIR / "jobs"
STATUS_DIR = BASE_DIR / "job_status"
HEALTH_FILE = BASE_DIR / "health.json"
STOP_FILE = BASE_DIR / "STOP"

JOB_DIR.mkdir(exist_ok=True)
STATUS_DIR.mkdir(exist_ok=True)

# =====================================================
# CONFIG
# =====================================================
RAM_LIMIT_GB = 120.0
CHECK_INTERVAL = 3

# =====================================================
# HEALTH
# =====================================================
def write_health(state, ready):
    mem = psutil.virtual_memory()
    health = {
        "state": state,
        "ready": ready,
        "ram_gb": round(mem.used / 1024**3, 2),
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(HEALTH_FILE, "w") as f:
        json.dump(health, f, indent=2)

# =====================================================
# JOB STATUS
# =====================================================
def write_status(job_id, payload):
    path = STATUS_DIR / f"{job_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# =====================================================
# MAIN LOOP
# =====================================================
def main():
    log("Starting WAN daemon worker")
    write_health("idle", True)

    while True:
        if STOP_FILE.exists():
            log("STOP signal detected → graceful shutdown")
            write_health("stopped", False)
            break

        # RAM CHECK
        mem = psutil.virtual_memory()
        ram_gb = mem.used / 1024**3
        if ram_gb > RAM_LIMIT_GB:
            log(f"RAM {ram_gb:.1f}GB > limit → draining")
            write_health("draining", False)
            time.sleep(CHECK_INTERVAL)
            continue

        write_health("idle", True)

        jobs = sorted(JOB_DIR.glob("*.json"))
        if not jobs:
            time.sleep(CHECK_INTERVAL)
            continue

        job_file = jobs[0]
        job_id = job_file.stem
        log(f"Picked job {job_id}")

        with open(job_file) as f:
            job = json.load(f)

        write_health("running", False)
        write_status(job_id, {
            "state": "running",
            "progress": 0,
            "started_at": datetime.utcnow().isoformat(),
        })

        try:
            cmd = [
                "python3", "generate.py",
                "--task", "t2v-A14B",
                "--ckpt_dir", "Wan2.2-T2V-A14B",
                "--prompt", job["prompt"],
                "--size", job["size"],
                "--frame_num", str(job["frame_num"]),
                "--sample_steps", "4",
                "--sample_shift", "10",
                "--save_file", job["output"],
            ]

            log("Executing generate.py")
            subprocess.run(cmd, check=True)

            write_status(job_id, {
                "state": "done",
                "progress": 100,
                "finished_at": datetime.utcnow().isoformat(),
            })
            log(f"Job {job_id} finished")

        except Exception as e:
            write_status(job_id, {
                "state": "failed",
                "error": str(e),
            })
            log(f"Job {job_id} failed: {e}")

        job_file.unlink(missing_ok=True)
        write_health("idle", True)

if __name__ == "__main__":
    main()
