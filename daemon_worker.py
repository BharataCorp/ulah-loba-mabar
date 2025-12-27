import os
import json
import time
import logging
import psutil
from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_WAN_DIR = BASE_DIR / "wan_custom"

if str(CUSTOM_WAN_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_WAN_DIR.parent))

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
# IMPORT WAN PIPELINE (CRITICAL FIX)
# =====================================================
from wan_custom.pipelines.t2v_pipeline import T2VPipeline

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
            log("Executing T2V pipeline (subprocess WAN-native)")

            output_path = T2VPipeline.generate(
                prompt=job["prompt"],
                target_duration=job.get("duration", 5),
                size=job["size"],
                sample_steps=4,
                output_path=job["output"],
            )

            write_status(job_id, {
                "state": "done",
                "progress": 100,
                "output": output_path,
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
