import os
import json
import time
import logging
import psutil
import signal
import sys
from datetime import datetime
from pathlib import Path

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
CUSTOM_WAN_DIR = BASE_DIR / "wan_custom"

if str(CUSTOM_WAN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CUSTOM_WAN_DIR.parent))

# =====================================================
# LOGGER
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="[DAEMON] %(message)s",
)
log = logging.info

# =====================================================
# PATHS
# =====================================================
JOB_DIR = BASE_DIR / "jobs"
STATUS_DIR = BASE_DIR / "job_status"
HEALTH_FILE = BASE_DIR / "health.json"

JOB_DIR.mkdir(exist_ok=True)
STATUS_DIR.mkdir(exist_ok=True)

# =====================================================
# CONFIG
# =====================================================
CHECK_INTERVAL = 3
MIN_AVAILABLE_GB = 20
IDLE_CLEANUP_INTERVAL = 60  # seconds

# =====================================================
# IMPORT PIPELINE
# =====================================================
from wan_custom.pipelines.t2v_pipeline import T2VPipeline

# =====================================================
# HELPERS
# =====================================================
def available_ram_gb():
    return psutil.virtual_memory().available / 1024**3

def write_health(state, ready):
    mem = psutil.virtual_memory()
    health = {
        "state": state,
        "ready": ready,
        "ram_used_gb": round(mem.used / 1024**3, 2),
        "ram_available_gb": round(mem.available / 1024**3, 2),
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(HEALTH_FILE, "w") as f:
        json.dump(health, f, indent=2)

def write_status(job_id, payload):
    path = STATUS_DIR / f"{job_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def idle_cache_cleanup():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

def drop_page_cache_safe():
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1\n")  # page cache only
    except Exception:
        pass

# =====================================================
# SIGNAL HANDLING (NO FORK)
# =====================================================
def handle_exit(signum, frame):
    log(f"Received signal {signum}, shutting down safely")
    write_health("stopped", False)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

# =====================================================
# MAIN LOOP
# =====================================================
def main():
    log("Starting WAN daemon worker")
    write_health("idle", True)

    last_idle_cleanup = 0

    while True:
        # -------------------------------------------------
        # RAM GUARD (CRITICAL)
        # -------------------------------------------------
        if available_ram_gb() < MIN_AVAILABLE_GB:
            log("Low available RAM → draining & cleanup")
            write_health("draining", False)
            idle_cache_cleanup()
            drop_page_cache_safe()
            time.sleep(10)
            continue

        jobs = sorted(JOB_DIR.glob("*.json"))

        # -------------------------------------------------
        # IDLE MODE
        # -------------------------------------------------
        if not jobs:
            write_health("idle", True)

            now = time.time()
            if now - last_idle_cleanup > IDLE_CLEANUP_INTERVAL:
                log("Idle cleanup")
                idle_cache_cleanup()
                drop_page_cache_safe()
                last_idle_cleanup = now

            time.sleep(CHECK_INTERVAL)
            continue

        # -------------------------------------------------
        # RUN JOB
        # -------------------------------------------------
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
            output_path = T2VPipeline.generate(
                prompt=job["prompt"],
                target_duration=job.get("duration", 5),
                size=job["size"],
                sample_steps=job.get("sample_steps", 16),
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

        finally:
            try:
                import torch, gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log("Post-job CUDA cleanup done")
            except Exception:
                pass

        job_file.unlink(missing_ok=True)
        write_health("idle", True)

if __name__ == "__main__":
    main()
