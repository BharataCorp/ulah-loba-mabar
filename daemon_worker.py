import os
import json
import time
import signal
import sys
import gc
import psutil
import torch
from datetime import datetime

from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.logger import get_logger
from wan.utils.utils import save_video

# =========================================================
# CONFIG
# =========================================================
GPU_ID = 0
MAX_RAM_GB = 120          # auto-drain threshold
CHECK_INTERVAL = 2        # seconds

BASE_DIR = os.getcwd()
JOB_DIR = os.path.join(BASE_DIR, "jobs")
STATUS_DIR = os.path.join(BASE_DIR, "status")
HEALTH_FILE = os.path.join(BASE_DIR, "health.json")
CKPT_DIR = os.path.join(BASE_DIR, "Wan2.2-T2V-A14B")

os.makedirs(JOB_DIR, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)

LOGGER = get_logger("WAN.DAEMON")

# =========================================================
# GLOBAL STATE
# =========================================================
WAN_MODEL = None
STATE = "starting"       # starting | idle | running | draining | stopping
CURRENT_JOB = None
SHOULD_EXIT = False


# =========================================================
# HEALTH & STATUS
# =========================================================
def write_health():
    data = {
        "state": STATE,
        "ready": STATE in ("idle", "running"),
        "ram_gb": round(psutil.virtual_memory().used / 1024**3, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(HEALTH_FILE, "w") as f:
        json.dump(data, f, indent=2)


def write_job_status(job_id, payload):
    path = os.path.join(STATUS_DIR, f"{job_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# MODEL LOADING
# =========================================================
def load_model_once():
    global WAN_MODEL, STATE

    if WAN_MODEL is not None:
        return WAN_MODEL

    LOGGER.info("🚀 Loading WAN T2V model (ONCE)")
    STATE = "starting"
    write_health()

    torch.cuda.set_device(GPU_ID)
    cfg = WAN_CONFIGS["t2v-A14B"]

    WAN_MODEL = WanT2V(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=GPU_ID,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )

    STATE = "idle"
    write_health()
    LOGGER.info("✅ WAN model loaded & cached")
    return WAN_MODEL


# =========================================================
# RAM MONITOR
# =========================================================
def ram_exceeded():
    used_gb = psutil.virtual_memory().used / 1024**3
    return used_gb >= MAX_RAM_GB


# =========================================================
# CLEANUP
# =========================================================
def cleanup():
    global WAN_MODEL
    LOGGER.info("[DAEMON] Cleaning up model & CUDA")

    try:
        if WAN_MODEL is not None:
            del WAN_MODEL
            WAN_MODEL = None

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        LOGGER.error(f"[DAEMON] Cleanup error: {e}")


def graceful_exit(signum=None, frame=None):
    global SHOULD_EXIT, STATE
    LOGGER.info("[DAEMON] Shutdown signal received → graceful stop")
    SHOULD_EXIT = True
    STATE = "stopping"
    write_health()


# =========================================================
# JOB PROCESSING (WITH PROGRESS KASAR)
# =========================================================
def process_job(job_file):
    global STATE, CURRENT_JOB

    job_id = job_file.replace(".json", "")
    CURRENT_JOB = job_id

    job_path = os.path.join(JOB_DIR, job_file)
    with open(job_path) as f:
        job = json.load(f)

    # ------------------------------
    # 0% → job picked
    # ------------------------------
    write_job_status(job_id, {
        "status": "running",
        "progress": 0,
        "stage": "job_picked",
        "started_at": datetime.utcnow().isoformat(),
    })

    STATE = "running"
    write_health()

    # ------------------------------
    # 5% → model ready
    # ------------------------------
    model = load_model_once()
    write_job_status(job_id, {
        "status": "running",
        "progress": 5,
        "stage": "model_ready",
    })

    # ------------------------------
    # 10% → generation start
    # ------------------------------
    write_job_status(job_id, {
        "status": "running",
        "progress": 10,
        "stage": "generating",
    })

    video = model.generate(
        job["prompt"],
        size=SIZE_CONFIGS[job["size"]],
        frame_num=job["frame_num"],
        shift=10,
        sampling_steps=job.get("sample_steps", 4),
        guide_scale=1.0,
        seed=job.get("seed", 42),
        offload_model=False,
    )

    # ------------------------------
    # 85% → sampling finished
    # ------------------------------
    write_job_status(job_id, {
        "status": "running",
        "progress": 85,
        "stage": "sampling_done",
    })

    # ------------------------------
    # 90% → saving
    # ------------------------------
    write_job_status(job_id, {
        "status": "running",
        "progress": 90,
        "stage": "saving_video",
    })

    save_video(
        tensor=video[None],
        save_file=job["output"],
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    # ------------------------------
    # 100% → done
    # ------------------------------
    write_job_status(job_id, {
        "status": "done",
        "progress": 100,
        "stage": "finished",
        "output": job["output"],
        "finished_at": datetime.utcnow().isoformat(),
    })

    os.remove(job_path)
    CURRENT_JOB = None

    if STATE != "draining":
        STATE = "idle"

    write_health()
    LOGGER.info(f"[DAEMON] Job {job_id} completed")


# =========================================================
# MAIN LOOP
# =========================================================
def main():
    global STATE

    LOGGER.info("[DAEMON] Starting WAN daemon worker")

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    load_model_once()

    while True:
        if SHOULD_EXIT:
            if CURRENT_JOB is None:
                LOGGER.info("[DAEMON] No active job → exiting")
                break
            time.sleep(1)
            continue

        if ram_exceeded() and STATE not in ("draining", "stopping"):
            LOGGER.warning("[DAEMON] RAM threshold exceeded → draining mode")
            STATE = "draining"
            write_health()

        if STATE in ("idle", "running"):
            jobs = [f for f in os.listdir(JOB_DIR) if f.endswith(".json")]
            if jobs:
                process_job(jobs[0])

        time.sleep(CHECK_INTERVAL)

    cleanup()
    STATE = "stopped"
    write_health()
    LOGGER.info("[DAEMON] Worker exited cleanly")


if __name__ == "__main__":
    main()
