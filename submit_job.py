import json
import argparse
import os
import sys
from datetime import datetime

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.getcwd()
JOB_DIR = os.path.join(BASE_DIR, "jobs")
HEALTH_FILE = os.path.join(BASE_DIR, "health.json")

os.makedirs(JOB_DIR, exist_ok=True)

# =========================================================
# WORKER HEALTH CHECK
# =========================================================
def check_worker_ready():
    if not os.path.exists(HEALTH_FILE):
        print("❌ Worker health not found. Is daemon_worker.py running?")
        sys.exit(1)

    try:
        with open(HEALTH_FILE) as f:
            health = json.load(f)
    except Exception:
        print("❌ Failed to read health.json (corrupted?)")
        sys.exit(1)

    if not health.get("ready", False):
        state = health.get("state", "unknown")
        ram = health.get("ram_gb", "?")
        print(
            f"❌ Worker not ready (state={state}, ram={ram}GB). "
            "Job rejected."
        )
        sys.exit(1)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Submit WAN T2V job (with drain protection)"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target_duration", type=int, required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    # -----------------------------------------------------
    # 🔥 REJECT SUBMIT IF WORKER DRAINING / NOT READY
    # -----------------------------------------------------
    check_worker_ready()

    # -----------------------------------------------------
    # FRAME CALCULATION (WAN RULE: 4n + 1)
    # -----------------------------------------------------
    fps = 16
    raw_frames = args.target_duration * fps
    frame_num = (raw_frames // 4) * 4 + 1

    # -----------------------------------------------------
    # JOB PAYLOAD
    # -----------------------------------------------------
    job = {
        "prompt": args.prompt,
        "frame_num": frame_num,
        "size": args.size,
        "output": args.output,
        "sample_steps": 16,
        "seed": 42,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    job_id = datetime.utcnow().strftime("job_%Y%m%d_%H%M%S")
    job_path = os.path.join(JOB_DIR, f"{job_id}.json")

    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)

    print(f"✅ Job submitted successfully: {job_id}")
    print(f"   Frames : {frame_num}")
    print(f"   Output : {args.output}")


if __name__ == "__main__":
    main()
