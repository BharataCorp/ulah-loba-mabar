#!/usr/bin/env python3
import os, json, time, base64, subprocess, shlex, sys
import boto3, requests
from datetime import datetime

# set message starting generate wan2_1 batch to log file to /var/log/wan2_1_batch.log to notice running system in background
# ================= LOGGING =================
LOG_PATH = "/var/log/wan2_1_batch.log"

def write_log(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception as e:
        # kalau log bermasalah, tetap print ke stdout
        print(f"ERROR LOGGING: {e} → message: {msg}")

write_log("=== START wan2_1_batch process ===")

# ========================== READ ENV ==========================
generate_number = os.environ.get("GENERATE_NUMBER", "gv_unknown")
target_duration = int(os.environ.get("TARGET_DURATION", "10"))
upload_base_path = os.environ.get(
    "UPLOAD_BASE_PATH",
    f"video/{datetime.now().strftime('%Y/%m/%d')}/unknown"
)

s3_endpoint = os.environ.get("S3_ENDPOINT", "")
s3_bucket = os.environ.get("S3_BUCKET", "")
s3_access_key = os.environ.get("S3_ACCESS_KEY", "")
s3_secret_key = os.environ.get("S3_SECRET_KEY", "")
public_base_url = os.environ.get("PUBLIC_BASE_URL", "")

model_s3_endpoint = os.environ.get("MODEL_S3_ENDPOINT", "")
model_s3_bucket = os.environ.get("MODEL_S3_BUCKET", "")
model_s3_access_key = os.environ.get("MODEL_S3_ACCESS_KEY", "")
model_s3_secret_key = os.environ.get("MODEL_S3_SECRET_KEY", "")

callback_url = os.environ.get("CALLBACK_URL", "")
callback_api_key = os.environ.get("CALLBACK_API_KEY", "")

project_dir = os.environ.get("PROJECT_DIR", "/root")
wan_task = os.environ.get("WAN_TASK", "t2v-1.3B")
wan_size = os.environ.get("WAN_SIZE", "832*480")
ckpt_dir = os.environ.get("CKPT_DIR", "/models/Wan2.1-T2V-1.3B")

prompts_b64 = os.environ.get("PROMPTS_B64", "W10=")

# ========================== HELPERS ==========================
def send_callback(endpoint, payload, method="POST"):
    if not callback_url:
        return
    url = f"{callback_url}/{endpoint}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if callback_api_key:
        headers["key"] = callback_api_key

    try:
        if method.upper() == "POST":
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            print(f"[CALLBACK POST:{endpoint}] {r.status_code} {r.text}")
        elif method.upper() == "PUT":
            r = requests.put(url, json=payload, headers=headers, timeout=30)
            print(f"[CALLBACK PUT:{endpoint}] {r.status_code} {r.text}")
    except Exception as e:
        print(f"[ERROR] Callback {endpoint} failed:", e)


# send callback: starting
send_callback("from_server_generate", {
    "title": "Starting Batch",
    "content": "Memulai proses batch.",
    "data": {
        "status": "STARTING_BATCH",
        "generate_number": generate_number
    }
})

try:
    prompts = json.loads(base64.b64decode(prompts_b64).decode("utf-8"))

    send_callback("from_server_generate", {
        "title": "Prompts Loaded",
        "content": f"Loaded {len(prompts)} prompts.",
        "data": {
            "status": "PROMPTS_LOADED",
            "total_prompts": len(prompts),
            "detail_prompts_json": prompts
        }
    })

except Exception as e:
    print("[ERROR] Failed to decode PROMPTS_B64:", e)
    prompts = []

# ============ SAFE TRUNCATE PROMPT (MAX 1800 CHAR) ============
def truncate_prompt(text, limit=1800):
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"

# check if empty prompts or not a list
if not isinstance(prompts, list) or len(prompts) == 0:
    print("[ERROR] No prompts provided.")
    send_callback("from_server_generate", {
        "title": "No Prompts",
        "content": "Tidak ada prompt yang diberikan.",
        "data": {"status": "FAILED"}
    })
    sys.exit(1)

# ========================== INIT S3 CLIENT ==========================
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{s3_endpoint}",
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
)

date_path = datetime.now().strftime("%Y/%m/%d")
if not upload_base_path:
    upload_base_path = f"video/{date_path}/{generate_number}"

video_urls = []


def ensure_duration(in_path, out_path, target_sec):
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", in_path],
            capture_output=True, text=True
        )
        orig = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
    except:
        orig = 0.0

    if orig <= 0.1:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-t", str(target_sec),
             "-c", "copy", out_path],
            check=False
        )
        return

    if orig > target_sec + 0.1:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-t", str(target_sec), "-c", "copy", out_path],
            check=False
        )
    elif orig < target_sec - 0.1:
        loop_count = max(1, int(target_sec // max(orig, 1)))
        subprocess.run(
            ["ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", in_path,
             "-t", str(target_sec), "-c", "libx264", "-pix_fmt", "yuv420p",
             out_path],
            check=False
        )
    else:
        subprocess.run(["cp", in_path, out_path], check=False)

try:
    import torch
    import numpy as np
except Exception as e:
    send_callback("from_server_generate", {
        "title": "Import Error",
        "content": "Gagal import torch dan numpy",
        "data": {
            "status": "FAILED",
            "failed_reason": str(e)
        }
    })
    sys.exit(1)

# ========================== GPU CHECK ==========================
def check_gpu():
    send_callback("from_server_generate", {
        "title": "Checking GPU",
        "content": "Memeriksa GPU...",
        "data": {"status": "CHECKING_GPU"}
    })

    try:
        import torch
        if not torch.cuda.is_available():
            msg = "GPU TIDAK TERDETEKSI"
            print(f"[FATAL] {msg}")
            send_callback("from_server_generate", {
                "title": "GPU Not Found",
                "content": msg,
                "data": {
                    "status": "FAILED",
                    "failed_reason": msg
                }
            })
            sys.exit(1)   # ← HENTIKAN
        else:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"[INFO] GPU Detected: {name} ({vram}GB)")

    except Exception as e:
        send_callback("from_server_generate", {
            "title": "GPU Check Failed",
            "content": str(e),
            "data": {
                "status": "FAILED",
                "failed_reason": str(e)
            }
        })
        sys.exit(1)   # STOP

check_gpu()

# ========================== DOWNLOAD MODELS ==========================
def download_models_if_needed():
    """
    Download seluruh folder model dari S3 (recursive),
    termasuk google/umt5-xxl dan assets, dll.
    """
    # Jika sudah ada minimal file wajib, skip
    required = [
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth"
    ]
    if all(os.path.exists(os.path.join(ckpt_dir, f)) for f in required):
        print(f"[INFO] Model folder already complete: {ckpt_dir}")
        return True

    print("[INFO] Downloading model recursively from S3...")
    try:
        model_s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{model_s3_endpoint}",
            aws_access_key_id=model_s3_access_key,
            aws_secret_access_key=model_s3_secret_key
        )

        prefix = "models/Wan2.1-T2V-1.3B/"
        paginator = model_s3.get_paginator("list_objects_v2")

        os.makedirs(ckpt_dir, exist_ok=True)

        for page in paginator.paginate(Bucket=model_s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key.replace(prefix, "")
                dst_path = os.path.join(ckpt_dir, rel_path)

                if key.endswith("/"):
                    continue

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                print(f"[DOWNLOAD] {rel_path} ...")
                model_s3.download_file(model_s3_bucket, key, dst_path)
                print(f"[OK] {rel_path}")

        return True

    except Exception as e:
        print("[ERROR] Failed download:", e)
        return False

if not download_models_if_needed():
    send_callback("from_server_generate", {
        "title": "Model Download Failed",
        "content": "Gagal download model.",
        "data": {"status": "FAILED"}
    })
    sys.exit(1)

# ========================== MAIN PIPELINE ==========================
try:
    send_callback("from_server_generate", {
        "title": "Starting Batch",
        "content": "Memulai proses generate.",
        "data": {
            "status": "STARTING_BATCH",
            "total_prompts": len(prompts)
        }
    })

    candidates = [
        os.path.join(project_dir, "ulah-loba-mabar", "Wan2.1"),
    ]
    generate_dir = next(
        (x for x in candidates if os.path.exists(os.path.join(x, "generate.py"))),
        None
    )
    if not generate_dir:
        raise FileNotFoundError("generate.py not found")

    # frame_num N4+1
    def calc_frame_num(sec, fps=16):
        raw = int(round(fps * sec))
        return 4 * round((raw - 1) / 4) + 1

    frame_num = calc_frame_num(target_duration)
    print(f"[INFO] frame_num={frame_num}")

    # LOOP
    has_succeeded = False
    for idx, raw_prompt in enumerate(prompts):

        safe_prompt = truncate_prompt(raw_prompt)

        send_callback("from_server_generate", {
            "title": "Start Generate",
            "content": f"Mulai generate index {idx}",
            "data": {
                "status": "STARTING_GENERATION",
                "order_index": idx,
                "prompt": safe_prompt
            }
        })

        tmp_out = f"/tmp/{generate_number}_{idx}.mp4"
        final_out = f"/tmp/{generate_number}_{idx}_final.mp4"

        try:
            prompt_file = f"/tmp/prompt_{idx}.txt"
            with open(prompt_file, "w", encoding="utf-8") as fp:
                fp.write(safe_prompt)

            cmd = (
                f"python3 generate.py "
                f"--task {wan_task} "
                f"--size {wan_size} "
                f"--ckpt_dir {ckpt_dir} "
                f"--prompt_file {prompt_file} "
                f"--frame_num {frame_num}"
            )

            subprocess.run(cmd, cwd=generate_dir, shell=True, check=True)

            produced = (
                tmp_out
                if os.path.exists(tmp_out)
                else os.path.join(generate_dir, "output.mp4")
            )

        except Exception as e:
            print("[ERROR] Generate failed:", e)
            send_callback("from_server_generate", {
                "title": "Generation Failed",
                "content": f"Gagal generate index {idx}",
                "data": {
                    "status": "FAILED",
                    "order_index": idx,
                    "failed_reason": str(e)
                }
            })
            sys.exit(1)

        ensure_duration(produced, final_out, target_duration)

        s3_key = f"{upload_base_path}/{idx}.mp4"

        try:
            s3.upload_file(
                final_out, s3_bucket, s3_key,
                ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"}
            )
            url = f"{public_base_url}/{s3_key}"
            video_urls.append(url)

            send_callback("from_server_generate", {
                "title": "Upload OK",
                "content": f"Upload OK: {idx}",
                "data": {
                    "status": "UPLOADED",
                    "order_index": idx,
                    "video_url": url
                }
            })
            has_succeeded = True

        except Exception as e:
            print("[ERROR] Upload failed:", e)
            send_callback("from_server_generate", {
                "title": "Upload Failed",
                "content": "Gagal upload",
                "data": {
                    "status": "GENERATE_FAILED",
                    "order_index": idx,
                    "failed_reason": str(e)
                }
            })

        time.sleep(1)

    if not has_succeeded:
        send_callback("from_server_generate", {
            "title": "Batch Failed",
            "content": "Semua generate gagal.",
            "data": {"status": "FAILED"}
        })
        sys.exit(1)
    else:
        send_callback("from_server_generate", {
            "title": "Completed",
            "content": "Batch selesai.",
            "data": {
                "status": "COMPLETED",
                "video_urls": video_urls
            }
        })

except Exception as e:
    print("[FATAL]", e)
    send_callback("from_server_generate", {
        "title": "Batch Failed",
        "content": "Fatal error.",
        "data": {
            "status": "FAILED",
            "failed_reason": str(e)
        }
    })
    sys.exit(1)

print("[DONE]")
