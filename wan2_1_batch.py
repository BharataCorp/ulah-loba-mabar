#!/usr/bin/env python3
import os, json, time, base64, subprocess, shlex, sys
import boto3, requests
from datetime import datetime

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
try:
    prompts = json.loads(base64.b64decode(prompts_b64).decode("utf-8"))
except Exception as e:
    print("[ERROR] Failed to decode PROMPTS_B64:", e)
    prompts = []

# ============ SAFE TRUNCATE PROMPT (MAX 1800 CHAR) ============
def truncate_prompt(text, limit=1800):
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"

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

# ========================== GPU CHECK ==========================
def check_gpu():
    send_callback("from_server_generate", {
        "title": "Checking GPU",
        "content": "Memeriksa GPU...",
        "data": {"status": "CHECKING_GPU"}
    })

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"[INFO] GPU Detected: {name} ({vram}GB)")
        else:
            print("[WARN] NO GPU FOUND")
    except Exception as e:
        print("[ERROR] GPU check failed:", e)

check_gpu()

# ========================== DOWNLOAD MODELS ==========================
def download_models_if_needed():
    config_path = os.path.join(ckpt_dir, "config.json")

    if os.path.exists(config_path):
        print(f"[INFO] Using existing model: {ckpt_dir}")
        return True

    print("[INFO] Models not found → Download from model S3...")

    try:
        model_s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{model_s3_endpoint}",
            aws_access_key_id=model_s3_access_key,
            aws_secret_access_key=model_s3_secret_key
        )

        model_root = "models/Wan2.1-T2V-1.3B"
        files = [
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ]

        os.makedirs(ckpt_dir, exist_ok=True)

        for f in files:
            key = f"{model_root}/{f}"
            dst = os.path.join(ckpt_dir, f)

            print(f"[DOWNLOAD] {f}...")
            model_s3.download_file(model_s3_bucket, key, dst)
            print(f"[OK] {f}")

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
            cmd = (
                f"python3 generate.py "
                f"--task {wan_task} "
                f"--size {wan_size} "
                f"--ckpt_dir {ckpt_dir} "
                f"--prompt {shlex.quote(safe_prompt)} "
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
            continue

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

        except Exception as e:
            print("[ERROR] Upload failed:", e)
            send_callback("from_server_generate", {
                "title": "Upload Failed",
                "content": "Gagal upload",
                "data": {
                    "status": "FAILED",
                    "order_index": idx,
                    "failed_reason": str(e)
                }
            })

        time.sleep(1)

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
