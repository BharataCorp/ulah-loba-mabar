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

callback_url = os.environ.get("CALLBACK_URL", "")
callback_api_key = os.environ.get("CALLBACK_API_KEY", "")

# Model S3 configuration (for downloading models if needed)
model_s3_endpoint = os.environ.get("MODEL_S3_ENDPOINT", "sgp1.vultrobjects.com")
model_s3_bucket = os.environ.get("MODEL_S3_BUCKET", "mabar-app")
model_s3_access_key = os.environ.get("MODEL_S3_ACCESS_KEY", s3_access_key)
model_s3_secret_key = os.environ.get("MODEL_S3_SECRET_KEY", s3_secret_key)

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

# ========================== INIT S3 ==========================
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
def send_callback(endpoint, payload):
    """Kirim callback ke API"""
    if not callback_url:
        return
    url = f"{callback_url}/{endpoint}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    if callback_api_key:
        headers["key"] = callback_api_key

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"[CALLBACK:{endpoint}] {r.status_code} {r.text}")
    except Exception as e:
        print(f"[ERROR] Callback {endpoint} failed:", e)


def ensure_duration(in_path, out_path, target_sec):
    """Pastikan durasi video sesuai target"""
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
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-t", str(target_sec), "-c", "copy", out_path], check=False)
        return

    if orig > target_sec + 0.1:
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-t", str(target_sec), "-c", "copy", out_path], check=False)
    elif orig < target_sec - 0.1:
        loop_count = max(1, int(target_sec // max(orig, 1)))
        subprocess.run(
            ["ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", in_path,
             "-t", str(target_sec), "-c", "libx264", "-pix_fmt", "yuv420p", out_path],
            check=False
        )
    else:
        subprocess.run(["cp", in_path, out_path], check=False)


# ========================== GPU CHECK ==========================
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
            name = torch.cuda.get_device_name(0)
            print(f"[INFO] Detected GPU: {name} ({vram} GB VRAM)")
            if vram < 8:
                print("[WARN] VRAM terlalu kecil (<8GB). flash_attn akan di-skip. "
                      "Gunakan requirements-lite.txt.")
            elif vram < 30:
                print("[WARN] GPU terdeteksi menengah (8–30GB). Bisa jalan tapi mungkin lambat.")
            else:
                print("[INFO] GPU besar terdeteksi (≥30GB VRAM). Disarankan install full requirements "
                      "dengan flash_attn.")
        else:
            print("[WARN] Tidak ada GPU terdeteksi, akan fallback ke CPU (sangat lambat).")
    except Exception as e:
        print("[ERROR] Gagal cek GPU:", e)

check_gpu()

# ========================== CHECK & DOWNLOAD MODELS ==========================
def download_models_if_needed():
    """Download models dari S3 Singapore jika belum ada di network volume"""
    # Check if models exist
    config_path = os.path.join(ckpt_dir, "config.json")
    
    if os.path.exists(config_path):
        print(f"[INFO] Models found in {ckpt_dir}, using cached version")
        return True
    
    print(f"[INFO] Models not found in {ckpt_dir}, downloading from S3...")
    
    try:
        # Initialize S3 client for model download
        s3_models = boto3.client('s3',
            endpoint_url=f'https://{model_s3_endpoint}',
            aws_access_key_id=model_s3_access_key,
            aws_secret_access_key=model_s3_secret_key
        )
        
        # Determine which model to download
        if 'I2V' in ckpt_dir or 'i2v' in wan_task.lower():
            model_path = 'models/Wan2.1-I2V-14B-480P'
            files = [
                'config.json',
                'diffusion_pytorch_model.safetensors',
                'Wan2.1_VAE.pth',
            ]
        else:
            model_path = 'models/Wan2.1-T2V-1.3B'
            files = [
                'config.json',
                'diffusion_pytorch_model.safetensors',
                'Wan2.1_VAE.pth',
                'models_t5_umt5-xxl-enc-bf16.pth',
            ]
        
        # Create directory
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Download each file
        for f in files:
            s3_key = f'{model_path}/{f}'
            local_path = os.path.join(ckpt_dir, f)
            
            print(f"[DOWNLOAD] {f} from S3 Singapore...")
            s3_models.download_file(model_s3_bucket, s3_key, local_path)
            
            size = os.path.getsize(local_path) / (1024*1024*1024)
            print(f"[OK] {f} ({size:.2f} GB)")
        
        print(f"[INFO] Models downloaded successfully to {ckpt_dir}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download models: {e}")
        return False

# Download models if needed (from network volume or S3)
if not download_models_if_needed():
    print("[FATAL] Cannot proceed without models")
    sys.exit(1)

# ========================== MAIN ==========================
try:
    send_callback("process", {
        "status": "PREPARING_PROCESSING",
        "generate_number": generate_number,
        "total_prompts": len(prompts),
        "upload_base_path": upload_base_path,
        "target_duration": target_duration,
        "wan_task": wan_task,
        "wan_size": wan_size,
        "ckpt_dir": ckpt_dir,
        "project_dir": project_dir,
        "public_base_url": public_base_url,
        "s3_bucket": s3_bucket,
        "date_path": date_path,
    })

    # --- Cari lokasi generate.py ---
    candidates = [
        os.path.join(project_dir, "Wan2.1"),                  # official clone
        os.path.join(project_dir, "wan2.1-runner", "Wan2.1"), # repo kamu
    ]
    generate_dir = next((c for c in candidates if os.path.exists(os.path.join(c, "generate.py"))), None)
    if not generate_dir:
        raise FileNotFoundError("generate.py not found in expected locations.")
    print(f"[INFO] Using generate.py from {generate_dir}")


    # ========================== FRAME_NUM CALC ==========================
    def calc_frame_num(target_sec, fps=16):
        """Hitung frame_num sesuai target detik (harus 4n+1)."""
        raw = int(round(fps * target_sec))
        return 4 * round((raw - 1) / 4) + 1


    # contoh kalkulasi global
    default_fps = 16  # T2V config biasanya 16 fps
    frame_num = calc_frame_num(target_duration, default_fps)
    print(f"[INFO] Auto-calculated frame_num={frame_num} (~{frame_num / default_fps:.2f} sec)")

    for idx, prompt in enumerate(prompts):
        print(f"[INFO] ({idx+1}/{len(prompts)}) Generating: {prompt}")
        tmp_out = f"/tmp/{generate_number}_{idx}.mp4"
        final_out = f"/tmp/{generate_number}_{idx}_final.mp4"
        video_url = None

        try:
            # --- GENERATE VIDEO ---
            cmd = (
                f"python3 generate.py "
                f"--task {wan_task} --size {wan_size} "
                f"--ckpt_dir {shlex.quote(ckpt_dir)} "
                f"--prompt {shlex.quote(prompt)} "
                f"--frame_num {frame_num}"
            )

            subprocess.run(cmd, cwd=generate_dir, shell=True, check=True)
            produced = os.path.join("/tmp", f"{generate_number}_{idx}.mp4")
            if not os.path.exists(produced):
                produced = os.path.join(generate_dir, "output.mp4")
        except Exception as e:
            print("[ERROR] generate.py failed:", e)
            send_callback("fail", {
                "status": "FAILED",
                "type_error": "GENERATE_FAILED",
                "failed_reason": str(e),
                "current_index": idx,
                "current_prompt": prompt,
                "video_urls": video_urls
            })
            produced = tmp_out
            open(produced, "wb").write(b"DUMMY")

        # --- DUMMY-SAFE FIX ---
        if os.environ.get("WAN_DUMMY") == "1":
            produced = f"/tmp/{generate_number}_{idx}.mp4"
            if not os.path.exists(produced):
                # fallback: bikin dummy kecil biar pasti ada file
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=256x256:rate=8",
                     "-t", str(target_duration), "-pix_fmt", "yuv420p", produced],
                    check=False
                )
            print(f"[DUMMY] Ensured dummy file exists at {produced}")

        # --- PAKSA DURASI ---
        ensure_duration(produced, final_out, target_duration)

        # --- UPLOAD ---
        s3_key = f"{upload_base_path}/{idx}.mp4"
        try:
            s3.upload_file(
                final_out,
                s3_bucket,
                s3_key,
                ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"}
            )
            video_url = f"{public_base_url}/{s3_key}"
            video_urls.append(video_url)
            print(f"[INFO] Uploaded (public): {video_url}")

            send_callback("upload", {
                "status": "SUCCESS",
                "current_index": idx,
                "video_url": video_url,
                "video_urls": video_urls,
            })
        except Exception as e:
            print("[ERROR] Upload failed:", e)
            send_callback("upload", {
                "status": "FAILED",
                "failed_reason": str(e),
                "current_index": idx,
                "video_url": "",
                "video_urls": video_urls,
            })

        # --- PROGRESS ---
        send_callback("progress", {
            "status": "GENERATING",
            "current_index": idx,
            "current_prompt": prompt,
            "current_url": video_url,
            "video_urls": video_urls,
        })

        time.sleep(3)

    # # ========================== SEGMENT MERGE ==========================
    # # Jika hasil tetap pendek (misal < target_duration - 1), gabungkan segmen
    # if len(video_urls) > 1 or target_duration > 10:
    #     try:
    #         list_path = f"/tmp/{generate_number}_concat.txt"
    #         with open(list_path, "w") as f:
    #             for idx in range(len(prompts)):
    #                 final_out = f"/tmp/{generate_number}_{idx}_final.mp4"
    #                 f.write(f"file '{final_out}'\n")
    #
    #         concat_out = f"/tmp/{generate_number}_concat_final.mp4"
    #         subprocess.run(
    #             ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
    #              "-i", list_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", concat_out],
    #             check=False
    #         )
    #
    #         # Upload concat juga
    #         s3_key = f"{upload_base_path}/concat_final.mp4"
    #         s3.upload_file(
    #             concat_out,
    #             s3_bucket,
    #             s3_key,
    #             ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"}
    #         )
    #         concat_url = f"{public_base_url}/{s3_key}"
    #         video_urls.append(concat_url)
    #         print(f"[INFO] Uploaded concatenated video: {concat_url}")
    #         send_callback("upload", {
    #             "status": "SUCCESS",
    #             "concat_url": concat_url,
    #             "video_urls": video_urls
    #         })
    #     except Exception as e:
    #         print("[WARN] Concat step failed:", e)

    # --- SUCCESS ---
    send_callback("success", {
        "status": "COMPLETED",
        "video_urls": video_urls
    })

except Exception as e:
    print("[FATAL] Pipeline failed:", e)
    send_callback("fail", {
        "status": "FAILED",
        "type_error": "FATAL_ERROR",
        "failed_reason": str(e),
        "video_urls": video_urls
    })
    sys.exit(1)

print("[DONE] All videos processed.")
