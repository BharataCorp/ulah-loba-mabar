# WAN2.2 Manual Setup + Manual Generate Test (Production-like, Tanpa Docker)

Tujuan:
- Install WAN manual di Linux
- Struktur sama dengan Pod (`/workspace`)
- Model download dari S3 pakai script kamu
- Runtime files repo disalin ke WAN
- Test generate manual via terminal

---

# 1. PRASYARAT

## GPU harus aktif
```bash
nvidia-smi
```

## Python wajib 3.10
```bash
python3 --version
```

Kalau belum:

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

---

# 2. BUAT ROOT WORKSPACE

```bash
sudo mkdir -p /workspace
sudo chown $USER:$USER /workspace
cd /workspace
```

---

# 3. BUAT VIRTUAL ENV GLOBAL

```bash
python3.10 -m venv /workspace/wan_env
source /workspace/wan_env/bin/activate
pip install --upgrade pip setuptools wheel
```

---

# 4. INSTALL SYSTEM DEPENDENCIES

```bash
sudo apt-get update && sudo apt-get install -y \
git ffmpeg wget curl \
build-essential ninja-build cmake \
libgl1 libglib2.0-0
```

---

# 5. INSTALL PYTHON CORE LIBS

```bash
pip install --no-cache-dir decord librosa boto3
```

---

# 6. INSTALL PYTORCH CUDA 12.4 BUILD

```bash
pip install --no-cache-dir \
torch==2.5.0 torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cu124
```

---

# 7. INSTALL FLASH ATTENTION

```bash
pip install --no-cache-dir \
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.6.3+cu124torch2.5-cp310-cp310-linux_x86_64.whl
```

Test:

```bash
python -c "import flash_attn; print('flash_attn OK')"
```

---

# 8. INSTALL DIFFUSION STACK

```bash
pip install --no-cache-dir \
transformers==4.52.2 \
diffusers==0.31.0 \
peft==0.18.0 \
accelerate>=1.2.0 \
safetensors einops tqdm numpy
```

---

# 9. CLONE WAN2.2

```bash
cd /workspace
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
```

Hapus dependency konflik:

```bash
sed -i '/torch/d' requirements.txt
sed -i '/transformers/d' requirements.txt
sed -i '/diffusers/d' requirements.txt
sed -i '/flash_attn/d' requirements.txt
```

Install internal deps:

```bash
pip install -r requirements.txt
mkdir -p /workspace/Wan2.2/output_videos
```

---

# 10. SET ENV UNTUK DOWNLOAD MODEL

```bash
export MODEL_S3_BUCKET="ISI_BUCKET Check di s3 anu bharata"
export MODEL_S3_ENDPOINT="ISI_ENDPOINT"
export MODEL_S3_ACCESS_KEY="ISI_ACCESS_KEY"
export MODEL_S3_SECRET_KEY="ISI_SECRET_KEY"
export MODEL_FOLDER="Wan2.2-T2V-A14B"

export WAN_ROOT="/workspace/Wan2.2"
```

---

# 11. CLONE REPO MABAR

```bash
cd /workspace

REPO_URL="ISI_REPO_KAMU"

if [ ! -d "ulah-loba-mabar/.git" ]; then
  git clone "$REPO_URL" ulah-loba-mabar --branch master --single-branch
else
  cd ulah-loba-mabar
  git fetch origin master
  git checkout master
  git pull
  cd ..
fi
```

---

# 12. INSTALL DEPENDENCY REPO

```bash
cd /workspace/ulah-loba-mabar
pip install -r pod_requirements.txt --no-cache-dir
```

---

# 13. DOWNLOAD MODEL WAN DARI S3 (STEP PALING PENTING)

```bash
cd /workspace/ulah-loba-mabar/download_models
python3 download_wan2.py
```

Model harus muncul di:

```
/workspace/Wan2.2/Wan2.2-T2V-A14B
```

Cek:

```bash
ls /workspace/Wan2.2/Wan2.2-T2V-A14B
```

---

# 14. COPY RUNTIME FILES KE WAN (WAJIB UNTUK ENV IDENTIK POD)

```bash
cd /workspace/ulah-loba-mabar

cp wan2_t2v_task.py /workspace/Wan2.2/

cp -r wan_custom /workspace/Wan2.2/
cp -r wan_custom /workspace/Wan2.2/wan_custom

cp -r services /workspace/Wan2.2/services
```

Masuk runtime:

```bash
cd /workspace/Wan2.2
```

Pastikan folder output ada:

```bash
if [ ! -d "output_videos" ]; then
  mkdir output_videos
fi
```

---

# 15. TEST GENERATE MANUAL

### Test pakai generate.py

```bash
python generate.py \
--task t2v-A14B \
--ckpt_dir /workspace/Wan2.2/Wan2.2-T2V-A14B \
--prompt "A cinematic shot of a futuristic city at sunset, ultra realistic, 4k" \
--size 480*832 \
--frame_num 81 \
--sample_steps 30 \
--sample_shift 5 \
--offload_model True \
--convert_model_dtype \
--save_file output_videos/test.mp4
```

---

### Test pakai generate_t2v.py

```bash
python3 generate_t2v.py \
--prompt "A cinematic morning scene in a cozy pastel baby room..." \
--target_duration 5 \
--size 480*832 \
--sample_steps 8 \
--sample_shift 10 \
--output output_videos/sample_test.mp4
```

---

# 16. OUTPUT LOKASI

```
/workspace/Wan2.2/output_videos/
```

---

# SELESAI

Environment kamu sekarang:

✔ Sama dengan Pod runtime  
✔ Model dari S3 sesuai production  
✔ Runtime files repo ikut tersalin  
✔ Bisa test manual generate langsung  
✔ Runner tetap bisa dipakai kalau perlu  
