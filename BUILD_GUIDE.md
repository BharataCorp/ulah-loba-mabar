# Wan2.1 Runner - Docker Build Guide

**Purpose**: Build Docker image untuk Runpod deployment

---

## 🐳 Quick Build

### Prerequisites

1. Docker installed
2. Wan2.1 model files downloaded (~25 GB)
3. Docker Hub account (untuk push image)

### Build Command

```bash
cd /mnt/nvme/my-job/github/BharataCorp/wan2.1-runner

# Build image
docker build -t yadenmustopa/wan2.1-runner:latest .

# Test locally (if you have GPU)
docker run --gpus all \
  -e GENERATE_NUMBER="test" \
  -e PROMPTS_B64="W10=" \
  yadenmustopa/wan2.1-runner:latest

# Push to Docker Hub
docker login
docker push yadenmustopa/wan2.1-runner:latest
```

---

## 📋 Step-by-Step Guide

### Step 1: Prepare Model Files

**Download Wan2.1 Models:**

```bash
# Create models directory
mkdir -p models/Wan2.1-T2V-1.3B
mkdir -p models/Wan2.1-I2V-14B-480P

# Download T2V model (~10 GB)
cd models/Wan2.1-T2V-1.3B
wget https://huggingface.co/Alibaba-PAI/Wan2.1-T2V-1.3B/resolve/main/config.json
wget https://huggingface.co/Alibaba-PAI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors
wget https://huggingface.co/Alibaba-PAI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
# ... download all files

# Download I2V model (~15 GB)
cd ../Wan2.1-I2V-14B-480P
# ... download I2V model files

cd ../..
```

**Or use HuggingFace CLI:**

```bash
# Install huggingface-cli
pip install huggingface-hub

# Download T2V
huggingface-cli download Alibaba-PAI/Wan2.1-T2V-1.3B \
  --local-dir models/Wan2.1-T2V-1.3B

# Download I2V
huggingface-cli download Alibaba-PAI/Wan2.1-I2V-14B-480P \
  --local-dir models/Wan2.1-I2V-14B-480P
```

---

### Step 2: Update Dockerfile

**Choose build strategy:**

#### **Option A: Include Models in Image** (Recommended)

Uncomment these lines in Dockerfile:

```dockerfile
# Copy model files
COPY models/Wan2.1-T2V-1.3B/ /root/models/Wan2.1-T2V-1.3B/
COPY models/Wan2.1-I2V-14B-480P/ /root/models/Wan2.1-I2V-14B-480P/
```

**Pros:**
- ✅ Fastest startup (2-3 min)
- ✅ No download needed
- ✅ Consistent

**Cons:**
- ❌ Large image (~20 GB)
- ❌ Longer build time
- ❌ Larger push time

---

#### **Option B: Download on Build** (Alternative)

Set MODEL_URL and uncomment download commands:

```dockerfile
ARG MODEL_URL=https://huggingface.co/Alibaba-PAI
RUN wget -q -O /root/models/Wan2.1-T2V-1.3B/config.json \
    ${MODEL_URL}/Wan2.1-T2V-1.3B/resolve/main/config.json
# ... etc
```

**Pros:**
- ✅ Smaller local files
- ✅ Always get latest models

**Cons:**
- ❌ Slower build
- ❌ Network dependency
- ❌ Build might fail if network issue

---

### Step 3: Build Image

```bash
# Navigate to wan2.1-runner directory
cd /mnt/nvme/my-job/github/BharataCorp/wan2.1-runner

# Build with tag
docker build -t yadenmustopa/wan2.1-runner:latest .

# Build with specific version
docker build -t yadenmustopa/wan2.1-runner:v1.0.0 .

# Build and monitor progress
docker build -t yadenmustopa/wan2.1-runner:latest . --progress=plain
```

**Expected Duration:**
- With models: 30-60 min
- Without models: 10-20 min

---

### Step 4: Test Image Locally (Optional)

```bash
# Test if image works
docker run --gpus all \
  -e GENERATE_NUMBER="test_$(date +%s)" \
  -e TARGET_DURATION="10" \
  -e PROMPTS_B64="WyJBIGNhdCBzaXR0aW5nIG9uIGEgY2hhaXIiXQ==" \
  -e S3_ENDPOINT="sgp1.vultrobjects.com" \
  -e S3_BUCKET="test-bucket" \
  -e S3_ACCESS_KEY="test" \
  -e S3_SECRET_KEY="test" \
  -e PUBLIC_BASE_URL="https://test.com" \
  -e CALLBACK_URL="https://test.com/callback" \
  -e WAN_TASK="t2v-1.3B" \
  -e WAN_SIZE="832*480" \
  -e PROJECT_DIR="/root" \
  -e CKPT_DIR="/root/models/Wan2.1-T2V-1.3B" \
  yadenmustopa/wan2.1-runner:latest

# Check if GPU detected
docker run --gpus all yadenmustopa/wan2.1-runner:latest \
  python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

### Step 5: Push to Docker Hub

```bash
# Login to Docker Hub
docker login
# Enter username: yadenmustopa
# Enter password: your_password

# Push image
docker push yadenmustopa/wan2.1-runner:latest

# Push specific version
docker push yadenmustopa/wan2.1-runner:v1.0.0
```

**Expected Duration:**
- First push: 1-2 hours (20+ GB)
- Subsequent pushes: 5-10 min (only changed layers)

---

## 🔍 Verify Image

### Check Image Size

```bash
docker images | grep wan2.1-runner
```

Expected:
```
yadenmustopa/wan2.1-runner  latest  abc123  20GB  10 minutes ago
```

### Check Image Contents

```bash
# List files
docker run yadenmustopa/wan2.1-runner:latest ls -la /root/

# Check Wan2.1 folder
docker run yadenmustopa/wan2.1-runner:latest ls -la /root/Wan2.1/

# Check models
docker run yadenmustopa/wan2.1-runner:latest ls -la /root/models/

# Check Python packages
docker run yadenmustopa/wan2.1-runner:latest pip3 list
```

---

## 📊 Build Options Comparison

| Strategy | Build Time | Image Size | Startup Time | Maintenance |
|----------|-----------|------------|--------------|-------------|
| **Include Models** | 30-60 min | ~20 GB | 2-3 min ⭐ | Update image untuk model changes |
| **Download on Build** | 1-2 hours | ~5 GB | 2-3 min | Rebuild untuk updates |
| **Download on Run** | 10 min | ~3 GB | 15-20 min ❌ | No rebuild needed |

**Recommended:** ✅ **Include Models** for production (fastest runtime)

---

## 🚨 Troubleshooting

### Build Failed - Out of Space

```bash
# Check disk space
df -h

# Clean up old images
docker system prune -a

# Use different build directory
docker build -t ... --build-arg DOCKER_BUILDKIT=1 .
```

---

### Build Failed - CUDA Issues

```bash
# Use different base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Or
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn8-runtime
```

---

### Push Failed - Too Large

```bash
# Compress layers
docker build --compress -t ... .

# Or use multi-stage build
FROM nvidia/cuda:11.8.0... AS builder
# ... build stage
FROM nvidia/cuda:11.8.0-runtime...
COPY --from=builder /root /root
```

---

### Test Failed - No GPU

```bash
# Check GPU available
nvidia-smi

# Install nvidia-docker
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

---

## 📝 File Structure

```
wan2.1-runner/
├── Dockerfile                    # ← Docker build config
├── wan2_1_batch.py              # ← Main execution script
├── requirements.txt             # ← Full dependencies
├── requirements-lite.txt        # ← Lite version (no flash_attn)
├── Wan2.1/                      # ← Official Wan2.1 repo
│   ├── generate.py              # ← Main generator
│   ├── wan/                     # ← Wan modules
│   └── ...
└── models/                      # ← Model files (optional in repo)
    ├── Wan2.1-T2V-1.3B/        # ← T2V model (~10 GB)
    └── Wan2.1-I2V-14B-480P/    # ← I2V model (~15 GB)
```

---

## 🎯 Quick Start

### Minimal Build (Testing)

```bash
# Build without models (smallest image)
docker build -t yadenmustopa/wan2.1-runner:test .

# Test
docker run --gpus all yadenmustopa/wan2.1-runner:test \
  python3 -c "print('OK')"
```

### Production Build

```bash
# 1. Download models first
mkdir -p models
cd models
# Download Wan2.1-T2V-1.3B and Wan2.1-I2V-14B-480P
cd ..

# 2. Update Dockerfile to COPY models
# Uncomment the COPY models/ lines

# 3. Build
docker build -t yadenmustopa/wan2.1-runner:latest .

# 4. Test
docker run --gpus all \
  -e PROJECT_DIR="/root" \
  -e CKPT_DIR="/root/models/Wan2.1-T2V-1.3B" \
  yadenmustopa/wan2.1-runner:latest \
  python3 -c "import os; print(os.path.exists('/root/Wan2.1/generate.py'))"

# 5. Push
docker push yadenmustopa/wan2.1-runner:latest
```

---

## ⏱️ Timeline

### One-Time Setup

| Task | Duration |
|------|----------|
| Download models | 1-2 hours (25 GB) |
| Build Docker image | 30-60 min |
| Test image locally | 10 min |
| Push to Docker Hub | 30-60 min |
| **Total** | **2-4 hours** |

### Per Request (After Setup)

| Task | Duration |
|------|----------|
| Runpod pull image | 2-3 min (first time) |
| Container start | 30 sec |
| Video generation | 1-2 min per video |
| Upload to S3 | 30 sec per video |
| **Total for 10 videos** | **~5-10 min** ⚡ |

---

## 🎉 After Build

Once image is built and pushed:

```bash
# Test Runpod integration
php spark runpod:test YOUR_API_KEY

# Generate videos
php spark generate_video:wan_model
```

**Current PHP code akan langsung jalan!** ✅

---

**Build Time**: 2-4 hours (one-time)  
**Runtime**: 2-3 min startup ⚡  
**Maintenance**: Update image when needed



