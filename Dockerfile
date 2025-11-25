FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="BharataCorp Mabar Video"
LABEL description="Wan2.1 Video Generation Runner for Runpod"

WORKDIR /root

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first (for better caching)
COPY requirements.txt /root/
COPY requirements-lite.txt /root/

# Install Python dependencies
# Use lite version for GPUs < 30GB (RTX 4090, RTX 3090, etc)
# flash_attn can be problematic, so we use lite version
RUN pip3 install --no-cache-dir -r requirements-lite.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy Wan2.1 official repository
COPY Wan2.1/ /root/Wan2.1/

# Copy batch processing script
COPY wan2_1_batch.py /root/

# Create model directories
RUN mkdir -p /root/models/Wan2.1-T2V-1.3B \
    && mkdir -p /root/models/Wan2.1-I2V-14B-480P

# Download model files (or copy if you have them locally)
# Option 1: Download during build (uncomment and set MODEL_URL)
# ARG MODEL_URL=https://your-model-storage.com
# RUN wget -q -O /root/models/Wan2.1-T2V-1.3B/config.json ${MODEL_URL}/Wan2.1-T2V-1.3B/config.json
# RUN wget -q -O /root/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors ${MODEL_URL}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
# ... (download all model files)

# Option 2: Copy from local (if you have models locally)
# Uncomment these if you have model files:
# COPY models/Wan2.1-T2V-1.3B/ /root/models/Wan2.1-T2V-1.3B/
# COPY models/Wan2.1-I2V-14B-480P/ /root/models/Wan2.1-I2V-14B-480P/

# Set environment variables (defaults)
ENV PROJECT_DIR=/root
ENV CKPT_DIR=/root/models/Wan2.1-T2V-1.3B

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Set entrypoint
CMD ["python3", "/root/wan2_1_batch.py"]



