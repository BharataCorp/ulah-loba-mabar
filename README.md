# Wan Runner For Runner Wan2.1 Text to video

Runner script untuk eksekusi prompt → video dengan model Wan2.1, upload hasil ke Vultr Object Storage (S3-compatible), lalu callback ke backend.

## Usage

Pastikan environment sudah di-set:

```bash
export GENERATE_VIDEO_ID="gv_123"
export S3_ENDPOINT="https://sgp1.vultrobjects.com"
export S3_BUCKET="mybucket"
export S3_ACCESS_KEY="xxxx"
export S3_SECRET_KEY="yyyy"
export PUBLIC_BASE_URL="https://mybucket.sgp1.vultrobjects.com"
export BASE_URL="https://my-backend.com"
export PROJECT_DIR="/root/project"
export PROMPTS='["prompt 1", "prompt 2"]'
```

## Warn for install dependencies
1. If using small GPU (<30GB VRAM, misal 4090/3090/4080/3080)
```shell
pip3 install -r requirements-lite.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

2. If using large GPU (>=30GB VRAM, misal 4090 24GB/3090 24GB/4080 32GB/3080 24GB/3090 Ti 24GB/4090 Ti 24GB)
```shell
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
