# Wan Runner For Runner Wan2.1 Text to video

Runner script untuk eksekusi prompt → video dengan model Wan2.1, upload hasil ke Vultr Object Storage (S3-compatible), lalu callback ke backend.

## Require directories
```tree
/workspace/
├── Wan2.2/
│   ├── Wan2.2-T2V-A14B/
│   ├── Wan2.2-I2V-A14B/
│
└── ulah-loba-mabar/
    ├── wan/
    │   ├── pipelines/
    │   │   ├── base_pipeline.py
    │   │   ├── t2v_pipeline.py
    │   │   └── i2v_pipeline.py
    │   ├── config.py
    │   └── logger.py

```

## Environment Variables
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

## Usage

*_Note:_*
> Before running the script, make sure to set the required environment variables as shown below.
> ```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
# CPU throttle (WAJIB untuk stabilitas)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
> ```

### WAN Usage Pipeline Generate
#### T2V Usage
Note :
> makesure to import 
> ```from wan_custom.pipelines.t2v_pipeline import T2VPipeline```

```bash
output = T2VPipeline.generate(
    prompt="A cinematic cat dancing on stage, dramatic lighting",
    target_duration=5,        # seconds
    size="832*480",
    sample_steps=25,
)

print("Video saved at:", output)
````

##### Example Prompt String(Simple)
```
output = T2VPipeline.generate(
    prompt="A cinematic cat dancing on stage, dramatic lighting",
    target_duration=5,        # seconds
    size="832*480",
    sample_steps=25,
)

print("Video saved at:", output)
```

##### Example Prompt JSON(Scene-based)
```
prompt_json = {
    "scenes": [
        {
            "actions": [
                "Ibu mengoleskan lotion ke tangan bayi",
                "Close-up tekstur lotion"
            ],
            "setting": "ruang keluarga pastel",
            "mood": "hangat dan lembut"
        }
    ],
    "prompt_for_wan_one_t2v": (
        "A warm cinematic baby care scene, "
        "soft daylight, mother applying baby lotion"
    )
}

output = T2VPipeline.generate(
    prompt=prompt_json,
    target_duration=8,
    size="832*480",
)

```

#### I2V Usage
Note :
> makesure to import 
> ```from wan_custom.pipelines.i2v_pipeline import I2VPipeline```
> Also prepare input image path
> 

##### Example Prompt String(Simple)
```bash
output = I2VPipeline.generate(
    image_path="/workspace/assets/product.jpg",
    prompt="A cinematic product reveal, soft lighting",
    target_duration=5,
    size="832*480",
)

print("Video saved at:", output)
```

##### Example Prompt JSON(Scene-based)
```bash
prompt_json = {
    "scenes": [
        {
            "actions": [
                "Produk muncul di tengah frame",
                "Slow zoom ke logo"
            ],
            "setting": "studio minimalis",
            "mood": "premium"
        }
    ],
    "prompt_for_wan_one_i2v": (
        "A premium cinematic product showcase, "
        "soft reflections, clean background"
    )
}

output = I2VPipeline.generate(
    image_path="/workspace/assets/product.jpg",
    prompt=prompt_json,
    target_duration=6,
    size="832*480",
)
````

#### TI2V Usage
Note :
> makesure to import 
> ```from wan_custom.pipelines.ti2v_pipeline import TI2VPipeline```
> Also prepare input text and image path
>

##### Example Prompt String(Simple)
```bash
from wan_custom.pipelines.ti2v_pipeline import TI2VPipeline

output = TI2VPipeline.generate(
    image_path="assets/product.jpg",
    prompt="A cinematic product showcase with soft lighting",
    target_duration=5,
    size="832*480",
)
print("Saved to:", output)
````

#### Animate Usage
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

from wan_custom.pipelines.animate_pipeline import AnimatePipeline

output = AnimatePipeline.generate(
    reference_image="assets/character.png",
    prompt="A cheerful brand mascot waving and smiling to the camera",
    target_duration=5,
    size="832*480",
)
print("Saved:", output)
```

#### Animate Swap Usage
```bash
from wan_custom.pipelines.animate_swap_pipeline import AnimateSwapPipeline

output = AnimateSwapPipeline.generate(
    source_video="input/video.mp4",
    reference_image="assets/character.png",
    output_path="output_videos/character_swap.mp4"
)
```

#### Mass Animate Swap Usage
```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

from wan_custom.pipelines.mass_animate_pipeline import MassAnimatePipeline

video_path = "/workspace/videos/master_video.mp4"

characters = {
    "ibu_hijab": "/workspace/characters/ibu_hijab.png",
    "ibu_modern": "/workspace/characters/ibu_modern.png",
    "influencer": "/workspace/characters/influencer.png",
}

output_dir = "/workspace/output_videos/affiliate_variants"

results = MassAnimatePipeline.generate_for_characters(
    video_path=video_path,
    character_images=characters,
    output_dir=output_dir,
    replace=True,   # True = face/character replacement
)

print(results)

```

#### S2V Usage
Note :
> makesure to import 
> ```from wan_custom.pipelines.s2v_pipeline import S2VPipeline```

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Optional (CPU throttling)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

from wan_custom.pipelines.s2v_pipeline import S2VPipeline

output = S2VPipeline.generate(
    prompt="A cheerful mother talking to her baby",
    audio_path="examples/talk.wav",
    image_path="examples/character.jpg",
    target_duration=5,  # seconds
    size="832*480",
)

print("Video generated:", output)

```

```bash
prompt_json = {
    "prompt_for_wan_one_t2v": (
        "A warm mother speaks gently to her baby, "
        "baby smiles and reacts to the voice"
    )
}

S2VPipeline.generate(
    prompt=prompt_json,
    audio_path="voice.wav",
    image_path="mother.jpg",
    target_duration=8,
)
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



