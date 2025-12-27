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
    sample_steps=16,
)

print("Video saved at:", output)
````

##### Example Prompt String(Simple)
```
output = T2VPipeline.generate(
    prompt="A cinematic cat dancing on stage, dramatic lighting",
    target_duration=5,        # seconds
    size="832*480",
    sample_steps=16,
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



## Usage test manual
### T2V Generate test manual
```bash
 python3 generate_t2v.py   --prompt "A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily"   --target_duration 5   --size 480*832   --sample_steps 8 --sample_shift 8 --output output_videos/sample_step8_sample_8_potrait.mp4
```

### I2V Generate test manual
```bash
 python3 generate_i2v.py \
  --image assets/product.jpg \
  --prompt "A cinematic product reveal, soft lighting" \
  --target_duration 5 \
  --size 832*480
```

```bash
prompt_json = {
    "scenes": [
        {
            "actions": ["Produk muncul di tengah frame", "Slow zoom ke logo"],
            "setting": "studio minimalis",
            "mood": "premium"
        }
    ],
    "prompt_for_wan_one_i2v": (
        "A premium cinematic product showcase, "
        "soft reflections, clean background"
    )
}

```


### TI2V Generate test manual
```bash
 python3 generate_ti2v.py \
  --image assets/product.jpg \
  --prompt "A cinematic product showcase with soft lighting" \
  --target_duration 5 \
  --size 832*480
```




### kesimpulan paling bagus adalah

 python3 generate_t2v.py   --prompt "A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily"   --target_duration 5   --size 480*832   --sample_steps 8 --sample_shift 10 --output output_videos/sample_step8_sample_10_potrait.mp4


    artinya --sample_steps 8 --sample_shift 10 , generate 6-7 menit untuk 5 detik video



### Question

Di test

root@2227e4ebbb3a:/workspace/Wan2.2# python3 generate_ti2v.py \
  --image example_assets/images/images.jpeg \
  --prompt "A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image" \
  --target_duration 5 \
  --size 704*1280 --output output_videos/ti2v.mp4
[2025-12-27 09:15:10] [INFO] [wan_custom.TI2V] Splitting 5s into chunks: [5]
[2025-12-27 09:15:10] [INFO] [wan_custom.TI2V] [Chunk 1/1] python3 generate.py --task ti2v-5B --ckpt_dir /workspace/Wan2.2/Wan2.2-TI2V-5B --prompt A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image --image example_assets/images/images.jpeg --size 704*1280 --frame_num 81 --sample_steps 16 --sample_shift 10 --offload_model True --t5_cpu --convert_model_dtype --save_file /workspace/Wan2.2/output_videos/ti2v_chunk1.mp4
[2025-12-27 09:15:13,421] INFO: Generation job args: Namespace(task='ti2v-5B', size='704*1280', frame_num=81, ckpt_dir='/workspace/Wan2.2/Wan2.2-TI2V-5B', offload_model=True, ulysses_size=1, t5_fsdp=False, t5_cpu=True, dit_fsdp=False, save_file='/workspace/Wan2.2/output_videos/ti2v_chunk1.mp4', prompt="A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image", use_prompt_extend=False, prompt_extend_method='local_qwen', prompt_extend_model=None, prompt_extend_target_lang='zh', base_seed=3937418005304531965, image='example_assets/images/images.jpeg', sample_solver='unipc', sample_steps=16, sample_shift=10.0, sample_guide_scale=5.0, convert_model_dtype=True, src_root_path=None, refert_num=77, replace_flag=False, use_relighting_lora=False, num_clip=None, audio=None, enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, pose_video=None, start_from_ref=False, infer_frames=80)
[2025-12-27 09:15:13,421] INFO: Generation model config: {'__name__': 'Config: Wan TI2V 5B', 't5_model': 'umt5_xxl', 't5_dtype': torch.bfloat16, 'text_len': 512, 'param_dtype': torch.bfloat16, 'num_train_timesteps': 1000, 'sample_fps': 24, 'sample_neg_prompt': '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', 'frame_num': 121, 't5_checkpoint': 'models_t5_umt5-xxl-enc-bf16.pth', 't5_tokenizer': 'google/umt5-xxl', 'vae_checkpoint': 'Wan2.2_VAE.pth', 'vae_stride': (4, 16, 16), 'patch_size': (1, 2, 2), 'dim': 3072, 'ffn_dim': 14336, 'freq_dim': 256, 'num_heads': 24, 'num_layers': 30, 'window_size': (-1, -1), 'qk_norm': True, 'cross_attn_norm': True, 'eps': 1e-06, 'sample_shift': 5.0, 'sample_steps': 50, 'sample_guide_scale': 5.0}
[2025-12-27 09:15:13,421] INFO: Input prompt: A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image
[2025-12-27 09:15:13,426] INFO: Input image: example_assets/images/images.jpeg
[2025-12-27 09:15:13,426] INFO: Creating WanTI2V pipeline.
[2025-12-27 09:16:04,575] INFO: loading /workspace/Wan2.2/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
[2025-12-27 09:16:14,408] INFO: loading /workspace/Wan2.2/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
[2025-12-27 09:16:16,581] INFO: Creating WanModel from /workspace/Wan2.2/Wan2.2-TI2V-5B
[2025-12-27 09:16:33,876] INFO: Generating video ...
100%|███████████████████████████████████████████| 16/16 [01:24<00:00,  5.27s/it]
[2025-12-27 09:18:49,709] INFO: Saving generated video to /workspace/Wan2.2/output_videos/ti2v_chunk1.mp4
[2025-12-27 09:18:51,787] INFO: Finished.
[2025-12-27 09:18:54] [INFO] [wan_custom.TI2V] Single chunk → saved as output_videos/ti2v.mp4
✅ Video saved to: output_videos/ti2v.mp4
root@2227e4ebbb3a:/workspace/Wan2.2# python3 generate_ti2v.py   --image example_assets/images/images.jpeg   --prompt "A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image"   --target_duration 5   --size 1280*704 --output output_videos/ti2v.mp4
[2025-12-27 09:20:08] [INFO] [wan_custom.TI2V] Splitting 5s into chunks: [5]
[2025-12-27 09:20:08] [INFO] [wan_custom.TI2V] [Chunk 1/1] python3 generate.py --task ti2v-5B --ckpt_dir /workspace/Wan2.2/Wan2.2-TI2V-5B --prompt A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image --image example_assets/images/images.jpeg --size 1280*704 --frame_num 81 --sample_steps 16 --sample_shift 10 --offload_model True --t5_cpu --convert_model_dtype --save_file /workspace/Wan2.2/output_videos/ti2v_chunk1.mp4
[2025-12-27 09:20:12,213] INFO: Generation job args: Namespace(task='ti2v-5B', size='1280*704', frame_num=81, ckpt_dir='/workspace/Wan2.2/Wan2.2-TI2V-5B', offload_model=True, ulysses_size=1, t5_fsdp=False, t5_cpu=True, dit_fsdp=False, save_file='/workspace/Wan2.2/output_videos/ti2v_chunk1.mp4', prompt="A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image", use_prompt_extend=False, prompt_extend_method='local_qwen', prompt_extend_model=None, prompt_extend_target_lang='zh', base_seed=9163324304871451290, image='example_assets/images/images.jpeg', sample_solver='unipc', sample_steps=16, sample_shift=10.0, sample_guide_scale=5.0, convert_model_dtype=True, src_root_path=None, refert_num=77, replace_flag=False, use_relighting_lora=False, num_clip=None, audio=None, enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, pose_video=None, start_from_ref=False, infer_frames=80)
[2025-12-27 09:20:12,213] INFO: Generation model config: {'__name__': 'Config: Wan TI2V 5B', 't5_model': 'umt5_xxl', 't5_dtype': torch.bfloat16, 'text_len': 512, 'param_dtype': torch.bfloat16, 'num_train_timesteps': 1000, 'sample_fps': 24, 'sample_neg_prompt': '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', 'frame_num': 121, 't5_checkpoint': 'models_t5_umt5-xxl-enc-bf16.pth', 't5_tokenizer': 'google/umt5-xxl', 'vae_checkpoint': 'Wan2.2_VAE.pth', 'vae_stride': (4, 16, 16), 'patch_size': (1, 2, 2), 'dim': 3072, 'ffn_dim': 14336, 'freq_dim': 256, 'num_heads': 24, 'num_layers': 30, 'window_size': (-1, -1), 'qk_norm': True, 'cross_attn_norm': True, 'eps': 1e-06, 'sample_shift': 5.0, 'sample_steps': 50, 'sample_guide_scale': 5.0}
[2025-12-27 09:20:12,213] INFO: Input prompt: A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image



kenapa hasilnya 3 detik saja ya, padahal sudah di set target_duration 5 detik?]

tanpa mengubahkeseluruhan scriptnya hanya mengubah kondisi dan mengganti rife 
SUPPORTED_SIZES = {
  "ti2v-5B": ["704*1280", "1280*704"]
}


Question:

root@2227e4ebbb3a:/workspace/Wan2.2# python3 generate_i2v.py   --image example_assets/images/images.jpeg   --prompt "A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image"   --target_duration 5 --size 480*832   --sample_steps 8 --sample_shift 10 --output output_videos/ti2v.mp4
[2025-12-27 10:03:53] [INFO] [wan_custom.I2V] Splitting 5s into chunks: [5]
[2025-12-27 10:03:53] [INFO] [wan_custom.I2V] [Chunk 1/1] python3 generate.py --task i2v-A14B --ckpt_dir /workspace/Wan2.2/Wan2.2-I2V-A14B --prompt A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image --image example_assets/images/images.jpeg --size 480*832 --frame_num 81 --sample_steps 8 --sample_shift 10 --offload_model True --t5_cpu --convert_model_dtype --save_file /workspace/Wan2.2/output_videos/ti2v_chunk1.mp4
[2025-12-27 10:03:56,863] INFO: Generation job args: Namespace(task='i2v-A14B', size='480*832', frame_num=81, ckpt_dir='/workspace/Wan2.2/Wan2.2-I2V-A14B', offload_model=True, ulysses_size=1, t5_fsdp=False, t5_cpu=True, dit_fsdp=False, save_file='/workspace/Wan2.2/output_videos/ti2v_chunk1.mp4', prompt="A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image", use_prompt_extend=False, prompt_extend_method='local_qwen', prompt_extend_model=None, prompt_extend_target_lang='zh', base_seed=4965415490791932445, image='example_assets/images/images.jpeg', sample_solver='unipc', sample_steps=8, sample_shift=10.0, sample_guide_scale=(3.5, 3.5), convert_model_dtype=True, src_root_path=None, refert_num=77, replace_flag=False, use_relighting_lora=False, num_clip=None, audio=None, enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, pose_video=None, start_from_ref=False, infer_frames=80)
[2025-12-27 10:03:56,863] INFO: Generation model config: {'__name__': 'Config: Wan I2V A14B', 't5_model': 'umt5_xxl', 't5_dtype': torch.bfloat16, 'text_len': 512, 'param_dtype': torch.bfloat16, 'num_train_timesteps': 1000, 'sample_fps': 16, 'sample_neg_prompt': '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', 'frame_num': 81, 't5_checkpoint': 'models_t5_umt5-xxl-enc-bf16.pth', 't5_tokenizer': 'google/umt5-xxl', 'vae_checkpoint': 'Wan2.1_VAE.pth', 'vae_stride': (4, 8, 8), 'patch_size': (1, 2, 2), 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'num_heads': 40, 'num_layers': 40, 'window_size': (-1, -1), 'qk_norm': True, 'cross_attn_norm': True, 'eps': 1e-06, 'low_noise_checkpoint': 'low_noise_model', 'high_noise_checkpoint': 'high_noise_model', 'sample_shift': 5.0, 'sample_steps': 40, 'boundary': 0.9, 'sample_guide_scale': (3.5, 3.5)}
[2025-12-27 10:03:56,863] INFO: Input prompt: A cinematic morning scene in a cozy pastel baby room: a young mother in a pastel hijab gently applies Locare Body Lotion to her laughing baby boy's arm. Soft natural light fills the room, showcasing the creamy lotion texture and baby's smooth skin. Close-up of mother's caring smile, baby's chubby cheeks, and Locare lotion bottle. Warm, loving atmosphere, 4K quality, clean details, smooth motion, no glitches. End with mother holding up the Locare bottle to the camera, baby waving happily, makesure: model using reference by image
[2025-12-27 10:03:56,867] INFO: Input image: example_assets/images/images.jpeg
[2025-12-27 10:03:56,867] INFO: Creating WanI2V pipeline.
[2025-12-27 10:04:51,891] INFO: loading /workspace/Wan2.2/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth
[2025-12-27 10:05:00,728] INFO: loading /workspace/Wan2.2/Wan2.2-I2V-A14B/Wan2.1_VAE.pth
[2025-12-27 10:05:01,098] INFO: Creating WanModel from /workspace/Wan2.2/Wan2.2-I2V-A14B
[2025-12-27 10:06:55,231] INFO: Generating video ...
100%|█████████████████████████████████████████████| 8/8 [03:46<00:00, 28.36s/it]
[2025-12-27 10:11:33,849] INFO: Saving generated video to /workspace/Wan2.2/output_videos/ti2v_chunk1.mp4
[2025-12-27 10:11:36,977] INFO: Finished.
[2025-12-27 10:11:42] [INFO] [wan_custom.I2V] Single chunk → saved as output_videos/ti2v.mp4
✅ Video saved to: output_videos/ti2v.mp4


kenapa hasilnya landscape ya, padahal sudah di set size 480*832 (portrait)?