#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --enable_deepcache \
    --video_path "data/ori/video/clip_1_part1_NoMoreBets.mp4" \
    --audio_path "data/ori/audio/tts_NoMoreBets.wav" \
    --video_out_path "data/output/lipSync_clip_1_part1_NoMoreBets.mp4"