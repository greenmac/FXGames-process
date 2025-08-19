#!/bin/bash

# Create a new conda environment
conda create -y -n latentsync python=3.11.13
conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
hf download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
hf download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints