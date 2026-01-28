#!/bin/bash
# GPU-accelerated BlazePose extraction with NVIDIA RTX 3050

# Force NVIDIA GPU for OpenGL/EGL
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export DRI_PRIME=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment
source /home/blaze/venvs/main/bin/activate

# Change to project directory
cd /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING

# Run extraction with GPU
python3 totalcapture_dataset/extract_blazepose_10ch_gpu.py \
  --videos totalcapture_dataset/Videos \
  --output everything_from_blazepose \
  --workers 8 \
  --max_side 1920 \
  --model_complexity 2
