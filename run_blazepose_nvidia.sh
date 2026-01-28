#!/bin/bash
# Script to run BlazePose extraction using NVIDIA RTX GPU instead of AMD Radeon

# Ensure DISPLAY is set (required for EGL context)
export DISPLAY=${DISPLAY:-:1}

# Force NVIDIA GPU for rendering (PRIME offload)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Force NVIDIA EGL for MediaPipe (this is the critical one!)
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# DRI device selection - try NVIDIA device first
# renderD128 is usually AMD, renderD129 might be NVIDIA
export DRI_PRIME=1
export MESA_VK_DEVICE_SELECT=10de:2583  # NVIDIA RTX 3050 device ID

# Additional NVIDIA settings
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __VK_LAYER_NV_optimus=NVIDIA_only

# CUDA settings (if MediaPipe uses CUDA backend)
export CUDA_VISIBLE_DEVICES=0

# Disable Mesa drivers to prevent conflicts
export MESA_LOADER_DRIVER_OVERRIDE=""

# Activate virtual environment
source /home/blaze/venvs/main/bin/activate

# Change to project directory
cd /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING

# Run the BlazePose extraction script
python3 totalcapture_dataset/extract_blazepose_10ch.py \
  --videos /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/Videos \
  --output /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/everything_from_blazepose \
  --workers 1 \
  --max_side 1920 \
  --model /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/pose_landmarker_heavy.task \
  --delegate gpu
