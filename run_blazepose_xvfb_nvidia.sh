#!/bin/bash
# Script to run BlazePose extraction using NVIDIA RTX GPU with Xvfb virtual display

# Kill any existing Xvfb on display :99
pkill -f "Xvfb :99" 2>/dev/null

# Start Xvfb on display :99 with GLX extension
echo "Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1920x1080x24 +extension GLX &
XVFB_PID=$!
sleep 2

# Set display to use Xvfb
export DISPLAY=:99

# Force NVIDIA GPU for rendering (PRIME offload)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Force NVIDIA EGL for MediaPipe
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# DRI device selection
export DRI_PRIME=1

# Additional NVIDIA settings
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __VK_LAYER_NV_optimus=NVIDIA_only

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

echo "Testing NVIDIA GPU selection..."
glxinfo -B 2>&1 | grep -E "(OpenGL vendor|OpenGL renderer)" || echo "⚠️  glxinfo failed, continuing anyway..."

# Activate virtual environment
source /home/blaze/venvs/main/bin/activate

# Change to project directory
cd /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING

# Run the BlazePose extraction script
echo -e "\n🚀 Starting BlazePose extraction with NVIDIA GPU...\n"
python3 totalcapture_dataset/extract_blazepose_10ch.py \
  --videos /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/Videos \
  --output /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/everything_from_blazepose \
  --workers 1 \
  --max_side 1920 \
  --model /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/pose_landmarker_heavy.task \
  --delegate gpu

# Cleanup: kill Xvfb
echo -e "\n\nCleaning up Xvfb..."
kill $XVFB_PID 2>/dev/null
