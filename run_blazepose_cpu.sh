#!/bin/bash
# Optimized CPU-based BlazePose extraction script
# This uses TensorFlow Lite XNNPACK delegate for maximum CPU performance

# Activate virtual environment
source /home/blaze/venvs/main/bin/activate

# Change to project directory
cd /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING

# Run with CPU delegate (most reliable option)
python3 totalcapture_dataset/extract_blazepose_10ch.py \
  --videos /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/Videos \
  --output /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/everything_from_blazepose \
  --workers 1 \
  --max_side 1920 \
  --model /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/pose_landmarker_heavy.task \
  --delegate cpu

echo -e "\n✅ Processing complete!"
