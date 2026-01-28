#!/bin/bash
# Optimized 10-channel BlazePose extraction with 16 workers + FHD resolution

# Activate virtual environment
source /home/blaze/venvs/main/bin/activate

# Change to project directory
cd /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING

# Run with maximum parallelization + FHD quality
python3 totalcapture_dataset/extract_blazepose_10ch_gpu.py \
  --videos totalcapture_dataset/Videos \
  --output everything_from_blazepose \
  --workers 16 \
  --max_side 1920 \
  --model_complexity 2

echo -e "\n✅ Processing complete!"
