# Extract All BlazePose Keypoints (33 Landmarks)

This script extracts all 33 BlazePose 3D world position landmarks from TotalCapture dataset videos.

## Output Format

- **Shape**: `(Total_Frames, 33, 4)` for each video
- **Columns**: `[X, Y, Z, Visibility]`
- **Missing Data**: If a frame is missing, the entire row is `[0, 0, 0, 0]`
- **Data Type**: `float32` (fp32)
- **File Format**: NumPy `.npy` files

## File Naming Convention

Videos are saved with the prefix `blaze_` followed by the video name:
- Input: `TC_S1_acting1_cam1.mp4`
- Output: `blaze_S1_acting1_cam1.npy`

## Directory Structure

```
main_all_keypoints/
├── extract_all_blazepose_keypoints.py
├── blazepose/
│   ├── S1/
│   │   ├── acting1/
│   │   │   ├── blaze_S1_acting1_cam1.npy
│   │   │   ├── blaze_S1_acting1_cam2.npy
│   │   │   └── ... (cam3-cam8)
│   │   ├── acting2/
│   │   └── ... (all 12 actions)
│   ├── S2/
│   └── S3/
└── gt/
```

## GPU Optimization (T4)

The script is optimized for T4 GPU with:
- **Sequential processing**: Processes one video at a time to avoid memory issues
- **Garbage collection**: Forces cleanup after each video
- **Progress tracking**: Shows real-time progress with `tqdm`
- **Skip existing files**: Automatically skips already processed videos
- **High-quality model**: Uses MediaPipe model complexity 2 (best accuracy)

## Usage

### 1. Basic Usage (Process All Videos)

```bash
cd main_all_keypoints
python extract_all_blazepose_keypoints.py
```

This assumes videos are in `../totalcapture_dataset/Videos/` and will save outputs to `./blazepose/`

### 2. Custom Directories

```bash
python extract_all_blazepose_keypoints.py \
    --videos /path/to/totalcapture_dataset/Videos \
    --output /path/to/output
```

### 3. Verify Output Structure

```bash
python extract_all_blazepose_keypoints.py --verify
```

This will:
- Show the directory structure
- Display file counts and sizes
- Verify array shapes and dtypes
- Show sample files from each directory

## Expected Processing Time

For TotalCapture dataset (3 subjects × 12 actions × 8 cameras = 288 videos):
- **T4 GPU**: ~2-4 hours (depends on video lengths)
- Each video: ~1-3 minutes (varies by frame count)

## Output Verification

After processing, verify your outputs:

```python
import numpy as np

# Load a file
data = np.load('blazepose/S1/acting1/blaze_S1_acting1_cam1.npy')

print(f"Shape: {data.shape}")  # Expected: (num_frames, 33, 4)
print(f"Dtype: {data.dtype}")  # Expected: float32
print(f"Min: {data.min()}, Max: {data.max()}")

# Check a specific frame
frame_100 = data[100]  # Shape: (33, 4)
print(f"Frame 100 keypoints:\n{frame_100}")

# Check for missing frames (all zeros)
missing_frames = np.all(data == 0, axis=(1, 2))
print(f"Missing frames: {missing_frames.sum()}/{len(data)}")
```

## BlazePose 33 Landmarks

MediaPipe BlazePose detects 33 landmarks:

### Body (11 landmarks: 0-10)
- 0: Nose
- 1: Left eye inner
- 2: Left eye
- 3: Left eye outer
- 4: Right eye inner
- 5: Right eye
- 6: Right eye outer
- 7: Left ear
- 8: Right ear
- 9: Mouth left
- 10: Mouth right

### Upper Body (12 landmarks: 11-22)
- 11: Left shoulder
- 12: Right shoulder
- 13: Left elbow
- 14: Right elbow
- 15: Left wrist
- 16: Right wrist
- 17: Left pinky
- 18: Right pinky
- 19: Left index
- 20: Right index
- 21: Left thumb
- 22: Right thumb

### Lower Body (10 landmarks: 23-32)
- 23: Left hip
- 24: Right hip
- 25: Left knee
- 26: Right knee
- 27: Left ankle
- 28: Right ankle
- 29: Left heel
- 30: Right heel
- 31: Left foot index
- 32: Right foot index

## Troubleshooting

### Out of Memory
If you encounter GPU memory issues:
- The script already processes videos sequentially
- Close other GPU applications
- Reduce `model_complexity` from 2 to 1 in the code (line 54)

### Missing Videos
If videos aren't found:
- Check the `--videos` path is correct
- Ensure directory structure matches: `Videos/s1/acting1/*.mp4`
- Verify video file extensions (`.mp4`, `.avi`, `.mov`, `.mkv`)

### Slow Processing
- Verify GPU is being used by MediaPipe (should auto-detect)
- Check CUDA installation: `nvidia-smi`
- Ensure no other processes are using GPU

## Dependencies

```bash
pip install mediapipe opencv-python numpy tqdm
```

Or use the project requirements:
```bash
pip install -r ../requirements_pose.txt
pip install tqdm  # If not already installed
```

## Resume Interrupted Processing

The script automatically skips files that already exist, so you can safely:
1. Stop processing at any time (Ctrl+C)
2. Re-run the same command
3. It will resume from where it left off

## Memory Usage

- **Per video**: ~100-500 MB depending on video length
- **Output file size**: ~4 KB per frame (e.g., 1000 frames ≈ 4 MB)
- **Total dataset**: ~1-2 GB for all 288 videos
