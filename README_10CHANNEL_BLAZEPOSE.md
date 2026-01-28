# BlazePose 10-Channel Landmark Extraction

## Overview

This project extracts 3D pose landmarks from TotalCapture dataset videos using Google MediaPipe BlazePose, outputting a comprehensive **10-channel representation** for each of the 33 body landmarks per frame.

## 10-Channel Format Specification

Each landmark contains **10 channels** of data per frame:

| Channel | Name | Description | Value Range | Units |
|---------|------|-------------|-------------|-------|
| **0** | World X | 3D world X coordinate (left/right) | Real numbers | meters |
| **1** | World Y | 3D world Y coordinate (up/down) | Real numbers | meters |
| **2** | World Z | 3D world Z coordinate (forward/backward) | Real numbers | meters |
| **3** | World Visibility | Confidence that landmark is visible in 3D space | 0.0 - 1.0 | probability |
| **4** | Anchor Flag | Tracking continuity (0=new/start, 1=continuous) | 0.0 or 1.0 | boolean |
| **5** | Screen X | 2D normalized screen X coordinate | 0.0 - 1.0 | normalized |
| **6** | Screen Y | 2D normalized screen Y coordinate | 0.0 - 1.0 | normalized |
| **7** | Screen Z | Relative depth from camera | Real numbers | relative |
| **8** | Screen Visibility | Confidence that landmark is visible on screen | 0.0 - 1.0 | probability |
| **9** | Anchor Backup | Duplicate of channel 4 for redundancy | 0.0 or 1.0 | boolean |

### Channel Details

#### World 3D Coordinates (Channels 0-2)
- **Origin**: Centered at the subject's hips (midpoint between hip landmarks)
- **Coordinate System**: 
  - X: Left (-) to Right (+)
  - Y: Down (-) to Up (+)
  - Z: Forward (+) to Backward (-)
- **Units**: Meters (real-world scale)
- **Use Case**: Motion analysis, 3D pose reconstruction, biomechanics

#### World Visibility (Channel 3)
- Probability that the landmark exists and is visible in 3D space
- `1.0` = Highly confident
- `0.0` = Not visible/occluded
- Use threshold (e.g., `> 0.5`) to filter unreliable landmarks

#### Anchor Flag (Channels 4, 9)
- **Value 0.0**: New detection (subject just appeared or tracking was lost)
- **Value 1.0**: Continuous tracking from previous frame
- **Purpose**: Identify temporal discontinuities in tracking
- **Channel 9** is a backup copy of channel 4

#### Screen 2D Coordinates (Channels 5-6)
- **Normalized** to image dimensions: `[0, 1]` range
- `(0, 0)` = Top-left corner
- `(1, 1)` = Bottom-right corner
- **Use Case**: 2D pose visualization, overlay on video frames

#### Screen Depth (Channel 7)
- Relative depth from camera (not absolute distance)
- Smaller values = Closer to camera
- **Note**: Not calibrated to real-world units

#### Screen Visibility (Channel 8)
- Probability that the landmark is visible in the 2D image
- Similar to World Visibility but for 2D screen space

## Output File Structure

### Directory Hierarchy

```
everything_from_blazepose/
├── S1/
│   ├── acting1/
│   │   ├── blaze_S1_acting1_cam1.npy
│   │   ├── blaze_S1_acting1_cam2.npy
│   │   ├── blaze_S1_acting1_cam3.npy
│   │   ├── blaze_S1_acting1_cam4.npy
│   │   ├── blaze_S1_acting1_cam5.npy
│   │   ├── blaze_S1_acting1_cam6.npy
│   │   ├── blaze_S1_acting1_cam7.npy
│   │   └── blaze_S1_acting1_cam8.npy
│   ├── acting2/
│   │   └── blaze_S1_acting2_cam*.npy
│   ├── acting3/
│   ├── freestyle1/
│   ├── freestyle2/
│   ├── freestyle3/
│   ├── rom1/
│   ├── rom2/
│   ├── rom3/
│   ├── walking1/
│   ├── walking2/
│   └── walking3/
├── S2/
│   └── (same structure)
├── S3/
│   └── (same structure)
├── S4/
│   └── (same structure)
└── S5/
    └── (same structure)
```

### File Naming Convention

**Pattern**: `blaze_{SUBJECT}_{ACTION}_cam{CAMERA}.npy`

**Examples**:
- `blaze_S1_acting1_cam1.npy` → Subject 1, Action "acting1", Camera 1
- `blaze_S3_walking2_cam5.npy` → Subject 3, Action "walking2", Camera 5

### Array Format

Each `.npy` file contains a NumPy array with:

```python
Shape: (FRAMES, 33, 10)
Dtype: float32
```

- **FRAMES**: Total number of video frames (varies per video, typically ~4000-6000)
- **33**: Number of BlazePose landmarks (fixed)
- **10**: Number of channels per landmark

**Memory per file**: Approximately `FRAMES × 33 × 10 × 4 bytes`
- Example: 4000 frames = ~5.3 MB

## BlazePose 33 Landmarks

MediaPipe BlazePose detects 33 landmarks:

```
Index  Landmark Name
-----  -------------
0      Nose
1      Left Eye (Inner)
2      Left Eye
3      Left Eye (Outer)
4      Right Eye (Inner)
5      Right Eye
6      Right Eye (Outer)
7      Left Ear
8      Right Ear
9      Mouth (Left)
10     Mouth (Right)
11     Left Shoulder
12     Right Shoulder
13     Left Elbow
14     Right Elbow
15     Left Wrist
16     Right Wrist
17     Left Pinky
18     Right Pinky
19     Left Index
20     Right Index
21     Left Thumb
22     Right Thumb
23     Left Hip
24     Right Hip
25     Left Knee
26     Right Knee
27     Left Ankle
28     Right Ankle
29     Left Heel
30     Right Heel
31     Left Foot Index
32     Right Foot Index
```

## Loading and Processing Data

### Basic Loading

```python
import numpy as np

# Load a single file
data = np.load('everything_from_blazepose/S1/acting1/blaze_S1_acting1_cam1.npy')

# Shape: (frames, 33, 10)
print(f"Shape: {data.shape}")
print(f"Frames: {data.shape[0]}")
print(f"Landmarks: {data.shape[1]}")
print(f"Channels: {data.shape[2]}")
```

### Extracting Specific Channels

```python
# Extract world 3D coordinates for all frames, all landmarks
world_3d = data[:, :, 0:3]  # Shape: (frames, 33, 3) - X, Y, Z

# Extract world visibility
world_vis = data[:, :, 3]  # Shape: (frames, 33)

# Extract screen 2D coordinates
screen_2d = data[:, :, 5:7]  # Shape: (frames, 33, 2) - X, Y

# Extract anchor flags
anchor_flags = data[:, :, 4]  # Shape: (frames, 33)
```

### Extracting Specific Landmarks

```python
# Get right wrist (landmark 16) across all frames
right_wrist = data[:, 16, :]  # Shape: (frames, 10)

# Get 3D position of right wrist
right_wrist_3d = data[:, 16, 0:3]  # Shape: (frames, 3)

# Get visibility of right wrist
right_wrist_visibility = data[:, 16, 3]  # Shape: (frames,)
```

### Filtering by Visibility

```python
# Filter frames where nose (landmark 0) is visible
nose_visibility = data[:, 0, 3]
visible_frames = data[nose_visibility > 0.5]  # Only frames with nose visible

# Count missing detections
missing_frames = np.sum(nose_visibility < 0.5)
print(f"Missing detections: {missing_frames}/{len(data)}")
```

### Finding Tracking Discontinuities

```python
# Identify where tracking was lost and restarted
anchor_flags = data[:, 0, 4]  # Use any landmark's anchor flag
discontinuities = np.where(anchor_flags == 0.0)[0]
print(f"Tracking restarted at frames: {discontinuities}")
```

### Computing Distances

```python
# Compute distance between shoulders across all frames
left_shoulder = data[:, 11, 0:3]   # Shape: (frames, 3)
right_shoulder = data[:, 12, 0:3]  # Shape: (frames, 3)

shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
print(f"Average shoulder width: {np.mean(shoulder_distance):.3f} meters")
```

### Batch Processing Multiple Files

```python
import glob
import os

# Load all videos for a subject/action
pattern = 'everything_from_blazepose/S1/acting1/*.npy'
files = sorted(glob.glob(pattern))

all_data = {}
for filepath in files:
    camera = os.path.basename(filepath).split('_')[-1].replace('.npy', '')
    all_data[camera] = np.load(filepath)
    print(f"Loaded {camera}: {all_data[camera].shape}")

# Access data by camera
cam1_data = all_data['cam1']
cam2_data = all_data['cam2']
```

### Handling Missing Data

```python
# Check for frames with zero values (no detection)
def has_detection(frame_data):
    """Check if frame has valid detection (not all zeros)"""
    return np.any(frame_data != 0.0)

# Count valid frames
valid_frames = [has_detection(data[i]) for i in range(len(data))]
valid_count = sum(valid_frames)
print(f"Valid frames: {valid_count}/{len(data)}")

# Get only valid frames
valid_data = data[[has_detection(data[i]) for i in range(len(data))]]
```

## Visualization Examples

### 3D Skeleton Plot

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define skeleton connections
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 20), (16, 22),  # Right arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
]

def plot_3d_skeleton(frame_data, frame_idx=0):
    """Plot 3D skeleton for a single frame"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get 3D coordinates
    coords = frame_data[frame_idx, :, 0:3]
    visibility = frame_data[frame_idx, :, 3]
    
    # Plot landmarks
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c=visibility, cmap='viridis', s=50)
    
    # Plot connections
    for start, end in CONNECTIONS:
        if visibility[start] > 0.5 and visibility[end] > 0.5:
            ax.plot([coords[start, 0], coords[end, 0]],
                   [coords[start, 1], coords[end, 1]],
                   [coords[start, 2], coords[end, 2]], 'b-')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Frame {frame_idx}')
    plt.show()

# Example usage
data = np.load('everything_from_blazepose/S1/acting1/blaze_S1_acting1_cam1.npy')
plot_3d_skeleton(data, frame_idx=100)
```

### 2D Overlay on Video

```python
import cv2

def overlay_2d_skeleton(video_path, npy_path, output_path):
    """Overlay 2D skeleton on original video"""
    cap = cv2.VideoCapture(video_path)
    data = np.load(npy_path)
    
    # Video writer setup
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(data):
            # Get 2D coordinates and visibility
            coords_2d = data[frame_idx, :, 5:7]  # Normalized [0, 1]
            visibility = data[frame_idx, :, 8]
            
            # Convert to pixel coordinates
            coords_px = coords_2d * [width, height]
            
            # Draw landmarks
            for i, (x, y) in enumerate(coords_px):
                if visibility[i] > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Draw connections
            for start, end in CONNECTIONS:
                if visibility[start] > 0.5 and visibility[end] > 0.5:
                    pt1 = tuple(coords_px[start].astype(int))
                    pt2 = tuple(coords_px[end].astype(int))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
```

## Running the Extraction

### Command

```bash
cd /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING
source /home/blaze/venvs/main/bin/activate

python3 totalcapture_dataset/extract_blazepose_10ch_gpu.py \
  --videos totalcapture_dataset/Videos \
  --output everything_from_blazepose \
  --workers 16 \
  --max_side 1920 \
  --model_complexity 2
```

### Options

| Option | Description | Default | Recommended |
|--------|-------------|---------|-------------|
| `--videos` | Path to input videos directory | `./Videos` | Your videos path |
| `--output` | Output directory for .npy files | `./everything_from_blazepose` | Any path |
| `--workers` | Number of parallel workers | 8 | 16 for 12-core CPU |
| `--max_side` | Max resolution (longest side in pixels) | 1920 | 1920 (FHD) |
| `--model_complexity` | Model accuracy (0=fast, 2=accurate) | 2 | 2 for best results |
| `--start-from` | Resume from specific subject/action | None | `s2 rom1` to resume |

### Verifying Output

```bash
# Verify extraction completed successfully
python3 totalcapture_dataset/extract_blazepose_10ch_gpu.py --verify

# Or manually check
python3 -c "
import numpy as np
data = np.load('everything_from_blazepose/S1/acting1/blaze_S1_acting1_cam1.npy')
print(f'Shape: {data.shape}')
print(f'Dtype: {data.dtype}')
print(f'Size: {data.nbytes / 1024 / 1024:.2f} MB')
"
```

## Performance

### Processing Speed
- **CPU**: TensorFlow Lite XNNPACK delegate (optimized SIMD)
- **Preprocessing**: AMD Radeon iGPU (EGL)
- **Throughput**: ~2-3 FPS per worker
- **Total Time**: ~2-3 hours for full dataset (5 subjects, 12 actions each, 8 cameras)

### Resource Usage
- **CPU**: 12 cores utilized
- **RAM**: ~8-12 GB (with 16 workers)
- **Disk Space**: ~20-30 GB for full dataset output

## Technical Notes

### MediaPipe Version
- **Version**: 0.10.14
- **API**: Classic `mp.solutions.pose.Pose` (not Tasks API)
- **Reason**: Tasks API has EGL compatibility issues with NVIDIA GPUs on Linux

### Data Precision
- **Type**: `float32` (32-bit floating point)
- **Precision**: ~7 decimal digits
- **Range**: Sufficient for motion analysis applications

### Missing Data Representation
- Frames with no detection: All 10 channels set to `0.0`
- Check `visibility` channels (3, 8) to identify missing data
- Use `anchor_flag` (channel 4) to detect tracking discontinuities

## Use Cases

### Motion Analysis
- Extract joint trajectories over time
- Compute joint angles and velocities
- Analyze gait patterns

### 3D Reconstruction
- Use multi-camera data for triangulation
- Reconstruct full 3D body pose
- Calibrate camera positions

### Activity Recognition
- Train classifiers on pose sequences
- Temporal pose feature extraction
- Action segmentation

### Biomechanics
- Measure body segment lengths
- Compute center of mass
- Analyze movement efficiency

## Troubleshooting

### Low Visibility Scores
- Check video quality and lighting
- Increase `min_detection_confidence` in script
- Use higher resolution input (`--max_side 1920`)

### High Memory Usage
- Reduce `--workers` count
- Process subjects sequentially
- Lower `--max_side` resolution

### Missing Files
- Check `--start-from` option to resume
- Verify input video directory structure
- Check disk space availability

## Citation

If you use this data, please cite:

```bibtex
@misc{blazepose10channel,
  title={BlazePose 10-Channel Landmark Extraction for TotalCapture Dataset},
  author={Your Name},
  year={2026},
  note={MediaPipe BlazePose-based extraction with 10-channel representation}
}
```

## License

This extraction code follows MediaPipe's Apache 2.0 license. The TotalCapture dataset has its own license terms.

## Contact

For questions or issues, please open an issue in the repository.
