# BlazePose 2D Screen Visualization

This script visualizes BlazePose 2D screen coordinates with hip-relative delta accumulation.

## Data Source
- **File**: `training_dataset_both_in_out/blazepose_25_7_nosync/S1/acting1/blaze_S1_acting1_cam1.npy`
- **Shape**: `(4115, 25, 7)` - 4115 frames, 25 keypoints, 7 channels
- **Channels Used**: 3 and 4 (screen x, y in 0-1 normalized range)

## Coordinate Transformation

### BlazePose Original Coordinates
- Origin: **Top-left (0, 0)**
- Range: **(0, 0)** to **(1, 1)**
- Y-axis: Points **downward** (top=0, bottom=1)

### Target Coordinates (Center Origin)
- Origin: **Center (0, 0)**
- Range: **(-1, -1)** to **(1, 1)**
- Y-axis: Points **upward** (bottom=-1, top=1)

### Transformation Formula
```python
x_new = (x - 0.5) * 2   # Maps 0→-1, 0.5→0, 1→1
y_new = (0.5 - y) * 2   # Maps 0→1, 0.5→0, 1→-1 (Y-axis flip)
```

### Mapping Table
| BlazePose Position | Target Position |
|-------------------|-----------------|
| (0, 0) top-left | (-1, 1) top-left |
| (1, 0) top-right | (1, 1) top-right |
| (0, 1) bottom-left | (-1, -1) bottom-left |
| (1, 1) bottom-right | (1, -1) bottom-right |
| (0.5, 0.5) center | (0, 0) center |

## Hip-Relative Delta Accumulation for Plotting

### Hip Center Calculation
- **Left Hip Index**: 15 (original BlazePose index 23)
- **Right Hip Index**: 16 (original BlazePose index 24)
- **Hip Center**: Midpoint of left and right hip

### Anchor Frame (Frame 0)
At the anchor frame (frame 0):
- **Hip position**: Translated to **(0, 0)** origin
- **Delta**: **(0, 0)** - no movement from previous frame
- All keypoints are relative to hip center

```
anchor_frame:
    hip_center = (0, 0)
    delta = (0, 0)
    all_keypoints = transformed_keypoints - original_hip_position
```

### Subsequent Frames (Frame N > 0)
For every frame after anchor:
1. **Calculate delta**: Movement from previous frame (in transformed coordinates)
2. **Add delta**: Apply delta to previous frame's positions
3. **Accumulate**: Motion builds up frame by frame

```
frame_n:
    delta = transformed[n] - transformed[n-1]  # Per-frame movement
    keypoints[n] = keypoints[n-1] + delta       # Accumulate motion
```

### Example Walkthrough
```
Frame 0 (Anchor):
    Original hip: (0.5, 0.6)  →  Hip at: (0, 0)
    Delta: (0, 0)
    All points shifted by (-0.5, -0.6)

Frame 1:
    Delta = frame1 - frame0 = (0.01, -0.02)
    Keypoints = frame0_keypoints + (0.01, -0.02)
    # Hip moves slightly from origin

Frame 2:
    Delta = frame2 - frame1 = (0.005, 0.01)
    Keypoints = frame1_keypoints + (0.005, 0.01)
    # Motion continues to accumulate
```

### Key Points
- **Anchor frame** establishes the reference with hip at origin
- **Delta** represents frame-to-frame movement only
- **Accumulation** preserves relative motion while centering the skeleton
- Result: Skeleton stays centered but shows limb movements naturally

## 25 Keypoints (After Removing Face Details)
Removed original BlazePose indices: 1, 2, 3, 4, 5, 6, 9, 10

| New Index | Original Index | Keypoint Name |
|-----------|---------------|---------------|
| 0 | 0 | nose |
| 1 | 7 | left_ear |
| 2 | 8 | right_ear |
| 3 | 11 | left_shoulder |
| 4 | 12 | right_shoulder |
| 5 | 13 | left_elbow |
| 6 | 14 | right_elbow |
| 7 | 15 | left_wrist |
| 8 | 16 | right_wrist |
| 9 | 17 | left_pinky |
| 10 | 18 | right_pinky |
| 11 | 19 | left_index |
| 12 | 20 | right_index |
| 13 | 21 | left_thumb |
| 14 | 22 | right_thumb |
| 15 | 23 | left_hip |
| 16 | 24 | right_hip |
| 17 | 25 | left_knee |
| 18 | 26 | right_knee |
| 19 | 27 | left_ankle |
| 20 | 28 | right_ankle |
| 21 | 29 | left_heel |
| 22 | 30 | right_heel |
| 23 | 31 | left_foot_index |
| 24 | 32 | right_foot_index |
