# BlazePose vs GT Local 6D Visualization

This script visualizes BlazePose 3D keypoints alongside Ground Truth skeleton computed using Local 6D rotations.

## Data Sources

### BlazePose Data
- **File**: `training_dataset_both_in_out/blazepose_25_7_nosync/S1/acting1/blaze_S1_acting1_cam1.npy`
- **Shape**: `(4115, 25, 7)` - 4115 frames, 25 keypoints, 7 channels
- **Channels Used**: 0, 1, 2 (world x, y, z coordinates)

### GT Data
- **BVH File**: `totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh` - Skeleton hierarchy and offsets
- **Quaternion File**: `totalcapture_dataset/positions/S1/acting1/gt_skel_gbl_ori.txt` - Global orientations
- **Motion File**: `totalcapture_dataset/bvh/testbonel/motion.txt` - Root positions

## GT Processing Pipeline

### Step 1: Parse BVH Skeleton Structure
Extract joint hierarchy, parent-child relationships, and local offsets from BVH file.

### Step 2: Load Global Quaternions
TotalCapture quaternions are in **[x, y, z, w]** format.

### Step 3: Convert Quaternion to Rotation Matrix
```python
def quaternion_to_matrix(q):
    x, y, z, w = q
    # Normalize quaternion
    # Build 3x3 rotation matrix
```

### Step 4: Compute Local 6D Rotations

#### For Root (Hip):
- Use **global rotation** directly
- Convert to 6D: Take first two columns of rotation matrix

```python
6D = [R[:, 0], R[:, 1]]  # First two columns concatenated
```

#### For Child Joints:
- Compute **local rotation** relative to parent
```python
R_local = R_parent.T @ R_child_global
```
- Convert local rotation to 6D representation

### Step 5: Forward Kinematics
Reconstruct joint positions from rotations and offsets:

```python
for each joint:
    if root:
        position = root_position
        R_global = R_from_6d(global_6d)
    else:
        R_local = R_from_6d(local_6d)
        R_global = R_parent_global @ R_local
        position = parent_position + R_parent_global @ offset
```

### Step 6: Camera Transformation
Transform world positions to Camera 1 coordinates:

```python
P_camera = R_cam @ P_world + T_cam
```

**Camera 1 Parameters:**
```python
R_cam = [[-0.99713,  0.00504, -0.07554],
         [ 0.02217, -0.93461, -0.35498],
         [-0.07239, -0.35564,  0.93182]]
T_cam = [0.8205, 0.597, 5.336]
```

### Step 7: Unit Conversion
Convert from inches to meters:
```python
position_meters = position_camera * 0.0254
```

## BlazePose Axis Alignment

BlazePose uses different axis conventions than GT. To align:

### Original BlazePose Axes
- X: Horizontal
- Y: Vertical (up is negative)
- Z: Depth

### Rotation Applied (90° Pitch)
```python
plot_x = blaze_x           # X unchanged
plot_y = blaze_z           # Y becomes Z (depth)
plot_z = -blaze_y          # Z becomes -Y (flip vertical)
```

This rotates the skeleton 90° around the X-axis to stand upright like GT.

## Hip-Relative Positioning

Both skeletons are normalized to have hip at origin:

1. **Compute hip position** for current frame
2. **Subtract hip offset** from all joint positions
3. **Root Motion Toggle**:
   - **OFF**: Hip always at (0, 0, 0)
   - **ON**: Hip follows trajectory starting from origin

## 6D Rotation Representation

### Matrix to 6D
```python
def matrix_to_6d(R):
    return [R[:, 0], R[:, 1]]  # First two columns
```

### 6D to Matrix (Gram-Schmidt)
```python
def rotation_6d_to_matrix(r6d):
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - dot(b1, a2) * b1)
    b3 = cross(b1, b2)
    return [b1, b2, b3]  # Column stack
```

## Visualization Colors
- **RED**: BlazePose keypoints and bones
- **BLUE**: GT Local 6D skeleton

## Controls
- **Frame Slider**: Navigate through frames
- **Root Motion Checkbox**: Toggle hip root motion on/off
