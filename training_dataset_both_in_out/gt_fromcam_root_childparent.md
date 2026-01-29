# Ground Truth Camera-Space Reconstruction (Root-Child/Parent) Pipeline

This document details the data processing pipeline implemented in `test_cam_6d_gt.py`. This script serves as a proof-of-concept for training models in **Camera Space** using **6D Rotation Representations**, ensuring that the data pipeline is mathematically consistent and reversible.

**CRITICAL NOTE**: The entire pipeline operates purely on matrix transformations defined by the extrinsic calibration parameters. **No manual coordinate axis swapping (e.g., flipping Y and Z) or ad-hoc flipping was performed.**

## Detailed Process Steps

### 1. BVH Skeleton Parsing
**Objective**: Extract the static skeletal structure (bone lengths).
- **Input File**: `totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh`
- **Logic**: The script reads the BVH hierarchy to find the `OFFSET` vector for each joint.
- **Why**: Deep learning models often predict rotations, but to visualize or compute losses on 3D positions, we need the fixed bone lengths to perform Forward Kinematics (FK).
- **Scale**: Offsets are scaled from inches to meters ($ \times 0.0254 $).

### 2. Data Loading (Ground Truth)
**Objective**: Load the raw standard-truth motion data.
- **Positions**: `gt_skel_gbl_pos.txt` (Global positions in World Frame, inches).
- **Orientations**: `gt_skel_gbl_ori.txt` (Global quaternions in World Frame, format $[x, y, z, w]$).
- **Subject**: S1 / Freestyle3.

### 3. Camera Calibration Loading
**Objective**: Retrieve the Extrinsic Matrix for the target camera.
- **Input File**: `calibration_params.json`
- **Target**: Camera 6.
- **Parameters**: 
    - Rotation Matrix ($R_{ext} \in \mathbb{R}^{3 \times 3}$)
    - Translation Vector ($T_{ext} \in \mathbb{R}^{3}$)

### 4. Transformation to Camera Coordinate System
**Objective**: Move all "World" data into the "Camera" frame of reference.
This is the most critical step. Instead of learning in an arbitrary world frame, we prepare the data as seen by the camera.

**Position Transformation**:
$$ P_{camera} = R_{ext} \cdot P_{world} + T_{ext} $$
*Applied to the Root (Hips) position.*

**Orientation Transformation**:
$$ R_{global\_camera} = R_{ext} \cdot R_{global\_world} $$
*Applied to all 21 joints. This rotates the entire orientation frame of reference to align with the camera.*

### 5. Hierarchical Decomposition (Global $\to$ Local)
**Objective**: Convert absolute orientations into relative (local) rotations.
Generative models typically predict local rotations relative to the parent joint, as this makes the motion invariant to global position/rotation.

**Logic**:
For any child joint $C$ and its parent $P$:
$$ R_{child\_global} = R_{parent\_global} \cdot R_{child\_local} $$
Therefore, we solve for the local rotation:
$$ R_{child\_local} = (R_{parent\_global})^T \cdot R_{child\_global} $$

*Note: For the Root (Hips), the "Parent" is the Camera Frame (Identity), so $R_{local} = R_{global}$.*

### 6. Continuous 6D Rotation Representation
**Objective**: Convert 3x3 matrices to a representation suitable for neural networks.
- **Conversion**: The 3x3 rotation matrix ($R_{local}$) is converted to a 6D vector ($r_{6d}$) by taking the **first two columns**.
- **Reversion (Verification)**: The 6D vector is converted back to a rotation matrix using Gram-Schmidt orthogonalization.
- **Purpose**: This step confirms that our data representation (which will be the label for training) contains all necessary rotational information to reconstruct the skeleton perfectly.

### 7. Full Reconstruction via Forward Kinematics (FK)
**Objective**: Verify the integrity of the entire pipeline by rebuilding the skeleton from the processed features.
The script reconstructs the global pose using **only** the Hips Start Position and the Local 6D Rotations.

**Recursive Algorithm**:
1.  **Initialize Root**: 
    $$ P_{root} = \text{Target Position (Camera Frame)} $$
    $$ R_{root\_global} = \text{Reconstructed 6D Rotation} $$
2.  **Propagate to Children**:
    $$ R_{child\_global} = R_{parent\_global} \cdot R_{child\_local(\text{from } 6D)} $$
    $$ P_{child} = P_{parent} + R_{parent\_global} \cdot \vec{\text{Offset}}_{child} $$

### 8. Visualization
**Objective**: Visual Proof.
- The script uses `matplotlib` 3D plotting to render the reconstructed skeleton.
- **Success Criteria**: The skeleton appears natural, connected, and moves according to the "Freestyle3" action, purely within the Camera 6 coordinate limits.
