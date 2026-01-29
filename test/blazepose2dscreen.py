"""
BlazePose 2D screen coordinates visualization.
Uses channels 3,4 (screen x,y in 0-1 range) from the 25x7 data.
Transforms to center-origin coordinates and applies hip-relative delta accumulation.
"""
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os

# ==========================================
# LOAD DATA
# ==========================================
data_file = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/training_dataset_both_in_out/blazepose_25_7_nosync/S1/acting1/blaze_S1_acting1_cam1.npy'
data = np.load(data_file)  # Shape: (frames, 25, 7)
print(f"Loaded data: {data.shape}")

# Extract channels 3,4 (screen x, y in 0-1 range)
# Shape: (frames, 25, 2)
screen_coords = data[:, :, 3:5].copy()
print(f"Screen coords shape: {screen_coords.shape}")

# ==========================================
# COORDINATE TRANSFORMATION
# ==========================================
# BlazePose: (0,0) top-left, (1,1) bottom-right
# Target: (0,0) center, (-1,-1) bottom-left, (1,1) top-right
# 
# Mapping:
#   BlazePose (0,0) top-left    -> Target (-1, 1) top-left
#   BlazePose (1,0) top-right   -> Target (1, 1) top-right
#   BlazePose (0,1) bottom-left -> Target (-1, -1) bottom-left
#   BlazePose (1,1) bottom-right-> Target (1, -1) bottom-right
#   BlazePose (0.5,0.5) center  -> Target (0, 0) center

def transform_to_center_origin(coords):
    """Transform from 0-1 range to -1 to 1 with center at origin.
    Bottom-left: (-1,-1), Top-right: (1,1)
    """
    transformed = np.zeros_like(coords)
    transformed[:, :, 0] = (coords[:, :, 0] - 0.5) * 2  # x: left(-1) to right(1)
    transformed[:, :, 1] = (0.5 - coords[:, :, 1]) * 2  # y: bottom(-1) to top(1), flip Y axis
    return transformed

# Transform all coordinates
transformed_coords = transform_to_center_origin(screen_coords)
print(f"Transformed coords range: x=[{transformed_coords[:,:,0].min():.3f}, {transformed_coords[:,:,0].max():.3f}], y=[{transformed_coords[:,:,1].min():.3f}, {transformed_coords[:,:,1].max():.3f}]")

# ==========================================
# HIP-RELATIVE DELTA ACCUMULATION
# ==========================================
# Hip index in 25-keypoint format (after removing indices 1,2,3,4,5,6,9,10)
# Original index 23 (left_hip) -> new index 15
# Original index 24 (right_hip) -> new index 16
# We'll use the midpoint of left and right hip as "hip center"
LEFT_HIP_IDX = 15
RIGHT_HIP_IDX = 16

def compute_hip_center(frame_coords):
    """Compute hip center as midpoint of left and right hip."""
    return (frame_coords[LEFT_HIP_IDX] + frame_coords[RIGHT_HIP_IDX]) / 2

# Compute hip-relative coordinates with delta accumulation
num_frames = len(transformed_coords)
hip_relative_coords = np.zeros_like(transformed_coords)

# Frame 0: translate all points so hip is at origin
frame0_hip = compute_hip_center(transformed_coords[0])
hip_relative_coords[0] = transformed_coords[0] - frame0_hip

# For subsequent frames: accumulate deltas
for i in range(1, num_frames):
    # Delta = current frame - previous frame (in original transformed space)
    delta = transformed_coords[i] - transformed_coords[i-1]
    # Add delta to previous hip-relative positions
    hip_relative_coords[i] = hip_relative_coords[i-1] + delta

print(f"Hip-relative coords computed for {num_frames} frames")

# ==========================================
# BLAZEPOSE BONE CONNECTIONS (25 keypoints)
# ==========================================
BLAZEPOSE_BONES = [
    # Face
    (0, 1),   # nose -> left_ear
    (0, 2),   # nose -> right_ear
    # Torso
    (3, 4),   # left_shoulder -> right_shoulder
    (3, 15),  # left_shoulder -> left_hip
    (4, 16),  # right_shoulder -> right_hip
    (15, 16), # left_hip -> right_hip
    # Left arm
    (3, 5),   # left_shoulder -> left_elbow
    (5, 7),   # left_elbow -> left_wrist
    (7, 9),   # left_wrist -> left_pinky
    (7, 11),  # left_wrist -> left_index
    (7, 13),  # left_wrist -> left_thumb
    # Right arm
    (4, 6),   # right_shoulder -> right_elbow
    (6, 8),   # right_elbow -> right_wrist
    (8, 10),  # right_wrist -> right_pinky
    (8, 12),  # right_wrist -> right_index
    (8, 14),  # right_wrist -> right_thumb
    # Left leg
    (15, 17), # left_hip -> left_knee
    (17, 19), # left_knee -> left_ankle
    (19, 21), # left_ankle -> left_heel
    (19, 23), # left_ankle -> left_foot_index
    (21, 23), # left_heel -> left_foot_index
    # Right leg
    (16, 18), # right_hip -> right_knee
    (18, 20), # right_knee -> right_ankle
    (20, 22), # right_ankle -> right_heel
    (20, 24), # right_ankle -> right_foot_index
    (22, 24), # right_heel -> right_foot_index
]

# ==========================================
# VISUALIZATION
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.15)

# Setup scatter and lines
scat = ax.scatter([], [], c='red', s=50)
lines = []
for _ in range(len(BLAZEPOSE_BONES)):
    line, = ax.plot([], [], 'r-', lw=2)
    lines.append(line)

# Axis limits
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('BlazePose 2D Screen - Hip Relative Delta Accumulation')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Draw origin cross
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

def update(val):
    frame = int(slider.val)
    coords = hip_relative_coords[frame]
    
    # Update scatter
    xs = coords[:, 0]
    ys = coords[:, 1]
    scat.set_offsets(np.column_stack([xs, ys]))
    
    # Update bones
    for i, (p1_idx, p2_idx) in enumerate(BLAZEPOSE_BONES):
        if p1_idx < len(coords) and p2_idx < len(coords):
            lx = [coords[p1_idx, 0], coords[p2_idx, 0]]
            ly = [coords[p1_idx, 1], coords[p2_idx, 1]]
            lines[i].set_data(lx, ly)
    
    # Show hip center position
    hip_center = compute_hip_center(coords)
    ax.set_title(f'Frame {frame} | Hip Center: ({hip_center[0]:.3f}, {hip_center[1]:.3f})')
    fig.canvas.draw_idle()

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
slider.on_changed(update)

# Initial draw
update(0)
plt.show()
