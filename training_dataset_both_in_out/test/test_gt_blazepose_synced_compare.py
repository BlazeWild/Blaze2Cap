"""
GT vs BlazePose SYNCED Data Visualization

Target: S1 acting1 cam1
Folders:
- gt_synced (22x6, Deltas for Hip)
- blazepose_synced (25x7, Absolute World Coords in 0-2)

GT Logic:
- Accumulate Hip Position and Rotation from Index 0 and 1.
- Reset Accumulation when Delta is zero (Anchor Frame).
- Use Forward Kinematics for full skeleton.

BlazePose Logic:
- Plot Columns 0, 1, 2 directly.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
GT_SYNCED_DIR = f'{BASE_DIR}/training_dataset_both_in_out/gt_synced'
BP_SYNCED_DIR = f'{BASE_DIR}/training_dataset_both_in_out/blazepose_synced'
BVH_FILE = f'{BASE_DIR}/totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh'

SUBJECT = 'S1'
ACTION = 'acting1'
CAMERA = 'cam1'

# ==========================================
# GT SKELETON
# ==========================================
GT_JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

GT_PARENT_INDICES = [
    -1, 0, 1, 2, 3, 4, 5,
    4, 7, 8, 9,
    4, 11, 12, 13,
    0, 15, 16,
    0, 18, 19,
]

OFFSET_SCALE = 0.0254

# ==========================================
# BLAZEPOSE SKELETON
# ==========================================
BLAZEPOSE_BONES = [
    (0, 1), (0, 2), (3, 4), (3, 15), (4, 16), (15, 16), 
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13), 
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14), 
    (15, 17), (17, 19), (19, 21), (19, 23), 
    (16, 18), (18, 20), (20, 22), (20, 24), 
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_bvh_structure(file_path):
    hierarchy = {}
    stack = []
    with open(file_path, 'r') as f:
        parent = None
        for line in f:
            line = line.strip()
            if line.startswith('ROOT') or line.startswith('JOINT'):
                name = line.split()[1]
                hierarchy[name] = {'parent': parent, 'offset': np.zeros(3)}
                parent = name
                stack.append(name)
            elif line.startswith('OFFSET'):
                parts = line.split()
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                if stack:
                    hierarchy[stack[-1]]['offset'] = offset
            elif line.startswith('End Site'):
                parent = stack[-1]
            elif line == '}':
                if stack:
                    stack.pop()
                    parent = stack[-1] if stack else None
    return hierarchy

def rotation_6d_to_matrix(r6d):
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])

def calculate_gt_pose(gt_frame, skeleton, acc_pos, acc_rot):
    """
    Calculate GT joint positions using accumulated hip data + local rotations.
    """
    positions = {}
    global_rotations = {}
    
    positions['Hips'] = acc_pos
    global_rotations['Hips'] = acc_rot
    
    for i, joint_name in enumerate(GT_JOINT_NAMES[1:], start=1):
        parent_idx = GT_PARENT_INDICES[i]
        parent_name = GT_JOINT_NAMES[parent_idx]
        
        local_6d = gt_frame[i + 1, :]
        local_rotation = rotation_6d_to_matrix(local_6d)
        
        R_parent = global_rotations[parent_name]
        R_global = R_parent @ local_rotation
        global_rotations[joint_name] = R_global
        
        offset_bvh = np.zeros(3)
        if joint_name in skeleton:
            offset_bvh = skeleton[joint_name]['offset'] * OFFSET_SCALE
            
        rotated_offset = R_parent @ offset_bvh
        positions[joint_name] = positions[parent_name] + rotated_offset
        
    return positions

def is_anchor(gt_frame):
    # Anchor if hip delta pos and rot are all zeros
    return np.allclose(gt_frame[0, :], 0) and np.allclose(gt_frame[1, :], 0)

def main():
    print("=" * 60)
    print(f"GT vs BlazePose SYNCED Comparison for {SUBJECT}/{ACTION}/{CAMERA}")
    print("=" * 60)
    
    skeleton = parse_bvh_structure(BVH_FILE)
    
    # Paths
    gt_path = Path(GT_SYNCED_DIR) / SUBJECT / ACTION / f'gt_{SUBJECT}_{ACTION}_{CAMERA}.npy'
    bp_path = Path(BP_SYNCED_DIR) / SUBJECT / ACTION / f'blaze_{SUBJECT}_{ACTION}_{CAMERA}.npy'
    
    print(f"GT File: {gt_path}")
    print(f"BP File: {bp_path}")
    
    if not gt_path.exists() or not bp_path.exists():
        print("Error: Files not found.")
        return

    gt_data = np.load(gt_path)
    bp_data = np.load(bp_path)
    
    frames = min(gt_data.shape[0], bp_data.shape[0])
    print(f"Comparison Frames: {frames}")
    
    # --- Precompute GT Positions (Accumulation) ---
    print("Precomputing GT positions...")
    gt_positions_all = []
    acc_pos = np.zeros(3)
    acc_rot = np.eye(3)
    
    for i in range(frames):
        frame = gt_data[i]
        
        if is_anchor(frame):
            acc_pos = np.zeros(3)
            acc_rot = np.eye(3)
        else:
            delta_pos = frame[0, :3]
            delta_rot = rotation_6d_to_matrix(frame[1, :])
            
            acc_pos = acc_pos + delta_pos
            acc_rot = acc_rot @ delta_rot
            
        positions = calculate_gt_pose(frame, skeleton, acc_pos, acc_rot)
        gt_positions_all.append(positions)
    print("Done.")

    # --- Visualization ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    def update(val):
        frame_idx = int(val)
        ax.clear()
        
        # Plot GT (Blue)
        gt_pos = gt_positions_all[frame_idx]
        for joint in GT_JOINT_NAMES:
            if joint in gt_pos:
                p = gt_pos[joint]
                # Plot X, Z, -Y
                ax.scatter(p[0], p[2], -p[1], c='blue', s=20, alpha=0.6)
        
        for i, joint in enumerate(GT_JOINT_NAMES):
            if i == 0: continue
            parent = GT_JOINT_NAMES[GT_PARENT_INDICES[i]]
            if joint in gt_pos and parent in gt_pos:
                p1, p2 = gt_pos[parent], gt_pos[joint]
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], 'b-', lw=1.5, alpha=0.6)
                
        # Plot BlazePose (Red)
        bp_frame = bp_data[frame_idx]
        bp_xyz = bp_frame[:, :3]
        
        for i in range(25):
            x, y, z = bp_xyz[i]
            # Plot directly (assuming data is already X, Z, -Y)
            ax.scatter(x, y, z, c='red', s=20, alpha=0.6)
            
        for (i, j) in BLAZEPOSE_BONES:
            p1 = bp_xyz[i]
            p2 = bp_xyz[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', lw=1.5, alpha=0.6)
            
        is_anchor_frame = is_anchor(gt_data[frame_idx])
        anchor_txt = "[ANCHOR]" if is_anchor_frame else ""
        
        ax.set_title(f"Frame {frame_idx} {anchor_txt} | Blue=GT(CamSpace) Red=BP(World)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y (Depth)")
        ax.set_zlabel("Z (Up)")
        
        # Set limits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 5)
        ax.set_zlim(-2, 2)
        
    update(0)
    
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=0, valstep=1)
    slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    main()
