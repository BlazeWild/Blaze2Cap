"""
GT (Nosync) vs BlazePose (Matched) Comparison

Plots:
1. GT Data from 'gt_22_6_nosync' (Blue)
   - Computed via Forward Kinematics from 22x6 format
   - Space: Camera Coordinates (Hip Position is in Camera Coords)
   
2. BlazePose Data from 'blazepose_coordinates_matched' (Red)
   - Columns 0,1,2 (World X, Y, Z)
   - Space: Rotated World Coordinates (X, Z_depth, -Y_up)

Note: Since GT is in Camera Space and BlazePose is in World Space (Rotated), 
they will likely be offset from each other. Functional alignment requires 
camera calibration application.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import random
import json

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
GT_NOSYNC_DIR = f'{BASE_DIR}/training_dataset_both_in_out/gt_22_6_nosync'
BP_MATCHED_DIR = f'{BASE_DIR}/training_dataset_both_in_out/blazepose_coordinates_matched'
BVH_FILE = f'{BASE_DIR}/totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh'
CALIBRATION_FILE = f'{BASE_DIR}/totalcapture_dataset/calibration_params.json'

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
    (0, 1), (0, 2),  # Nose to Ears
    (3, 4),          # Shoulders
    (3, 15), (4, 16), (15, 16), # Torso 
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13), # Left Arm
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14), # Right Arm
    (15, 17), (17, 19), (19, 21), (19, 23), # Left Leg
    (16, 18), (18, 20), (20, 22), (20, 24), # Right Leg
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

def calculate_gt_pose(gt_frame, skeleton):
    """
    Calculate GT joint positions (Forward Kinematics).
    GT Frame: [HipPos(3), HipRot(6)+pad, ChildLocal(20*6)]
    """
    positions = {}
    global_rotations = {}
    
    # Hip (Camera Coords)
    hip_pos = gt_frame[0, :3]
    hip_6d = gt_frame[1, :]
    hip_rotation = rotation_6d_to_matrix(hip_6d)
    
    positions['Hips'] = hip_pos
    global_rotations['Hips'] = hip_rotation
    
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

def main():
    print("=" * 60)
    print("GT (Nosync) vs BlazePose (Matched) Comparison")
    print("=" * 60)
    
    skeleton = parse_bvh_structure(BVH_FILE)
    
    # Find list of GT files
    gt_files = list(Path(GT_NOSYNC_DIR).rglob("*.npy"))
    if not gt_files:
        print("No GT files found.")
        return

    # Pick random GT file
    gt_path = random.choice(gt_files)
    
    # Construct matching BlazePose path
    # GT Path: .../gt_22_6_nosync/S1/acting1/gt_S1_acting1_cam1.npy
    # BP Path: .../blazepose_coordinates_matched/S1/acting1/blaze_S1_acting1_cam1.npy
    
    rel_path = gt_path.relative_to(GT_NOSYNC_DIR)
    # Filename conversion: gt_X -> blaze_X
    bp_filename = gt_path.name.replace('gt_', 'blaze_')
    bp_path = Path(BP_MATCHED_DIR) / rel_path.parent / bp_filename
    
    print(f"GT File: {gt_path}")
    print(f"BP File: {bp_path}")
    
    if not bp_path.exists():
        print("Matching BlazePose file not found.")
        return
        
    gt_data = np.load(gt_path)
    bp_data = np.load(bp_path)
    
    frames = min(gt_data.shape[0], bp_data.shape[0])
    print(f"Comparison Frames: {frames}")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    def update(val):
        frame_idx = int(val)
        ax.clear()
        
        # --- PLOT GT (Blue) ---
        gt_frame = gt_data[frame_idx]
        gt_positions = calculate_gt_pose(gt_frame, skeleton)
        
        # Transform for Visualization (Z-up)
        # GT is Camera Coords: X_right, Y_down, Z_forward
        # Plot: X=X, Y=Z, Z=-Y
        
        for joint in GT_JOINT_NAMES:
            if joint in gt_positions:
                p = gt_positions[joint]
                ax.scatter(p[0], p[2], -p[1], c='blue', s=20)
                
        for i, joint in enumerate(GT_JOINT_NAMES):
            if i == 0: continue
            parent = GT_JOINT_NAMES[GT_PARENT_INDICES[i]]
            if joint in gt_positions and parent in gt_positions:
                p1, p2 = gt_positions[parent], gt_positions[joint]
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], 'b-', lw=1.5)

        # --- PLOT BLAZEPOSE (Red) ---
        bp_frame = bp_data[frame_idx]
        bp_xyz = bp_frame[:, :3] # Already rotated world coords?
        
        # NOTE: BP Matched is:
        # 0: X
        # 1: Z (depth)
        # 2: -Y (up)
        # So we can plot directly to X, Y, Z of the visualizer which uses Z-up convention?
        # Let's assume the channels map directly to X, Y(depth), Z(up) for plotting
        
        for i in range(25):
            x, y, z = bp_xyz[i]
            # Plot directly
            ax.scatter(x, y, z, c='red', s=20)
            
        for (i, j) in BLAZEPOSE_BONES:
            p1 = bp_xyz[i]
            p2 = bp_xyz[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', lw=1.5)
            
        ax.set_title(f"Frame {frame_idx} | Blue=GT(CamSpace) Red=BP(WorldRotated)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y (Depth)")
        ax.set_zlabel("Z (Up)")
        
        # Find visual limits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 5) # Depth usually positive
        ax.set_zlim(-2, 2)
        
    update(0)
    
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=0, valstep=1)
    slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    main()
