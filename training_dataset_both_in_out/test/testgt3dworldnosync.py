"""
Test GT 3D World Visualization (Nosync)

Visualizes GT data from `gt_22_6_nosync`.
This data contains:
- Hip Position (Camera Coordinates)
- Hip Rotation (6D)
- Child Joint Rotations (6D, Local)

Reconstructs using Forward Kinematics.
Plots in Camera Coordinate Frame (X right, Y depth, Z up).
"""

import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os
import argparse
import random

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
DATA_DIR = os.path.join(BASE_DIR, 'training_dataset_both_in_out/gt_22_6_nosync')
BVH_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='Filename to visualize (e.g. gt_S1_acting1_cam1.npy)')
args = parser.parse_args()

# ==========================================
# SKELETON DEF
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
    target_path = None
    
    if args.filename:
        found_files = list(Path(DATA_DIR).rglob(args.filename))
        if found_files:
            target_path = found_files[0]
        else:
            print(f"Error: File {args.filename} not found in {DATA_DIR}")
            return
    else:
        # Fallback to random
        all_files = list(Path(DATA_DIR).rglob("*.npy"))
        if not all_files:
            print(f"No .npy files found in {DATA_DIR}")
            return
        target_path = random.choice(all_files)
        print("No filename specified, picking random.")

    print(f"Selected: {target_path}")
    
    data = np.load(target_path) 
    print(f"Data shape: {data.shape}")
    
    skeleton = parse_bvh_structure(BVH_FILE)
    num_frames = data.shape[0]
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    def update(val):
        frame_idx = int(slider.val)
        ax.clear()
        
        gt_frame = data[frame_idx]
        gt_positions = calculate_gt_pose(gt_frame, skeleton)
        
        # Plot GT
        # Data is in Camera Coords (X right, Y depth, Z up from calibration?)
        # Wait, nosync data from unsynced.py comes from cam1local6dfromquat logic.
        # It's usually X right, Y down/depth, Z up?
        # Let's assume typical plot: X=X, Y=Z(depth), Z=-Y(up) or some variation.
        # But 'testgtplot.py' used: scatter(p[0], p[2], -p[1])
        # Let's try that.
        
        for joint in GT_JOINT_NAMES:
            if joint in gt_positions:
                p = gt_positions[joint]
                ax.scatter(p[0], p[2], -p[1], c='blue', s=20) # X, Z, -Y
                
        for i, joint in enumerate(GT_JOINT_NAMES):
            if i == 0: continue
            parent = GT_JOINT_NAMES[GT_PARENT_INDICES[i]]
            if joint in gt_positions and parent in gt_positions:
                p1, p2 = gt_positions[parent], gt_positions[joint]
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], 'b-', lw=1.5)
        
        ax.set_title(f"Frame {frame_idx} | {target_path.name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y (Depth)")
        ax.set_zlabel("Z (Up)")
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 5)
        ax.set_zlim(-2, 2)
        
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
    slider.on_changed(update)
    
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
