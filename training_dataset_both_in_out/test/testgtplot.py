"""
Test GT 22x6 Data Visualization

Uses BVH hierarchy to calculate bone lengths.
Matches the forward kinematics approach from cam1local6dfromquat.py
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
GT_DIR = f'{BASE_DIR}/training_dataset_both_in_out/gt_22_6_nosync'
BVH_FILE = f'{BASE_DIR}/totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh'
CALIBRATION_FILE = f'{BASE_DIR}/totalcapture_dataset/calibration_params.json'

# Joint names matching GT 22x6 format
JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

# Parent indices for each joint
PARENT_INDICES = [
    -1, 0, 1, 2, 3, 4, 5,  # Spine chain
    4, 7, 8, 9,            # Right arm
    4, 11, 12, 13,         # Left arm  
    0, 15, 16,             # Right leg
    0, 18, 19,             # Left leg
]

OFFSET_SCALE = 0.0254


def load_calibration():
    """Load camera calibration parameters."""
    with open(CALIBRATION_FILE, 'r') as f:
        calib = json.load(f)
    cameras = {}
    for cam in calib['cameras']:
        cameras[cam['camera_id']] = {
            'rotation': np.array(cam['rotation_matrix']),
            'translation': np.array(cam['translation_vector'])
        }
    return cameras


def parse_bvh_structure(file_path):
    """Parses BVH to get the hierarchy and local offsets"""
    hierarchy = {}
    joint_names = []
    stack = []
    
    with open(file_path, 'r') as f:
        parent = None
        for line in f:
            line = line.strip()
            if line.startswith('ROOT') or line.startswith('JOINT'):
                name = line.split()[1]
                joint_names.append(name)
                hierarchy[name] = {'parent': parent, 'children': [], 'offset': np.zeros(3)}
                if parent:
                    hierarchy[parent]['children'].append(name)
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
    """Convert 6D rotation back to 3x3 matrix using Gram-Schmidt."""
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def calculate_pose(gt_frame, skeleton, R_cam_inv):
    """
    Calculate joint positions using forward kinematics.
    
    The GT data stores:
    - Index 0: Camera-transformed hip position (scaled to meters)
    - Index 1: Camera-oriented hip global 6D (R_cam @ R_hip_world)
    - Indices 2-21: World-coordinate local 6D (R_parent^T @ R_child)
    
    To reconstruct in camera coordinates for visualization:
    1. Use hip position directly (already in camera coords)
    2. Use hip rotation directly (already camera-oriented) 
    3. For children: R_global = R_parent @ R_local, then offset
    """
    positions = {}
    global_rotations = {}
    
    # Hip position (already in camera coords, meters)
    hip_pos = gt_frame[0, :3]
    
    # Hip rotation (already camera-oriented)
    hip_6d = gt_frame[1, :]
    hip_rotation = rotation_6d_to_matrix(hip_6d)
    
    positions['Hips'] = hip_pos
    global_rotations['Hips'] = hip_rotation
    
    # Child joints
    for i, joint_name in enumerate(JOINT_NAMES[1:], start=1):
        parent_idx = PARENT_INDICES[i]
        parent_name = JOINT_NAMES[parent_idx]
        
        # Get local rotation (indices 2-21)
        local_6d = gt_frame[i + 1, :]
        local_rotation = rotation_6d_to_matrix(local_6d)
        
        # Get parent's global rotation (which is already camera-oriented for hip,
        # and we propagate that through the chain)
        R_parent = global_rotations[parent_name]
        
        # Global rotation: R_global = R_parent @ R_local
        R_global = R_parent @ local_rotation
        global_rotations[joint_name] = R_global
        
        # Get offset from BVH (in BVH coordinates)
        if joint_name in skeleton:
            offset_bvh = skeleton[joint_name]['offset'] * OFFSET_SCALE
        else:
            offset_bvh = np.zeros(3)
        
        # The offset needs to be rotated by the parent's rotation
        # Since parent rotation is camera-oriented, this places child in camera coords
        rotated_offset = R_parent @ offset_bvh
        positions[joint_name] = positions[parent_name] + rotated_offset
    
    return positions


def main():
    print("=" * 60)
    print("GT 22x6 Data Visualization")
    print("=" * 60)
    
    # Parse BVH
    print("Parsing BVH file...")
    skeleton = parse_bvh_structure(BVH_FILE)
    
    # Load calibration
    cameras = load_calibration()
    
    # Find GT files
    gt_files = list(Path(GT_DIR).rglob("*.npy"))
    random_file = random.choice(gt_files)
    
    # Extract camera ID from filename
    filename = random_file.stem
    cam_id = int(filename.split('cam')[1])
    
    print(f"Loading: {random_file}")
    print(f"Camera: {cam_id}")
    
    # Get camera params
    R_cam = cameras[cam_id]['rotation']
    R_cam_inv = R_cam.T
    
    # Load GT data
    gt_data = np.load(random_file)
    num_frames = gt_data.shape[0]
    print(f"Frames: {num_frames}")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    def update(val):
        frame_idx = int(val)
        ax.clear()
        
        positions = calculate_pose(gt_data[frame_idx], skeleton, R_cam_inv)
        
        # Plot in camera coordinates
        # Camera: X=right, Y=down, Z=forward (depth)
        # For 3D plot: X=right, Y=depth (forward), Z=up
        # So: plot_x = cam_x, plot_y = cam_z, plot_z = -cam_y
        
        for joint in JOINT_NAMES:
            if joint in positions:
                p = positions[joint]
                ax.scatter(p[0], p[2], -p[1], c='red', s=30)
        
        # Draw bones
        for i, joint in enumerate(JOINT_NAMES):
            if i == 0:
                continue
            parent = JOINT_NAMES[PARENT_INDICES[i]]
            if joint in positions and parent in positions:
                p1, p2 = positions[parent], positions[joint]
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], 'b-', lw=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z (depth)')  
        ax.set_zlabel('-Y (up)')
        ax.set_title(f'Frame {frame_idx}/{num_frames-1} - Camera {cam_id}')
        
        # Axis limits in meters
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 5)
        ax.set_zlim(-2, 2)
        
        fig.canvas.draw_idle()
    
    update(0)
    
    ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    slider.on_changed(update)
    
    plt.show()


if __name__ == "__main__":
    main()
