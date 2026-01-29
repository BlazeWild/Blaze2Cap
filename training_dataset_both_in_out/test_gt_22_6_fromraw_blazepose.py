"""
Test GT 22x6 From Raw (BlazePose-style Visualization)

Transforms GT TotalCapture skeleton (S1/Acting1) to Camera 1 coordinates 
and visualizes it in 3D, similar to test3dworld.py.
"""
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import json
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
DATA_POS_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/positions/S1/freestyle3/gt_skel_gbl_pos.txt')
DATA_ORI_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/positions/S1/freestyle3/gt_skel_gbl_ori.txt')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/calibration_params.json')

# Scale factor to convert from inches to meters (TotalCapture positions are in inches)
SCALE_FACTOR = 0.0254

# Joint names in order (21 joints)
JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

# Parent indices (from gt_22_6_unsynced.py)
PARENT_INDICES = [
    -1,  # Hips
    0,   # Spine -> Hips
    1,   # Spine1 -> Spine
    2,   # Spine2 -> Spine1
    3,   # Spine3 -> Spine2
    4,   # Neck -> Spine3
    5,   # Head -> Neck
    4,   # RightShoulder -> Spine3
    7,   # RightArm -> RightShoulder
    8,   # RightForeArm -> RightArm
    9,   # RightHand -> RightForeArm
    4,   # LeftShoulder -> Spine3
    11,  # LeftArm -> LeftShoulder
    12,  # LeftForeArm -> LeftArm
    13,  # LeftHand -> LeftForeArm
    0,   # RightUpLeg -> Hips
    15,  # RightLeg -> RightUpLeg
    16,  # RightFoot -> RightLeg
    0,   # LeftUpLeg -> Hips
    18,  # LeftLeg -> LeftUpLeg
    19,  # LeftFoot -> LeftLeg
]

# Bones list for plotting (p1_idx, p2_idx)
BONES = []
for i, parent_idx in enumerate(PARENT_INDICES):
    if parent_idx != -1:
        BONES.append((parent_idx, i))

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_calibration():
    """Load camera calibration parameters."""
    print(f"Loading calibration from: {CALIBRATION_FILE}")
    with open(CALIBRATION_FILE, 'r') as f:
        calib = json.load(f)
    
    cameras = {}
    for cam in calib['cameras']:
        cam_id = cam['camera_id']
        cameras[cam_id] = {
            'rotation': np.array(cam['rotation_matrix']),
            'translation': np.array(cam['translation_vector'])
        }
    return cameras

def world_to_camera_position(pos_world, R_cam, T_cam):
    """
    Transform position from world coordinates to camera coordinates.
    P_camera = R_cam @ P_world + T_cam
    """
    return R_cam @ pos_world + T_cam

def parse_pos_file(file_path):
    """
    Parse position file (x, y, z).
    Returns: list of dicts, each dict maps joint_name -> [x, y, z]
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return [], []
        
    frames = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        joint_names_in_file = [h.strip() for h in header if h.strip()]
        
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            frame_data = {}
            for i, name in enumerate(joint_names_in_file):
                if i < len(parts) and parts[i].strip():
                    vals = [float(v) for v in parts[i].split()]
                    if len(vals) == 3:
                        frame_data[name] = vals
            frames.append(frame_data)
    
    return frames, joint_names_in_file

# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("Test GT 22x6 From Raw (Camera 1 Visualization)")
    print("=" * 60)
    
    # Load Calibration
    cameras = load_calibration()
    cam1_params = cameras.get(6)
    if not cam1_params:
        print("Camera 1 calibration not found!")
        return

    R_cam = cam1_params['rotation']
    T_cam = cam1_params['translation']
    print("Camera 1 loaded.")
    print(f"R shape: {R_cam.shape}")
    print(f"T shape: {T_cam.shape}")

    # Load Pos Data (S1/Acting1)
    print(f"Loading positions from: {DATA_POS_FILE}")
    pos_frames, _ = parse_pos_file(DATA_POS_FILE)
    if not pos_frames:
        print("No position data found.")
        return
    
    num_frames = len(pos_frames)
    print(f"Loaded {num_frames} frames.")

    # Prepare data for plotting: (Frames, Joints, 3)
    # Joints in order of JOINT_NAMES
    all_cam_coords = np.zeros((num_frames, len(JOINT_NAMES), 3))

    for f_idx, frame_data in enumerate(pos_frames):
        for j_idx, joint_name in enumerate(JOINT_NAMES):
            if joint_name in frame_data:
                # 1. Get raw world pos (inches)
                p_world_in = np.array(frame_data[joint_name])
                
                # 2. Convert to meters
                p_world_m = p_world_in * SCALE_FACTOR
                
                # 3. Transform to Camera Frame
                # No manual swapping logic, just P_cam = R @ P + T
                p_cam = world_to_camera_position(p_world_m, R_cam, T_cam)
                
                all_cam_coords[f_idx, j_idx, :] = p_cam
            else:
                # Should not happen for valid joints, but handle gracefully
                pass

    print(f"Transformed coordinates shape: {all_cam_coords.shape}")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)

    # Setup scatter and lines
    scat = ax.scatter([], [], [], c='red', s=20)
    lines = []
    for _ in range(len(BONES)):
        line, = ax.plot([], [], [], 'b-', lw=1.5)
        lines.append(line)

    # Compute axis limits
    all_x = all_cam_coords[:, :, 0].flatten()
    all_y = all_cam_coords[:, :, 1].flatten()
    all_z = all_cam_coords[:, :, 2].flatten()
    
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
    
    ax.set_xlabel('X (Camera)')
    ax.set_ylabel('Y (Camera)')
    ax.set_zlabel('Z (Camera)')
    ax.set_title(f'S1 Acting1 Cam1: {DATA_POS_FILE}')

    def update(val):
        frame = int(slider.val)
        coords = all_cam_coords[frame] # (21, 3)
        
        # Update scatter
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        scat._offsets3d = (xs, ys, zs)
        
        # Update bones
        for i, (p1, p2) in enumerate(BONES):
            lines[i].set_data_3d(
                [coords[p1, 0], coords[p2, 0]],
                [coords[p1, 1], coords[p2, 1]],
                [coords[p1, 2], coords[p2, 2]]
            )
        
        ax.set_title(f'Frame {frame}/{num_frames-1}')
        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
    slider.on_changed(update)

    # Initial draw
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
