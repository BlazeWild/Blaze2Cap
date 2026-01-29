"""
Test Camera-Space 6D Ground Truth Visualization

1. Loads Global Positions (S1/Freestyle3) and Orientations.
2. Loads Camera 6 Calibration.
3. Transforms Hip Position and All Orientations to Camera 6 Frame.
4. Decomposes Orientations into Local Rotations (Child relative to Parent).
5. Converts Local Rotations to 6D (and back) to verify 6D representation.
6. Reconstructs Skeleton via Forward Kinematics (FK) using BVH Offsets.
7. Visualizes the result.
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
import re

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
POS_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/positions/S1/freestyle3/gt_skel_gbl_pos.txt')
ORI_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/positions/S1/freestyle3/gt_skel_gbl_ori.txt')
BVH_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh')
CALIB_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/calibration_params.json')

TARGET_CAM_ID = 6
SCALE_FACTOR = 0.0254 # inches to meters

# Joint names in order (21 joints, matching previous scripts)
JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

# Parent indices (index in JOINT_NAMES, -1 for root)
PARENT_INDICES = [
    -1,  # Hips (root)
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

# Bones for plotting
BONES = []
for i, p_idx in enumerate(PARENT_INDICES):
    if p_idx != -1:
        BONES.append((p_idx, i))

# ==========================================
# MATH UTILS (Rotations & 6D)
# ==========================================

def quaternion_to_matrix(q):
    """[x, y, z, w] -> 3x3 matrix"""
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n == 0: return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def matrix_to_6d(R):
    """3x3 -> 6D vector (first two columns)"""
    return np.concatenate([R[:, 0], R[:, 1]])

def rotation_6d_to_matrix(r6d):
    """
    6D vector -> 3x3 rotation matrix using Gram-Schmidt orthogonalization.
    r6d: (6,) array
    """
    x_raw = r6d[0:3]
    y_raw = r6d[3:6]
    
    # Normalize x
    x = x_raw / np.linalg.norm(x_raw)
    
    # Project y onto plane orthogonal to x
    y = y_raw - np.dot(x, y_raw) * x
    y = y / np.linalg.norm(y)
    
    # z = x cross y
    z = np.cross(x, y)
    
    R = np.column_stack((x, y, z))
    return R

def compute_local_rotation(R_parent_global, R_child_global):
    """
    R_child_global = R_parent_global @ R_child_local
    => R_child_local = R_parent_global^T @ R_child_global
    """
    return R_parent_global.T @ R_child_global

# ==========================================
# PARSERS
# ==========================================

def parse_bvh_offsets(bvh_path):
    """
    Simple parser to extract OFFSET for each joint in JOINT_NAMES.
    Assumes standard format where OFFSET comes after JOINT/ROOT definition.
    Current BVH has Scale 1.0 (implied, offsets are in inches usually for this dataset).
    """
    offsets = {}
    current_joint = None
    
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith('ROOT') or line.startswith('JOINT'):
            # "ROOT Hips" or "JOINT Spine"
            parts = line.split()
            if len(parts) >= 2:
                joint_name = parts[1]
                if joint_name in JOINT_NAMES:
                    current_joint = joint_name
        elif line.startswith('OFFSET'):
            if current_joint:
                parts = line.split()
                # OFFSET x y z
                vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                # Convert inches to meters immediately using SCALE_FACTOR
                # (Assuming BVH offsets are in same unit as pos file, inches)
                offsets[current_joint] = vec * SCALE_FACTOR 
                current_joint = None # Reset to avoid capturing end site offsets if any unexpected nesting
    
    # Fill missing with zeros if any (shouldn't happen for valid list)
    for name in JOINT_NAMES:
        if name not in offsets:
            print(f"Warning: No offset found for {name}, using zero.")
            offsets[name] = np.zeros(3)
            
    return offsets

def load_calibration():
    with open(CALIB_FILE, 'r') as f:
        data = json.load(f)
    for cam in data['cameras']:
        if cam['camera_id'] == TARGET_CAM_ID:
            return {
                'R': np.array(cam['rotation_matrix']),
                'T': np.array(cam['translation_vector'])
            }
    raise ValueError(f"Camera {TARGET_CAM_ID} not found in calibration")

def load_pos_ori(pos_path, ori_path):
    # Retrieve raw frames
    pos_frames = []
    with open(pos_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        names = [x.strip() for x in header if x.strip()]
        for line in lines[1:]:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            frame = {}
            for i, name in enumerate(names):
                if i < len(parts) and parts[i].strip():
                    frame[name] = np.array([float(v) for v in parts[i].split()])
            pos_frames.append(frame)

    ori_frames = []
    with open(ori_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        names = [x.strip() for x in header if x.strip()]
        for line in lines[1:]:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            frame = {}
            for i, name in enumerate(names):
                if i < len(parts) and parts[i].strip():
                    frame[name] = np.array([float(v) for v in parts[i].split()])
            ori_frames.append(frame)
            
    return pos_frames, ori_frames

# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Processing S1/Freestyle3 for Camera {TARGET_CAM_ID}...")
    
    # 1. Load Data
    bvh_offsets = parse_bvh_offsets(BVH_FILE)
    calib = load_calibration()
    R_cam = calib['R'] # 3x3
    T_cam = calib['T'] # 3,
    
    pos_frames, ori_frames = load_pos_ori(POS_FILE, ORI_FILE)
    num_frames = min(len(pos_frames), len(ori_frames))
    print(f"Loaded {num_frames} frames.")
    
    # 2. Main Processing Loop
    # We will reconstruct positions for plotting: (Frames, Joints, 3)
    reconstructed_pos = np.zeros((num_frames, len(JOINT_NAMES), 3))
    
    print("Processing frames (Transform -> 6D -> Reconstruct FK)...")
    for f in range(num_frames):
        p_data = pos_frames[f]
        o_data = ori_frames[f]
        
        # --- A. Process Hips (Root) ---
        # 1. World Position
        hips_world_in = p_data.get('Hips', np.zeros(3))
        hips_world_m = hips_world_in * SCALE_FACTOR
        
        # 2. Transform to Cam Frame (Pos)
        hips_cam = R_cam @ hips_world_m + T_cam
        
        # 3. Orientation
        hips_quat = o_data.get('Hips', np.array([0,0,0,1])) # xyzw
        R_hips_world = quaternion_to_matrix(hips_quat)
        
        # 4. Transform to Cam Frame (Rot)
        # R_cam_global = R_ext_cam @ R_world_global
        R_hips_cam = R_cam @ R_hips_world
        
        # --- B. Process Hierarchy ---
        # We need to store global rotations (in cam frame) for FK
        global_rotations_cam = {} 
        global_rotations_cam['Hips'] = R_hips_cam
        
        # We also want to prove we can go via Local 6D
        # Store reconstructed globals
        reconstructed_globals = {}
        
        # Hips "Local" is its Global in local-root-space (which is camera space effectively for root)
        r6d_hips = matrix_to_6d(R_hips_cam)
        R_hips_rec = rotation_6d_to_matrix(r6d_hips) # Verify 6D integrity
        reconstructed_globals['Hips'] = R_hips_rec
        
        # Store final Position
        reconstructed_pos[f, 0, :] = hips_cam # Hips
        
        # Iterate children in order
        for i, joint_name in enumerate(JOINT_NAMES):
            if i == 0: continue # Skip Hips
            
            p_idx = PARENT_INDICES[i]
            parent_name = JOINT_NAMES[p_idx]
            
            # 1. Get Child Global Ori (World)
            q_child = o_data.get(joint_name, np.array([0,0,0,1]))
            R_child_world = quaternion_to_matrix(q_child)
            
            # 2. Transform to Cam Frame
            R_child_cam = R_cam @ R_child_world
            global_rotations_cam[joint_name] = R_child_cam
            
            # 3. Compute Local Rotation matrix relative to Parent (in Cam Frame)
            # Parent Global (Cam)
            R_parent_cam = global_rotations_cam[parent_name]
            
            # R_child_cam = R_parent_cam @ R_local
            # -> R_local = R_parent_cam.T @ R_child_cam
            R_local = R_parent_cam.T @ R_child_cam
            
            # 4. Convert to 6D and back (Verify requirement)
            r6d = matrix_to_6d(R_local)
            R_local_rec = rotation_6d_to_matrix(r6d)
            
            # 5. Reconstruct Global Rotation (FK step 1)
            # R_child_global_rec = R_parent_global_rec @ R_local_rec
            # Use the *reconstructed* parent rotation to propagate FK chain
            R_parent_rec = reconstructed_globals[parent_name]
            R_child_rec = R_parent_rec @ R_local_rec
            reconstructed_globals[joint_name] = R_child_rec
            
            # 6. Reconstruct Position (FK step 2)
            # P_child = P_parent + R_parent_global @ Offset
            # Offset is constant from BVH
            parent_pos_rec = reconstructed_pos[f, p_idx, :]
            offset = bvh_offsets[joint_name] 
            
            # Apply parent's rotation to the offset
            # (Note: BVH offset is vector in Parent's Frame)
            rotated_offset = R_parent_rec @ offset
            child_pos_rec = parent_pos_rec + rotated_offset
            
            reconstructed_pos[f, i, :] = child_pos_rec

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    # Calculate limits for static plot
    all_x = reconstructed_pos[:, :, 0].flatten()
    all_y = reconstructed_pos[:, :, 1].flatten()
    all_z = reconstructed_pos[:, :, 2].flatten()
    
    # Center view
    mid_x = (all_x.max() + all_x.min()) / 2
    mid_y = (all_y.max() + all_y.min()) / 2
    mid_z = (all_z.max() + all_z.min()) / 2
    max_range = max(all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (Cam)')
    ax.set_ylabel('Y (Cam)')
    ax.set_zlabel('Z (Cam)')
    ax.set_title(f'S1 Freestyle3 Cam{TARGET_CAM_ID} (FK Reconstructed)')
    
    # Objects
    scat = ax.scatter([], [], [], c='r', s=20)
    lines = [ax.plot([], [], [], 'b-')[0] for _ in BONES]
    
    def update(val):
        f = int(slider.val)
        frame_data = reconstructed_pos[f] # (21, 3)
        
        scat._offsets3d = (frame_data[:,0], frame_data[:,1], frame_data[:,2])
        
        for line, (p1, p2) in zip(lines, BONES):
            line.set_data(
                [frame_data[p1,0], frame_data[p2,0]],
                [frame_data[p1,1], frame_data[p2,1]]
            )
            line.set_3d_properties([frame_data[p1,2], frame_data[p2,2]])
            
        ax.set_title(f'Frame {f}')
        fig.canvas.draw_idle()
        
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valfmt='%d')
    slider.on_changed(update)
    
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
