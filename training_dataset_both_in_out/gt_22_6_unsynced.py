"""
Ground Truth 22x6 Dataset Generator (Camera-Oriented)

Generates ground truth data for each subject/action/camera combination with:
- Index 0: Hip root position (x,y,z,0,0,0) - world position transformed to camera coords, scaled to meters
- Index 1: Hip global 6D rotation (camera-oriented)
- Indices 2-21: Local 6D rotations for 20 child joints (relative to parent)

Input:
    - gt_skel_gbl_ori.txt: Global quaternions [x,y,z,w] for 21 joints
    - gt_skel_gbl_pos.txt: Global positions (x,y,z) for 21 joints (only hip position used)
    - calibration_params.json: Camera rotation and translation matrices

Output: gt_22_6_nosync/{Subject}/{Action}/gt_{Subject}_{Action}_cam{N}.npy
    Shape: (frames, 22, 6)
"""

import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
POSITIONS_DIR = os.path.join(BASE_DIR, 'totalcapture_dataset/positions')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/calibration_params.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_dataset_both_in_out/gt_22_6_nosync')

# Scale factor to convert from inches to meters (TotalCapture positions are in inches)
SCALE_FACTOR = 0.0254

# Joint names in order (21 joints from ori/pos files)
JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

# Parent indices for each joint (index in JOINT_NAMES, -1 for root)
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

# Number of cameras
NUM_CAMERAS = 8

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_calibration():
    """Load camera calibration parameters."""
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


def quaternion_to_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    TotalCapture quaternions are in [x, y, z, w] format.
    """
    x, y, z, w = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])


def matrix_to_6d(R):
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def global_to_local_rotation(R_parent, R_child_global):
    """Convert global rotation to local: R_local = R_parent^T @ R_child_global"""
    return R_parent.T @ R_child_global


def world_to_camera_rotation(R_world, R_cam):
    """
    Transform rotation from world coordinates to camera coordinates.
    R_camera = R_cam @ R_world
    """
    return R_cam @ R_world


def world_to_camera_position(pos_world, R_cam, T_cam):
    """
    Transform position from world coordinates to camera coordinates.
    P_camera = R_cam @ P_world + T_cam
    """
    return R_cam @ pos_world + T_cam


def parse_ori_file(file_path):
    """
    Parse orientation file (quaternions).
    Returns: list of dicts, each dict maps joint_name -> [x, y, z, w]
    """
    frames = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        joint_names = [h.strip() for h in header if h.strip()]
        
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            frame_data = {}
            for i, name in enumerate(joint_names):
                if i < len(parts) and parts[i].strip():
                    vals = [float(v) for v in parts[i].split()]
                    if len(vals) == 4:
                        frame_data[name] = vals
            frames.append(frame_data)
    
    return frames, joint_names


def parse_pos_file(file_path):
    """
    Parse position file (x, y, z).
    Returns: list of dicts, each dict maps joint_name -> [x, y, z]
    """
    frames = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        joint_names = [h.strip() for h in header if h.strip()]
        
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            frame_data = {}
            for i, name in enumerate(joint_names):
                if i < len(parts) and parts[i].strip():
                    vals = [float(v) for v in parts[i].split()]
                    if len(vals) == 3:
                        frame_data[name] = vals
            frames.append(frame_data)
    
    return frames, joint_names


def process_sequence(ori_file, pos_file, camera_params):
    """
    Process a single sequence for one camera.
    
    Returns: (frames, 22, 6) array
        - Index 0: Hip position (x,y,z,0,0,0) in camera coords, scaled to meters
        - Index 1: Hip global 6D in camera coords
        - Indices 2-21: Local 6D for child joints
    """
    ori_frames, _ = parse_ori_file(ori_file)
    pos_frames, _ = parse_pos_file(pos_file)
    
    num_frames = min(len(ori_frames), len(pos_frames))
    
    R_cam = camera_params['rotation']
    T_cam = camera_params['translation']
    
    # Output: (frames, 22, 6)
    output = np.zeros((num_frames, 22, 6), dtype=np.float32)
    
    for frame_idx in range(num_frames):
        ori_data = ori_frames[frame_idx]
        pos_data = pos_frames[frame_idx]
        
        # Get hip position (world coords)
        if 'Hips' in pos_data:
            hip_pos_world = np.array(pos_data['Hips'])
        else:
            hip_pos_world = np.zeros(3)
        
        # Transform hip position to camera coords and scale to meters
        hip_pos_camera = world_to_camera_position(hip_pos_world, R_cam, T_cam)
        hip_pos_scaled = hip_pos_camera * SCALE_FACTOR
        
        # Index 0: Hip position (x,y,z,0,0,0)
        output[frame_idx, 0, :3] = hip_pos_scaled
        output[frame_idx, 0, 3:] = 0.0  # Padding
        
        # Compute global rotation matrices for all joints (in world coords first)
        world_rotations = {}
        for joint_name in JOINT_NAMES:
            if joint_name in ori_data:
                q = ori_data[joint_name]
                world_rotations[joint_name] = quaternion_to_matrix(q)
            else:
                world_rotations[joint_name] = np.eye(3)
        
        # Index 1: Hip global 6D in camera coordinates
        R_hip_world = world_rotations['Hips']
        R_hip_camera = world_to_camera_rotation(R_hip_world, R_cam)
        output[frame_idx, 1, :] = matrix_to_6d(R_hip_camera)
        
        # Indices 2-21: Local 6D for child joints
        for joint_idx, joint_name in enumerate(JOINT_NAMES[1:], start=2):
            parent_idx = PARENT_INDICES[joint_idx - 1]  # -1 because we start from index 1 in JOINT_NAMES
            parent_name = JOINT_NAMES[parent_idx]
            
            R_parent = world_rotations[parent_name]
            R_child = world_rotations[joint_name]
            
            # Compute local rotation: R_local = R_parent^T @ R_child
            R_local = global_to_local_rotation(R_parent, R_child)
            output[frame_idx, joint_idx, :] = matrix_to_6d(R_local)
    
    return output


def main():
    print("=" * 60)
    print("Ground Truth 22x6 Dataset Generator (Camera-Oriented)")
    print("=" * 60)
    print(f"Input:  {POSITIONS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Load calibration
    cameras = load_calibration()
    print(f"Loaded calibration for {len(cameras)} cameras")
    
    # Find all subject/action combinations
    subjects = sorted([d for d in os.listdir(POSITIONS_DIR) if os.path.isdir(os.path.join(POSITIONS_DIR, d))])
    
    total_files = 0
    total_frames = 0
    
    for subject in subjects:
        subject_dir = os.path.join(POSITIONS_DIR, subject)
        actions = sorted([d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))])
        
        for action in tqdm(actions, desc=f"Processing {subject}"):
            action_dir = os.path.join(subject_dir, action)
            
            ori_file = os.path.join(action_dir, 'gt_skel_gbl_ori.txt')
            pos_file = os.path.join(action_dir, 'gt_skel_gbl_pos.txt')
            
            if not os.path.exists(ori_file) or not os.path.exists(pos_file):
                print(f"  Skipping {subject}/{action}: Missing ori/pos files")
                continue
            
            # Process for each camera
            for cam_id in range(1, NUM_CAMERAS + 1):
                cam_params = cameras[cam_id]
                
                # Process sequence
                output_data = process_sequence(ori_file, pos_file, cam_params)
                
                # Save output
                output_subdir = os.path.join(OUTPUT_DIR, subject, action)
                os.makedirs(output_subdir, exist_ok=True)
                
                output_file = os.path.join(output_subdir, f'gt_{subject}_{action}_cam{cam_id}.npy')
                np.save(output_file, output_data)
                
                total_files += 1
                total_frames += output_data.shape[0]
    
    print()
    print("=" * 60)
    print(f"Completed! Generated {total_files} files, {total_frames} total frames")
    print(f"Output saved to: {OUTPUT_DIR}")
    print()
    print("Output format (22 joints x 6 channels):")
    print("  Index 0: Hip position (x,y,z,0,0,0) - camera coords, meters")
    print("  Index 1: Hip global 6D (camera-oriented)")
    print("  Indices 2-21: Local 6D for 20 child joints")


if __name__ == "__main__":
    main()
