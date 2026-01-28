import numpy as np
import os
import json
import glob
from scipy.spatial.transform import Rotation as R

# Get the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

# confg
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "totalcapture_dataset", "positions")
CALIB_FILE = os.path.join(WORKSPACE_ROOT, "totalcapture_dataset", "calibration_params.json")
OUTPUT_FOLDER = os.path.join(WORKSPACE_ROOT, "training_data_targets")
INCHES_TO_METERS = 0.0254

## TotalCapture Specific 21-Bone Hierarchy
# 0:Hips, 1:Spine, 2:Spine1, 3:Spine2, 4:Spine3, 5:Neck, 6:Head
# 7:RShldr, 8:RArm, 9:RForeArm, 10:RHand
# 11:LShldr, 12:LArm, 13:LForeArm, 14:LHand
# 15:RUpLeg, 16:RLeg, 17:RFoot
# 18:LUpLeg, 19:LLeg, 20:LFoot

PARENTS = [
    -1, # 0: Hips
    0,  # 1: Spine
    1,  # 2: Spine1
    2,  # 3: Spine2
    3,  # 4: Spine3
    4,  # 5: Neck
    5,  # 6: Head
    4,  # 7: RShoulder
    7,  # 8: RArm
    8,  # 9: RForeArm
    9,  # 10: RHand
    4,  # 11: LShoulder
    11, # 12: LArm
    12, # 13: LForeArm
    13, # 14: LHand
    0,  # 15: RUpLeg
    15, # 16: RLeg
    16, # 17: RFoot
    0,  # 18: LUpLeg
    18, # 19: LLeg
    19  # 20: LFoot
]

# helper functions
def load_txt_file(filepath, cols_per_bone):
    """Parses tab-separated files.SKips header"""
    try:
        raw=np.loadtxt(filepath, skiprows=1)
        # reshape to (fames, 21bones, cols)
        return raw.reshape((raw.shape[0], 21, cols_per_bone))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
def quat_to_6d(quates):
    """Convert quaternions to 6D rotation representation."""
    rot_mats = R.from_quat(quates).as_matrix()  # (N, 3, 3)
    return rot_mats[:, :, :2].reshape(-1, 6)  # (N, 6)

def load_cameras(json_path):
    """Load camera calibration parameters from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    cams = {}
    for cam in data['cameras']:
        cid = cam['camera_id']
        # load rotation (world-> camera)
        rot_mat = np.array(cam['rotation_matrix'])
        # load translation (not strictly needed for velocity , but good to have)
        trans_vec = np.array(cam['translation_vector'])
        r_obj = R.from_matrix(rot_mat)
        cams[cid] = r_obj
        
    return cams

 # core processing functions
def process_single_action(pos_path, ori_path, cameras, subject_name, action_name):
    # load global data:
    global_pos = load_txt_file(pos_path, 3) # (F, 21, 3)
    global_ori = load_txt_file(ori_path, 4) # (F, 21, 4) # quaternions
    
    if global_pos is None or global_ori is None:
        return
     
    # unit conversion
    global_pos *= INCHES_TO_METERS
    F = global_pos.shape[0]
    
    # Root position and velocity
    root_global_pos = global_pos[:,0,:] # (F, 3)
    
    # Calculate global velocity vector
    global_vel = np.zeros_like(root_global_pos)
    global_vel[1:,:] = root_global_pos[1:] - root_global_pos[:-1]
    
    # Generate 8 camera views
    for cam_id, cam_R in cameras.items():
        # STEP 1: Transform ALL global orientations to camera space
        # Q_cam = Q_cam_rot * Q_world for each joint
        cam_global_ori = []
        for j in range(21):
            r_world = R.from_quat(global_ori[:, j, :])  # (F,) rotation objects
            r_cam = cam_R * r_world  # Transform to camera space
            cam_global_ori.append(r_cam)
        
        # STEP 2: Compute local orientations relative to immediate parent (in camera space)
        child_poses_list = []
        for i in range(1, 21):
            parent_idx = PARENTS[i]
            # Get camera-space rotations
            r_parent_cam = cam_global_ori[parent_idx]
            r_child_cam = cam_global_ori[i]
            
            # Calculate local: Q_local = Q_parent_inv * Q_child (in camera space)
            r_local = r_parent_cam.inv() * r_child_cam
            
            # Convert to 6D
            pose_6d = quat_to_6d(r_local.as_quat())  # (F, 6)
            child_poses_list.append(pose_6d)
        
        # Stack: (F, 20, 6) - local orientations for bones 1-20
        child_pose_block = np.stack(child_poses_list, axis=1)
        
        # Row 0: Camera-space velocity
        cam_vel_vec = cam_R.apply(global_vel)  # (F, 3)
        row_0 = np.zeros((F, 6))
        row_0[:, :3] = cam_vel_vec
        
        # Row 1: Camera-space root (Hip) orientation
        root_cam_ori = cam_global_ori[0]  # Already in camera space
        row_1 = quat_to_6d(root_cam_ori.as_quat())  # (F, 6)
        
        # Reshape for broadcasting
        r0 = row_0[:, None, :]  # (F, 1, 6)
        r1 = row_1[:, None, :]  # (F, 1, 6)
        
        # Concatenate 
        final_target = np.concatenate([r0, r1, child_pose_block], axis=1).astype(np.float32)
        # final shape: (F, 22, 6)
        
        # save 
        # Create subfolder for subject/action
        output_subfolder = os.path.join(OUTPUT_FOLDER, subject_name, action_name)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # filename: target_S1_acting1_cam1.npy
        save_name = f"target_{subject_name}_{action_name}_cam{cam_id}.npy"
        save_path = os.path.join(output_subfolder, save_name)
        np.save(save_path, final_target)


# =========================================================
# 4. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # 1. Load Camera Calibration
    print("Loading Cameras...")
    cameras = load_cameras(CALIB_FILE)
    
    # 2. Walk through Folder Structure
    # Looking for: dataset/S1/acting1/gt_skel_gbl_pos.txt
    print(f"Scanning {DATASET_ROOT}...")
    
    # Find all 'gt_skel_gbl_pos.txt' files recursively
    search_pattern = os.path.join(DATASET_ROOT, "**", "gt_skel_gbl_pos.txt")
    pos_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(pos_files)} actions to process.")
    
    for pos_path in pos_files:
        # Determine paths and names
        # pos_path example: .../S1/acting1/gt_skel_gbl_pos.txt
        folder_dir = os.path.dirname(pos_path)
        ori_path = os.path.join(folder_dir, "gt_skel_gbl_ori.txt")
        
        if not os.path.exists(ori_path):
            print(f"Skipping {pos_path}: Missing ori file.")
            continue
            
        # Extract Subject and Action from folder path
        # Assuming structure: .../S1/acting1/...
        parts = pos_path.split(os.sep)
        action_name = parts[-2] # 'acting1'
        subject_name = parts[-3] # 'S1'
        
        print(f"Processing: {subject_name} - {action_name}")
        
        # Process and Generate 8 Files
        process_single_action(pos_path, ori_path, cameras, subject_name, action_name)
        
    print("Done! All camera views generated.")      
    