import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths
TARGET_FILE = os.path.join(WORKSPACE_ROOT, "training_data_targets", "S1", "acting1", "target_S1_acting1_cam1.npy")
BVH_TEMPLATE = os.path.join(SCRIPT_DIR, "bvh", "acting1_BlenderZXY_YmZ.bvh")
OUTPUT_BVH = os.path.join(WORKSPACE_ROOT, "training_data_targets", "S1", "acting1", "target_S1_acting1_cam1.bvh")

# Conversion factor
INCHES_TO_METERS = 0.0254

# Bone hierarchy (same as in make_gt_datain_ori.py)
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

BONE_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RShoulder", "RArm", "RForeArm", "RHand",
    "LShoulder", "LArm", "LForeArm", "LHand",
    "RUpLeg", "RLeg", "RFoot",
    "LUpLeg", "LLeg", "LFoot"
]

def convert_6d_to_rotation_matrix(rot_6d):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    col1 = rot_6d[:3]
    col2 = rot_6d[3:6]
    
    # Normalize first column
    col1_norm = np.linalg.norm(col1)
    if col1_norm > 1e-8:
        col1 = col1 / col1_norm
    else:
        col1 = np.array([1, 0, 0])
    
    # Gram-Schmidt orthogonalization
    col2 = col2 - np.dot(col2, col1) * col1
    col2_norm = np.linalg.norm(col2)
    if col2_norm > 1e-8:
        col2 = col2 / col2_norm
    else:
        col2 = np.array([0, 1, 0])
    
    # Third column is cross product
    col3 = np.cross(col1, col2)
    
    # Construct rotation matrix
    rot_mat = np.column_stack([col1, col2, col3])
    
    return rot_mat

def read_bvh_hierarchy(bvh_file):
    """Read the hierarchy section from BVH template."""
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
    
    hierarchy_lines = []
    in_hierarchy = True
    
    for line in lines:
        if line.strip().startswith("MOTION"):
            break
        hierarchy_lines.append(line.rstrip())
    
    return hierarchy_lines

def convert_to_euler_zxy(rot_mat):
    """Convert rotation matrix to ZXY Euler angles (in degrees) for BVH."""
    r = R.from_matrix(rot_mat)
    # BVH uses ZXY order (Zrotation Xrotation Yrotation)
    euler_zxy = r.as_euler('ZXY', degrees=True)
    return euler_zxy

def forward_kinematics_with_rotations(root_rot_6d, child_rots_6d):
    """
    Compute global rotations for all joints.
    
    Args:
        root_rot_6d: (6,) root orientation in camera space
        child_rots_6d: (20, 6) child joint LOCAL orientations
    
    Returns:
        global_rotations: list of 21 rotation matrices
    """
    global_rotations = [None] * 21
    
    # Root joint
    root_rot_mat = convert_6d_to_rotation_matrix(root_rot_6d)
    global_rotations[0] = root_rot_mat
    
    # All other joints
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        
        # Get LOCAL rotation for this child
        local_rot_6d = child_rots_6d[i-1]
        local_rot_mat = convert_6d_to_rotation_matrix(local_rot_6d)
        
        # Compute GLOBAL rotation
        parent_global_rot = global_rotations[parent_idx]
        global_rotations[i] = parent_global_rot @ local_rot_mat
    
    return global_rotations

def parse_bvh_hierarchy_order(bvh_file):
    """
    Parse the joint names in order from the BVH file to match the output data properly.
    Returns:
        joint_names: List of joint names in DFS order as they appear in the file.
    """
    joint_names = []
    
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("ROOT") or line.startswith("JOINT"):
            parts = line.split()
            if len(parts) >= 2:
                joint_names.append(parts[1])
    
    return joint_names

def create_bvh_from_target(target_file, template_bvh, output_bvh):
    """Create BVH file from target NPY data."""
    
    print(f"Loading target file: {target_file}")
    data = np.load(target_file)
    print(f"Data shape: {data.shape}")  # (F, 22, 6)
    
    num_frames = data.shape[0]
    
    # Read hierarchy from template
    print(f"Reading BVH template: {template_bvh}")
    hierarchy_lines = read_bvh_hierarchy(template_bvh)
    
    # Parse hierarchy order
    bvh_joints = parse_bvh_hierarchy_order(template_bvh)
    print(f"Found {len(bvh_joints)} joints in BVH template: {bvh_joints}")
    
    # Mapping from our 21-bone model to BVH (name matching)
    # Our names: Hips, Spine, Spine1, Spine2, Spine3, Neck, Head, 
    # RShoulder, RArm, RForeArm, RHand, 
    # LShoulder, LArm, LForeArm, LHand, 
    # RUpLeg, RLeg, RFoot, 
    # LUpLeg, LLeg, LFoot
    
    # BVH template names typically match these, but might have slight variations or extra joints.
    # We will try to map by exact name or simple replacements.
    
    # Map index in BONE_NAMES (0-20) to index in bvh_joints list
    bone_to_bvh_map = {}
    
    # Create name mapping dictionary for common mismatches
    name_map = {
        "RightShoulder": "RShoulder",
        "RightArm": "RArm",
        "RightForeArm": "RForeArm",
        "RightHand": "RHand",
        "LeftShoulder": "LShoulder",
        "LeftArm": "LArm",
        "LeftForeArm": "LForeArm",
        "LeftHand": "LHand",
        "RightUpLeg": "RUpLeg",
        "RightLeg": "RLeg",
        "RightFoot": "RFoot",
        "LeftUpLeg": "LUpLeg",
        "LeftLeg": "LLeg",
        "LeftFoot": "LFoot"
    }
    # Add identity mapping
    for name in BONE_NAMES:
        name_map[name] = name
        
    # Prepare motion data
    print("Converting rotations to Euler angles...")
    motion_lines = []
    
    # Accumulate positions for root joint
    root_position = np.array([0.0, 0.0, 0.0])
    
    for frame_idx in range(num_frames):
        frame_data = data[frame_idx]  # (22, 6)
        
        # Row 0: velocity (camera space)
        velocity = frame_data[0, :3]
        
        # Accumulate root position
        root_position += velocity
        
        # Row 1: root orientation (camera space)
        root_rot_6d = frame_data[1]
        
        # Rows 2-21: child local orientations
        child_rots_6d = frame_data[2:]  # (20, 6)
        
        # Compute all global rotations for our 21 joints
        global_rotations = forward_kinematics_with_rotations(root_rot_6d, child_rots_6d)
        
        # Create a dictionary of Euler angles for our 21 joints
        joint_eulers = {}
        for i, name in enumerate(BONE_NAMES):
            joint_eulers[name] = convert_to_euler_zxy(global_rotations[i])
            
        # Convert root position to inches
        pos_inches = root_position / INCHES_TO_METERS
        
        # Build the frame values line based on BVH hierarchy order
        frame_values = []
        
        for bvh_joint_name in bvh_joints:
            # Check if this BVH joint corresponds to one of our 21 joints
            our_joint_name = name_map.get(bvh_joint_name)
            
            if our_joint_name in joint_eulers:
                euler = joint_eulers[our_joint_name]
                if bvh_joint_name == "Hips": # Root
                    # Position + Rotation
                    frame_values.extend([pos_inches[0], pos_inches[1], pos_inches[2]])
                    frame_values.extend(euler)
                else:
                    # Dummy Position + Rotation (since template has 6 channels)
                    frame_values.extend([0.0, 0.0, 0.0])
                    frame_values.extend(euler)
            else:
                # Extra joint (like *End or *Thumb or *ToeBase)
                # We don't have data for it, so zero position and zero rotation
                # Check template for channels count - usually 6 or 3.
                # Assuming 6 channels as per template inspection (all joints have 6 channels)
                # If template has End Site without channels, it is not in bvh_joints list?
                # Wait, parse_bvh_hierarchy_order only picks lines starting with ROOT or JOINT.
                # End Sites are not JOINTS. They don't have channels usually.
                # But verify if any JOINTS in the list are extra.
                
                # Extra joints: RightHandEnd, RightHandThumb1, RightToeBase, etc.
                # We output identity transform
                frame_values.extend([0.0, 0.0, 0.0]) # Pos
                frame_values.extend([0.0, 0.0, 0.0]) # Rot
        
        # Format as string
        motion_line = " ".join([f"{v:.6f}" for v in frame_values])
        motion_lines.append(motion_line)
        
        if (frame_idx + 1) % 500 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames")
    
    # Write BVH file
    print(f"\nWriting BVH file: {output_bvh}")
    with open(output_bvh, 'w') as f:
        # Write hierarchy
        for line in hierarchy_lines:
            f.write(line + '\n')
        
        # Write motion header
        f.write('MOTION\n')
        f.write(f'Frames: {num_frames}\n')
        f.write('Frame Time: 0.016667\n')  # 60 FPS
        
        # Write motion data
        for line in motion_lines:
            f.write(line + '\n')
    
    print(f"BVH file created successfully!")
    print(f"Total frames: {num_frames}")

def main():
    create_bvh_from_target(TARGET_FILE, BVH_TEMPLATE, OUTPUT_BVH)

if __name__ == "__main__":
    main()
