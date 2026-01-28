"""
Test script to verify the skeleton reconstruction from target data.
Compares reconstructed positions with ground truth positions.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths
TARGET_FILE = os.path.join(WORKSPACE_ROOT, "training_data_targets", "S1", "acting1", "target_S1_acting1_cam1.npy")
GT_POS_FILE = os.path.join(SCRIPT_DIR, "positions", "S1", "acting1", "gt_skel_gbl_pos.txt")
BVH_FILE = os.path.join(SCRIPT_DIR, "bvh", "acting1_BlenderZXY_YmZ.bvh")

INCHES_TO_METERS = 0.0254

PARENTS = [
    -1, 0, 1, 2, 3, 4, 5,  # 0-6: Hips, Spine, Spine1, Spine2, Spine3, Neck, Head
    4, 7, 8, 9,            # 7-10: RShoulder, RArm, RForeArm, RHand
    4, 11, 12, 13,         # 11-14: LShoulder, LArm, LForeArm, LHand
    0, 15, 16,             # 15-17: RUpLeg, RLeg, RFoot
    0, 18, 19              # 18-20: LUpLeg, LLeg, LFoot
]

BONE_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RShoulder", "RArm", "RForeArm", "RHand",
    "LShoulder", "LArm", "LForeArm", "LHand",
    "RUpLeg", "RLeg", "RFoot",
    "LUpLeg", "LLeg", "LFoot"
]

def parse_bvh_offsets(bvh_file):
    """Parse bone offsets from BVH."""
    offsets = {}
    current_joint = None
    
    bvh_to_bone = {
        "Hips": "Hips", "Spine": "Spine", "Spine1": "Spine1", "Spine2": "Spine2",
        "Spine3": "Spine3", "Neck": "Neck", "Head": "Head",
        "RightShoulder": "RShoulder", "RightArm": "RArm", "RightForeArm": "RForeArm", "RightHand": "RHand",
        "LeftShoulder": "LShoulder", "LeftArm": "LArm", "LeftForeArm": "LForeArm", "LeftHand": "LHand",
        "RightUpLeg": "RUpLeg", "RightLeg": "RLeg", "RightFoot": "RFoot",
        "LeftUpLeg": "LUpLeg", "LeftLeg": "LLeg", "LeftFoot": "LFoot"
    }
    
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("ROOT") or line.startswith("JOINT"):
            parts = line.split()
            if len(parts) >= 2:
                bvh_name = parts[1]
                if bvh_name in bvh_to_bone:
                    current_joint = bvh_to_bone[bvh_name]
        elif line.startswith("OFFSET") and current_joint:
            parts = line.split()
            if len(parts) == 4:
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * INCHES_TO_METERS
                offsets[current_joint] = offset
    
    return offsets

def convert_6d_to_rotation_matrix(rot_6d):
    """Convert 6D rotation to 3x3 rotation matrix."""
    col1 = rot_6d[:3]
    col2 = rot_6d[3:6]
    
    col1_norm = np.linalg.norm(col1)
    if col1_norm > 1e-8:
        col1 = col1 / col1_norm
    else:
        col1 = np.array([1, 0, 0])
    
    col2 = col2 - np.dot(col2, col1) * col1
    col2_norm = np.linalg.norm(col2)
    if col2_norm > 1e-8:
        col2 = col2 / col2_norm
    else:
        col2 = np.array([0, 1, 0])
    
    col3 = np.cross(col1, col2)
    rot_mat = np.column_stack([col1, col2, col3])
    
    return rot_mat

def forward_kinematics(root_rot_6d, child_rots_6d, bone_offsets):
    """Compute joint positions using forward kinematics."""
    positions = np.zeros((21, 3))
    global_rotations = [None] * 21
    
    # Root joint
    root_rot_mat = convert_6d_to_rotation_matrix(root_rot_6d)
    global_rotations[0] = root_rot_mat
    positions[0] = np.array([0, 0, 0])
    
    # Other joints
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        bone_name = BONE_NAMES[i]
        
        # Local rotation
        local_rot_6d = child_rots_6d[i-1]
        local_rot_mat = convert_6d_to_rotation_matrix(local_rot_6d)
        
        # Global rotation
        parent_global_rot = global_rotations[parent_idx]
        global_rotations[i] = parent_global_rot @ local_rot_mat
        
        # Position
        if bone_name in bone_offsets:
            offset = bone_offsets[bone_name]
        else:
            offset = np.array([0.01, 0.01, 0.01])
        
        rotated_offset = parent_global_rot @ offset
        positions[i] = positions[parent_idx] + rotated_offset
    
    return positions

def load_gt_positions(file_path):
    """Load ground truth positions."""
    data = np.loadtxt(file_path, skiprows=1)
    data = data.reshape((data.shape[0], 21, 3))
    return data * INCHES_TO_METERS

def plot_comparison(reconstructed, ground_truth, frame_idx):
    """Plot reconstructed vs ground truth skeleton."""
    fig = plt.figure(figsize=(16, 8))
    
    # Reconstructed
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                c='red', marker='o', s=100, label='Joints')
    
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        ax1.plot([reconstructed[parent_idx, 0], reconstructed[i, 0]],
                 [reconstructed[parent_idx, 1], reconstructed[i, 1]],
                 [reconstructed[parent_idx, 2], reconstructed[i, 2]],
                 'b-', linewidth=2)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Reconstructed - Frame {frame_idx}')
    ax1.view_init(elev=20, azim=45)
    
    # Ground truth
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c='green', marker='o', s=100, label='Joints')
    
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        ax2.plot([ground_truth[parent_idx, 0], ground_truth[i, 0]],
                 [ground_truth[parent_idx, 1], ground_truth[i, 1]],
                 [ground_truth[parent_idx, 2], ground_truth[i, 2]],
                 'g-', linewidth=2)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Ground Truth - Frame {frame_idx}')
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

def main():
    print("Step 1: Loading data...")
    target_data = np.load(TARGET_FILE)
    gt_positions = load_gt_positions(GT_POS_FILE)
    bone_offsets = parse_bvh_offsets(BVH_FILE)
    
    print(f"  Target shape: {target_data.shape}")
    print(f"  GT positions shape: {gt_positions.shape}")
    print(f"  Bone offsets: {len(bone_offsets)}")
    
    print("\nStep 2: Reconstructing skeleton from target data...")
    frame_idx = 0
    frame_data = target_data[frame_idx]
    
    # Get velocity and orientation data
    velocity = frame_data[0, :3]
    root_rot_6d = frame_data[1]
    child_rots_6d = frame_data[2:]
    
    print(f"  Velocity: {velocity}")
    print(f"  Root rotation shape: {root_rot_6d.shape}")
    print(f"  Child rotations shape: {child_rots_6d.shape}")
    
    # Reconstruct positions
    reconstructed = forward_kinematics(root_rot_6d, child_rots_6d, bone_offsets)
    
    print("\nStep 3: Computing errors...")
    gt_frame = gt_positions[frame_idx]
    
    # Compute per-joint errors
    errors = np.linalg.norm(reconstructed - gt_frame, axis=1)
    
    print("\n" + "="*70)
    print(f"{'Joint':<15} | {'Reconstructed (m)':<25} | {'GT (m)':<25} | {'Error (m)':<10}")
    print("-"*70)
    for i in range(21):
        rec_str = f"[{reconstructed[i,0]:7.4f}, {reconstructed[i,1]:7.4f}, {reconstructed[i,2]:7.4f}]"
        gt_str = f"[{gt_frame[i,0]:7.4f}, {gt_frame[i,1]:7.4f}, {gt_frame[i,2]:7.4f}]"
        print(f"{BONE_NAMES[i]:<15} | {rec_str:<25} | {gt_str:<25} | {errors[i]:>9.4f}")
    print("="*70)
    
    print(f"\nMean error: {errors.mean():.4f} m")
    print(f"Max error: {errors.max():.4f} m")
    print(f"Min error: {errors.min():.4f} m")
    
    print("\nStep 4: Plotting comparison...")
    fig = plot_comparison(reconstructed, gt_frame, frame_idx)
    plt.show()
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
