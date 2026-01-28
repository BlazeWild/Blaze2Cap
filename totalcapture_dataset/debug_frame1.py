"""
Debug script to check frame 1 reconstruction in detail.
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
GT_ORI_FILE = os.path.join(SCRIPT_DIR, "positions", "S1", "acting1", "gt_skel_gbl_ori.txt")
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

def load_gt_data():
    """Load ground truth positions and orientations."""
    # Load positions
    pos_data = np.loadtxt(GT_POS_FILE, skiprows=1)
    pos_data = pos_data.reshape((pos_data.shape[0], 21, 3)) * INCHES_TO_METERS
    
    # Load orientations  
    ori_data = np.loadtxt(GT_ORI_FILE, skiprows=1)
    ori_data = ori_data.reshape((ori_data.shape[0], 21, 4))
    
    return pos_data, ori_data

def convert_6d_to_rotation_matrix(rot_6d):
    """Convert 6D rotation to 3x3 rotation matrix."""
    col1 = rot_6d[:3]
    col2 = rot_6d[3:6]
    
    # Normalize first column
    col1_norm = np.linalg.norm(col1)
    if col1_norm > 1e-8:
        col1 = col1 / col1_norm
    else:
        col1 = np.array([1, 0, 0])
    
    # Orthogonalize second column
    col2 = col2 - np.dot(col2, col1) * col1
    col2_norm = np.linalg.norm(col2)
    if col2_norm > 1e-8:
        col2 = col2 / col2_norm
    else:
        col2 = np.array([0, 1, 0])
    
    # Third column via cross product
    col3 = np.cross(col1, col2)
    rot_mat = np.column_stack([col1, col2, col3])
    
    return rot_mat

def forward_kinematics(root_rot_6d, child_rots_6d, bone_offsets):
    """Compute joint positions using forward kinematics."""
    positions = np.zeros((21, 3))
    global_rotations = [None] * 21
    
    # Root joint (at origin)
    root_rot_mat = convert_6d_to_rotation_matrix(root_rot_6d)
    global_rotations[0] = root_rot_mat
    positions[0] = np.array([0, 0, 0])
    
    # Other joints
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        bone_name = BONE_NAMES[i]
        
        # Local rotation from target data
        local_rot_6d = child_rots_6d[i-1]
        local_rot_mat = convert_6d_to_rotation_matrix(local_rot_6d)
        
        # Global rotation = parent_global * local
        parent_global_rot = global_rotations[parent_idx]
        global_rotations[i] = parent_global_rot @ local_rot_mat
        
        # Position = parent_pos + parent_global_rot * bone_offset
        if bone_name in bone_offsets:
            offset = bone_offsets[bone_name]
        else:
            offset = np.array([0.01, 0.01, 0.01])  # fallback
        
        rotated_offset = parent_global_rot @ offset
        positions[i] = positions[parent_idx] + rotated_offset
    
    return positions, global_rotations

def plot_comparison(reconstructed, ground_truth, frame_idx):
    """Plot reconstructed vs ground truth side by side."""
    fig = plt.figure(figsize=(18, 8))
    
    # Reconstructed
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                c='red', marker='o', s=100, alpha=0.8, label='Joints')
    
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        ax1.plot([reconstructed[parent_idx, 0], reconstructed[i, 0]],
                 [reconstructed[parent_idx, 1], reconstructed[i, 1]],
                 [reconstructed[parent_idx, 2], reconstructed[i, 2]],
                 'b-', linewidth=2)
    
    for i, name in enumerate(BONE_NAMES):
        ax1.text(reconstructed[i, 0], reconstructed[i, 1], reconstructed[i, 2],
                name, fontsize=6, color='darkred')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Reconstructed from Target Data (Frame {frame_idx})', fontweight='bold')
    ax1.view_init(elev=10, azim=45)
    
    # Ground truth
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c='green', marker='o', s=100, alpha=0.8, label='Joints')
    
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        ax2.plot([ground_truth[parent_idx, 0], ground_truth[i, 0]],
                 [ground_truth[parent_idx, 1], ground_truth[i, 1]],
                 [ground_truth[parent_idx, 2], ground_truth[i, 2]],
                 'g-', linewidth=2)
    
    for i, name in enumerate(BONE_NAMES):
        ax2.text(ground_truth[i, 0], ground_truth[i, 1], ground_truth[i, 2],
                name, fontsize=6, color='darkgreen')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Ground Truth (Frame {frame_idx})', fontweight='bold')
    ax2.view_init(elev=10, azim=45)
    
    # Set same limits for both plots
    all_pos = np.vstack([reconstructed, ground_truth])
    max_range = np.array([
        all_pos[:, 0].max() - all_pos[:, 0].min(),
        all_pos[:, 1].max() - all_pos[:, 1].min(),
        all_pos[:, 2].max() - all_pos[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_pos[:, 0].max() + all_pos[:, 0].min()) * 0.5
    mid_y = (all_pos[:, 1].max() + all_pos[:, 1].min()) * 0.5
    mid_z = (all_pos[:, 2].max() + all_pos[:, 2].min()) * 0.5
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig

def main():
    print("="*80)
    print("DEBUG: Frame 1 - S1 Acting1 Cam1")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    target_data = np.load(TARGET_FILE)
    gt_pos, gt_ori = load_gt_data()
    bone_offsets = parse_bvh_offsets(BVH_FILE)
    
    print(f"   Target data shape: {target_data.shape}")
    print(f"   GT positions shape: {gt_pos.shape}")
    print(f"   GT orientations shape: {gt_ori.shape}")
    
    # Extract frame 1
    frame_idx = 1
    target_frame = target_data[frame_idx]
    gt_pos_frame = gt_pos[frame_idx]
    gt_ori_frame = gt_ori[frame_idx]
    
    print(f"\n2. Frame {frame_idx} data:")
    print(f"   Target frame shape: {target_frame.shape}")
    
    # Check velocity
    velocity = target_frame[0, :3]
    print(f"\n   Row 0 (velocity): {velocity}")
    
    # Check root orientation
    root_rot_6d = target_frame[1]
    print(f"   Row 1 (root orientation 6D): {root_rot_6d}")
    
    # Check child orientations
    child_rots_6d = target_frame[2:]
    print(f"   Rows 2-21 (child orientations) shape: {child_rots_6d.shape}")
    
    # Ground truth
    print(f"\n3. Ground truth frame {frame_idx}:")
    print(f"   Root position: {gt_pos_frame[0]}")
    print(f"   Root orientation (quat): {gt_ori_frame[0]}")
    
    # Print bone lengths from offsets
    print(f"\n4. Bone offsets (from BVH):")
    for i, bone_name in enumerate(BONE_NAMES):
        if bone_name in bone_offsets:
            offset = bone_offsets[bone_name]
            length = np.linalg.norm(offset)
            print(f"   {i:2d}. {bone_name:15s}: length={length:.4f}m, offset={offset}")
        else:
            print(f"   {i:2d}. {bone_name:15s}: NOT FOUND")
    
    # Check computed bone lengths from GT positions
    print(f"\n5. Bone lengths computed from GT positions (frame {frame_idx}):")
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        bone_vector = gt_pos_frame[i] - gt_pos_frame[parent_idx]
        bone_length = np.linalg.norm(bone_vector)
        print(f"   {i:2d}. {BONE_NAMES[i]:15s}: length={bone_length:.4f}m, parent={BONE_NAMES[parent_idx]}")
    
    # Reconstruct skeleton from target data
    print(f"\n6. Reconstructing skeleton from target data...")
    reconstructed_pos, global_rots = forward_kinematics(root_rot_6d, child_rots_6d, bone_offsets)
    
    # Compute errors
    errors = np.linalg.norm(reconstructed_pos - gt_pos_frame, axis=1)
    
    print(f"\n7. Per-joint position errors:")
    print(f"   {'Joint':<15} | {'Reconstructed':<30} | {'Ground Truth':<30} | {'Error (m)'}")
    print(f"   {'-'*95}")
    for i in range(21):
        rec = reconstructed_pos[i]
        gt = gt_pos_frame[i]
        rec_str = f"[{rec[0]:7.4f}, {rec[1]:7.4f}, {rec[2]:7.4f}]"
        gt_str = f"[{gt[0]:7.4f}, {gt[1]:7.4f}, {gt[2]:7.4f}]"
        error_str = f"{errors[i]:8.4f}"
        print(f"   {BONE_NAMES[i]:<15} | {rec_str:<30} | {gt_str:<30} | {error_str}")
    
    print(f"\n8. Error statistics:")
    print(f"   Mean error:  {errors.mean():.4f} m ({errors.mean()*100:.2f} cm)")
    print(f"   Max error:   {errors.max():.4f} m at {BONE_NAMES[np.argmax(errors)]}")
    print(f"   Min error:   {errors.min():.4f} m at {BONE_NAMES[np.argmin(errors)]}")
    
    print("\n" + "="*80)
    print("Showing comparison plot...")
    print("="*80)
    
    # Plot comparison
    fig = plot_comparison(reconstructed_pos, gt_pos_frame, frame_idx)
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
