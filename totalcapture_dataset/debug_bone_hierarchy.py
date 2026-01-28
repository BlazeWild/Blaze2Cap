"""
Debug script to visualize bone hierarchy and understand how positions are computed
from local 6D rotations and bone offsets.
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

def forward_kinematics_verbose(root_rot_6d, child_rots_6d, bone_offsets):
    """
    Compute joint positions using forward kinematics with detailed logging.
    
    For each bone:
    - Local rotation = stored in target data (6D)
    - Global rotation = parent's global rotation * local rotation
    - Position = parent's position + parent's global rotation * bone offset
    """
    positions = np.zeros((21, 3))
    global_rotations = [None] * 21
    
    print("\n" + "="*100)
    print("FORWARD KINEMATICS STEP-BY-STEP")
    print("="*100)
    
    # Root joint (at origin)
    print("\n[0] Hips (ROOT)")
    print(f"    Position: [0, 0, 0] (fixed at origin)")
    print(f"    Local rotation (6D): {root_rot_6d}")
    root_rot_mat = convert_6d_to_rotation_matrix(root_rot_6d)
    print(f"    Global rotation matrix:")
    print(f"        {root_rot_mat[0]}")
    print(f"        {root_rot_mat[1]}")
    print(f"        {root_rot_mat[2]}")
    
    global_rotations[0] = root_rot_mat
    positions[0] = np.array([0, 0, 0])
    
    # Other joints
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        bone_name = BONE_NAMES[i]
        parent_name = BONE_NAMES[parent_idx]
        
        print(f"\n[{i}] {bone_name} (parent: {parent_name})")
        
        # Local rotation from target data
        local_rot_6d = child_rots_6d[i-1]
        print(f"    Local rotation (6D): {local_rot_6d}")
        local_rot_mat = convert_6d_to_rotation_matrix(local_rot_6d)
        
        # Global rotation = parent_global * local
        parent_global_rot = global_rotations[parent_idx]
        global_rotations[i] = parent_global_rot @ local_rot_mat
        
        # Bone offset
        if bone_name in bone_offsets:
            offset = bone_offsets[bone_name]
        else:
            offset = np.array([0.01, 0.01, 0.01])
            print(f"    WARNING: No offset found, using fallback!")
        
        offset_length = np.linalg.norm(offset)
        print(f"    Bone offset (local): {offset} (length={offset_length:.4f}m)")
        
        # Rotated offset
        rotated_offset = parent_global_rot @ offset
        print(f"    Rotated offset (global): {rotated_offset}")
        
        # Position
        positions[i] = positions[parent_idx] + rotated_offset
        print(f"    Parent position: {positions[parent_idx]}")
        print(f"    Final position: {positions[i]}")
    
    return positions, global_rotations

def plot_skeleton_hierarchy(positions):
    """Plot skeleton with hierarchy visualization."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color code by body part
    colors = {
        'spine': 'red',
        'left_arm': 'blue',
        'right_arm': 'green',
        'left_leg': 'cyan',
        'right_leg': 'magenta',
        'head': 'yellow'
    }
    
    def get_color(idx):
        if idx in [0, 1, 2, 3, 4]:  # Hips to Spine3
            return colors['spine']
        elif idx in [5, 6]:  # Neck, Head
            return colors['head']
        elif idx in [11, 12, 13, 14]:  # Left arm
            return colors['left_arm']
        elif idx in [7, 8, 9, 10]:  # Right arm
            return colors['right_arm']
        elif idx in [18, 19, 20]:  # Left leg
            return colors['left_leg']
        elif idx in [15, 16, 17]:  # Right leg
            return colors['right_leg']
        return 'gray'
    
    # Plot bones
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        color = get_color(i)
        ax.plot([positions[parent_idx, 0], positions[i, 0]],
                [positions[parent_idx, 1], positions[i, 1]],
                [positions[parent_idx, 2], positions[i, 2]],
                color=color, linewidth=3, alpha=0.7)
    
    # Plot joints
    for i in range(21):
        color = get_color(i)
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2],
                  c=color, marker='o', s=150, alpha=0.9, edgecolors='black', linewidths=2)
        
        # Add label
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2],
               f'{i}:{BONE_NAMES[i]}', fontsize=8, color='black', fontweight='bold')
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('Reconstructed Skeleton from Local 6D Rotations\n(Root at Origin)', 
                 fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=10, azim=45)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['spine'], label='Spine'),
        Patch(facecolor=colors['head'], label='Head/Neck'),
        Patch(facecolor=colors['left_arm'], label='Left Arm'),
        Patch(facecolor=colors['right_arm'], label='Right Arm'),
        Patch(facecolor=colors['left_leg'], label='Left Leg'),
        Patch(facecolor=colors['right_leg'], label='Right Leg')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig

def main():
    print("="*100)
    print("BONE HIERARCHY RECONSTRUCTION ANALYSIS")
    print("="*100)
    
    # Load data
    print("\n1. Loading data...")
    target_data = np.load(TARGET_FILE)
    bone_offsets = parse_bvh_offsets(BVH_FILE)
    
    print(f"   Target shape: {target_data.shape}")
    print(f"   Bone offsets loaded: {len(bone_offsets)}")
    
    # Frame 1
    frame_idx = 1
    target_frame = target_data[frame_idx]
    
    print(f"\n2. Frame {frame_idx} data:")
    velocity = target_frame[0, :3]
    root_rot_6d = target_frame[1]
    child_rots_6d = target_frame[2:]
    
    print(f"   Velocity: {velocity}")
    print(f"   Root rotation (6D): {root_rot_6d}")
    print(f"   Child rotations shape: {child_rots_6d.shape}")
    
    # Reconstruct with verbose logging
    print(f"\n3. Computing forward kinematics...")
    positions, global_rots = forward_kinematics_verbose(root_rot_6d, child_rots_6d, bone_offsets)
    
    # Summary
    print("\n" + "="*100)
    print("FINAL POSITIONS SUMMARY")
    print("="*100)
    for i in range(21):
        parent_idx = PARENTS[i]
        if parent_idx >= 0:
            parent_name = BONE_NAMES[parent_idx]
            bone_length = np.linalg.norm(positions[i] - positions[parent_idx])
            print(f"[{i:2d}] {BONE_NAMES[i]:15s}: pos={positions[i]}, "
                  f"parent={parent_name}, bone_length={bone_length:.4f}m")
        else:
            print(f"[{i:2d}] {BONE_NAMES[i]:15s}: pos={positions[i]} (ROOT)")
    
    # Plot
    print("\n" + "="*100)
    print("Showing visualization...")
    print("="*100)
    fig = plot_skeleton_hierarchy(positions)
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
