import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R
import os

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths
TARGET_FILE = os.path.join(WORKSPACE_ROOT, "training_data_targets", "S1", "acting1", "target_S1_acting1_cam1.npy")
BVH_FILE = os.path.join(SCRIPT_DIR, "bvh", "acting1_BlenderZXY_YmZ.bvh")

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

def parse_bvh_offsets(bvh_file):
    """Parse bone offsets (lengths) from BVH file."""
    offsets = {}
    current_joint = None
    
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
    
    # Map BVH joint names to our bone names
    bvh_to_bone = {
        "Hips": "Hips",
        "Spine": "Spine",
        "Spine1": "Spine1",
        "Spine2": "Spine2",
        "Spine3": "Spine3",
        "Neck": "Neck",
        "Head": "Head",
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
    
    for line in lines:
        line = line.strip()
        
        # Check for ROOT or JOINT declarations
        if line.startswith("ROOT") or line.startswith("JOINT"):
            parts = line.split()
            if len(parts) >= 2:
                bvh_name = parts[1]
                if bvh_name in bvh_to_bone:
                    current_joint = bvh_to_bone[bvh_name]
        
        # Check for OFFSET
        elif line.startswith("OFFSET") and current_joint:
            parts = line.split()
            if len(parts) == 4:
                # Convert from inches to meters
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * INCHES_TO_METERS
                offsets[current_joint] = offset
    
    return offsets

def convert_6d_to_rotation_matrix(rot_6d):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rot_6d: (6,) array containing first two columns of rotation matrix
    
    Returns:
        (3, 3) rotation matrix
    """
    # Reshape to get first two columns
    col1 = rot_6d[:3]
    col2 = rot_6d[3:6]
    
    # Normalize first column
    col1_norm = np.linalg.norm(col1)
    if col1_norm > 1e-8:
        col1 = col1 / col1_norm
    else:
        col1 = np.array([1, 0, 0])
    
    # Gram-Schmidt orthogonalization for second column
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

def forward_kinematics(root_rot_6d, child_rots_6d, bone_offsets):
    """
    Compute joint positions using forward kinematics.
    
    Args:
        root_rot_6d: (6,) root orientation in 6D format (camera-space)
        child_rots_6d: (20, 6) child joint LOCAL orientations in 6D format (relative to parent)
        bone_offsets: dict mapping bone names to offset vectors
    
    Returns:
        positions: (21, 3) array of joint positions
    """
    positions = np.zeros((21, 3))
    global_rotations = [None] * 21
    
    # Root joint (index 0)
    root_rot_mat = convert_6d_to_rotation_matrix(root_rot_6d)
    global_rotations[0] = root_rot_mat
    positions[0] = np.array([0, 0, 0])  # Root at origin
    
    # Process all other joints using forward kinematics
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        bone_name = BONE_NAMES[i]
        
        # Get LOCAL rotation for this child (index i-1 in child array)
        local_rot_6d = child_rots_6d[i-1]
        local_rot_mat = convert_6d_to_rotation_matrix(local_rot_6d)
        
        # Compute GLOBAL rotation by composing with parent's global rotation
        # global_child = global_parent * local_child
        parent_global_rot = global_rotations[parent_idx]
        global_rotations[i] = parent_global_rot @ local_rot_mat
        
        # Get bone offset from parent to this joint
        if bone_name in bone_offsets:
            offset = bone_offsets[bone_name]
        else:
            # Default small offset if not found
            offset = np.array([0.01, 0.01, 0.01])
        
        # Transform offset by PARENT's global rotation and add to parent position
        rotated_offset = parent_global_rot @ offset
        positions[i] = positions[parent_idx] + rotated_offset
    
    return positions

def plot_skeleton(positions, frame_num=0, title="Skeleton", ax=None):
    """Plot skeleton in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='red', marker='o', s=100, alpha=0.8, label='Joints')
    
    # Define bone connections with colors
    connections = []
    for i in range(1, 21):
        parent_idx = PARENTS[i]
        connections.append((parent_idx, i))
    
    # Color coding for different body parts
    spine_bones = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]  # Spine chain
    right_arm = [(4, 7), (7, 8), (8, 9), (9, 10)]  # Right arm
    left_arm = [(4, 11), (11, 12), (12, 13), (13, 14)]  # Left arm
    right_leg = [(0, 15), (15, 16), (16, 17)]  # Right leg
    left_leg = [(0, 18), (18, 19), (19, 20)]  # Left leg
    
    # Plot bones with different colors
    for parent_idx, child_idx in spine_bones:
        ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                [positions[parent_idx, 1], positions[child_idx, 1]],
                [positions[parent_idx, 2], positions[child_idx, 2]],
                'b-', linewidth=3, label='Spine' if parent_idx == 0 else '')
    
    for parent_idx, child_idx in right_arm:
        ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                [positions[parent_idx, 1], positions[child_idx, 1]],
                [positions[parent_idx, 2], positions[child_idx, 2]],
                'r-', linewidth=3, label='Right Arm' if parent_idx == 4 else '')
    
    for parent_idx, child_idx in left_arm:
        ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                [positions[parent_idx, 1], positions[child_idx, 1]],
                [positions[parent_idx, 2], positions[child_idx, 2]],
                'g-', linewidth=3, label='Left Arm' if parent_idx == 4 else '')
    
    for parent_idx, child_idx in right_leg:
        ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                [positions[parent_idx, 1], positions[child_idx, 1]],
                [positions[parent_idx, 2], positions[child_idx, 2]],
                'm-', linewidth=3, label='Right Leg' if parent_idx == 0 else '')
    
    for parent_idx, child_idx in left_leg:
        ax.plot([positions[parent_idx, 0], positions[child_idx, 0]],
                [positions[parent_idx, 1], positions[child_idx, 1]],
                [positions[parent_idx, 2], positions[child_idx, 2]],
                'c-', linewidth=3, label='Left Leg' if parent_idx == 0 else '')
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    ax.set_title(f'{title} - Frame {frame_num}', fontsize=12)
    
    # Set equal aspect ratio
    max_range = 1.0  # Fixed range for better viewing
    mid_x = positions[:, 0].mean()
    mid_y = positions[:, 1].mean()
    mid_z = positions[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    return ax

def main():
    print(f"Loading target file: {TARGET_FILE}")
    print(f"Loading BVH file: {BVH_FILE}")
    
    # Load the NPY file
    if not os.path.exists(TARGET_FILE):
        print(f"Error: Target file not found: {TARGET_FILE}")
        return
    
    data = np.load(TARGET_FILE)
    print(f"Data shape: {data.shape}")  # Should be (F, 22, 6)
    num_frames = data.shape[0]
    
    # Parse bone offsets from BVH
    bone_offsets = parse_bvh_offsets(BVH_FILE)
    print(f"Parsed {len(bone_offsets)} bone offsets from BVH")
    
    # Check which bones have offsets
    print("\nBone offsets (meters):")
    for i, bone_name in enumerate(BONE_NAMES):
        if bone_name in bone_offsets:
            offset = bone_offsets[bone_name]
            length = np.linalg.norm(offset)
            print(f"  {i:2d}. {bone_name:15s}: {offset} (length: {length:.4f}m)")
        else:
            print(f"  {i:2d}. {bone_name:15s}: NOT FOUND")
    
    print(f"\nTotal frames: {num_frames}")
    
    # Compute positions for all frames
    print("Computing positions for all frames...")
    all_positions = []
    for frame_idx in range(num_frames):
        frame_data = data[frame_idx]  # (22, 6)
        root_rot_6d = frame_data[1]
        child_rots_6d = frame_data[2:]  # (20, 6)
        positions = forward_kinematics(root_rot_6d, child_rots_6d, bone_offsets)
        all_positions.append(positions)
    
    all_positions = np.array(all_positions)  # (F, 21, 3)
    print(f"All positions shape: {all_positions.shape}")
    
    # Create interactive plot with slider
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial frame
    current_frame = [0]
    
    def update_plot(frame_idx):
        frame_idx = int(frame_idx)
        positions = all_positions[frame_idx]
        plot_skeleton(positions, frame_idx, title=f"S1 Acting1 Camera 1", ax=ax)
        fig.canvas.draw_idle()
    
    # Plot initial frame
    update_plot(0)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)
    slider.on_changed(update_plot)
    
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
