"""
Visualization script to compare:
- BlazePose keypoints (RED) from training_dataset_both_in_out/blazepose_25_7_nosync/S1/acting1/blaze_S1_acting1_cam1.npy
- GT Local 6D keypoints (BLUE) from cam1local6dfromquat.py logic for s1_acting1_cam1
"""
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# ==========================================
# CAMERA CALIBRATION PARAMETERS (Camera 1)
# ==========================================
CAM_ROTATION = np.array([
    [-0.99713, 0.00504186, -0.0755413],
    [0.0221672, -0.93461, -0.354982],
    [-0.0723915, -0.355637, 0.931816]
])
CAM_TRANSLATION = np.array([0.820506, 0.59704, 5.33591])

# Conversion factor: inches to meters
INCH_TO_METER = 0.0254

# BlazePose 25 keypoints bone connections
# Original 33 indices with removed: 1,2,3,4,5,6,9,10
# Mapping from original 33 to new 25 indices:
# Original: 0 -> New: 0 (nose)
# Original: 7 -> New: 1 (left_ear)
# Original: 8 -> New: 2 (right_ear)
# Original: 11 -> New: 3 (left_shoulder)
# Original: 12 -> New: 4 (right_shoulder)
# Original: 13 -> New: 5 (left_elbow)
# Original: 14 -> New: 6 (right_elbow)
# Original: 15 -> New: 7 (left_wrist)
# Original: 16 -> New: 8 (right_wrist)
# Original: 17 -> New: 9 (left_pinky)
# Original: 18 -> New: 10 (right_pinky)
# Original: 19 -> New: 11 (left_index)
# Original: 20 -> New: 12 (right_index)
# Original: 21 -> New: 13 (left_thumb)
# Original: 22 -> New: 14 (right_thumb)
# Original: 23 -> New: 15 (left_hip)
# Original: 24 -> New: 16 (right_hip)
# Original: 25 -> New: 17 (left_knee)
# Original: 26 -> New: 18 (right_knee)
# Original: 27 -> New: 19 (left_ankle)
# Original: 28 -> New: 20 (right_ankle)
# Original: 29 -> New: 21 (left_heel)
# Original: 30 -> New: 22 (right_heel)
# Original: 31 -> New: 23 (left_foot_index)
# Original: 32 -> New: 24 (right_foot_index)

BLAZEPOSE_BONES = [
    # Face
    (0, 1),   # nose -> left_ear
    (0, 2),   # nose -> right_ear
    # Torso
    (3, 4),   # left_shoulder -> right_shoulder
    (3, 15),  # left_shoulder -> left_hip
    (4, 16),  # right_shoulder -> right_hip
    (15, 16), # left_hip -> right_hip
    # Left arm
    (3, 5),   # left_shoulder -> left_elbow
    (5, 7),   # left_elbow -> left_wrist
    (7, 9),   # left_wrist -> left_pinky
    (7, 11),  # left_wrist -> left_index
    (7, 13),  # left_wrist -> left_thumb
    # Right arm
    (4, 6),   # right_shoulder -> right_elbow
    (6, 8),   # right_elbow -> right_wrist
    (8, 10),  # right_wrist -> right_pinky
    (8, 12),  # right_wrist -> right_index
    (8, 14),  # right_wrist -> right_thumb
    # Left leg
    (15, 17), # left_hip -> left_knee
    (17, 19), # left_knee -> left_ankle
    (19, 21), # left_ankle -> left_heel
    (19, 23), # left_ankle -> left_foot_index
    (21, 23), # left_heel -> left_foot_index
    # Right leg
    (16, 18), # right_hip -> right_knee
    (18, 20), # right_knee -> right_ankle
    (20, 22), # right_ankle -> right_heel
    (20, 24), # right_ankle -> right_foot_index
    (22, 24), # right_heel -> right_foot_index
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def quaternion_to_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0: return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def matrix_to_6d(R):
    """Convert 3x3 rotation matrix to 6D representation."""
    return np.concatenate([R[:, 0], R[:, 1]])

def rotation_6d_to_matrix(r6d):
    """Convert 6D rotation back to 3x3 matrix using Gram-Schmidt."""
    a1 = r6d[:3]
    a2 = r6d[3:6]
    
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    
    return np.column_stack([b1, b2, b3])

def global_to_local_rotation(R_parent, R_child_global):
    """Convert global rotation to local: R_local = R_parent^T @ R_child_global"""
    return R_parent.T @ R_child_global

def world_to_camera(point_world, R_cam, T_cam):
    """Transform point from world to camera coordinates."""
    return R_cam @ point_world + T_cam

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
                end_name = parent + '_End'
                hierarchy[end_name] = {'parent': parent, 'children': [], 'offset': np.zeros(3)}
                hierarchy[parent]['children'].append(end_name)
                stack.append(end_name)
            elif line == '}':
                if stack:
                    stack.pop()
                    parent = stack[-1] if stack else None
    
    return hierarchy, joint_names

# ==========================================
# LOAD DATA
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# BlazePose data (25 keypoints x 7 channels, use channels 0,1,2 for x,y,z)
blazepose_file = os.path.join(project_root, 'training_dataset_both_in_out/blazepose_25_7_nosync/S1/acting1/blaze_S1_acting1_cam1.npy')
blazepose_data = np.load(blazepose_file)  # Shape: (frames, 25, 7)
print(f"Loaded BlazePose data: {blazepose_data.shape}")

# GT Local 6D data - load from BVH and quaternions
bvh_file = os.path.join(project_root, 'totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh')
quat_file = os.path.join(project_root, 'totalcapture_dataset/positions/S1/acting1/gt_skel_gbl_ori.txt')
motion_file = os.path.join(project_root, 'totalcapture_dataset/bvh/testbonel/motion.txt')

# Load Skeleton Structure
skeleton, joint_order = parse_bvh_structure(bvh_file)
print(f"Loaded skeleton with {len(joint_order)} joints: {joint_order}")

# Load Root Positions
motion_data = np.loadtxt(motion_file)
if motion_data.ndim == 1: motion_data = motion_data.reshape(1, -1)
print(f"Motion data: {motion_data.shape}")

# Load Global Quaternions and convert to 6D
quat_data = []
rotation_6d_data = []

with open(quat_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split('\t')
    q_names = [h.strip() for h in header if h.strip()]
    
    for line in lines[1:]:
        if not line.strip(): continue
        parts = line.strip().split('\t')
        frame_q = {}
        for i, name in enumerate(q_names):
            if i < len(parts) and parts[i].strip():
                vals = [float(v) for v in parts[i].split()]
                if len(vals) == 4:
                    frame_q[name] = vals 
        quat_data.append(frame_q)

# Pre-compute 6D representations
print("Computing 6D rotations (global for Hip, local for children)...")
for frame_idx, frame_quats in enumerate(quat_data):
    frame_6d = {}
    frame_matrices = {}
    
    for joint in joint_order:
        if joint in frame_quats:
            R_global = quaternion_to_matrix(frame_quats[joint])
            frame_matrices[joint] = R_global
    
    for joint in joint_order:
        if joint not in frame_matrices:
            continue
            
        data = skeleton[joint]
        parent = data['parent']
        
        if parent is None:
            frame_6d[joint] = {
                'type': 'global',
                '6d': matrix_to_6d(frame_matrices[joint]),
                'matrix': frame_matrices[joint]
            }
        else:
            if parent in frame_matrices:
                R_local = global_to_local_rotation(frame_matrices[parent], frame_matrices[joint])
            else:
                R_local = frame_matrices[joint]
            
            frame_6d[joint] = {
                'type': 'local',
                '6d': matrix_to_6d(R_local),
                'matrix': R_local
            }
    
    rotation_6d_data.append(frame_6d)

print(f"Processed {len(rotation_6d_data)} GT frames")

# ==========================================
# FORWARD KINEMATICS + CAMERA TRANSFORM
# ==========================================
def calculate_gt_pose_camera(frame_idx):
    """Calculate GT pose and transform to Camera 1 coordinates"""
    idx = min(frame_idx, len(motion_data)-1, len(rotation_6d_data)-1)
    
    # Get Root Position (world coordinates)
    raw_pos = motion_data[idx][:3]
    root_pos_world = np.array([raw_pos[0], raw_pos[1], raw_pos[2]])
    
    # Get 6D rotations
    current_6d = rotation_6d_data[idx]
    
    # Storage for world positions and global rotations
    world_positions = {}
    global_rotations = {}
    
    for joint in joint_order:
        data = skeleton[joint]
        parent = data['parent']
        
        if parent is None:
            world_positions[joint] = root_pos_world
            if joint in current_6d:
                R_global = rotation_6d_to_matrix(current_6d[joint]['6d'])
                global_rotations[joint] = R_global
            else:
                global_rotations[joint] = np.eye(3)
        else:
            if joint in current_6d:
                R_local = rotation_6d_to_matrix(current_6d[joint]['6d'])
            else:
                R_local = np.eye(3)
            
            if parent in global_rotations:
                R_parent_global = global_rotations[parent]
            else:
                R_parent_global = np.eye(3)
            
            R_global = R_parent_global @ R_local
            global_rotations[joint] = R_global
            
            offset = data['offset']
            rotated_offset = R_parent_global @ offset
            world_positions[joint] = world_positions[parent] + rotated_offset
            
    # Handle End Sites
    for name, data in skeleton.items():
        if '_End' in name and name not in world_positions:
            parent = data['parent']
            if parent in world_positions:
                if parent in global_rotations:
                    R_parent = global_rotations[parent]
                else:
                    R_parent = np.eye(3)
                world_positions[name] = world_positions[parent] + (R_parent @ data['offset'])
    
    # Transform all positions to Camera 1 coordinates and convert inches to meters
    camera_positions = {}
    for joint, pos_world in world_positions.items():
        pos_camera = world_to_camera(pos_world, CAM_ROTATION, CAM_TRANSLATION)
        camera_positions[joint] = pos_camera * INCH_TO_METER  # Convert inches to meters
    
    return camera_positions

def get_blazepose_keypoints(frame_idx):
    """Get BlazePose keypoints (x, y, z) for a frame"""
    idx = min(frame_idx, len(blazepose_data) - 1)
    # Channels 0, 1, 2 are x, y, z
    return blazepose_data[idx, :, :3]  # Shape: (25, 3)

# ==========================================
# VISUALIZATION
# ==========================================
num_frames = min(len(blazepose_data), len(rotation_6d_data))
print(f"Total frames to visualize: {num_frames}")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initial Calculation
gt_positions = calculate_gt_pose_camera(0)
blaze_keypoints = get_blazepose_keypoints(0)

# Setup Plots - GT in BLUE
gt_scat = ax.scatter([], [], [], c='blue', s=30, label='GT Local 6D', alpha=0.8)
gt_lines = []
for _ in range(len(skeleton)):
    line, = ax.plot([], [], [], 'b-', lw=1.5, alpha=0.7)
    gt_lines.append(line)

# BlazePose in RED
blaze_scat = ax.scatter([], [], [], c='red', s=30, label='BlazePose', alpha=0.8)
blaze_lines = []
for _ in range(len(BLAZEPOSE_BONES)):
    line, = ax.plot([], [], [], 'r-', lw=1.5, alpha=0.7)
    blaze_lines.append(line)

# Axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('BlazePose (RED) vs GT Local 6D (BLUE) - S1 Acting1 Cam1')
ax.legend()

# Root motion toggle state
root_motion_enabled = [False]  # Use list to allow modification in nested function

# Store initial hip position for root motion offset
initial_gt_hip_pos = None

def update(val):
    global initial_gt_hip_pos
    frame = int(slider.val)
    
    # Update GT (BLUE)
    pos_dict = calculate_gt_pose_camera(frame)
    
    # Get hip position (first joint is Hips)
    hip_joint = 'Hips'
    if hip_joint in pos_dict:
        hip_pos = pos_dict[hip_joint].copy()
        
        # Store initial hip position on first frame
        if initial_gt_hip_pos is None:
            initial_gt_hip_pos = hip_pos.copy()
        
        # Calculate offset to make hip at origin
        if root_motion_enabled[0]:
            # Root motion ON: offset from initial position (start at 0,0,0)
            offset = initial_gt_hip_pos
        else:
            # Root motion OFF: hip always at origin
            offset = hip_pos
        
        # Apply offset to all joints
        for joint in pos_dict:
            pos_dict[joint] = pos_dict[joint] - offset
    
    # For GT: X=right, Y=down, Z=forward (depth)
    gt_xs = [p[0] for p in pos_dict.values()]
    gt_ys = [p[2] for p in pos_dict.values()]  # Z (depth) as Y in plot
    gt_zs = [-p[1] for p in pos_dict.values()]  # -Y (flip for up direction)
    
    gt_scat._offsets3d = (gt_xs, gt_ys, gt_zs)
    
    # Update GT Bones
    line_idx = 0
    for child, data in skeleton.items():
        parent = data['parent']
        if parent and parent in pos_dict and child in pos_dict:
            p1 = pos_dict[parent]
            p2 = pos_dict[child]
            
            lx = [p1[0], p2[0]]
            ly = [p1[2], p2[2]]
            lz = [-p1[1], -p2[1]]
            
            if line_idx < len(gt_lines):
                gt_lines[line_idx].set_data(lx, ly)
                gt_lines[line_idx].set_3d_properties(lz)
                line_idx += 1
    
    # Update BlazePose (RED) - using channels 0,1,2 for x,y,z
    # Apply 90-degree pitch rotation: swap Y and Z, negate Y to stand upright like GT
    blaze_kp = get_blazepose_keypoints(frame)
    blaze_xs = blaze_kp[:, 0]           # X stays the same
    blaze_ys = blaze_kp[:, 2]           # Y becomes Z (depth)
    blaze_zs = -blaze_kp[:, 1]          # Z becomes -Y (flip vertical)
    
    blaze_scat._offsets3d = (blaze_xs.tolist(), blaze_ys.tolist(), blaze_zs.tolist())
    
    # Update BlazePose Bones with same rotation
    for i, (p1_idx, p2_idx) in enumerate(BLAZEPOSE_BONES):
        if p1_idx < len(blaze_kp) and p2_idx < len(blaze_kp):
            lx = [blaze_kp[p1_idx, 0], blaze_kp[p2_idx, 0]]
            ly = [blaze_kp[p1_idx, 2], blaze_kp[p2_idx, 2]]
            lz = [-blaze_kp[p1_idx, 1], -blaze_kp[p2_idx, 1]]
            blaze_lines[i].set_data(lx, ly)
            blaze_lines[i].set_3d_properties(lz)
    
    motion_status = "ON" if root_motion_enabled[0] else "OFF"
    ax.set_title(f'Frame {frame} - BlazePose (RED) vs GT Local 6D (BLUE) | Root Motion: {motion_status}')
    fig.canvas.draw_idle()

def toggle_root_motion(label):
    root_motion_enabled[0] = not root_motion_enabled[0]
    update(slider.val)

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
slider.on_changed(update)

# Root motion toggle button
ax_check = plt.axes([0.02, 0.4, 0.15, 0.1])
check_button = CheckButtons(ax_check, ['Root Motion'], [False])
check_button.on_clicked(toggle_root_motion)

# Initial draw
update(0)
plt.show()
