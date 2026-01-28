import matplotlib
# Use TkAgg for interactive plotting
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

# ==========================================
# 1. SETTINGS & COORDINATE SYSTEM
# ==========================================
# TotalCapture data: motion.txt has positions, quaternions from gt_skel_gbl_ori.txt
# BVH offsets are in skeleton's local coordinate system
# This script converts global quaternions to:
#   - Global 6D for Hip (root)
#   - Local 6D for all child joints

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def quaternion_to_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    TotalCapture quaternions are in [x, y, z, w] format.
    """
    x, y, z, w = q  # TotalCapture format is [x, y, z, w]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0: return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    
    # Standard quaternion to rotation matrix (right-handed)
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def matrix_to_6d(R):
    """
    Convert 3x3 rotation matrix to 6D representation.
    6D = first two columns of the rotation matrix flattened.
    """
    return np.concatenate([R[:, 0], R[:, 1]])

def rotation_6d_to_matrix(r6d):
    """
    Convert 6D rotation representation back to 3x3 rotation matrix.
    Uses Gram-Schmidt orthogonalization.
    """
    a1 = r6d[:3]
    a2 = r6d[3:6]
    
    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    
    # Orthogonalize and normalize second column
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    
    # Third column is cross product
    b3 = np.cross(b1, b2)
    
    return np.column_stack([b1, b2, b3])

def global_to_local_rotation(R_parent, R_child_global):
    """
    Convert global rotation to local rotation.
    R_local = R_parent^T @ R_child_global
    """
    return R_parent.T @ R_child_global

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
                # Use raw BVH offsets - quaternions handle rotation
                bvh_x = float(parts[1])
                bvh_y = float(parts[2])
                bvh_z = float(parts[3])
                offset = np.array([bvh_x, bvh_y, bvh_z])
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
# 3. LOAD DATA
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
bvh_file = os.path.join(os.path.dirname(script_dir), 'acting1_BlenderZXY_YmZ.bvh')
quat_file = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/positions/S1/acting1/gt_skel_gbl_ori.txt'
motion_file = os.path.join(script_dir, 'motion.txt')

# 1. Load Skeleton Structure
skeleton, joint_order = parse_bvh_structure(bvh_file)

# 2. Load Root Positions (Motion.txt)
motion_data = np.loadtxt(motion_file)
if motion_data.ndim == 1: motion_data = motion_data.reshape(1, -1)

# 3. Load Global Quaternions and convert to 6D (global for hip, local for children)
quat_data = []
rotation_6d_data = []  # Store 6D rotations per frame

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
    
    # First pass: compute all global rotation matrices
    for joint in joint_order:
        if joint in frame_quats:
            R_global = quaternion_to_matrix(frame_quats[joint])
            frame_matrices[joint] = R_global
    
    # Second pass: convert to 6D (global for hip, local for children)
    for joint in joint_order:
        if joint not in frame_matrices:
            continue
            
        data = skeleton[joint]
        parent = data['parent']
        
        if parent is None:
            # Hip: use global 6D
            frame_6d[joint] = {
                'type': 'global',
                '6d': matrix_to_6d(frame_matrices[joint]),
                'matrix': frame_matrices[joint]
            }
        else:
            # Child: compute local 6D
            if parent in frame_matrices:
                R_local = global_to_local_rotation(frame_matrices[parent], frame_matrices[joint])
            else:
                R_local = frame_matrices[joint]  # Fallback to global if parent missing
            
            frame_6d[joint] = {
                'type': 'local',
                '6d': matrix_to_6d(R_local),
                'matrix': R_local  # Store local matrix
            }
    
    rotation_6d_data.append(frame_6d)

print(f"Processed {len(rotation_6d_data)} frames")

# ==========================================
# 4. FORWARD KINEMATICS (using 6D rotations)
# ==========================================
def calculate_pose(frame_idx):
    idx = min(frame_idx, len(motion_data)-1, len(rotation_6d_data)-1)
    
    # 1. Get Root Position from motion.txt - use raw values
    raw_pos = motion_data[idx][:3]
    root_pos = np.array([raw_pos[0], raw_pos[1], raw_pos[2]])
    
    # 2. Get 6D rotations for this frame
    current_6d = rotation_6d_data[idx]
    
    # Storage for calculated World positions and global rotations
    world_positions = {}
    global_rotations = {}
    
    for joint in joint_order:
        data = skeleton[joint]
        parent = data['parent']
        
        if parent is None:
            # Root/Hip: use global position and global 6D rotation
            world_positions[joint] = root_pos
            if joint in current_6d:
                R_global = rotation_6d_to_matrix(current_6d[joint]['6d'])
                global_rotations[joint] = R_global
            else:
                global_rotations[joint] = np.eye(3)
        else:
            # Child: use local 6D rotation
            if joint in current_6d:
                R_local = rotation_6d_to_matrix(current_6d[joint]['6d'])
            else:
                R_local = np.eye(3)
            
            # Compute global rotation: R_global = R_parent_global @ R_local
            if parent in global_rotations:
                R_parent_global = global_rotations[parent]
            else:
                R_parent_global = np.eye(3)
            
            R_global = R_parent_global @ R_local
            global_rotations[joint] = R_global
            
            # Calculate Position using parent's global rotation
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
                
    return world_positions

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initial Calculation
positions = calculate_pose(0)

# Setup Plots
scat = ax.scatter([], [], [], c='r', s=15)
lines = []
for _ in range(len(skeleton)):
    line, = ax.plot([], [], [], 'k-', lw=1)
    lines.append(line)

# Axis limits and labels
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_zlim(-10, 50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Local 6D Rotations (Global Hip, Local Children)')

def update(val):
    frame = int(slider.val)
    pos_dict = calculate_pose(frame)
    
    # Transform positions for visualization
    # vis_X = raw_X, vis_Y = -raw_Z, vis_Z = raw_Y
    xs = [p[0] for p in pos_dict.values()]   # X (no flip)
    ys = [-p[2] for p in pos_dict.values()]  # -Z
    zs = [p[1] for p in pos_dict.values()]   # Y (height becomes Z in plot)
    
    scat._offsets3d = (xs, ys, zs)
    
    # Update Bones
    line_idx = 0
    for child, data in skeleton.items():
        parent = data['parent']
        if parent and parent in pos_dict and child in pos_dict:
            p1 = pos_dict[parent]
            p2 = pos_dict[child]
            
            lx = [p1[0], p2[0]]
            ly = [-p1[2], -p2[2]]
            lz = [p1[1], p2[1]]
            
            if line_idx < len(lines):
                lines[line_idx].set_data(lx, ly)
                lines[line_idx].set_3d_properties(lz)
                line_idx += 1
    
    fig.canvas.draw_idle()

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(motion_data)-1, valinit=0, valfmt='%d')
slider.on_changed(update)

# Initial draw
update(0)
plt.show()
