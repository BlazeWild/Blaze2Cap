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
# 1. CAMERA CALIBRATION PARAMETERS
# ==========================================

# --- Camera 1 (commented out) ---
# CAM_ROTATION = np.array([
#     [-0.99713, 0.00504186, -0.0755413],
#     [0.0221672, -0.93461, -0.354982],
#     [-0.0723915, -0.355637, 0.931816]
# ])
# CAM_TRANSLATION = np.array([0.820506, 0.59704, 5.33591])
# CAM_FX, CAM_FY = 1284.32, 1286.38
# CAM_CX, CAM_CY = 959.5, 539.5

# --- Camera 2 (active) ---
CAM_ROTATION = np.array([
    [0.00446316, 0.00528034, -0.999976],
    [0.33666, -0.941619, -0.0034703],
    [-0.941616, -0.336637, -0.00598029]
])
CAM_TRANSLATION = np.array([-2.20664, 1.22766, 3.89912])
CAM_FX, CAM_FY = 1395.1, 1400.23
CAM_CX, CAM_CY = 959.5, 539.5

ACTIVE_CAMERA = 2  # For display title

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def quaternion_to_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    TotalCapture quaternions are in [x, y, z, w] format.
    """
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
    """
    Transform point from world coordinates to camera coordinates.
    P_camera = R_cam @ P_world + T_cam
    """
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

# Load Skeleton Structure
skeleton, joint_order = parse_bvh_structure(bvh_file)

# Load Root Positions
motion_data = np.loadtxt(motion_file)
if motion_data.ndim == 1: motion_data = motion_data.reshape(1, -1)

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

print(f"Processed {len(rotation_6d_data)} frames")

# ==========================================
# 4. FORWARD KINEMATICS + CAMERA TRANSFORM
# ==========================================
def calculate_pose_camera(frame_idx):
    """Calculate pose and transform to Camera 1 coordinates"""
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
    
    # Transform all positions to Camera 1 coordinates
    camera_positions = {}
    for joint, pos_world in world_positions.items():
        camera_positions[joint] = world_to_camera(pos_world, CAM_ROTATION, CAM_TRANSLATION)
    
    return camera_positions

# ==========================================
# 5. VISUALIZATION (Camera 1 View)
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initial Calculation
positions = calculate_pose_camera(0)

# Setup Plots
scat = ax.scatter([], [], [], c='r', s=15)
lines = []
for _ in range(len(skeleton)):
    line, = ax.plot([], [], [], 'k-', lw=1)
    lines.append(line)

# Axis limits for camera coordinates
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(0, 50)
ax.set_xlabel('X (Camera)')
ax.set_ylabel('Y (Camera)')
ax.set_zlabel('Z (Depth)')
ax.set_title(f'Camera {ACTIVE_CAMERA} View - Local 6D Rotations')

def update(val):
    frame = int(slider.val)
    pos_dict = calculate_pose_camera(frame)
    
    # In camera coordinates: X=right, Y=down, Z=forward (depth)
    # For 3D plot, we use: Plot X = Cam X, Plot Y = Cam Z (depth), Plot Z = -Cam Y (up)
    xs = [p[0] for p in pos_dict.values()]
    ys = [p[2] for p in pos_dict.values()]  # Z (depth) as Y in plot
    zs = [-p[1] for p in pos_dict.values()]  # -Y (flip for up direction)
    
    scat._offsets3d = (xs, ys, zs)
    
    # Update Bones
    line_idx = 0
    for child, data in skeleton.items():
        parent = data['parent']
        if parent and parent in pos_dict and child in pos_dict:
            p1 = pos_dict[parent]
            p2 = pos_dict[child]
            
            lx = [p1[0], p2[0]]
            ly = [p1[2], p2[2]]
            lz = [-p1[1], -p2[1]]
            
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
