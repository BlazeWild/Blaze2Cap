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
# 1. SETTINGS & AXIS MAPPING
# ==========================================
# TotalCapture data: motion.txt has positions in a specific coordinate system
# BVH offsets are in the skeleton's local coordinate system
# Global quaternions rotate the offsets to world space

# Position axis mapping - will be applied in visualization only
# Match eulerdata_length.py final output coordinate system
# Raw positions use TotalCapture coordinate system
# Transformation to visualization: X->-X, Y->-Z, Z->Y

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
# --- UPDATE PATHS HERE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes files are in strict locations as per your previous code
bvh_file = os.path.join(os.path.dirname(script_dir), 'acting1_BlenderZXY_YmZ.bvh')
quat_file = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/positions/S1/acting1/gt_skel_gbl_ori.txt'
motion_file = os.path.join(script_dir, 'motion.txt')

# 1. Load Skeleton Structure
skeleton, joint_order = parse_bvh_structure(bvh_file)

# 2. Load Root Positions (Motion.txt)
# Assumes format: [Frame, X, Y, Z, ...] or just [X, Y, Z, ...]
motion_data = np.loadtxt(motion_file)
if motion_data.ndim == 1: motion_data = motion_data.reshape(1, -1)

# 3. Load Global Quaternions
quat_data = []
with open(quat_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split('\t')
    # Filter empty strings from header
    q_names = [h.strip() for h in header if h.strip()]
    
    for line in lines[1:]:
        if not line.strip(): continue
        parts = line.strip().split('\t')
        frame_q = {}
        for i, name in enumerate(q_names):
            if i < len(parts) and parts[i].strip():
                vals = [float(v) for v in parts[i].split()]
                if len(vals) == 4:
                    # TotalCapture format is [x, y, z, w]
                    frame_q[name] = vals 
        quat_data.append(frame_q)

# ==========================================
# 4. FORWARD KINEMATICS (GLOBAL ONLY)
# ==========================================
def calculate_pose(frame_idx):
    # Safe guard for frame index
    idx = min(frame_idx, len(motion_data)-1, len(quat_data)-1)
    
    # 1. Get Root Position from motion.txt - use raw values
    raw_pos = motion_data[idx][:3]
    root_pos = np.array([raw_pos[0], raw_pos[1], raw_pos[2]])
    
    # 2. Get Quaternions for this frame
    current_quats = quat_data[idx]
    
    # Storage for calculated World positions
    world_positions = {}
    
    # We must process in hierarchical order to ensure parents are ready
    # The joint_order from BVH parsing is usually depth-first, which is safe
    for joint in joint_order:
        data = skeleton[joint]
        parent = data['parent']
        
        # Determine Rotation Matrix (R) for the PARENT
        # Why Parent? Because the offset is a vector emerging FROM the parent.
        # We rotate the offset by the Parent's global rotation.
        if parent is None:
            # Root: Position is explicit
            world_positions[joint] = root_pos
        else:
            # Child: Needs Parent Pos + Rotated Offset
            
            # Get Parent's Global Rotation
            # Note: We look up the rotation of the PARENT bone
            if parent in current_quats:
                q = current_quats[parent]
                R_parent = quaternion_to_matrix(q)
            else:
                R_parent = np.eye(3)

            # Calculate Position
            # Pos = Parent_Pos + (R_parent * Offset)
            offset = data['offset']
            
            # --- CRITICAL FIX ---
            # If the skeleton is tangled, the offset vector might not align 
            # with the quaternion's expected axis.
            # Total Capture BVH offsets are usually strictly structural.
            # We trust the Global Quaternion to handle the World Rotation.
            
            rotated_offset = R_parent @ offset
            world_positions[joint] = world_positions[parent] + rotated_offset
            
    # Handle End Sites
    for name, data in skeleton.items():
        if '_End' in name and name not in world_positions:
            parent = data['parent']
            if parent in world_positions:
                if parent in current_quats:
                    R_parent = quaternion_to_matrix(current_quats[parent])
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
# Pre-create line objects
for _ in range(len(skeleton)):
    line, = ax.plot([], [], [], 'k-', lw=1)
    lines.append(line)

# Axis limits and labels - match eulerdata_length.py (Z is vertical)
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_zlim(-10, 50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Calculated from Global Quaternions')

def update(val):
    frame = int(slider.val)
    pos_dict = calculate_pose(frame)
    
    # Transform positions to match eulerdata_length.py coordinate system
    # Remove X negation to fix mirroring issue
    # vis_X = raw_X, vis_Y = -raw_Z, vis_Z = raw_Y
    
    # Update Scatter
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
            
            # Apply same transformation (no X flip)
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