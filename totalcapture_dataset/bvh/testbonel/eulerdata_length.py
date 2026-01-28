import matplotlib
# Use TkAgg for the interactive window (Native Zoom/Pan support)
try:
    matplotlib.use('TkAgg')
except:
    print("Warning: TkAgg backend not available. Using default.")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import re

# ==========================================
# COORDINATE SYSTEM CORRECTION SETTINGS
# ==========================================
# The hip trajectory shows horizontal movement, but skeleton runs perpendicular to ground
# Apply base rotation to align skeleton with ground plane
APPLY_HIP_CORRECTION = True
HIP_CORRECTION_X = 90   # Rotate 90° around X to align with ground plane
HIP_CORRECTION_Y = 180  # Rotation around Y axis in degrees  
HIP_CORRECTION_Z = 0    # Rotation around Z axis in degrees

# Global position axis mapping and sign correction
# If motion is going up/down instead of on the ground plane, swap Y/Z here.
GLOBAL_POS_MAP = {'X': 'X', 'Y': 'Z', 'Z': 'Y'}
GLOBAL_POS_FLIP = {'X': -1, 'Y': 1, 'Z': -1}

# ==========================================
# 1. PARSE BVH FILE FOR HIERARCHY
# ==========================================

def parse_bvh_hierarchy(bvh_file):
    """Parse BVH file to extract hierarchy and channel information"""
    hierarchy = {}
    joint_parsing_order = []
    connections = []
    joint_channels = {}
    
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
    
    joint_stack = []
    current_joint = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('ROOT') or line.startswith('JOINT'):
            joint_name = line.split()[1]
            joint_parsing_order.append(joint_name)
            current_joint = joint_name
            joint_stack.append(joint_name)
            
        elif line.startswith('End Site'):
            # Mark that we're in an end site
            current_joint = 'END_SITE'
            
        elif line.startswith('OFFSET'):
            parts = line.split()
            offset = [float(parts[1]), float(parts[2]), float(parts[3])]
            
            if current_joint == 'END_SITE' and len(joint_stack) > 0:
                # Create end site name
                parent = joint_stack[-1]
                end_name = parent + 'End'
                hierarchy[end_name] = offset + [parent]
                connections.append((parent, end_name))
            elif current_joint and current_joint != 'END_SITE':
                parent = joint_stack[-2] if len(joint_stack) > 1 else None
                hierarchy[current_joint] = offset + [parent]
                if parent:
                    connections.append((parent, current_joint))
                    
        elif line.startswith('CHANNELS'):
            if current_joint and current_joint != 'END_SITE':
                parts = line.split()
                channel_count = int(parts[1])
                channels = parts[2:2 + channel_count]
                joint_channels[current_joint] = channels
            
        elif line == '}':
            if current_joint == 'END_SITE':
                current_joint = joint_stack[-1] if joint_stack else None
            elif joint_stack:
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None
                
        elif line.startswith('MOTION'):
            break
    
    return hierarchy, joint_parsing_order, connections, joint_channels

# Parse the BVH file
script_dir = os.path.dirname(os.path.abspath(__file__))
bvh_file = os.path.join(os.path.dirname(script_dir), 'acting1_BlenderZXY_YmZ.bvh')

print(f"Parsing BVH file: {bvh_file}")
hierarchy, joint_parsing_order, connections, joint_channels = parse_bvh_hierarchy(bvh_file)

print(f"Found {len(hierarchy)} joints in hierarchy")
print(f"Joint order: {joint_parsing_order}")
print(f"\nCoordinate System Correction:")
print(f"  APPLY_HIP_CORRECTION = {APPLY_HIP_CORRECTION}")
if APPLY_HIP_CORRECTION:
    print(f"  Hip Correction: X={HIP_CORRECTION_X}°, Y={HIP_CORRECTION_Y}°, Z={HIP_CORRECTION_Z}°")
    print(f"  (Edit these values at the top of the script if skeleton looks wrong)")
print()

# ==========================================
# 2. MATH HELPERS
# ==========================================

def get_rotation_matrix_axis(axis, deg):
    angle = np.radians(deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'X':
        return np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])
    if axis == 'Y':
        return np.array([[c, 0, s, 0],
                         [0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [0, 0, 0, 1]])
    if axis == 'Z':
        return np.array([[c, -s, 0, 0],
                         [s, c, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    raise ValueError(f"Unknown rotation axis: {axis}")

def get_rotation_matrix_from_order(rotation_order, angles):
    """
    Build rotation matrix using the actual BVH rotation channel order.
    For channel order [Z, X, Y], apply rotations in that order:
    R = Rz @ Rx @ Ry
    """
    R = np.eye(4)
    for axis in rotation_order:
        R = R @ get_rotation_matrix_axis(axis, angles[axis])
    return R

def get_translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x], 
                     [0, 1, 0, y], 
                     [0, 0, 1, z], 
                     [0, 0, 0, 1]])

def calculate_frame_positions(frame_data):
    """
    Calculate 3D positions for all joints from motion data.
    
    CRITICAL: In BVH format:
    - ROOT joint: Uses global position from motion data + local rotation
    - Child joints: Use bone offset (from hierarchy) + local rotation from motion data
    - The position channels in motion data for non-root joints should be IGNORED
      (or are typically zero in standard BVH files)
    
    NOTE: This BVH file is "YmZ" (Y-mapped-to-Z) format from Blender
    The coordinate system and rotations need special handling for proper visualization
    """
    global_matrices = {}
    positions = {}
    val_idx = 0

    for joint in joint_parsing_order:
        offset = hierarchy[joint][:3]  # Bone offset from parent (bone length/direction)
        parent_name = hierarchy[joint][3]

        channels_list = joint_channels.get(joint, [])
        channels = frame_data[val_idx : val_idx + len(channels_list)]
        val_idx += len(channels_list)

        pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        rot = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        rotation_order = []

        for ch_name, value in zip(channels_list, channels):
            axis = ch_name[0].upper()
            if ch_name.endswith('position'):
                pos[axis] = value
            elif ch_name.endswith('rotation'):
                rot[axis] = value
                rotation_order.append(axis)

        # Build local rotation matrix from euler angles in the correct order
        R_local = get_rotation_matrix_from_order(rotation_order, rot)
        
        # Build local transformation matrix
        if parent_name is None:  # ROOT joint (Hips)
            # For root: Apply global position and rotation directly from motion data
            # Position channels give the global position of the hip in world space
            # Rotation channels give the global orientation of the hip
            
            # Apply global hip position from motion data (with axis mapping and flips)
            mapped_x = pos[GLOBAL_POS_MAP['X']] * GLOBAL_POS_FLIP['X']
            mapped_y = pos[GLOBAL_POS_MAP['Y']] * GLOBAL_POS_FLIP['Y']
            mapped_z = pos[GLOBAL_POS_MAP['Z']] * GLOBAL_POS_FLIP['Z']
            T_position = get_translation_matrix(mapped_x, mapped_y, mapped_z)
            
            # CRITICAL: Apply base rotation correction to align skeleton with ground plane
            # The hip rotations are in a coordinate system where the skeleton runs perpendicular to ground
            if APPLY_HIP_CORRECTION:
                # Build correction rotation matrix (applied BEFORE the motion rotation)
                R_correction = get_rotation_matrix_from_order(
                    ['Z', 'X', 'Y'],
                    {'X': HIP_CORRECTION_X, 'Y': HIP_CORRECTION_Y, 'Z': HIP_CORRECTION_Z}
                )
                # Local transform = Translation * Correction * Rotation
                M_local = T_position @ R_correction @ R_local
            else:
                # Local transform = Translation * Rotation
                M_local = T_position @ R_local
                
            global_matrices[joint] = M_local
        else:  # All child joints
            # For children: use bone offset from hierarchy (ignore motion position data)
            T_offset = get_translation_matrix(offset[0], offset[1], offset[2])
            # Local transform = Offset * Rotation
            M_local = T_offset @ R_local
            # Global transform = Parent's global transform * Local transform
            global_matrices[joint] = global_matrices[parent_name] @ M_local

        # Extract 3D position from transformation matrix
        pos = global_matrices[joint] @ np.array([0, 0, 0, 1])
        positions[joint] = pos[:3]

    # Handle End Sites using fixed offsets from Hierarchy
    for joint, data in hierarchy.items():
        if joint not in positions:
            offset = data[:3]
            parent_name = data[3]
            if parent_name in global_matrices:
                T_end = get_translation_matrix(*offset)
                M_end = global_matrices[parent_name] @ T_end
                pos_end = M_end @ np.array([0, 0, 0, 1])
                positions[joint] = pos_end[:3]
                
    return positions

# ==========================================
# 3. LOAD & PRE-CALCULATE
# ==========================================
print("Loading motion.txt...")
try:
    file_path = os.path.join(script_dir, 'motion.txt')
    motion_data = np.loadtxt(file_path)
    if len(motion_data.shape) == 1:
        motion_data = motion_data.reshape(1, -1)
except Exception as e:
    print(f"Error: {e}. Please ensure 'motion.txt' is in the same folder.")
    exit()

print(f"Processing {len(motion_data)} frames...")
all_frame_positions = []
for i, frame in enumerate(motion_data):
    if i % 100 == 0:
        print(f"  Frame {i}/{len(motion_data)}")
    all_frame_positions.append(calculate_frame_positions(frame))

print("Done processing frames!")

# ==========================================
# 4. PLOTTING WITH SLIDER (FIXED SCALING)
# ==========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initialize Plot Objects (Frame 0)
initial_pos = all_frame_positions[0]
xs = [p[0] for p in initial_pos.values()]
ys = [p[1] for p in initial_pos.values()]
zs = [p[2] for p in initial_pos.values()]

sc = ax.scatter(xs, ys, zs, c='red', s=20)

lines = []
for start, end in connections:
    p1 = initial_pos.get(start, [0,0,0])
    p2 = initial_pos.get(end, [0,0,0])
    line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='black')
    lines.append((line, start, end))

# --- SCALE FIX ---
first_frame_coords = np.array(list(initial_pos.values()))
x_min, x_max = first_frame_coords[:,0].min(), first_frame_coords[:,0].max()
y_min, y_max = first_frame_coords[:,1].min(), first_frame_coords[:,1].max()
z_min, z_max = first_frame_coords[:,2].min(), first_frame_coords[:,2].max()

max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
mid_x = (x_max + x_min) * 0.5
mid_y = (y_max + y_min) * 0.5
mid_z = (z_max + z_min) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Motion Viewer')

ax_slider = plt.axes([0.25, 0.1, 0.50, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, len(motion_data)-1, valinit=0, valfmt='%0.0f')

def update(val):
    frame_idx = int(slider.val)
    current_pos = all_frame_positions[frame_idx]
    
    new_xs = [p[0] for p in current_pos.values()]
    new_ys = [p[1] for p in current_pos.values()]
    new_zs = [p[2] for p in current_pos.values()]
    sc._offsets3d = (new_xs, new_ys, new_zs)
    
    for line, start, end in lines:
        if start in current_pos and end in current_pos:
            p1 = current_pos[start]
            p2 = current_pos[end]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
            
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()