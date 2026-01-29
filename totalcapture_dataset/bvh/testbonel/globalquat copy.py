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
import json
import re

# ==========================================
# COORDINATE SYSTEM CORRECTION SETTINGS
# ==========================================
APPLY_HIP_CORRECTION = True
HIP_CORRECTION_X = 90
HIP_CORRECTION_Y = 180
HIP_CORRECTION_Z = 0

GLOBAL_POS_MAP = {'X': 'X', 'Y': 'Z', 'Z': 'Y'}
GLOBAL_POS_FLIP = {'X': -1, 'Y': 1, 'Z': -1}

# ==========================================
# 1. PARSE BVH FILE FOR HIERARCHY AND OFFSETS
# ==========================================

def parse_bvh_hierarchy(bvh_file):
    """Parse BVH file to extract hierarchy and offsets (bone structure)"""
    hierarchy = {}
    joint_parsing_order = []
    connections = []
    
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
            current_joint = 'END_SITE'
            
        elif line.startswith('OFFSET'):
            parts = line.split()
            offset = [float(parts[1]), float(parts[2]), float(parts[3])]
            
            if current_joint == 'END_SITE' and len(joint_stack) > 0:
                parent = joint_stack[-1]
                end_name = parent + 'End'
                hierarchy[end_name] = offset + [parent]
                connections.append((parent, end_name))
            elif current_joint and current_joint != 'END_SITE':
                parent = joint_stack[-2] if len(joint_stack) > 1 else None
                hierarchy[current_joint] = offset + [parent]
                if parent:
                    connections.append((parent, current_joint))
                    
        elif line == '}':
            if current_joint == 'END_SITE':
                current_joint = joint_stack[-1] if joint_stack else None
            elif joint_stack:
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None
                
        elif line.startswith('MOTION'):
            break
    
    return hierarchy, joint_parsing_order, connections

# ==========================================
# 2. QUATERNION MATH UTILITIES
# ==========================================

def quaternion_conjugate(quat):
    """
    Compute quaternion conjugate (for unit quaternions, equals inverse)
    Input: [w, x, y, z]
    Output: [w, -x, -y, -z]
    """
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

def quaternion_inverse(quat):
    """
    Compute quaternion inverse
    For unit quaternions: q^-1 = conjugate(q)
    For non-unit: q^-1 = conjugate(q) / |q|^2
    """
    w, x, y, z = quat
    norm_sq = w*w + x*x + y*y + z*z
    conj = quaternion_conjugate(quat)
    return conj / norm_sq if norm_sq > 0 else conj

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2
    Input: q1 = [w1, x1, y1, z1], q2 = [w2, x2, y2, z2]
    Output: q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

# ==========================================
# 3. QUATERNION TO 6D CONVERSION
# ==========================================

def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix
    """
    w, x, y, z = quat
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R

def rotation_matrix_to_6d(R):
    """
    Convert 3x3 rotation matrix to 6D representation
    6D representation: first two columns of rotation matrix
    """
    return R[:, :2].flatten()  # Returns [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]

def quaternion_to_6d(quat):
    """
    Convert quaternion to 6D rotation representation
    """
    R = quaternion_to_rotation_matrix(quat)
    return rotation_matrix_to_6d(R)

def rotation_6d_to_matrix(d6):
    """
    Convert 6D representation back to rotation matrix for visualization
    """
    # Reshape to get first two columns
    col0 = np.array([d6[0], d6[1], d6[2]])
    col1 = np.array([d6[3], d6[4], d6[5]])
    
    # Normalize first column
    col0 = col0 / np.linalg.norm(col0)
    
    # Make col1 orthogonal to col0
    col1 = col1 - np.dot(col0, col1) * col0
    col1 = col1 / np.linalg.norm(col1)
    
    # Third column is cross product
    col2 = np.cross(col0, col1)
    
    R = np.column_stack([col0, col1, col2])
    return R

# ==========================================
# 4. MATH HELPERS
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

def get_translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x], 
                     [0, 1, 0, y], 
                     [0, 0, 1, z], 
                     [0, 0, 0, 1]])

def mat3_to_mat4(R3):
    """Convert 3x3 rotation matrix to 4x4 homogeneous matrix"""
    R4 = np.eye(4)
    R4[:3, :3] = R3
    return R4

# ==========================================
# 5. LOAD DATA AND CAMERA CALIBRATION
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
bvh_file = os.path.join(os.path.dirname(script_dir), 'acting1_BlenderZXY_YmZ.bvh')
quat_file = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/positions/S1/acting1/gt_skel_gbl_ori.txt'
calib_file = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/totalcapture_dataset/calibration_params.json'

print(f"Parsing BVH file: {bvh_file}")
hierarchy, joint_parsing_order, connections = parse_bvh_hierarchy(bvh_file)

# Load camera calibration parameters
print(f"Loading camera calibration: {calib_file}")
with open(calib_file, 'r') as f:
    calib_data = json.load(f)

# Extract camera1 parameters
camera1 = None
for cam in calib_data['cameras']:
    if cam['camera_id'] == 1:
        camera1 = cam
        break

if camera1 is None:
    raise ValueError("Camera 1 not found in calibration file")

# Get rotation matrix and translation vector
R_cam = np.array(camera1['rotation_matrix'])  # 3x3 rotation matrix
t_cam = np.array(camera1['translation_vector'])  # 3x1 translation vector

print(f"Camera 1 calibration loaded:")
print(f"  Rotation matrix shape: {R_cam.shape}")
print(f"  Translation vector: {t_cam}")

# Create 4x4 extrinsic transformation matrix
# P_camera = R_cam * (P_world - t_cam)
# In homogeneous coordinates: P_camera = [R | -R*t] * P_world
T_cam_extrinsic = np.eye(4)
T_cam_extrinsic[:3, :3] = R_cam
T_cam_extrinsic[:3, 3] = -R_cam @ t_cam

print(f"Camera extrinsic matrix (world to camera):")
print(T_cam_extrinsic)

print(f"Loading quaternion data: {quat_file}")
# Read quaternion file
with open(quat_file, 'r') as f:
    lines = f.readlines()

# First line contains joint names
header = lines[0].strip().split('\t')
joint_names_quat = [name.strip() for name in header if name.strip()]

# Read quaternion data (w, x, y, z for each joint)
# Format: each line has tab-separated blocks, each block contains 4 space-separated quaternion values
quat_data = []
for line in lines[1:]:
    if line.strip():
        # Split by tabs to get each joint's quaternion block
        blocks = line.strip().split('\t')
        frame_quats = {}
        for i, joint_name in enumerate(joint_names_quat):
            if i < len(blocks) and blocks[i].strip():
                # Split the block by spaces to get the 4 quaternion values
                quat_values = blocks[i].strip().split()
                if len(quat_values) >= 4:
                    quat = [float(quat_values[0]), float(quat_values[1]), 
                           float(quat_values[2]), float(quat_values[3])]
                    frame_quats[joint_name] = quat
        quat_data.append(frame_quats)

print(f"Loaded {len(quat_data)} frames with quaternions for {len(joint_names_quat)} joints")

# Also load root positions from motion.txt
motion_file = os.path.join(script_dir, 'motion.txt')
print(f"Loading root positions from: {motion_file}")
motion_data = np.loadtxt(motion_file)
if len(motion_data.shape) == 1:
    motion_data = motion_data.reshape(1, -1)

# ==========================================
# 5. CALCULATE POSITIONS USING GLOBAL QUATS
# ==========================================

def calculate_frame_positions_with_global_quats(frame_idx):
    """
    Calculate 3D positions using:
    - Root position from BVH motion data
    - Joint offsets from BVH hierarchy
    - Global quaternions from gt_skel_gbl_ori.txt converted to 6D
    - Hip uses global 6D
    - Other joints use local 6D (derived from parent)
    """
    global_matrices = {}
    positions = {}
    rotation_6d = {}  # Store 6D representations
    
    # Get quaternions for this frame
    frame_quats = quat_data[frame_idx] if frame_idx < len(quat_data) else quat_data[0]
    
    # Get root position from motion data (first 3 values are root position)
    root_pos = motion_data[frame_idx][:3] if frame_idx < len(motion_data) else motion_data[0][:3]
    
    # Store position in dictionary for easy mapping (same as eulerdata_length.py)
    pos = {'X': root_pos[0], 'Y': root_pos[1], 'Z': root_pos[2]}
    
    # Apply position mapping and flips (same as eulerdata_length.py)
    mapped_x = pos[GLOBAL_POS_MAP['X']] * GLOBAL_POS_FLIP['X']
    mapped_y = pos[GLOBAL_POS_MAP['Y']] * GLOBAL_POS_FLIP['Y']
    mapped_z = pos[GLOBAL_POS_MAP['Z']] * GLOBAL_POS_FLIP['Z']
    
    # Transform root position to camera coordinates
    world_pos_homogeneous = np.array([mapped_x, mapped_y, mapped_z, 1.0])
    camera_pos_homogeneous = T_cam_extrinsic @ world_pos_homogeneous
    cam_x, cam_y, cam_z = camera_pos_homogeneous[:3]
    
    for joint in joint_parsing_order:
        offset = hierarchy[joint][:3]
        parent_name = hierarchy[joint][3]
        
        # Get global quaternion for this joint
        if joint in frame_quats:
            global_quat = frame_quats[joint]
            # Convert to 6D
            global_6d = quaternion_to_6d(global_quat)
            # Convert back to rotation matrix for transformation
            R_global_3x3 = rotation_6d_to_matrix(global_6d)
            R_global = mat3_to_mat4(R_global_3x3)
        else:
            # No rotation data, use identity
            R_global = np.eye(4)
            global_6d = rotation_matrix_to_6d(np.eye(3))
        
        if parent_name is None:  # ROOT joint (Hips)
            # For root: use camera-transformed position and global 6D rotation
            T_position = get_translation_matrix(cam_x, cam_y, cam_z)
            
            # Apply hip correction (same order as eulerdata_length.py)
            if APPLY_HIP_CORRECTION:
                # Build correction rotation matrix in Z -> X -> Y order
                R_correction = np.eye(4)
                R_correction = R_correction @ get_rotation_matrix_axis('Z', HIP_CORRECTION_Z)
                R_correction = R_correction @ get_rotation_matrix_axis('X', HIP_CORRECTION_X)
                R_correction = R_correction @ get_rotation_matrix_axis('Y', HIP_CORRECTION_Y)
                M_local = T_position @ R_correction @ R_global
            else:
                M_local = T_position @ R_global
            
            global_matrices[joint] = M_local
            rotation_6d[joint] = global_6d  # Hip uses global 6D
            
        else:  # Child joints
            # For children: calculate LOCAL 6D from global quaternions
            if parent_name in frame_quats and joint in frame_quats:
                parent_quat = frame_quats[parent_name]
                joint_quat = frame_quats[joint]
                
                # Compute LOCAL quaternion: q_local = q_parent^-1 * q_child
                parent_quat_inv = quaternion_inverse(parent_quat)
                local_quat = quaternion_multiply(parent_quat_inv, joint_quat)
                
                # Convert local quaternion to 6D
                local_6d = quaternion_to_6d(local_quat)
                rotation_6d[joint] = local_6d
                
                # Convert to rotation matrix for transformation
                R_local_3x3 = rotation_6d_to_matrix(local_6d)
                R_local_4x4 = mat3_to_mat4(R_local_3x3)
            else:
                R_local_4x4 = np.eye(4)
                rotation_6d[joint] = rotation_matrix_to_6d(np.eye(3))
            
            # Apply offset and local rotation
            T_offset = get_translation_matrix(offset[0], offset[1], offset[2])
            M_local = T_offset @ R_local_4x4
            global_matrices[joint] = global_matrices[parent_name] @ M_local
        
        # Extract position
        pos = global_matrices[joint] @ np.array([0, 0, 0, 1])
        positions[joint] = pos[:3]
    
    # Handle End Sites
    for joint, data in hierarchy.items():
        if joint not in positions:
            offset = data[:3]
            parent_name = data[3]
            if parent_name in global_matrices:
                T_end = get_translation_matrix(*offset)
                M_end = global_matrices[parent_name] @ T_end
                pos_end = M_end @ np.array([0, 0, 0, 1])
                positions[joint] = pos_end[:3]
    
    return positions, rotation_6d

# ==========================================
# 6. PRE-CALCULATE ALL FRAMES
# ==========================================

print(f"Processing {min(len(quat_data), len(motion_data))} frames...")
all_frame_positions = []
all_rotation_6d = []

num_frames = min(len(quat_data), len(motion_data))
for i in range(num_frames):
    if i % 100 == 0:
        print(f"  Frame {i}/{num_frames}")
    pos, rot6d = calculate_frame_positions_with_global_quats(i)
    all_frame_positions.append(pos)
    all_rotation_6d.append(rot6d)

print("Done processing frames!")
print(f"\nCamera Coordinate Transformation:")
print(f"  - Transformed world coordinates to camera1 coordinate frame")
print(f"  - Camera position: {t_cam}")
print(f"\nRotation 6D representation info:")
print(f"  - Hip uses GLOBAL 6D rotation (from global quaternions)")
print(f"  - Other joints use LOCAL 6D rotation (q_local = q_parent^-1 * q_child)")
print(f"  - 6D format: [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]")

# ==========================================
# 7. PLOTTING WITH SLIDER
# ==========================================

fig = plt.figure(figsize=(12, 8))
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

# Fixed scaling
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

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Motion Viewer (Camera1 Coordinates - Global Quaternions → Local 6D)')

# Add text box to display 6D info
info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontfamily='monospace', fontsize=8)

ax_slider = plt.axes([0.25, 0.1, 0.50, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valfmt='%0.0f')

def update(val):
    frame_idx = int(slider.val)
    current_pos = all_frame_positions[frame_idx]
    current_rot6d = all_rotation_6d[frame_idx]
    
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
    
    # Update info text with Hip 6D rotation
    if 'Hips' in current_rot6d:
        hip_6d = current_rot6d['Hips']
        info_str = f"Frame: {frame_idx}\n"
        info_str += f"Hip 6D (global): [{hip_6d[0]:.3f}, {hip_6d[1]:.3f}, {hip_6d[2]:.3f},\n"
        info_str += f"                  {hip_6d[3]:.3f}, {hip_6d[4]:.3f}, {hip_6d[5]:.3f}]"
        info_text.set_text(info_str)
        
    fig.canvas.draw_idle()

slider.on_changed(update)
update(0)  # Initialize with first frame info
plt.show()
