"""
Test Raw BlazePose vs GT Camera-Child (Root Centered)

Plots:
1. BlazePose (Red): From indices 0,1,2 of blazepose_25_7_nosync
2. GT (Blue): Reconstructed from Camera 6 FK, but with Root Position forced to (0,0,0)

Subject: S1/Freestyle3/Cam6
"""

import matplotlib
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
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
GT_POS_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/positions/S1/freestyle3/gt_skel_gbl_pos.txt')
GT_ORI_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/positions/S1/freestyle3/gt_skel_gbl_ori.txt')
BVH_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/bvh/acting1_BlenderZXY_YmZ.bvh')
CALIB_FILE = os.path.join(BASE_DIR, 'totalcapture_dataset/calibration_params.json')
BLAZE_FILE = os.path.join(BASE_DIR, 'training_dataset_both_in_out/blazepose_25_7_nosync/S1/freestyle3/blaze_S1_freestyle3_cam6.npy')

TARGET_CAM_ID = 6
SCALE_FACTOR = 0.0254 # inches to meters

# Joint names in order (21 joints, GT)
GT_JOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

# Parent indices (GT)
GT_PARENT_INDICES = [
    -1, 0, 1, 2, 3, 4, 5,
    4, 7, 8, 9,
    4, 11, 12, 13,
    0, 15, 16,
    0, 18, 19
]

GT_BONES = []
for i, p_idx in enumerate(GT_PARENT_INDICES):
    if p_idx != -1:
        GT_BONES.append((p_idx, i))

# BlazePose Bones (MediaPipe Topology, 25 points, ignoring 25-32 face/hand details mostly)
# We plot standard skeleton connections
BLAZE_BONES = [
    (0, 1), (0, 2), # Eye/Nose
    (3, 4), (3, 15), (4, 16), (15, 16), # Shoulders/Hips box
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13), # Left Arm
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14), # Right Arm
    (15, 17), (17, 19), (19, 21), (19, 23), # Left Leg
    (16, 18), (18, 20), (20, 22), (20, 24)  # Right Leg
]

# ==========================================
# MATH UTILS
# ==========================================

def quaternion_to_matrix(q):
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n == 0: return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def matrix_to_6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])

def rotation_6d_to_matrix(r6d):
    x_raw = r6d[0:3]
    y_raw = r6d[3:6]
    x = x_raw / np.linalg.norm(x_raw)
    y = y_raw - np.dot(x, y_raw) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))

# ==========================================
# PARSERS & LOADER
# ==========================================

def parse_bvh_offsets(bvh_path):
    offsets = {}
    current_joint = None
    with open(bvh_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ROOT') or line.startswith('JOINT'):
                parts = line.split()
                if len(parts) >= 2:
                    current_joint = parts[1]
            elif line.startswith('OFFSET'):
                if current_joint in GT_JOINT_NAMES:
                    parts = line.split()
                    vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    offsets[current_joint] = vec * SCALE_FACTOR
    for name in GT_JOINT_NAMES:
        if name not in offsets: offsets[name] = np.zeros(3)
    return offsets

def load_calibration():
    with open(CALIB_FILE, 'r') as f:
        data = json.load(f)
    for cam in data['cameras']:
        if cam['camera_id'] == TARGET_CAM_ID:
            return {
                'R': np.array(cam['rotation_matrix']),
                'T': np.array(cam['translation_vector'])
            }
    raise ValueError("Cam not found")

def load_gt_frames(pos_path, ori_path):
    # Retrieve raw frames
    pos_frames = []
    with open(pos_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        names = [x.strip() for x in header if x.strip()]
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if not line.strip(): continue
            frame = {}
            for i, name in enumerate(names):
                if i < len(parts) and parts[i].strip():
                    frame[name] = np.array([float(v) for v in parts[i].split()])
            pos_frames.append(frame)

    ori_frames = []
    with open(ori_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        names = [x.strip() for x in header if x.strip()]
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if not line.strip(): continue
            frame = {}
            for i, name in enumerate(names):
                if i < len(parts) and parts[i].strip():
                    frame[name] = np.array([float(v) for v in parts[i].split()])
            ori_frames.append(frame)
    return pos_frames, ori_frames

# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Comparing BlazePose vs GT (Zero Root) for S1/Freestyle3 Cam {TARGET_CAM_ID}")
    
    # 1. Load Data
    bvh_offsets = parse_bvh_offsets(BVH_FILE)
    calib = load_calibration()
    R_cam = calib['R']
    
    gt_pos_frames, gt_ori_frames = load_gt_frames(GT_POS_FILE, GT_ORI_FILE)
    
    blaze_data = np.load(BLAZE_FILE) # (Frames, 33/25 nodes, C)
    print(f"BlazePose Shape: {blaze_data.shape}")
    
    num_frames = min(len(gt_pos_frames), len(gt_ori_frames), blaze_data.shape[0])
    print(f"Common frames: {num_frames}")
    
    # Reconstruct GT using FK (Modified: Root Pos = 0)
    gt_reconstructed = np.zeros((num_frames, len(GT_JOINT_NAMES), 3))
    
    print("Reconstructing GT...")
    for f in range(num_frames):
        o_data = gt_ori_frames[f]
        
        # --- Root ---
        # FORCED ZERO POSITION for alignment
        hips_cam = np.array([0.0, 0.0, 0.0])
        gt_reconstructed[f, 0, :] = hips_cam
        
        # Root Ori (Still transformed to Camera Frame)
        hips_quat = o_data.get('Hips', np.array([0,0,0,1]))
        R_hips_cam = R_cam @ quaternion_to_matrix(hips_quat)
        
        global_rotations_cam = {'Hips': R_hips_cam}
        
        # --- Children ---
        for i, joint_name in enumerate(GT_JOINT_NAMES):
            if i == 0: continue
            
            p_idx = GT_PARENT_INDICES[i]
            parent_name = GT_JOINT_NAMES[p_idx]
            
            # Global Ori Cam
            q_child = o_data.get(joint_name, np.array([0,0,0,1]))
            R_child_cam = R_cam @ quaternion_to_matrix(q_child)
            global_rotations_cam[joint_name] = R_child_cam
            
            # Local Ori (for correctness check, same as test_cam_6d_gt)
            R_parent_cam = global_rotations_cam[parent_name]
            
            # P_child = P_parent + R_parent_global @ Offset
            parent_pos = gt_reconstructed[f, p_idx, :]
            offset = bvh_offsets[joint_name]
            
            child_pos = parent_pos + R_parent_cam @ offset
            gt_reconstructed[f, i, :] = child_pos

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    # Plot objects
    gt_scat = ax.scatter([], [], [], c='blue', s=20, label='GT (Zero Root)')
    gt_lines = [ax.plot([], [], [], 'b-')[0] for _ in GT_BONES]
    
    blaze_scat = ax.scatter([], [], [], c='red', s=20, label='BlazePose')
    blaze_lines = [ax.plot([], [], [], 'r-')[0] for _ in BLAZE_BONES]
    
    # Set Limits (Auto centering based on all data)
    # Blaze data channels 0,1,2
    blaze_xyz = blaze_data[:num_frames, :, 0:3]
    
    combined_x = np.concatenate([gt_reconstructed[:,:,0].flatten(), blaze_xyz[:,:,0].flatten()])
    combined_y = np.concatenate([gt_reconstructed[:,:,1].flatten(), blaze_xyz[:,:,1].flatten()])
    combined_z = np.concatenate([gt_reconstructed[:,:,2].flatten(), blaze_xyz[:,:,2].flatten()])
    
    max_range = np.array([
        combined_x.max() - combined_x.min(),
        combined_y.max() - combined_y.min(),
        combined_z.max() - combined_z.min()
    ]).max() / 2.0

    mid_x = (combined_x.max() + combined_x.min()) / 2
    mid_y = (combined_y.max() + combined_y.min()) / 2
    mid_z = (combined_z.max() + combined_z.min()) / 2
    
# 3. Set limits so all axes display the exact same volume of space
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 4. CRITICAL: Force the visual aspect ratio to be 1:1:1
    # Without this, Matplotlib stretches the axes to fit the rectangular window
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('BlazePose (Red) vs GT Zero-Root FK (Blue)')
    ax.legend()
    
    def update(val):
        f = int(slider.val)
        
        # GT Update
        g_data = gt_reconstructed[f]
        gt_scat._offsets3d = (g_data[:,0], g_data[:,1], g_data[:,2])
        for line, (p1, p2) in zip(gt_lines, GT_BONES):
            line.set_data([g_data[p1,0], g_data[p2,0]], [g_data[p1,1], g_data[p2,1]])
            line.set_3d_properties([g_data[p1,2], g_data[p2,2]])
            
        # Blaze Update
        b_data = blaze_xyz[f] # (25, 3)
        blaze_scat._offsets3d = (b_data[:,0], b_data[:,1], b_data[:,2])
        for i, (p1, p2) in enumerate(BLAZE_BONES):
            if p1 < len(b_data) and p2 < len(b_data):
                blaze_lines[i].set_data([b_data[p1,0], b_data[p2,0]], [b_data[p1,1], b_data[p2,1]])
                blaze_lines[i].set_3d_properties([b_data[p1,2], b_data[p2,2]])
        
        ax.set_title(f'Frame {f}')
        fig.canvas.draw_idle()
    
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valfmt='%d')
    slider.on_changed(update)
    
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
