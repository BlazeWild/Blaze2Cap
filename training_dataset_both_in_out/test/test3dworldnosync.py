"""
Test 3D World Coordinates Visualization (Nosync)

Visualizes blazepose_25_7_nosync data for a specific file.
Plots channels 0,1,2 (world x,y,z) with coordinate transform for upright viewing.

BlazePose Raw World Coords: X horizontal, Y down, Z depth
Visualization Transform: Plot X=X, Plot Y=Z, Plot Z=-Y
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
import random
from pathlib import Path
import os

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
DATA_DIR = os.path.join(BASE_DIR, 'training_dataset_both_in_out/blazepose_25_7_nosync')

# Specific file to visualize
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='Filename to visualize (e.g. blaze_S3_walking3_cam5.npy)')
args = parser.parse_args()

# BlazePose bone connections (25 keypoints)
# After filtering 33->25, indices change.
# Original indices kept: all except [1,2,3,4,5,6,9,10]
# New map: 0->0, 7->1, 8->2, 11->3, etc.
BLAZEPOSE_BONES = [
    (0, 1), (0, 2),  # Face (nose to ears)
    (3, 4), (3, 15), (4, 16), (15, 16),  # Torso (shoulders, hips)
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13),  # Left arm
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14),  # Right arm
    (15, 17), (17, 19), (19, 21), (19, 23), # Left leg
    (16, 18), (18, 20), (20, 22), (20, 24), # Right leg
]

def main():
    TARGET_FILE_PATH = None
    
    if args.filename:
        # Search for file in DATA_DIR
        found_files = list(Path(DATA_DIR).rglob(args.filename))
        if found_files:
            TARGET_FILE_PATH = found_files[0]
        else:
            print(f"Error: File {args.filename} not found in {DATA_DIR}")
            return
    else:
        # Fallback to defaults
        default_file = os.path.join(DATA_DIR, 'S3/walking3/blaze_S3_walking3_cam5.npy')
        if os.path.exists(default_file):
            TARGET_FILE_PATH = Path(default_file)
        else:
             # Random fallback
            all_files = list(Path(DATA_DIR).rglob("*.npy"))
            if not all_files:
                print(f"No .npy files found in {DATA_DIR}")
                return
            TARGET_FILE_PATH = random.choice(all_files)
            print("No filename specified and default missing, picking random.")

    print(f"Selected: {TARGET_FILE_PATH}")
    
    data = np.load(TARGET_FILE_PATH)  # Shape: (frames, 25, 7)
    print(f"Data shape: {data.shape}")

    # Extract world coordinates (channels 0,1,2)
    world_coords = data[:, :, 0:3]  # (frames, 25, 3)
    num_frames = world_coords.shape[0]

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)

    # Setup scatter and lines
    scat = ax.scatter([], [], [], c='red', s=50)
    lines = []
    for _ in range(len(BLAZEPOSE_BONES)):
        line, = ax.plot([], [], [], 'b-', lw=2)
        lines.append(line)

    def update(val):
        frame = int(slider.val)
        coords = world_coords[frame] # (25, 3) [x, y, z] raw
        
        # Transform for visualization (Z-up)
        # Raw: X=right, Y=down, Z=depth
        # Plot: X=rawX, Y=rawZ(depth), Z=-rawY(up)
        
        plot_x = coords[:, 0]
        plot_y = coords[:, 2] # Use depth as Y
        plot_z = -coords[:, 1] # Use -Y as Z (up)
        
        # Update scatter
        scat._offsets3d = (plot_x, plot_y, plot_z)
        
        # Update bones
        for i, (p1, p2) in enumerate(BLAZEPOSE_BONES):
            if p1 < len(coords) and p2 < len(coords):
                lines[i].set_data_3d(
                    [plot_x[p1], plot_x[p2]],
                    [plot_y[p1], plot_y[p2]],
                    [plot_z[p1], plot_z[p2]]
                )
        
        ax.set_title(f'Frame {frame}/{num_frames-1} | {TARGET_FILE_PATH.name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y (Depth)')
        ax.set_zlabel('Z (Up)')
        
        # Set fixed limits based on assumed range to avoid jitter
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-2, 1)
        
        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
    slider.on_changed(update)

    # Initial draw
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
