"""
Test 3D World Coordinates Visualization (Coordinate Matched)

Visualizes blazepose_coordinates_matched data.
Plots channels 0,1,2 (world x,y,z).
Note: These coordinates are already rotated in the dataset (X, Z, -Y) compared to original raw.
So we can plot them directly to X,Y,Z axes.
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
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
DATA_DIR = os.path.join(BASE_DIR, 'training_dataset_both_in_out/blazepose_coordinates_matched_and_delta')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='Filename to visualize (e.g. blaze_S3_walking3_cam5.npy)')
args = parser.parse_args()

# BlazePose bone connections (25 keypoints)
BLAZEPOSE_BONES = [
    (0, 1), (0, 2),  # Face
    (3, 4), (3, 15), (4, 16), (15, 16),  # Torso
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13),  # Left arm
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14),  # Right arm
    (15, 17), (17, 19), (19, 21), (19, 23), # Left leg
    (16, 18), (18, 20), (20, 22), (20, 24), # Right leg
]

def main():
    target_path = None
    
    if args.filename:
        found_files = list(Path(DATA_DIR).rglob(args.filename))
        if found_files:
            target_path = found_files[0]
        else:
            print(f"Error: File {args.filename} not found in {DATA_DIR}")
            return
    else:
        # Fallback to random
        all_files = list(Path(DATA_DIR).rglob("*.npy"))
        if not all_files:
            print(f"No .npy files found in {DATA_DIR}")
            return
        target_path = random.choice(all_files)
        print("No filename specified, picking random.")

    print(f"Selected: {target_path}")
    
    data = np.load(target_path)  # Shape: (frames, 25, 7)
    print(f"Data shape: {data.shape}")

    # Extract world coordinates (channels 0,1,2)
    # In coordinate_matched dataset, these are already rotated:
    # 0 -> X
    # 1 -> Z (depth)
    # 2 -> -Y (vertical up)
    world_coords = data[:, :, 0:3]
    num_frames = world_coords.shape[0]

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)

    scat = ax.scatter([], [], [], c='blue', s=50)
    lines = []
    for _ in range(len(BLAZEPOSE_BONES)):
        line, = ax.plot([], [], [], 'b-', lw=2)
        lines.append(line)

    def update(val):
        frame = int(slider.val)
        coords = world_coords[frame] # (25, 3)
        
        # Since data is already rotated (X, Depth, Up), we map directly:
        # Plot X = Data[0]
        # Plot Y = Data[1] (Depth)
        # Plot Z = Data[2] (Up)
        
        plot_x = coords[:, 0]
        plot_y = coords[:, 1]
        plot_z = coords[:, 2]
        
        scat._offsets3d = (plot_x, plot_y, plot_z)
        
        for i, (p1, p2) in enumerate(BLAZEPOSE_BONES):
            if p1 < len(coords) and p2 < len(coords):
                lines[i].set_data_3d(
                    [plot_x[p1], plot_x[p2]],
                    [plot_y[p1], plot_y[p2]],
                    [plot_z[p1], plot_z[p2]]
                )
        
        ax.set_title(f'Frame {frame}/{num_frames-1} | {target_path.name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y (Depth)')
        ax.set_zlabel('Z (Up)')
        
        # Set fixed limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 5) # Depth is usually positive
        ax.set_zlim(-2, 2)
        
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
