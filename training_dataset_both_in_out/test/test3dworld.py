"""
Test 3D World Coordinates Visualization
Picks a random .npy file from blazepose_coordinates_matched and plots channels 0,1,2 (world x,y,z)
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

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/training_dataset_both_in_out/blazepose_coordinates_matched'

# BlazePose bone connections (25 keypoints)
BLAZEPOSE_BONES = [
    (0, 1), (0, 2),  # Face
    (3, 4), (3, 15), (4, 16), (15, 16),  # Torso
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13),  # Left arm
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14),  # Right arm
    (15, 17), (17, 19), (19, 21), (19, 23), (21, 23),  # Left leg
    (16, 18), (18, 20), (20, 22), (20, 24), (22, 24),  # Right leg
]

# ==========================================
# LOAD RANDOM FILE
# ==========================================
all_files = list(Path(DATA_DIR).rglob("*.npy"))
if not all_files:
    print(f"No .npy files found in {DATA_DIR}")
    exit(1)

random_file = random.choice(all_files)
print(f"Selected: {random_file}")

data = np.load(random_file)  # Shape: (frames, 25, 7)
print(f"Data shape: {data.shape}")

# Extract world coordinates (channels 0,1,2)
world_coords = data[:, :, 0:3]  # (frames, 25, 3)
num_frames = world_coords.shape[0]
print(f"World coords shape: {world_coords.shape}")

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

# Compute axis limits from all frames
all_x = world_coords[:, :, 0].flatten()
all_y = world_coords[:, :, 1].flatten()
all_z = world_coords[:, :, 2].flatten()

margin = 0.2
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D World Coords: {random_file.name}')

def update(val):
    frame = int(slider.val)
    coords = world_coords[frame]
    
    # Update scatter
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    scat._offsets3d = (xs, ys, zs)
    
    # Update bones
    for i, (p1, p2) in enumerate(BLAZEPOSE_BONES):
        if p1 < len(coords) and p2 < len(coords):
            lines[i].set_data_3d(
                [coords[p1, 0], coords[p2, 0]],
                [coords[p1, 1], coords[p2, 1]],
                [coords[p1, 2], coords[p2, 2]]
            )
    
    ax.set_title(f'Frame {frame}/{num_frames-1} | {random_file.name}')
    fig.canvas.draw_idle()

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
slider.on_changed(update)

# Initial draw
update(0)
plt.show()
