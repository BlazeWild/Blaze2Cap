"""
Test 2D Screen Coordinates Visualization (Anchor-Aware)

Picks a random .npy file from blazepose_coordinates_matched and plots channels 3,4.
Reconstructs positions by accumulating deltas, respecting anchor frames (flag=0).

Logic:
- If Anchor (flag=0): Position = Delta (resets to hip-relative position)
- If Normal (flag=1): Position = Previous Position + Delta
"""
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

# Extract data
delta_screen = data[:, :, 3:5]  # (frames, 25, 2)
anchor_flags = data[:, 0, 6]    # (frames,) - assuming same flag for all keypoints in frame
num_frames = delta_screen.shape[0]

print(f"Compute reconstruction with {np.sum(anchor_flags == 0)} anchor frames...")

# Accumulate deltas with anchor reset logic
screen_positions = np.zeros_like(delta_screen)
prev_pos = np.zeros((25, 2))

for i in range(num_frames):
    # Check anchor flag (0 = anchor, 1 = normal)
    is_anchor = (anchor_flags[i] == 0)
    
    if is_anchor:
        # Anchor frame: Delta IS the position (hip-relative)
        current_pos = delta_screen[i]
    else:
        # Normal frame: Pos = Prev + Delta
        current_pos = prev_pos + delta_screen[i]
    
    screen_positions[i] = current_pos
    prev_pos = current_pos

print(f"Screen positions computed.")

# ==========================================
# VISUALIZATION
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.15)

# Setup scatter and lines
scat = ax.scatter([], [], c='red', s=50)
lines = []
for _ in range(len(BLAZEPOSE_BONES)):
    line, = ax.plot([], [], 'b-', lw=2)
    lines.append(line)

# Compute axis limits from all frames
all_x = screen_positions[:, :, 0].flatten()
all_y = screen_positions[:, :, 1].flatten()

margin = 0.2
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.set_xlabel('Screen X')
ax.set_ylabel('Screen Y')
ax.set_title(f'2D Screen Coords: {random_file.name}')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Draw origin cross
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

def update(val):
    frame = int(slider.val)
    coords = screen_positions[frame]
    is_anchor = (anchor_flags[frame] == 0)
    
    # Update scatter
    xs, ys = coords[:, 0], coords[:, 1]
    scat.set_offsets(np.column_stack([xs, ys]))
    
    # Update bones
    for i, (p1, p2) in enumerate(BLAZEPOSE_BONES):
        if p1 < len(coords) and p2 < len(coords):
            lines[i].set_data(
                [coords[p1, 0], coords[p2, 0]],
                [coords[p1, 1], coords[p2, 1]]
            )
            
    anchor_txt = "[ANCHOR]" if is_anchor else ""
    ax.set_title(f'Frame {frame}/{num_frames-1} {anchor_txt} | {random_file.name}')
    fig.canvas.draw_idle()

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')
slider.on_changed(update)

# Initial draw
update(0)
plt.show()
