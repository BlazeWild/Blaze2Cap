import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print("Warning: TkAgg backend not available.")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Load motion data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'motion.txt')

print("Loading motion.txt...")
motion_data = np.loadtxt(file_path)
if len(motion_data.shape) == 1:
    motion_data = motion_data.reshape(1, -1)

print(f"Loaded {len(motion_data)} frames")

# Extract hip positions (first 3 channels: Xpos, Ypos, Zpos)
hip_positions = motion_data[:, 0:3]

# Extract hip rotations (next 3 channels: Zrot, Xrot, Yrot)
hip_rotations = motion_data[:, 3:6]

print(f"\nHip Position Range:")
print(f"  X: {hip_positions[:,0].min():.2f} to {hip_positions[:,0].max():.2f}")
print(f"  Y: {hip_positions[:,1].min():.2f} to {hip_positions[:,1].max():.2f}")
print(f"  Z: {hip_positions[:,2].min():.2f} to {hip_positions[:,2].max():.2f}")

print(f"\nHip Rotation Range:")
print(f"  Z: {hip_rotations[:,0].min():.2f} to {hip_rotations[:,0].max():.2f}")
print(f"  X: {hip_rotations[:,1].min():.2f} to {hip_rotations[:,1].max():.2f}")
print(f"  Y: {hip_rotations[:,2].min():.2f} to {hip_rotations[:,2].max():.2f}")

# Create 3D plot of hip trajectory
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory as a line
ax.plot(hip_positions[:,0], hip_positions[:,1], hip_positions[:,2], 
        c='blue', linewidth=1, label='Hip Trajectory')

# Plot start and end points
ax.scatter(hip_positions[0,0], hip_positions[0,1], hip_positions[0,2], 
           c='green', s=100, marker='o', label='Start')
ax.scatter(hip_positions[-1,0], hip_positions[-1,1], hip_positions[-1,2], 
           c='red', s=100, marker='x', label='End')

# Plot every 100th frame
sample_indices = range(0, len(hip_positions), 100)
ax.scatter(hip_positions[sample_indices,0], 
           hip_positions[sample_indices,1], 
           hip_positions[sample_indices,2], 
           c='orange', s=20, alpha=0.5, label='Every 100th frame')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Hip Position Trajectory Over All Frames')
ax.legend()

# Make axes equal scale
max_range = np.array([
    hip_positions[:,0].max()-hip_positions[:,0].min(),
    hip_positions[:,1].max()-hip_positions[:,1].min(),
    hip_positions[:,2].max()-hip_positions[:,2].min()
]).max() / 2.0

mid_x = (hip_positions[:,0].max()+hip_positions[:,0].min()) * 0.5
mid_y = (hip_positions[:,1].max()+hip_positions[:,1].min()) * 0.5
mid_z = (hip_positions[:,2].max()+hip_positions[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Create additional plots for position over time
fig2, axes = plt.subplots(3, 1, figsize=(12, 8))
frames = np.arange(len(hip_positions))

axes[0].plot(frames, hip_positions[:,0], 'r-', linewidth=0.5)
axes[0].set_ylabel('X Position')
axes[0].set_title('Hip Position Components Over Time')
axes[0].grid(True)

axes[1].plot(frames, hip_positions[:,1], 'g-', linewidth=0.5)
axes[1].set_ylabel('Y Position')
axes[1].grid(True)

axes[2].plot(frames, hip_positions[:,2], 'b-', linewidth=0.5)
axes[2].set_ylabel('Z Position')
axes[2].set_xlabel('Frame')
axes[2].grid(True)

plt.tight_layout()

# Create rotation plots
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8))

axes3[0].plot(frames, hip_rotations[:,0], 'r-', linewidth=0.5)
axes3[0].set_ylabel('Z Rotation (deg)')
axes3[0].set_title('Hip Rotation Components Over Time')
axes3[0].grid(True)

axes3[1].plot(frames, hip_rotations[:,1], 'g-', linewidth=0.5)
axes3[1].set_ylabel('X Rotation (deg)')
axes3[1].grid(True)

axes3[2].plot(frames, hip_rotations[:,2], 'b-', linewidth=0.5)
axes3[2].set_ylabel('Y Rotation (deg)')
axes3[2].set_xlabel('Frame')
axes3[2].grid(True)

plt.tight_layout()

print("\nShowing plots...")
plt.show()
