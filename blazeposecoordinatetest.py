import cv2
import numpy as np
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D

#!/usr/bin/env python3
# blazeposecoordinatetest.py
# Requires: mediapipe, opencv-python, matplotlib, numpy
# Usage: place image.png in same folder and run.

import matplotlib.pyplot as plt

IMAGE_PATH = "image.png"

mp_pose = mp.solutions.pose

# load image
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not open {IMAGE_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# run BlazePose (Pose) to get world landmarks (33 points)
with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5) as pose:
    results = pose.process(img_rgb)

if not results or not results.pose_world_landmarks:
    raise RuntimeError("No pose landmarks detected")

landmarks = results.pose_world_landmarks.landmark  # list of 33
coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)  # world coordinates (meters)

# compute hip center (average of left/right hip: indices 23,24)
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
hip_center = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2.0

# make positions relative to hip center
coords_rel = coords - hip_center

# prepare names ordered by index
landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

# plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("BlazePose (world) landmarks relative to hip (center)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# set axes range as requested
ax.set_xlim3d(-1.5, 1.5)
ax.set_ylim3d(-1.5, 1.5)
ax.set_zlim3d(-1.5, 1.5)

# scatter points and annotate
xs = coords_rel[:, 0]
ys = coords_rel[:, 1]
zs = coords_rel[:, 2]
ax.scatter(xs, ys, zs, c='b', s=30)

for i, (x, y, z) in enumerate(coords_rel):
    # label each landmark with its numeric index only
    ax.text(x, y, z, f"{i}", size=8, zorder=1, color='black')

# draw connections using official POSE_CONNECTIONS
for connection in mp_pose.POSE_CONNECTIONS:
    a = connection[0].value if hasattr(connection[0], "value") else connection[0]
    b = connection[1].value if hasattr(connection[1], "value") else connection[1]
    # In some mediapipe versions POSE_CONNECTIONS stores int tuples, in others PoseLandmark enums.
    # Normalize to ints:
    try:
        a_idx = int(a)
        b_idx = int(b)
    except Exception:
        a_idx, b_idx = connection
    p1 = coords_rel[a_idx]
    p2 = coords_rel[b_idx]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red', linewidth=2)

plt.tight_layout()
plt.show()