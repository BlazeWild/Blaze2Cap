#!/usr/bin/env python3
"""
Test visualization script for GT and Mediapipe predictions.
- Converts GT from inches to meters
- Normalizes GT with hip center at origin (0, 0, 0)
- Applies coordinate transformation to mediapipe: (x, -y, z)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Define skeleton connections
CONNECTIONS = [
    (0, 1),   # LeftArm to RightArm (shoulders)
    (0, 2),   # LeftArm to LeftForeArm
    (1, 3),   # RightArm to RightForeArm
    (2, 4),   # LeftForeArm to LeftHand
    (3, 5),   # RightForeArm to RightHand
    (0, 6),   # LeftArm to LeftUpLeg (torso left)
    (1, 7),   # RightArm to RightUpLeg (torso right)
    (6, 7),   # LeftUpLeg to RightUpLeg (hips)
    (6, 8),   # LeftUpLeg to LeftLeg
    (7, 9),   # RightUpLeg to RightLeg
    (8, 10),  # LeftLeg to LeftFoot
    (9, 11)   # RightLeg to RightFoot
]

# Keypoint names
KEYPOINT_NAMES = [
    'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
    'LeftHand', 'RightHand', 'LeftUpLeg', 'RightUpLeg',
    'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'
]

INCHES_TO_METERS = 0.0254

def parse_coord_string(coord_str):
    """Parse coordinate string 'x, y, z' to numpy array."""
    if not coord_str or coord_str.strip() == "":
        return None
    coords = [float(x.strip()) for x in coord_str.split(',')]
    return np.array(coords)

def load_data(mediapipe_csv, gt_csv):
    """
    Load mediapipe predictions and ground truth data.
    - Converts mediapipe: (x, y, z) → (x, -y, z)
    - Converts GT from inches to meters
    - Normalizes GT with hip center at origin (0, 0, 0)
    """
    mediapipe_data = []
    gt_data = []
    
    print(f"Loading data from:")
    print(f"  Mediapipe: {mediapipe_csv}")
    print(f"  GT: {gt_csv}")
    
    # Load mediapipe data
    # Apply transformation: (-x, -y, z)
    with open(mediapipe_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_info = {
                'video_frame_number': int(row['video_frame_number']),
                'keypoints': []
            }
            for kp_name in KEYPOINT_NAMES:
                coord = parse_coord_string(row[kp_name])
                # Transform mediapipe coordinates: (x, y, z) → (-x, -y, z)
                if coord is not None:
                    coord = np.array([-coord[0], -coord[1], coord[2]])
                frame_info['keypoints'].append(coord)
            mediapipe_data.append(frame_info)
    
    # Load ground truth data
    # Convert from inches to meters and normalize with hip center at origin
    with open(gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        frame_idx = 0
        for row in reader:
            frame_info = {
                'frame_index': frame_idx,
                'keypoints': []
            }
            keypoints = []
            for kp_name in KEYPOINT_NAMES:
                coord = parse_coord_string(row[kp_name])
                # Convert from inches to meters
                if coord is not None:
                    coord = coord * INCHES_TO_METERS
                keypoints.append(coord)
            
            # Calculate hip center (midpoint between LeftUpLeg and RightUpLeg)
            # LeftUpLeg is index 6, RightUpLeg is index 7
            left_hip = keypoints[6]
            right_hip = keypoints[7]
            
            if left_hip is not None and right_hip is not None:
                hip_center = (left_hip + right_hip) / 2.0
                # Normalize all keypoints by subtracting hip center
                for kp in keypoints:
                    if kp is not None:
                        normalized_kp = kp - hip_center
                        frame_info['keypoints'].append(normalized_kp)
                    else:
                        frame_info['keypoints'].append(None)
            else:
                frame_info['keypoints'] = keypoints
            
            gt_data.append(frame_info)
            frame_idx += 1
    
    print(f"\nLoaded {len(mediapipe_data)} mediapipe frames and {len(gt_data)} GT frames")
    print(f"GT converted from inches to meters (×{INCHES_TO_METERS})")
    print(f"Mediapipe coordinates transformed: (x, y, z) → (-x, -y, z)")
    print(f"GT coordinates normalized with hip center at origin (0, 0, 0)")
    return mediapipe_data, gt_data

def draw_skeleton(ax, keypoints, color, label, alpha=1.0):
    """
    Draw skeleton on the 3D axis.
    """
    # Extract valid coordinates
    valid_points = []
    valid_indices = []
    for i, kp in enumerate(keypoints):
        if kp is not None:
            valid_points.append(kp)
            valid_indices.append(i)
    
    if not valid_points:
        return
    
    points = np.array(valid_points)
    
    # Draw keypoints
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=color, marker='o', s=50, label=label, alpha=alpha)
    
    # Draw connections
    for conn in CONNECTIONS:
        i, j = conn
        if keypoints[i] is not None and keypoints[j] is not None:
            line_points = np.array([keypoints[i], keypoints[j]])
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                   c=color, linewidth=2, alpha=alpha)

class KeypointVisualizer:
    """Interactive visualizer with slider."""
    
    def __init__(self, mediapipe_data, gt_data):
        self.mediapipe_data = mediapipe_data
        self.gt_data = gt_data
        self.current_frame = 0
        self.playing = False
        
        # Use minimum length
        self.max_frames = min(len(mediapipe_data), len(gt_data))
        
        # Initialize camera view angles
        self.elev = 20
        self.azim = 45
        
        # Create figure and axis
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Adjust layout to make room for slider
        self.fig.subplots_adjust(bottom=0.2)
        
        # Create slider axis
        ax_slider = self.fig.add_axes([0.15, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=self.max_frames - 1,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self.update_frame)
        
        # Create play/pause button
        ax_button = self.fig.add_axes([0.82, 0.1, 0.08, 0.03])
        self.button = Button(ax_button, 'Play')
        self.button.on_clicked(self.toggle_play)
        
        # Calculate axis limits from all data
        self.calculate_limits()
        
        # Initial plot
        self.update_plot()
        
        # Connect timer for animation
        self.timer = self.fig.canvas.new_timer(interval=50)  # 50ms = ~20 FPS
        self.timer.add_callback(self.animate)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def calculate_limits(self):
        """Calculate axis limits based on all data."""
        all_coords = []
        
        for frame_data in self.mediapipe_data + self.gt_data:
            for kp in frame_data['keypoints']:
                if kp is not None:
                    all_coords.append(kp)
        
        if all_coords:
            coords = np.array(all_coords)
            self.x_min, self.x_max = coords[:, 0].min(), coords[:, 0].max()
            self.y_min, self.y_max = coords[:, 1].min(), coords[:, 1].max()
            self.z_min, self.z_max = coords[:, 2].min(), coords[:, 2].max()
            
            # Add padding
            padding = 0.2
            x_range = self.x_max - self.x_min
            y_range = self.y_max - self.y_min
            z_range = self.z_max - self.z_min
            
            self.x_min -= x_range * padding
            self.x_max += x_range * padding
            self.y_min -= y_range * padding
            self.y_max += y_range * padding
            self.z_min -= z_range * padding
            self.z_max += z_range * padding
        else:
            self.x_min, self.x_max = -1, 1
            self.y_min, self.y_max = -1, 1
            self.z_min, self.z_max = -1, 1
    
    def update_plot(self):
        """Update the 3D plot for current frame."""
        # Save current view angles before clearing (read directly from axis state)
        current_elev = self.ax.elev
        current_azim = self.ax.azim
        
        # Update instance variables
        self.elev = current_elev
        self.azim = current_azim
        
        self.ax.clear()
        
        frame_idx = int(self.current_frame)
        mediapipe_frame = self.mediapipe_data[frame_idx]
        gt_frame = self.gt_data[frame_idx]
        
        # Draw ground truth (blue)
        draw_skeleton(self.ax, gt_frame['keypoints'], 'blue', 'Ground Truth', alpha=0.7)
        
        # Draw mediapipe prediction (red)
        draw_skeleton(self.ax, mediapipe_frame['keypoints'], 'red', 'Mediapipe', alpha=0.9)
        
        # Set labels and title
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title(f'Frame {frame_idx} - Video Frame {mediapipe_frame["video_frame_number"]} - GT Frame {gt_frame["frame_index"]}')
        
        # Set axis limits
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_zlim(self.z_min, self.z_max)
        
        # Add legend
        self.ax.legend()
        
        # Restore the saved viewing angle (preserves user's camera position)
        self.ax.view_init(elev=current_elev, azim=current_azim)
        
        self.fig.canvas.draw_idle()
    
    def update_frame(self, val):
        """Callback for slider change."""
        self.current_frame = int(val)
        self.update_plot()
    
    def toggle_play(self, event):
        """Toggle play/pause animation."""
        self.playing = not self.playing
        if self.playing:
            self.button.label.set_text('Pause')
            self.timer.start()
        else:
            self.button.label.set_text('Play')
            self.timer.stop()
    
    def animate(self):
        """Animation callback."""
        if self.playing:
            self.current_frame = (self.current_frame + 1) % self.max_frames
            self.slider.set_val(self.current_frame)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'left':
            # Previous frame
            new_frame = max(0, self.current_frame - 1)
            self.slider.set_val(new_frame)
        elif event.key == 'right':
            # Next frame
            new_frame = min(self.max_frames - 1, self.current_frame + 1)
            self.slider.set_val(new_frame)
    
    def show(self):
        """Show the window."""
        plt.show()

def visualize_keypoints(mediapipe_csv, gt_csv):
    """
    Create interactive window visualization with frame slider.
    """
    mediapipe_data, gt_data = load_data(mediapipe_csv, gt_csv)
    
    print("\nCreating interactive window visualization...")
    print(f"Total frames: {min(len(mediapipe_data), len(gt_data))}")
    print(f"\nFeatures:")
    print(f"  - Use the slider at the bottom to navigate frames")
    print(f"  - Click 'Play' to animate through frames")
    print(f"  - Drag to rotate, scroll to zoom")
    print(f"  - Blue = Ground Truth, Red = Mediapipe Prediction")
    print(f"\nOpening window...")
    
    visualizer = KeypointVisualizer(mediapipe_data, gt_data)
    visualizer.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize 3D keypoints with ground truth and mediapipe predictions'
    )
    parser.add_argument('--mediapipe', type=str, required=True, 
                       help='Path to mediapipe CSV file (blazepose predictions)')
    parser.add_argument('--gt', type=str, required=True,
                       help='Path to ground truth CSV file (gt_skel_gbl_pos.csv)')
    
    args = parser.parse_args()
    
    visualize_keypoints(args.mediapipe, args.gt)
