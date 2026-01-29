import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os
import sys
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

# Define MediaPipe Pose connections
MP_POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Legs
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

def download_model():
    """Download the MediaPipe pose landmarker model if not present."""
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        print(f"Model not found. Downloading {model_path}...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to {model_path}")
    return model_path

def extract_landmarks_from_video(video_path):
    """
    Extracts 33 pose landmarks from a video file using MediaPipe.
    Returns a numpy array of shape (frames, 33, 4) -> (x, y, z, visibility)
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    # Get model path
    model_path = download_model()
    
    # Create pose landmarker
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    landmarks_list = []
    
    frame_idx = 0
    timestamp_ms = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose with timestamp
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        # Extract 3D world landmarks ONLY (in meters, real-world coordinates)
        frame_landmarks = np.zeros((33, 4)) # x, y, z, visibility
        
        if detection_result.pose_world_landmarks:
            # Use world landmarks for 3D visualization (real-world coordinates in meters)
            # Origin is at the center between hips
            for i, lm in enumerate(detection_result.pose_world_landmarks[0]):
                frame_landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
        # No fallback - we only want 3D world positions, not normalized 2D landmarks
                
        landmarks_list.append(frame_landmarks)
        
        frame_idx += 1
        timestamp_ms += int(1000 / fps) if fps > 0 else 33  # Increment timestamp
        
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    detector.close()
    
    return np.array(landmarks_list)

def visualize_pose_3d(landmarks_data):
    """
    Visualizes captured 3D landmarks with a slider.
    landmarks_data: (frames, 33, 4)
    """
    if landmarks_data is None or len(landmarks_data) == 0:
        print("No landmark data to visualize.")
        return

    num_frames = len(landmarks_data)
    print(f"Visualizing {num_frames} frames...")

    # Setup the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)

    # Initial frame
    current_frame = 0
    
    # Pre-calculate bounds for consistent axis scaling
    # Filter out zeros (missing detections) for bound calculation
    valid_data = landmarks_data[landmarks_data[:, :, 3] > 0.5] # Use visibility threshold
    if len(valid_data) == 0:
        valid_data = landmarks_data[:, :, :3].reshape(-1, 3) # Fallback
        
    all_x = landmarks_data[:, :, 0].flatten()
    all_y = landmarks_data[:, :, 1].flatten()
    all_z = landmarks_data[:, :, 2].flatten()
    
    # Calculate global min/max across all frames
    # MediaPipe world landmarks are in meters with origin at hip center:
    # x: right (positive to the right from person's perspective)
    # y: up (positive upward, contrary to image coordinates)  
    # z: forward (positive towards camera/viewer)
    # We'll visualize as: X=left-right, Z=depth, -Y=height (inverted for upright display)
    
    mid_x = (np.max(all_x) + np.min(all_x)) / 2
    mid_y = (np.max(all_y) + np.min(all_y)) / 2
    mid_z = (np.max(all_z) + np.min(all_z)) / 2
    
    max_range = max(
        np.max(all_x) - np.min(all_x),
        np.max(all_y) - np.min(all_y),
        np.max(all_z) - np.min(all_z)
    ) / 2.0

    def update_plot(frame_idx):
        ax.clear()
        
        # Get data for this frame
        frame_data = landmarks_data[int(frame_idx)]
        
        # Check if we have valid data for this frame (not all zeros)
        if np.all(frame_data == 0):
            ax.text(mid_x, mid_y, mid_z, "No Pose Detected", color='red')
        else:
            xs = frame_data[:, 0]
            ys = frame_data[:, 1]
            # Invert Y and Z for better visualization if needed
            # Usually MediaPipe World: Y points down. We want Y up or Z up.
            # Let's map: MP_y -> -Matplot_y (if we want Y up) or just invert.
            # Let's try plotting raw but invert Y axis limits so 'up' is up.
            zs = frame_data[:, 2]

            # Plot joints
            # Visibility color map
            visibility = frame_data[:, 3]
            colors = ['red' if v < 0.5 else 'blue' for v in visibility]
            
            ax.scatter(xs, zs, -ys, c=colors, marker='o') # Remapping: y -> -z, z -> y for visual upright
            
            # Plot connections
            for connection in MP_POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if visibility[start_idx] > 0.3 and visibility[end_idx] > 0.3:
                    x_pair = [xs[start_idx], xs[end_idx]]
                    y_pair = [-ys[start_idx], -ys[end_idx]] # Inverted Y
                    z_pair = [zs[start_idx], zs[end_idx]]   # Z mapped to Y in plot
                    
                    # Plot: X, Z, -Y to make it look upright and facing correcly
                    ax.plot(x_pair, z_pair, y_pair, color='green')

        ax.set_xlabel('X')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('-Y (Height)')
        ax.set_title(f'Frame: {int(frame_idx)}')
        
        # Set consistent limits
        # Remapped limits: X->X, Z->Y, -Y->Z
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(-(mid_y + max_range), -(mid_y - max_range))

    # Add Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )

    def on_changed(val):
        update_plot(val)

    slider.on_changed(on_changed)
    
    # Initialize
    update_plot(0)
    plt.show()

def main():
    # Use current directory to find the video
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = "test_video.mp4"
    
    # Tries checking in current dir or script dir
    possible_paths = [
        os.path.join(script_dir, video_filename),
        os.path.join(os.getcwd(), video_filename),
        os.path.join(script_dir, "test", video_filename)
    ]
    
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break
            
    if video_path is None:
        # Create a dummy video file for testing? Or just error out. 
        # Since user asked specifically for test_video.mp4, we expect it.
        print(f"Could not find {video_filename} in:")
        for p in possible_paths:
            print(f" - {p}")
        
        # Fallback to file dialog if not found?
        try:
            import tkinter as tk
            from tkinter import filedialog
            print("Opening file dialog to select video...")
            root = tk.Tk()
            root.withdraw()
            video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
            root.destroy()
        except:
            pass

    if video_path and os.path.exists(video_path):
        landmarks = extract_landmarks_from_video(video_path)
        if landmarks is not None:
            visualize_pose_3d(landmarks)
    else:
        print("No valid video file selected.")

if __name__ == "__main__":
    main()
