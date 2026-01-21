import cv2
import mediapipe as mp
import csv
import argparse
import os

# MediaPipe landmark indices we want to keep (12 keypoints)
# Order matches the TotalCapture ground truth keypoints
REQUIRED_MEDIAPIPE_LANDMARKS = [
    (12, "LeftArm"),       # LEFT_SHOULDER -> maps to LeftArm in TotalCapture
    (11, "RightArm"),      # RIGHT_SHOULDER -> maps to RightArm
    (14, "LeftForeArm"),   # LEFT_ELBOW -> maps to LeftForeArm
    (13, "RightForeArm"),  # RIGHT_ELBOW -> maps to RightForeArm
    (16, "LeftHand"),      # LEFT_WRIST -> maps to LeftHand
    (15, "RightHand"),     # RIGHT_WRIST -> maps to RightHand
    (24, "LeftUpLeg"),     # LEFT_HIP -> maps to LeftUpLeg
    (23, "RightUpLeg"),    # RIGHT_HIP -> maps to RightUpLeg
    (26, "LeftLeg"),       # LEFT_KNEE -> maps to LeftLeg
    (25, "RightLeg"),      # RIGHT_KNEE -> maps to RightLeg
    (28, "LeftFoot"),      # LEFT_ANKLE -> maps to LeftFoot
    (27, "RightFoot")      # RIGHT_ANKLE -> maps to RightFoot
]

def extract_pose_landmarks_12kp(video_path, output_csv=None, keypoints_dir="12_keypoints"):
    """
    Extract 12 filtered 3D world landmarks from video using MediaPipe BlazePose.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file (optional, auto-generated if None)
        keypoints_dir: Base directory for saving 12 keypoint files
    """
    # Auto-generate output path if not provided
    if output_csv is None:
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # Extract subject and action from video name (e.g., s1_acting1, s2_rom3)
        parts = video_name.split('_')
        if len(parts) >= 2:
            subject = parts[0]  # e.g., s1, s2, s3
            action = parts[1]   # e.g., acting1, rom3, walking2
            
            # Create directory structure: 12_keypoints/s1/acting1/
            output_dir = os.path.join(keypoints_dir, subject, action)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as blazepose_{videoname}.csv
            output_csv = os.path.join(output_dir, f"blazepose_{video_name}_12kp.csv")
        else:
            # Fallback if naming pattern doesn't match
            output_dir = os.path.join(keypoints_dir, "other")
            os.makedirs(output_dir, exist_ok=True)
            output_csv = os.path.join(output_dir, f"blazepose_{video_name}_12kp.csv")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        # Create header: video_frame_number, mediapipe_predicted_frame, then 12 landmark names
        header = ['video_frame_number', 'mediapipe_predicted_frame'] + [name for _, name in REQUIRED_MEDIAPIPE_LANDMARKS]
        
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        video_frame_number = 0
        mediapipe_predicted_count = 0
        skipped_frames = 0
        
        print("Processing frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(frame_rgb)
            
            # Prepare row data
            row = [video_frame_number]
            
            if results.pose_world_landmarks:
                # MediaPipe successfully detected pose
                row.append(mediapipe_predicted_count)
                
                # Extract only the 12 required landmarks in correct order
                for idx, name in REQUIRED_MEDIAPIPE_LANDMARKS:
                    landmark = results.pose_world_landmarks.landmark[idx]
                    row.append(f"{landmark.x}, {landmark.y}, {landmark.z}")
                
                mediapipe_predicted_count += 1
            else:
                # MediaPipe skipped this frame (no detection)
                row.append(-1)  # -1 indicates skipped frame
                
                # Fill with None for landmarks
                for i in range(12):
                    row.append(None)
                
                skipped_frames += 1
            
            # Write row to CSV
            writer.writerow(row)
            
            video_frame_number += 1
            
            # Progress indicator
            if video_frame_number % 30 == 0:
                progress = (video_frame_number / total_frames) * 100
                print(f"Progress: {video_frame_number}/{total_frames} ({progress:.1f}%) - "
                      f"Detected: {mediapipe_predicted_count}, Skipped: {skipped_frames}")
    
    # Cleanup
    cap.release()
    pose.close()
    
    print(f"\nComplete!")
    print(f"Total video frames: {video_frame_number}")
    print(f"MediaPipe predictions: {mediapipe_predicted_count}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Output saved to: {output_csv}")


def batch_process_videos(video_dir, keypoints_dir="12_keypoints"):
    """
    Process multiple videos in a directory (including subdirectories).
    
    Args:
        video_dir: Directory containing video files (searches recursively)
        keypoints_dir: Base directory to save 12 keypoint CSV files
    """
    if not os.path.exists(keypoints_dir):
        os.makedirs(keypoints_dir)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    
    # Walk through directory recursively
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print('='*60)
        
        # Let the function auto-generate output path
        extract_pose_landmarks_12kp(video_path, output_csv=None, keypoints_dir=keypoints_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract 12 filtered 3D pose landmarks from video using MediaPipe BlazePose'
    )
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output CSV file (optional, auto-generated if not provided)')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos (for batch processing)')
    parser.add_argument('--keypoints_dir', type=str, default='12_keypoints', help='Base directory to save keypoint files')
    
    args = parser.parse_args()
    
    if args.video:
        # Single video processing
        extract_pose_landmarks_12kp(args.video, args.output, args.keypoints_dir)
    elif args.video_dir:
        # Batch processing
        batch_process_videos(args.video_dir, args.keypoints_dir)
    else:
        print("Error: Please provide either --video or --video_dir")
        print("\nExamples:")
        print("  Single video:")
        print("    python extract_blazepose_12kp.py --video s1_acting1.mp4")
        print("\n  Batch processing:")
        print("    python extract_blazepose_12kp.py --video_dir ./videos")
