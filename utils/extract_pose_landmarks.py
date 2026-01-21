import cv2
import mediapipe as mp
import csv
import argparse
import os

def extract_pose_landmarks(video_path, output_csv):
    """
    Extract 3D world landmarks from video using MediaPipe BlazePose.
    
    Args:
        video_path: Path to input video file
        output_csv: Path to output CSV file
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0, 1, or 2. Higher = more accurate but slower
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
    
    # MediaPipe landmark names (33 landmarks)
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        # Create header: video_frame_number, mediapipe_predicted_frame, then landmark names
        header = ['video_frame_number', 'mediapipe_predicted_frame'] + landmark_names
        
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
                
                # Extract world landmarks (3D coordinates in meters) - grouped as "x, y, z"
                for landmark in results.pose_world_landmarks.landmark:
                    row.append(f"{landmark.x}, {landmark.y}, {landmark.z}")
                
                mediapipe_predicted_count += 1
            else:
                # MediaPipe skipped this frame (no detection)
                row.append(-1)  # -1 indicates skipped frame
                
                # Fill with None for landmarks
                for i in range(33):
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


def batch_process_videos(video_dir, output_dir):
    """
    Process multiple videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save CSV files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = [f for f in os.listdir(video_dir) 
                   if os.path.splitext(f)[1].lower() in video_extensions]
    
    print(f"Found {len(video_files)} video files")
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {video_file}")
        print('='*60)
        
        video_path = os.path.join(video_dir, video_file)
        output_csv = os.path.join(output_dir, 
                                  os.path.splitext(video_file)[0] + '_landmarks.csv')
        
        extract_pose_landmarks(video_path, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract 3D pose landmarks from video using MediaPipe BlazePose'
    )
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos (for batch processing)')
    parser.add_argument('--output_dir', type=str, help='Directory to save CSV files (for batch processing)')
    
    args = parser.parse_args()
    
    if args.video and args.output:
        # Single video processing
        extract_pose_landmarks(args.video, args.output)
    elif args.video_dir and args.output_dir:
        # Batch processing
        batch_process_videos(args.video_dir, args.output_dir)
    else:
        print("Error: Please provide either (--video and --output) or (--video_dir and --output_dir)")
        print("\nExamples:")
        print("  Single video:")
        print("    python extract_pose_landmarks.py --video input.mp4 --output landmarks.csv")
        print("\n  Batch processing:")
        print("    python extract_pose_landmarks.py --video_dir ./videos --output_dir ./landmarks")
