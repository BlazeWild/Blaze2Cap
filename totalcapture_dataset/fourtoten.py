"""
Extract all 33 BlazePose 3D world landmarks for TotalCapture dataset videos.

Output format:
- Shape: (Total_Frames, 33, 10) for each video
- 10 Channels: [World X, Y, Z, World Vis, Anchor, Screen X, Y, Z, Screen Vis, Anchor Backup]
- World 3D: Real 3D positions relative to hips (pose_world_landmarks) in meters
- Screen 2D: Normalized screen coordinates (pose_landmarks) in [0, 1]
- Anchor: 0 if new/start frame, 1 if continuous tracking
- Missing Data: If a frame is missing, all 10 channels are [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- Format: Float32 numpy arrays saved as .npy files
- Naming: blaze_S1_acting1_cam1.npy (matching video name)
- GPU Optimized: Batch processing for T4 GPU
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc


def extract_all_keypoints_from_video(
    video_path,
    output_dir,
    model_complexity=2,
    max_side=720,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    cv_threads=0
):
    """
    Extract all 33 BlazePose 3D world landmarks from a single video.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output .npy file
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Parse video filename to create output name
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]  # e.g., TC_S1_acting1_cam1
    
    # Create output filename: blaze_S1_acting1_cam1.npy
    if video_name.startswith("TC_"):
        output_name = f"blaze_{video_name[3:]}.npy"  # Remove "TC_" prefix
    else:
        output_name = f"blaze_{video_name}.npy"
    
    output_path = os.path.join(output_dir, output_name)
    
    # Skip if already processed
    if os.path.exists(output_path):
        print(f"  ⏭️  Skipping (already exists): {output_name}")
        return True
    
    # OpenCV optimizations
    cv2.setUseOptimized(True)
    if cv_threads is not None and cv_threads >= 0:
        cv2.setNumThreads(cv_threads)

    # Initialize MediaPipe Pose with optimized settings
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Error: Could not open video {video_path}")
        pose.close()
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_dim = None
    if max_side and orig_w > 0 and orig_h > 0:
        max_orig_side = max(orig_w, orig_h)
        if max_orig_side > max_side:
            scale = max_side / max_orig_side
            resize_dim = (int(orig_w * scale), int(orig_h * scale))
    
    # Initialize output array: (total_frames, 33, 10) with zeros
    # Shape: [frame_idx, landmark_idx, (world_x, world_y, world_z, world_vis, anchor, screen_x, screen_y, screen_z, screen_vis, anchor_backup)]
    landmarks_array = np.zeros((total_frames, 33, 10), dtype=np.float32)
    
    frame_idx = 0
    processed_count = 0
    previous_frame_detected = False  # Track if previous frame had detection
    
    # Process frames with progress bar
    with tqdm(total=total_frames, desc=f"  Processing {output_name}", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if resize_dim is not None:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Process with MediaPipe
            results = pose.process(frame_rgb)
            
            if results.pose_world_landmarks and results.pose_landmarks:
                # Extract all 33 landmarks with 10-channel schema
                # Anchor flag: 0 if new/start, 1 if continuous tracking
                anchor_flag = 1.0 if previous_frame_detected else 0.0
                
                world_landmarks = results.pose_world_landmarks.landmark
                screen_landmarks = results.pose_landmarks.landmark
                
                for landmark_idx in range(min(33, len(world_landmarks), len(screen_landmarks))):
                    world_lm = world_landmarks[landmark_idx]
                    screen_lm = screen_landmarks[landmark_idx]
                    
                    # Channels 0-2: World 3D coordinates (meters)
                    landmarks_array[frame_idx, landmark_idx, 0] = world_lm.x
                    landmarks_array[frame_idx, landmark_idx, 1] = world_lm.y
                    landmarks_array[frame_idx, landmark_idx, 2] = world_lm.z
                    
                    # Channel 3: World visibility
                    landmarks_array[frame_idx, landmark_idx, 3] = world_lm.visibility
                    
                    # Channel 4: Anchor flag
                    landmarks_array[frame_idx, landmark_idx, 4] = anchor_flag
                    
                    # Channels 5-6: Screen 2D normalized coordinates
                    landmarks_array[frame_idx, landmark_idx, 5] = screen_lm.x
                    landmarks_array[frame_idx, landmark_idx, 6] = screen_lm.y
                    
                    # Channel 7: Screen relative depth
                    landmarks_array[frame_idx, landmark_idx, 7] = screen_lm.z
                    
                    # Channel 8: Screen visibility
                    landmarks_array[frame_idx, landmark_idx, 8] = screen_lm.visibility
                    
                    # Channel 9: Backup anchor flag
                    landmarks_array[frame_idx, landmark_idx, 9] = anchor_flag
                
                processed_count += 1
                previous_frame_detected = True  # Current frame detected, set for next frame
            else:
                # Frame was skipped - leave as zeros (all 10 channels)
                previous_frame_detected = False  # Current frame not detected
            
            frame_idx += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    pose.close()
    
    # Save as numpy array in float32 format
    np.save(output_path, landmarks_array)
    
    success_rate = (processed_count / total_frames * 100) if total_frames > 0 else 0
    print(f"  ✅ Saved: {output_name} | Frames: {total_frames} | Detected: {processed_count} ({success_rate:.1f}%)")
    
    # Force garbage collection to free GPU memory
    gc.collect()
    
    return True


def _process_single_video(args):
    return extract_all_keypoints_from_video(*args)


def process_subject_action(
    subject_dir,
    action,
    output_base_dir,
    model_complexity=2,
    max_side=720,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    cv_threads=0,
    workers=8
):
    """
    Process all camera videos for a specific subject and action.
    
    Args:
        subject_dir: Path to subject directory (e.g., totalcapture_dataset/Videos/s1)
        action: Action name (e.g., acting1, walking2)
        output_base_dir: Base directory for output files
        
    Returns:
        int: Number of videos successfully processed
    """
    action_dir = os.path.join(subject_dir, action)
    
    if not os.path.isdir(action_dir):
        return 0
    
    # Get subject name from path
    subject_name = os.path.basename(subject_dir).upper()  # e.g., S1, S2, S3
    
    # Create output directory structure: main_all_keypoints/blazepose/S1/acting1/
    output_dir = os.path.join(output_base_dir, "blazepose", subject_name, action)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files for this action
    video_files = sorted([f for f in os.listdir(action_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    
    if not video_files:
        print(f"  ⚠️  No videos found in {action_dir}")
        return 0
    
    print(f"\n{'='*80}")
    print(f"📹 Processing: {subject_name}/{action} ({len(video_files)} videos)")
    print('='*80)
    
    success_count = 0
    task_args = []
    for video_file in video_files:
        video_path = os.path.join(action_dir, video_file)
        task_args.append((
            video_path,
            output_dir,
            model_complexity,
            max_side,
            min_detection_confidence,
            min_tracking_confidence,
            cv_threads
        ))

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_single_video, args) for args in task_args]
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
    else:
        for args in task_args:
            if _process_single_video(args):
                success_count += 1
    
    return success_count


def batch_process_all_videos(
    videos_base_dir,
    output_base_dir,
    model_complexity=2,
    max_side=720,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    cv_threads=0,
    workers=8,
    start_from=None
):
    """
    Process all videos in the TotalCapture dataset structure.
    
    Args:
        videos_base_dir: Base directory containing s1, s2, s3 folders
        output_base_dir: Base directory for output (main_all_keypoints)
        start_from: Tuple (subject, action) to start from, e.g., ('s2', 'rom2')
    """
    # TotalCapture structure: Videos/s1/acting1/, Videos/s1/acting2/, etc.
    subjects = ['s1', 's2', 's3']
    actions = ['acting1', 'acting2', 'acting3', 
               'freestyle1', 'freestyle2', 'freestyle3',
               'rom1', 'rom2', 'rom3',
               'walking1', 'walking2', 'walking3']
    
    total_videos_processed = 0
    total_videos_expected = 0
    
    print("\n" + "="*80)
    print("🚀 STARTING BLAZEPOSE EXTRACTION - ALL 33 KEYPOINTS")
    print("="*80)
    print(f"📂 Input: {videos_base_dir}")
    print(f"💾 Output: {output_base_dir}/blazepose/")
    print(f"🎯 Format: (Frames, 33, 10) - Float32 - 10-Channel Schema")
    print(f"🌍 3D World Positions: Relative to hips (pose_world_landmarks)")
    print(f"🖥️  GPU: Optimized for L4 24GB with batch processing")
    print("="*80)
    
    # Determine if we should skip to a specific starting point
    should_skip = start_from is not None
    start_subject, start_action = start_from if start_from else (None, None)
    
    for subject in subjects:
        subject_dir = os.path.join(videos_base_dir, subject)
        
        if not os.path.isdir(subject_dir):
            print(f"⚠️  Warning: Subject directory not found: {subject_dir}")
            continue
        
        # Skip subjects before start_subject
        if should_skip and subject != start_subject:
            if subjects.index(subject) < subjects.index(start_subject):
                print(f"\n⏭️  Skipping subject: {subject.upper()} (already processed)")
                continue
        
        print(f"\n{'#'*80}")
        print(f"👤 SUBJECT: {subject.upper()}")
        print('#'*80)
        
        for action in actions:
            # Skip actions before start_action for the start_subject
            if should_skip and subject == start_subject and action != start_action:
                if actions.index(action) < actions.index(start_action):
                    print(f"  ⏭️  Skipping {action} (already processed)")
                    continue
            
            # Once we reach the start point, stop skipping
            if should_skip and subject == start_subject and action == start_action:
                should_skip = False
                print(f"  ▶️  Resuming from: {subject.upper()}/{action}")
            # Count expected videos
            action_dir = os.path.join(subject_dir, action)
            if os.path.isdir(action_dir):
                video_count = len([f for f in os.listdir(action_dir) 
                                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                total_videos_expected += video_count
            
            # Process all videos for this subject/action combination
            processed = process_subject_action(
                subject_dir,
                action,
                output_base_dir,
                model_complexity=model_complexity,
                max_side=max_side,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                cv_threads=cv_threads,
                workers=workers
            )
            total_videos_processed += processed
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 PROCESSING COMPLETE!")
    print("="*80)
    print(f"✅ Videos processed: {total_videos_processed}/{total_videos_expected}")
    print(f"💾 Output directory: {output_base_dir}/blazepose/")
    print(f"📊 Array format: (Frames, 33, 10) - Float32")
    print("="*80)


def verify_output_structure(output_base_dir):
    """
    Verify and display the output directory structure.
    
    Args:
        output_base_dir: Base directory for output (main_all_keypoints)
    """
    blazepose_dir = os.path.join(output_base_dir, "blazepose")
    
    if not os.path.exists(blazepose_dir):
        print("❌ Output directory does not exist yet.")
        return
    
    print("\n" + "="*80)
    print("📁 OUTPUT STRUCTURE VERIFICATION")
    print("="*80)
    
    total_files = 0
    for root, dirs, files in os.walk(blazepose_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            rel_path = os.path.relpath(root, blazepose_dir)
            print(f"\n📂 {rel_path}/")
            print(f"   {len(npy_files)} .npy files")
            
            # Show first few files as example
            for f in sorted(npy_files)[:3]:
                file_path = os.path.join(root, f)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Load and verify shape
                try:
                    data = np.load(file_path)
                    print(f"   ✓ {f} - Shape: {data.shape} - Size: {file_size:.2f} MB - dtype: {data.dtype}")
                except Exception as e:
                    print(f"   ✗ {f} - Error: {e}")
            
            if len(npy_files) > 3:
                print(f"   ... and {len(npy_files) - 3} more files")
            
            total_files += len(npy_files)
    
    print("\n" + "="*80)
    print(f"📊 Total .npy files: {total_files}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract all 33 BlazePose 3D world landmarks from TotalCapture videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos
  python extract_all_blazepose_keypoints.py

  # Specify custom directories
  python extract_all_blazepose_keypoints.py --videos ../totalcapture_dataset/Videos --output ./output

  # Verify output structure
  python extract_all_blazepose_keypoints.py --verify
        """
    )
    
    parser.add_argument(
        '--videos',
        type=str,
        default='../totalcapture_dataset/Videos',
        help='Path to TotalCapture Videos directory (default: ../totalcapture_dataset/Videos)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Base output directory (default: current directory - will create blazepose/ subdirectory)'
    )

    parser.add_argument(
        '--model_complexity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='MediaPipe model complexity: 0 (fast), 1 (balanced), 2 (accurate)'
    )

    parser.add_argument(
        '--max_side',
        type=int,
        default=1920,
        help='Resize frames so the longest side is <= max_side (0 disables resizing, default 1920 for L4 24GB)'
    )

    parser.add_argument(
        '--min_detection_confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence for MediaPipe Pose'
    )

    parser.add_argument(
        '--min_tracking_confidence',
        type=float,
        default=0.5,
        help='Minimum tracking confidence for MediaPipe Pose'
    )

    parser.add_argument(
        '--cv_threads',
        type=int,
        default=2,
        help='OpenCV CPU threads (default 2 for L4, 0 lets OpenCV decide)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel worker processes for videos (set 1 for sequential, default 8 for L4 24GB)'
    )
    
    parser.add_argument(
        '--start-from',
        type=str,
        nargs=2,
        metavar=('SUBJECT', 'ACTION'),
        help='Start processing from specific subject and action, e.g., --start-from s2 rom2'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify output structure without processing'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    videos_dir = os.path.abspath(args.videos)
    output_dir = os.path.abspath(args.output)
    
    if args.verify:
        # Just verify existing output
        verify_output_structure(output_dir)
    else:
        # Check if videos directory exists
        if not os.path.isdir(videos_dir):
            print(f"❌ Error: Videos directory not found: {videos_dir}")
            print(f"   Please check the path and try again.")
            exit(1)
        
        # Process all videos
        start_from = tuple(args.start_from) if args.start_from else None
        batch_process_all_videos(
            videos_dir,
            output_dir,
            model_complexity=args.model_complexity,
            max_side=args.max_side if args.max_side > 0 else None,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            cv_threads=args.cv_threads,
            workers=args.workers,
            start_from=start_from
        )
        
        # Verify output
        print("\n")
        verify_output_structure(output_dir)
