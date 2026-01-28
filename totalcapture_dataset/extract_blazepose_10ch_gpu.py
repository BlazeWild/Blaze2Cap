"""
Extract all 33 BlazePose landmarks with 10-channel schema for TotalCapture dataset.
GPU-optimized version using classic MediaPipe API.

Output format:
- Shape: (Total_Frames, 33, 10) for each video
- 10 Channels per keypoint:
  [0-2]: World 3D coordinates (x, y, z) in meters - relative to hips
  [3]:   World visibility score (0.0 to 1.0)
  [4]:   Anchor flag (0=new/start frame, 1=continuous tracking)
  [5-6]: Screen 2D normalized coordinates (x, y) in [0, 1]
  [7]:   Screen relative depth (z)
  [8]:   Screen visibility score (0.0 to 1.0)
  [9]:   Backup anchor flag (copy of channel 4)
- Format: Float32 numpy arrays saved as .npy files
- Naming: blaze_S1_acting1_cam1.npy (matching video name)
- Output Directory: everything_from_blazepose/S1/acting1/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc


def extract_10channel_keypoints_from_video(
    video_path,
    output_dir,
    model_complexity=2,
    max_side=1920,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    cv_threads=0
):
    """
    Extract all 33 BlazePose landmarks with 10-channel schema from a single video.
    
    10-Channel Schema:
    [0-2]: World 3D (x, y, z) in meters
    [3]:   World visibility
    [4]:   Anchor flag (0=new/start, 1=continuous)
    [5-6]: Screen 2D normalized (x, y)
    [7]:   Screen relative depth (z)
    [8]:   Screen visibility
    [9]:   Backup anchor flag
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output .npy file
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        
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

    # Initialize MediaPipe Pose with classic API (GPU-compatible)
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
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resize_dim = None
    if max_side and orig_w > 0 and orig_h > 0:
        max_orig_side = max(orig_w, orig_h)
        if max_orig_side > max_side:
            scale = max_side / max_orig_side
            resize_dim = (int(orig_w * scale), int(orig_h * scale))
    
    # First pass: Count ALL frames to get exact count
    print(f"  📖 Counting frames...", end="", flush=True)
    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    
    actual_frame_count = frame_count
    cap.release()
    print(f" {actual_frame_count} frames")
    
    # Initialize output array with EXACT frame count: (frames, 33, 10)
    landmarks_array = np.zeros((actual_frame_count, 33, 10), dtype=np.float32)
    
    # Re-open video for second pass - process all frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Error: Could not re-open video {video_path}")
        pose.close()
        return False
    
    processed_count = 0
    subject_detected_in_previous_frame = False  # Track continuous subject presence
    frame_idx = 0
    
    # Process all frames with progress bar
    with tqdm(total=actual_frame_count, desc=f"  Processing {output_name}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if resize_dim is not None:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Process with MediaPipe (VIDEO mode - processes every frame)
            results = pose.process(frame_rgb)
            
            if results.pose_world_landmarks and results.pose_landmarks:
                # Extract all 33 landmarks with 10-channel schema
                world_landmarks = results.pose_world_landmarks.landmark
                screen_landmarks = results.pose_landmarks.landmark
                
                # Determine anchor flag: 0 if new/start, 1 if continuous
                anchor_flag = 1.0 if subject_detected_in_previous_frame else 0.0
                
                for landmark_idx in range(min(33, len(world_landmarks), len(screen_landmarks))):
                    world_lm = world_landmarks[landmark_idx]
                    screen_lm = screen_landmarks[landmark_idx]
                    
                    # Channel 0-2: World 3D coordinates (meters, relative to hips)
                    landmarks_array[frame_idx, landmark_idx, 0] = world_lm.x
                    landmarks_array[frame_idx, landmark_idx, 1] = world_lm.y
                    landmarks_array[frame_idx, landmark_idx, 2] = world_lm.z
                    
                    # Channel 3: World visibility score
                    landmarks_array[frame_idx, landmark_idx, 3] = world_lm.visibility
                    
                    # Channel 4: Anchor flag (0=new/start, 1=continuous)
                    landmarks_array[frame_idx, landmark_idx, 4] = anchor_flag
                    
                    # Channel 5-6: Screen 2D normalized coordinates
                    landmarks_array[frame_idx, landmark_idx, 5] = screen_lm.x
                    landmarks_array[frame_idx, landmark_idx, 6] = screen_lm.y
                    
                    # Channel 7: Screen relative depth
                    landmarks_array[frame_idx, landmark_idx, 7] = screen_lm.z
                    
                    # Channel 8: Screen visibility score
                    landmarks_array[frame_idx, landmark_idx, 8] = screen_lm.visibility
                    
                    # Channel 9: Backup anchor flag
                    landmarks_array[frame_idx, landmark_idx, 9] = anchor_flag
                
                processed_count += 1
                subject_detected_in_previous_frame = True  # Mark for next frame
            else:
                # Frame was skipped - leave as zeros (all 10 channels)
                subject_detected_in_previous_frame = False  # Reset tracking
            
            frame_idx += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    pose.close()
    
    # Save as numpy array in float32 format
    np.save(output_path, landmarks_array)
    
    success_rate = (processed_count / actual_frame_count * 100) if actual_frame_count > 0 else 0
    print(f"  ✅ Saved: {output_name} | Shape: {landmarks_array.shape} | Frames: {actual_frame_count} | Detected: {processed_count} ({success_rate:.1f}%)")
    
    # Force garbage collection to free GPU memory
    gc.collect()
    
    return True


def _process_single_video(args):
    return extract_10channel_keypoints_from_video(*args)


def process_subject_action(
    subject_dir,
    action,
    output_base_dir,
    model_complexity=2,
    max_side=1920,
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
        output_base_dir: Base directory for output files (everything_from_blazepose)
        
    Returns:
        int: Number of videos successfully processed
    """
    action_dir = os.path.join(subject_dir, action)
    
    if not os.path.isdir(action_dir):
        return 0
    
    # Get subject name from path
    subject_name = os.path.basename(subject_dir).upper()  # e.g., S1, S2, S3
    
    # Create output directory structure: everything_from_blazepose/S1/acting1/
    output_dir = os.path.join(output_base_dir, subject_name, action)
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
        # Sequential processing
        for args in task_args:
            if _process_single_video(args):
                success_count += 1
    
    return success_count


def batch_process_all_videos(
    videos_base_dir,
    output_base_dir,
    model_complexity=2,
    max_side=1920,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    cv_threads=0,
    workers=8,
    start_from=None
):
    """
    Process all videos in the TotalCapture dataset structure.
    
    Args:
        videos_base_dir: Base directory containing s1, s2, s3, s4, s5 folders
        output_base_dir: Base directory for output (everything_from_blazepose)
        start_from: Tuple (subject, action) to start from, e.g., ('s2', 'rom2')
    """
    # TotalCapture structure: Videos/s1/acting1/, Videos/s1/acting2/, etc.
    subjects = ['s1', 's2', 's3', 's4', 's5']
    actions = ['acting1', 'acting2', 'acting3', 
               'freestyle1', 'freestyle2', 'freestyle3',
               'rom1', 'rom2', 'rom3',
               'walking1', 'walking2', 'walking3']
    
    total_videos_processed = 0
    total_videos_expected = 0
    
    print("\n" + "="*80)
    print("🚀 STARTING BLAZEPOSE EXTRACTION - 10-CHANNEL SCHEMA (GPU OPTIMIZED)")
    print("="*80)
    print(f"📂 Input: {videos_base_dir}")
    print(f"💾 Output: {output_base_dir}/")
    print(f"🎯 Format: (Frames, 33, 10) - Float32")
    print(f"📊 10-Channel Schema:")
    print(f"   [0-2]: World 3D (x,y,z) meters - relative to hips")
    print(f"   [3]:   World visibility (0.0-1.0)")
    print(f"   [4]:   Anchor flag (0=new/start, 1=continuous)")
    print(f"   [5-6]: Screen 2D normalized (x,y)")
    print(f"   [7]:   Screen relative depth (z)")
    print(f"   [8]:   Screen visibility (0.0-1.0)")
    print(f"   [9]:   Backup anchor flag")
    print(f"🖥️  GPU: Automatic via MediaPipe classic API")
    print(f"🎨 Model Complexity: {model_complexity} (2=high accuracy)")
    print(f"📐 Resolution: {max_side}px (FHD)")
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
    print(f"💾 Output directory: {output_base_dir}/")
    print(f"📊 Array format: (Frames, 33, 10) - Float32")
    print("="*80)


def verify_output_structure(output_base_dir):
    """
    Verify and display the output directory structure.
    
    Args:
        output_base_dir: Base directory for output (everything_from_blazepose)
    """
    if not os.path.exists(output_base_dir):
        print("❌ Output directory does not exist yet.")
        return
    
    print("\n" + "="*80)
    print("📁 OUTPUT STRUCTURE VERIFICATION")
    print("="*80)
    
    total_files = 0
    for root, dirs, files in os.walk(output_base_dir):
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            rel_path = os.path.relpath(root, output_base_dir)
            print(f"\n📂 {rel_path}/")
            print(f"   {len(npy_files)} .npy files")
            
            # Show first few files as example
            for f in sorted(npy_files)[:3]:
                file_path = os.path.join(root, f)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Load and verify shape
                try:
                    data = np.load(file_path)
                    print(f"   ✓ {f}")
                    print(f"      Shape: {data.shape} | dtype: {data.dtype} | Size: {file_size:.2f} MB")
                    
                    # Verify it's the expected 10-channel format
                    if data.ndim == 3 and data.shape[1] == 33 and data.shape[2] == 10:
                        print(f"      ✅ Valid 10-channel format (Frames, 33, 10)")
                    else:
                        print(f"      ⚠️  Unexpected shape - expected (Frames, 33, 10)")
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
        description='Extract all 33 BlazePose landmarks with 10-channel schema from TotalCapture videos (GPU-optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
10-Channel Schema:
  [0-2]: World 3D (x, y, z) in meters - relative to hips
  [3]:   World visibility score (0.0 to 1.0)
  [4]:   Anchor flag (0=new/start frame, 1=continuous tracking)
  [5-6]: Screen 2D normalized (x, y) in [0, 1]
  [7]:   Screen relative depth (z)
  [8]:   Screen visibility score (0.0 to 1.0)
  [9]:   Backup anchor flag (copy of channel 4)

Examples:
  # Process all videos (default: everything_from_blazepose/)
  python extract_blazepose_10ch_gpu.py

  # Specify custom directories
  python extract_blazepose_10ch_gpu.py --videos ./Videos --output ./my_output

  # Verify output structure
  python extract_blazepose_10ch_gpu.py --verify

  # Resume from specific point
  python extract_blazepose_10ch_gpu.py --start-from s2 rom2
        """
    )
    
    parser.add_argument(
        '--videos',
        type=str,
        default='./Videos',
        help='Path to TotalCapture Videos directory (default: ./Videos)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./everything_from_blazepose',
        help='Output directory (default: ./everything_from_blazepose)'
    )

    parser.add_argument(
        '--model_complexity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='MediaPipe model complexity: 0 (fast), 1 (balanced), 2 (accurate, default)'
    )

    parser.add_argument(
        '--max_side',
        type=int,
        default=1920,
        help='Resize frames so longest side is <= max_side (0 disables resizing, default 1920 for FHD)'
    )

    parser.add_argument(
        '--min_detection_confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence for MediaPipe Pose (default: 0.5)'
    )

    parser.add_argument(
        '--min_tracking_confidence',
        type=float,
        default=0.5,
        help='Minimum tracking confidence for MediaPipe Pose (default: 0.5)'
    )

    parser.add_argument(
        '--cv_threads',
        type=int,
        default=2,
        help='OpenCV CPU threads (default: 2, set 0 to let OpenCV decide)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel worker processes (default: 8, set 1 for sequential)'
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
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
