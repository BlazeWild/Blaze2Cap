"""
Final Dataset Preparation Script

This script:
1. Reads original GT position .txt files and reshapes to (frames, 17, 3)
2. Converts from inches to meters
3. Transforms to hip-relative coordinates (hip at origin 0,0,0)
4. Transforms coordinate system (x,y,z) -> (x,-y,z)
5. Synchronizes frames between BlazePose NPY and GT:
   - Only includes frames where subject is detected in BlazePose
   - Handles frame count mismatches
6. Creates separate GT for each camera (8 GTs per action)
7. Outputs to final_numpy_dataset/blazepose_numpy and gt_numpy
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def read_gt_txt_file(txt_path):
    """
    Read GT position txt file and reshape to (frames, 17, 3)
    File format: Tab-separated with header row
    Each data row has 21 joint positions with x,y,z for each (63 values total)
    We need joints 0-16 (first 17 joints), so 51 values
    """
    try:
        # Read the txt file, skipping header row
        data = np.loadtxt(txt_path, skiprows=1)
        
        # Check if it's the right shape
        if data.ndim == 1:
            # Single frame
            data = data.reshape(1, -1)
        
        n_frames = data.shape[0]
        n_values = data.shape[1]
        
        # File has 21 joints * 3 coords = 63 values per frame
        # We need first 17 joints = 51 values
        if n_values < 51:
            print(f"Warning: {txt_path} has {n_values} values per frame (expected at least 51)")
            return None
        
        # Take only first 51 values (17 joints * 3 coords)
        data = data[:, :51]
        
        # Reshape to (frames, 17, 3)
        data = data.reshape(n_frames, 17, 3)
        
        return data
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None


def inches_to_meters(positions):
    """Convert positions from inches to meters"""
    return positions * 0.0254


def transform_to_hip_relative(positions, hip_index=0):
    """
    Transform coordinates to be hip-relative (hip at origin 0,0,0)
    
    Args:
        positions: (frames, 17, 3) array
        hip_index: index of hip joint (default 0)
    
    Returns:
        Hip-relative positions
    """
    # Get hip positions for all frames
    hip_positions = positions[:, hip_index:hip_index+1, :]  # (frames, 1, 3)
    
    # Subtract hip position from all joints
    relative_positions = positions - hip_positions
    
    return relative_positions


def transform_coordinate_system(positions):
    """
    Transform coordinate system: (x,y,z) -> (x,-y,z)
    Flips the y-axis
    """
    transformed = positions.copy()
    transformed[:, :, 1] = -transformed[:, :, 1]  # Flip y-axis
    return transformed


def is_subject_detected(blazepose_frame):
    """
    Check if subject is detected in a BlazePose frame
    Subject is not detected if all values are zero
    
    Args:
        blazepose_frame: (33, 4) array
    
    Returns:
        True if subject detected, False otherwise
    """
    # Check if any non-zero values exist
    return np.any(blazepose_frame != 0)


def sync_frames(blazepose_data, gt_data):
    """
    Synchronize frames between BlazePose and GT
    
    Args:
        blazepose_data: (n_frames, 33, 4) array
        gt_data: (m_frames, 17, 3) array
    
    Returns:
        Tuple of (synced_blazepose, synced_gt, valid_frame_indices)
    """
    n_video_frames = len(blazepose_data)
    n_gt_frames = len(gt_data)
    
    # Determine the minimum number of frames to consider
    n_frames = min(n_video_frames, n_gt_frames)
    
    # Find frames where subject is detected in BlazePose
    valid_indices = []
    for i in range(n_frames):
        if is_subject_detected(blazepose_data[i]):
            valid_indices.append(i)
    
    valid_indices = np.array(valid_indices)
    
    if len(valid_indices) == 0:
        print(f"  Warning: No valid frames found!")
        return None, None, None
    
    # Extract valid frames
    synced_blazepose = blazepose_data[valid_indices]
    synced_gt = gt_data[valid_indices]
    
    return synced_blazepose, synced_gt, valid_indices


def process_activity(subject, activity, cameras, 
                     blazepose_base, positions_base, output_base):
    """
    Process one activity for a subject across all cameras
    
    Args:
        subject: Subject ID (e.g., 'S1')
        activity: Activity name (e.g., 'acting1')
        cameras: List of camera names (e.g., ['cam1', 'cam2', ...])
        blazepose_base: Base path to BlazePose NPY files
        positions_base: Base path to GT position files
        output_base: Base path for output
    
    Returns:
        Dictionary with statistics
    """
    # Read GT txt file (shared across all cameras)
    gt_txt_path = positions_base / subject / activity / "gt_skel_gbl_pos.txt"
    
    if not gt_txt_path.exists():
        print(f"  GT file not found: {gt_txt_path}")
        return None
    
    # Read and process GT
    gt_raw = read_gt_txt_file(gt_txt_path)
    if gt_raw is None:
        return None
    
    print(f"  GT original shape: {gt_raw.shape}")
    
    # Process GT: inches -> meters -> hip-relative -> coordinate transform
    gt_meters = inches_to_meters(gt_raw)
    gt_hip_relative = transform_to_hip_relative(gt_meters, hip_index=0)
    gt_transformed = transform_coordinate_system(gt_hip_relative)
    
    stats = {
        'subject': subject,
        'activity': activity,
        'gt_original_frames': len(gt_raw),
        'cameras_processed': []
    }
    
    # Process each camera separately
    for camera in cameras:
        # Read BlazePose NPY file
        npy_name = f"blaze_{subject}_{activity}_{camera}.npy"
        npy_path = blazepose_base / subject / activity / npy_name
        
        if not npy_path.exists():
            print(f"    {camera}: NPY not found")
            continue
        
        blazepose_data = np.load(npy_path)
        
        # Synchronize frames
        synced_blazepose, synced_gt, valid_indices = sync_frames(
            blazepose_data, gt_transformed
        )
        
        if synced_blazepose is None:
            print(f"    {camera}: No valid frames after sync")
            continue
        
        # Create output directories
        blazepose_out_dir = output_base / "blazepose_numpy" / subject / activity
        gt_out_dir = output_base / "gt_numpy" / subject / activity
        blazepose_out_dir.mkdir(parents=True, exist_ok=True)
        gt_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save synced data
        blazepose_out_name = f"blaze_{subject}_{activity}_{camera}.npy"
        gt_out_name = f"gt_{subject}_{activity}_{camera}.npy"
        
        np.save(blazepose_out_dir / blazepose_out_name, synced_blazepose)
        np.save(gt_out_dir / gt_out_name, synced_gt)
        
        # Record statistics
        cam_stats = {
            'camera': camera,
            'blazepose_original': len(blazepose_data),
            'gt_original': len(gt_transformed),
            'synced_frames': len(synced_blazepose),
            'blazepose_shape': synced_blazepose.shape,
            'gt_shape': synced_gt.shape
        }
        stats['cameras_processed'].append(cam_stats)
        
        print(f"    {camera}: {len(blazepose_data)} -> {len(synced_blazepose)} frames "
              f"(BlazePose: {synced_blazepose.shape}, GT: {synced_gt.shape})")
    
    return stats


def main():
    # Base paths
    workspace_root = Path(__file__).parent.parent
    blazepose_base = workspace_root / "main_all_keypoints" / "blazepose"
    positions_base = workspace_root / "totalcapture_dataset" / "positions"
    output_base = workspace_root / "final_numpy_dataset"
    
    # Create output directories
    (output_base / "blazepose_numpy").mkdir(parents=True, exist_ok=True)
    (output_base / "gt_numpy").mkdir(parents=True, exist_ok=True)
    
    # Subject-activity mapping
    subject_activities = {
        'S1': ['acting1', 'acting2', 'acting3', 
               'freestyle1', 'freestyle2', 'freestyle3',
               'rom1', 'rom2', 'rom3',
               'walking1', 'walking2', 'walking3'],
        'S2': ['acting1', 'acting2', 'acting3', 
               'freestyle1', 'freestyle2', 'freestyle3',
               'rom1', 'rom2', 'rom3',
               'walking1', 'walking2', 'walking3'],
        'S3': ['acting1', 'acting2', 'acting3', 
               'freestyle1', 'freestyle2', 'freestyle3',
               'rom1', 'rom2', 'rom3',
               'walking1', 'walking2', 'walking3'],
        'S4': ['acting3', 'freestyle1', 'freestyle3', 'rom3', 'walking2'],
        'S5': ['acting3', 'freestyle1', 'freestyle3', 'rom3', 'walking2']
    }
    
    cameras = [f'cam{i}' for i in range(1, 9)]
    
    print("=" * 80)
    print("FINAL DATASET PREPARATION")
    print("=" * 80)
    print(f"\nInput:")
    print(f"  BlazePose: {blazepose_base}")
    print(f"  GT Positions: {positions_base}")
    print(f"\nOutput:")
    print(f"  {output_base}")
    print(f"\nProcessing:")
    print(f"  - Converting inches to meters")
    print(f"  - Transforming to hip-relative coordinates")
    print(f"  - Applying coordinate transform (x,y,z) -> (x,-y,z)")
    print(f"  - Synchronizing frames (only where subject detected)")
    print(f"  - Creating separate GT for each camera")
    print()
    
    all_stats = []
    
    # Process each subject and activity
    total_activities = sum(len(acts) for acts in subject_activities.values())
    progress = tqdm(total=total_activities, desc="Overall Progress")
    
    for subject, activities in subject_activities.items():
        print(f"\n{'='*80}")
        print(f"Processing {subject}")
        print(f"{'='*80}")
        
        for activity in activities:
            print(f"\n{subject} - {activity}:")
            
            stats = process_activity(
                subject, activity, cameras,
                blazepose_base, positions_base, output_base
            )
            
            if stats:
                all_stats.append(stats)
            
            progress.update(1)
    
    progress.close()
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    total_cameras = 0
    total_synced_frames = 0
    
    for stats in all_stats:
        n_cams = len(stats['cameras_processed'])
        total_cameras += n_cams
        
        for cam_stat in stats['cameras_processed']:
            total_synced_frames += cam_stat['synced_frames']
    
    print(f"\nTotal activities processed: {len(all_stats)}")
    print(f"Total camera views processed: {total_cameras}")
    print(f"Total synced frames: {total_synced_frames}")
    
    # Save detailed statistics
    stats_file = output_base / "processing_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("FINAL DATASET PREPARATION - DETAILED STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in all_stats:
            f.write(f"\n{stats['subject']} - {stats['activity']}\n")
            f.write(f"  GT Original Frames: {stats['gt_original_frames']}\n")
            f.write(f"  Cameras Processed: {len(stats['cameras_processed'])}\n")
            
            for cam_stat in stats['cameras_processed']:
                f.write(f"\n    {cam_stat['camera']}:\n")
                f.write(f"      BlazePose Original: {cam_stat['blazepose_original']} frames\n")
                f.write(f"      GT Original: {cam_stat['gt_original']} frames\n")
                f.write(f"      Synced Frames: {cam_stat['synced_frames']} frames\n")
                f.write(f"      BlazePose Shape: {cam_stat['blazepose_shape']}\n")
                f.write(f"      GT Shape: {cam_stat['gt_shape']}\n")
    
    print(f"\n✓ Detailed statistics saved to: {stats_file}")
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Output location: {output_base}")


if __name__ == "__main__":
    main()
