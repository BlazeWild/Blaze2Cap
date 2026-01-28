"""
Script to compare frame counts across:
- Video files from totalcapture_dataset/Videos
- NPY files from main_all_keypoints/blazepose
- Ground truth CSV files from totalcapture_dataset/positions
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import csv


def count_video_frames(video_path):
    """Count frames in a video file using metadata only."""
    if not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count if frame_count > 0 else None
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None


def count_npy_frames(npy_path):
    """Count frames in an npy file."""
    if not os.path.exists(npy_path):
        return None
    
    try:
        data = np.load(str(npy_path))
        return len(data)
    except Exception as e:
        print(f"Error reading npy {npy_path}: {e}")
        return None


def count_csv_frames(csv_path):
    """Count rows in CSV file (excluding header)."""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None


def main():
    # Base paths
    workspace_root = Path(__file__).parent.parent
    videos_base = workspace_root / "totalcapture_dataset" / "Videos"
    blazepose_base = workspace_root / "main_all_keypoints" / "blazepose"
    positions_base = workspace_root / "totalcapture_dataset" / "positions"
    
    # Output CSV
    output_csv = workspace_root / "test" / "frame_count_comparison.csv"
    
    # Prepare results
    results = []
    
    # Subject names and their activities
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
    
    # Camera numbers (cam1-cam8)
    cameras = [f'cam{i}' for i in range(1, 9)]
    
    print("Starting frame count comparison...")
    print(f"Workspace root: {workspace_root}")
    
    for subject, activities in subject_activities.items():
        print(f"\nProcessing subject: {subject}")
        
        # Directory paths (note: videos use lowercase s in folder name)
        subject_video_dir = videos_base / subject.lower()
        subject_blazepose_dir = blazepose_base / subject
        subject_positions_dir = positions_base / subject
        
        if not subject_video_dir.exists():
            print(f"  Video directory not found: {subject_video_dir}")
            continue
            
        for activity in activities:
            print(f"  Processing activity: {activity}")
            
            # Get GT CSV frame count (same for all cameras)
            # GT CSV is inside the activity folder
            gt_csv_path = subject_positions_dir / activity / "gt_skel_gbl_pos.csv"
            gt_frame_count = count_csv_frames(gt_csv_path)
            
            for camera in cameras:
                # Video path: TC_S1_acting1_cam1.mp4
                video_name = f"TC_{subject}_{activity}_{camera}.mp4"
                video_path = subject_video_dir / activity / video_name
                
                # NPY path: blaze_S1_acting1_cam1.npy
                npy_name = f"blaze_{subject}_{activity}_{camera}.npy"
                npy_path = subject_blazepose_dir / activity / npy_name
                
                # Count frames
                video_frames = count_video_frames(video_path)
                npy_frames = count_npy_frames(npy_path)
                
                # Add to results
                results.append({
                    'Subject': subject,
                    'Activity': activity,
                    'Video': camera,
                    'Video_Frames': video_frames if video_frames is not None else 'N/A',
                    'NPY_Frames': npy_frames if npy_frames is not None else 'N/A',
                    'GT_CSV_Frames': gt_frame_count if gt_frame_count is not None else 'N/A'
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Frame count comparison completed!")
    print(f"✓ Results saved to: {output_csv}")
    print(f"\nTotal entries: {len(results)}")
    
    # Show summary statistics
    print("\n=== Summary Statistics ===")
    for col in ['Video_Frames', 'NPY_Frames', 'GT_CSV_Frames']:
        valid_counts = df[df[col] != 'N/A'][col]
        if len(valid_counts) > 0:
            print(f"\n{col}:")
            print(f"  Available: {len(valid_counts)}/{len(df)}")
            print(f"  Range: {valid_counts.min()} - {valid_counts.max()}")
    
    # Check for mismatches
    print("\n=== Checking for Frame Count Mismatches ===")
    df_valid = df[(df['Video_Frames'] != 'N/A') & 
                  (df['NPY_Frames'] != 'N/A') & 
                  (df['GT_CSV_Frames'] != 'N/A')].copy()
    
    if len(df_valid) > 0:
        df_valid['Video_NPY_Match'] = df_valid['Video_Frames'] == df_valid['NPY_Frames']
        df_valid['Video_GT_Match'] = df_valid['Video_Frames'] == df_valid['GT_CSV_Frames']
        df_valid['NPY_GT_Match'] = df_valid['NPY_Frames'] == df_valid['GT_CSV_Frames']
        
        mismatches = df_valid[~(df_valid['Video_NPY_Match'] & 
                                 df_valid['Video_GT_Match'] & 
                                 df_valid['NPY_GT_Match'])]
        
        if len(mismatches) > 0:
            print(f"Found {len(mismatches)} mismatches:")
            for _, row in mismatches.iterrows():
                print(f"  {row['Subject']} {row['Activity']} {row['Video']}: "
                      f"Video={row['Video_Frames']}, "
                      f"NPY={row['NPY_Frames']}, "
                      f"GT={row['GT_CSV_Frames']}")
        else:
            print("✓ All frame counts match!")
    else:
        print("No valid data to compare.")


if __name__ == "__main__":
    main()
