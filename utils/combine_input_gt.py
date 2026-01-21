#!/usr/bin/env python3
"""
Combine MediaPipe BlazePose predictions with ground truth data.
- Converts ground truth from inches to meters
- Aligns ground truth with MediaPipe predictions (only non-skipped frames)
- Creates input and ground truth CSV files for all 8 cameras
"""

import csv
import os
import argparse

def convert_inches_to_meters(coord_string):
    """
    Convert coordinate string from inches to meters.
    Args:
        coord_string: String like "x, y, z" in inches
    Returns:
        String like "x, y, z" in meters
    """
    if not coord_string or coord_string.strip() == "":
        return ""
    
    coords = [float(x.strip()) for x in coord_string.split(',')]
    # Convert inches to meters (1 inch = 0.0254 meters)
    coords_meters = [c * 0.0254 for c in coords]
    return f"{coords_meters[0]}, {coords_meters[1]}, {coords_meters[2]}"

def read_ground_truth(gt_csv_path):
    """
    Read ground truth CSV and convert inches to meters.
    Returns dict mapping 1-indexed frame numbers to coordinate data.
    """
    gt_data = {}
    
    with open(gt_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        keypoint_names = reader.fieldnames
        
        for frame_idx, row in enumerate(reader, start=1):
            # Convert all coordinates from inches to meters
            converted_row = {}
            for keypoint in keypoint_names:
                converted_row[keypoint] = convert_inches_to_meters(row[keypoint])
            gt_data[frame_idx] = converted_row
    
    return gt_data, keypoint_names

def process_action(subject, action, base_dir="12_keypoints", output_dir="non_skipped_frames_csv"):
    """
    Process one action (e.g., s1_acting1) with 8 camera views.
    
    Args:
        subject: Subject folder (e.g., 's1')
        action: Action name (e.g., 'acting1')
        base_dir: Base directory containing keypoint files
        output_dir: Output directory for combined CSV files
    """
    # Paths
    gt_csv_path = os.path.join(base_dir, subject, action, "gt_skel_gbl_pos.csv")
    blazepose_dir = os.path.join(base_dir, "TC", subject.upper(), "")
    
    # Read ground truth
    print(f"Reading ground truth from {gt_csv_path}")
    gt_data, keypoint_names = read_ground_truth(gt_csv_path)
    print(f"Loaded {len(gt_data)} ground truth frames")
    
    # Find all 8 camera BlazePose files
    camera_files = []
    for cam in range(1, 9):
        blazepose_file = os.path.join(blazepose_dir, 
                                     f"blazepose_TC_{subject.upper()}_{action}_cam{cam}_12kp.csv")
        if os.path.exists(blazepose_file):
            camera_files.append((cam, blazepose_file))
        else:
            print(f"Warning: Missing {blazepose_file}")
    
    if not camera_files:
        print(f"Error: No camera files found for {subject}_{action}")
        return
    
    print(f"Found {len(camera_files)} camera files")
    
    # Prepare output data
    input_rows = []
    gt_rows = []
    
    # Process each camera
    for cam_num, blazepose_file in camera_files:
        print(f"Processing camera {cam_num}: {os.path.basename(blazepose_file)}")
        
        with open(blazepose_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                video_frame_num = int(row['video_frame_number'])
                mediapipe_pred_frame = row['mediapipe_predicted_frame']
                
                # Only include frames where MediaPipe made a prediction (not -1 and not empty)
                if mediapipe_pred_frame and mediapipe_pred_frame.strip() != "" and mediapipe_pred_frame.strip() != "-1":
                    # Ground truth is 1-indexed, video_frame_number is 0-indexed
                    gt_frame_idx = video_frame_num + 1
                    
                    # Check if ground truth exists for this frame
                    if gt_frame_idx not in gt_data:
                        print(f"Warning: No ground truth for frame {gt_frame_idx}")
                        continue
                    
                    # Build input row (MediaPipe prediction)
                    input_row = {
                        'camera': cam_num,
                        'video_frame_number': video_frame_num,
                        'mediapipe_predicted_frame': mediapipe_pred_frame
                    }
                    for keypoint in keypoint_names:
                        input_row[keypoint] = row[keypoint]
                    input_rows.append(input_row)
                    
                    # Build ground truth row (converted to meters)
                    gt_row = {
                        'camera': cam_num,
                        'video_frame_number': video_frame_num,
                        'gt_frame_index': gt_frame_idx
                    }
                    gt_row.update(gt_data[gt_frame_idx])
                    gt_rows.append(gt_row)
    
    print(f"\nTotal aligned samples: {len(input_rows)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write input CSV
    input_csv = os.path.join(output_dir, f"{subject}_{action}_input.csv")
    input_fieldnames = ['camera', 'video_frame_number', 'mediapipe_predicted_frame'] + list(keypoint_names)
    
    with open(input_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=input_fieldnames)
        writer.writeheader()
        writer.writerows(input_rows)
    
    print(f"Saved input: {input_csv}")
    
    # Write ground truth CSV
    gt_csv = os.path.join(output_dir, f"{subject}_{action}_gt.csv")
    gt_fieldnames = ['camera', 'video_frame_number', 'gt_frame_index'] + list(keypoint_names)
    
    with open(gt_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=gt_fieldnames)
        writer.writeheader()
        writer.writerows(gt_rows)
    
    print(f"Saved ground truth: {gt_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine MediaPipe predictions with ground truth'
    )
    parser.add_argument('--subject', type=str, required=True, help='Subject (e.g., s1)')
    parser.add_argument('--action', type=str, required=True, help='Action (e.g., acting1)')
    parser.add_argument('--base_dir', type=str, default='12_keypoints', 
                       help='Base directory with keypoint files')
    parser.add_argument('--output_dir', type=str, default='non_skipped_frames_csv',
                       help='Output directory for combined CSV files')
    
    args = parser.parse_args()
    
    process_action(args.subject, args.action, args.base_dir, args.output_dir)
