#!/usr/bin/env python3
"""
Analyze coordinate system mapping between mediapipe (input) and TotalCapture (GT).
"""

import csv
import numpy as np

def parse_coord_string(coord_str):
    """Parse coordinate string 'x, y, z' to numpy array."""
    if not coord_str or coord_str.strip() == "":
        return None
    coords = [float(x.strip()) for x in coord_str.split(',')]
    return np.array(coords)

# Read sample data
input_file = "d:\\Ashok\\_AI\\_COMPUTER_VISION\\____RESEARCH\\___MOTION_T_LIGHTNING\\non_skipped_frames_csv\\s1_acting1_input.csv"
gt_file = "d:\\Ashok\\_AI\\_COMPUTER_VISION\\____RESEARCH\\___MOTION_T_LIGHTNING\\non_skipped_frames_csv\\s1_acting1_gt.csv"

# Read both files
with open(input_file, 'r') as f:
    input_reader = csv.DictReader(f)
    input_rows = list(input_reader)

with open(gt_file, 'r') as f:
    gt_reader = csv.DictReader(f)
    gt_rows = list(gt_reader)

# Compare first 10 frames, LeftArm keypoint
print("=" * 100)
print("COORDINATE SYSTEM ANALYSIS: Mediapipe (Input) vs TotalCapture (GT)")
print("=" * 100)
print("\nAnalyzing LeftArm keypoint for first 10 frames:\n")
print(f"{'Frame':<6} {'Input (x,y,z)':<40} {'GT (x,y,z)':<40}")
print("-" * 100)

for i in range(min(10, len(input_rows))):
    input_left_arm = parse_coord_string(input_rows[i]['LeftArm'])
    gt_left_arm = parse_coord_string(gt_rows[i]['LeftArm'])
    
    print(f"{i:<6} {str(input_left_arm):<40} {str(gt_left_arm):<40}")

print("\n" + "=" * 100)
print("TESTING POSSIBLE COORDINATE TRANSFORMATIONS")
print("=" * 100)

# Test different transformations
sample_input = parse_coord_string(input_rows[0]['LeftArm'])
sample_gt = parse_coord_string(gt_rows[0]['LeftArm'])

print(f"\nSample Input LeftArm (frame 0):  {sample_input}")
print(f"Sample GT LeftArm (frame 0):     {sample_gt}\n")

transformations = {
    "x, y, z (no change)": sample_input,
    "-x, y, z": np.array([-sample_input[0], sample_input[1], sample_input[2]]),
    "x, -y, z": np.array([sample_input[0], -sample_input[1], sample_input[2]]),
    "x, y, -z": np.array([sample_input[0], sample_input[1], -sample_input[2]]),
    "-x, y, -z": np.array([-sample_input[0], sample_input[1], -sample_input[2]]),
    "-x, -y, z": np.array([-sample_input[0], -sample_input[1], sample_input[2]]),
    "x, -y, -z": np.array([sample_input[0], -sample_input[1], -sample_input[2]]),
    "-x, -y, -z": np.array([-sample_input[0], -sample_input[1], -sample_input[2]]),
    "y, x, z": np.array([sample_input[1], sample_input[0], sample_input[2]]),
    "z, y, x": np.array([sample_input[2], sample_input[1], sample_input[0]]),
    "x, z, y": np.array([sample_input[0], sample_input[2], sample_input[1]]),
    "-x, -z, y": np.array([-sample_input[0], -sample_input[2], sample_input[1]]),
    "x, -z, -y": np.array([sample_input[0], -sample_input[2], -sample_input[1]]),
}

print(f"Testing transformations (GT target: {sample_gt}):\n")
print(f"{'Transformation':<25} {'Result':<40} {'Match Score':<15}")
print("-" * 100)

best_match = None
best_score = float('inf')

for transform_name, transformed in transformations.items():
    distance = np.linalg.norm(transformed - sample_gt)
    match_score = f"{distance:.6f}"
    
    # Check all samples
    if transform_name != "x, y, z (no change)":
        all_distances = []
        for j in range(min(10, len(input_rows))):
            inp = parse_coord_string(input_rows[j]['LeftArm'])
            gt = parse_coord_string(gt_rows[j]['LeftArm'])
            
            if transform_name == "-x, y, z":
                trans = np.array([-inp[0], inp[1], inp[2]])
            elif transform_name == "x, -y, z":
                trans = np.array([inp[0], -inp[1], inp[2]])
            elif transform_name == "x, y, -z":
                trans = np.array([inp[0], inp[1], -inp[2]])
            elif transform_name == "-x, y, -z":
                trans = np.array([-inp[0], inp[1], -inp[2]])
            elif transform_name == "-x, -y, z":
                trans = np.array([-inp[0], -inp[1], inp[2]])
            elif transform_name == "x, -y, -z":
                trans = np.array([inp[0], -inp[1], -inp[2]])
            elif transform_name == "-x, -y, -z":
                trans = np.array([-inp[0], -inp[1], -inp[2]])
            elif transform_name == "y, x, z":
                trans = np.array([inp[1], inp[0], inp[2]])
            elif transform_name == "z, y, x":
                trans = np.array([inp[2], inp[1], inp[0]])
            elif transform_name == "x, z, y":
                trans = np.array([inp[0], inp[2], inp[1]])
            elif transform_name == "-x, -z, y":
                trans = np.array([-inp[0], -inp[2], inp[1]])
            elif transform_name == "x, -z, -y":
                trans = np.array([inp[0], -inp[2], -inp[1]])
            
            all_distances.append(np.linalg.norm(trans - gt))
        
        avg_distance = np.mean(all_distances)
        match_score = f"{avg_distance:.6f}"
        
        if avg_distance < best_score:
            best_score = avg_distance
            best_match = transform_name
    
    print(f"{transform_name:<25} {str(transformed):<40} {match_score:<15}")

print("\n" + "=" * 100)
print(f"BEST MATCH: {best_match} with average distance {best_score:.6f}")
print("=" * 100)
