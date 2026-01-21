#!/usr/bin/env python3
"""
Process CSV files to convert TotalCapture GT coordinates from (-x, y, -z) to mediapipe format (x, y, z).
This ensures GT and mediapipe predictions are in the same coordinate system for comparison.
"""

import os
import csv
import numpy as np
from pathlib import Path

# Define the CSV directory
CSV_DIR = Path("d:\\Ashok\\_AI\\_COMPUTER_VISION\\____RESEARCH\\___MOTION_T_LIGHTNING\\non_skipped_frames_csv")

# Keypoint names
KEYPOINT_NAMES = [
    'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
    'LeftHand', 'RightHand', 'LeftUpLeg', 'RightUpLeg',
    'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'
]

def parse_coord_string(coord_str):
    """Parse coordinate string 'x, y, z' to list of floats."""
    if not coord_str or coord_str.strip() == "":
        return None
    coords = [float(x.strip()) for x in coord_str.split(',')]
    return coords

def convert_coordinates(coords):
    """Convert from TotalCapture (-x, y, -z) to mediapipe (x, y, z)."""
    if coords is None:
        return None
    return [-coords[0], coords[1], -coords[2]]

def process_gt_csv(input_file, output_file):
    """
    Process GT CSV file and convert coordinates.
    """
    print(f"Processing: {input_file}")
    
    rows = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Convert GT coordinates
            for kp_name in KEYPOINT_NAMES:
                coords = parse_coord_string(row[kp_name])
                if coords:
                    converted = convert_coordinates(coords)
                    row[kp_name] = f"{converted[0]}, {converted[1]}, {converted[2]}"
            
            rows.append(row)
    
    # Write converted data to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Converted GT saved to: {output_file}")

def main():
    """Main function to process all CSV files."""
    print("Processing CSV files...")
    print(f"Directory: {CSV_DIR}\n")
    
    # Find all GT CSV files
    gt_files = sorted(CSV_DIR.glob("*_gt.csv"))
    
    if not gt_files:
        print("No GT CSV files found!")
        return
    
    for gt_file in gt_files:
        # Create output filename with "_converted" suffix
        output_file = gt_file.parent / gt_file.name.replace("_gt.csv", "_gt_converted.csv")
        process_gt_csv(gt_file, output_file)
    
    print(f"\nProcessed {len(gt_files)} GT CSV files")
    print("All files have been converted from TotalCapture (-x, y, -z) to mediapipe (x, y, z) format")

if __name__ == "__main__":
    main()
