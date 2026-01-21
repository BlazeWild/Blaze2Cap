import os
import csv

# Mapping of TotalCapture joint names (from gt_skel_gbl_pos.csv)
TOTALCAPTURE_JOINTS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

# Indices of the 12 keypoints we want to keep from TotalCapture
REQUIRED_KEYPOINTS_INDICES = {
    7: 'RightShoulder',
    8: 'RightArm',
    9: 'RightForeArm',
    10: 'RightHand',
    11: 'LeftShoulder',
    12: 'LeftArm',
    13: 'LeftForeArm',
    14: 'LeftHand',
    15: 'RightUpLeg',
    16: 'RightLeg',
    17: 'RightFoot',
    18: 'LeftUpLeg',
    19: 'LeftLeg',
    20: 'LeftFoot'
}

def filter_ground_truth_csv(input_csv, output_csv):
    """Filter ground truth CSV to keep only 12 keypoints."""
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Filter header to keep only required keypoints
        filtered_header = [header[i] for i in sorted(REQUIRED_KEYPOINTS_INDICES.keys())]
        
        # Read and filter all rows
        filtered_rows = []
        for row in reader:
            if row:  # Skip empty rows
                filtered_row = [row[i] for i in sorted(REQUIRED_KEYPOINTS_INDICES.keys())]
                filtered_rows.append(filtered_row)
    
    # Write filtered CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(filtered_header)
        writer.writerows(filtered_rows)
    
    print(f"Filtered: {input_csv} -> {output_csv}")

def batch_filter_ground_truth(positions_dir, output_dir):
    """Filter all ground truth CSV files to 12 keypoints."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filtered_count = 0
    
    for root, dirs, files in os.walk(positions_dir):
        for file in files:
            if file == 'gt_skel_gbl_pos.csv':
                input_path = os.path.join(root, file)
                
                # Determine output path maintaining directory structure
                relative_path = os.path.relpath(root, positions_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                output_path = os.path.join(output_subdir, 'gt_skel_gbl_pos_12kp.csv')
                
                filter_ground_truth_csv(input_path, output_path)
                filtered_count += 1
    
    print(f"\nTotal ground truth files filtered: {filtered_count}")

if __name__ == "__main__":
    positions_dir = "/teamspace/studios/this_studio/totalcapture_dataset/positions"
    output_dir = "/teamspace/studios/this_studio/12_keypoints"
    
    batch_filter_ground_truth(positions_dir, output_dir)
