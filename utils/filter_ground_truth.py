import os
import csv

# Mapping: TotalCapture joint names that we need
REQUIRED_GT_KEYPOINTS = [
    "LeftArm",      # LEFT_SHOULDER
    "RightArm",     # RIGHT_SHOULDER
    "LeftForeArm",  # LEFT_ELBOW
    "RightForeArm", # RIGHT_ELBOW
    "LeftHand",     # LEFT_WRIST
    "RightHand",    # RIGHT_WRIST
    "LeftUpLeg",    # LEFT_HIP
    "RightUpLeg",   # RIGHT_HIP
    "LeftLeg",      # LEFT_KNEE
    "RightLeg",     # RIGHT_KNEE
    "LeftFoot",     # LEFT_ANKLE
    "RightFoot"     # RIGHT_ANKLE
]

def filter_gt_csv(input_csv, output_csv):
    """Filter ground truth CSV to keep only 12 required keypoints."""
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Find indices of required keypoints
        keypoint_indices = []
        filtered_header = []
        for kp in REQUIRED_GT_KEYPOINTS:
            if kp in header:
                idx = header.index(kp)
                keypoint_indices.append(idx)
                filtered_header.append(kp)
        
        # Write filtered CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(filtered_header)
            
            for row in reader:
                filtered_row = [row[idx] for idx in keypoint_indices]
                writer.writerow(filtered_row)
    
    print(f"Filtered: {output_csv}")

def batch_filter_ground_truth():
    """Filter all ground truth CSV files."""
    positions_dir = "/teamspace/studios/this_studio/totalcapture_dataset/positions"
    output_base = "/teamspace/studios/this_studio/12_keypoints"
    
    for root, dirs, files in os.walk(positions_dir):
        for file in files:
            if file == "gt_skel_gbl_pos.csv":
                input_path = os.path.join(root, file)
                
                # Extract subject and action from path
                parts = root.split('/')
                subject = parts[-2].lower()  # S1 -> s1
                action = parts[-1]           # acting1
                
                # Create output path: 12_keypoints/s1/acting1/gt_skel_gbl_pos.csv
                output_path = os.path.join(output_base, subject, action, "gt_skel_gbl_pos.csv")
                
                filter_gt_csv(input_path, output_path)

if __name__ == "__main__":
    batch_filter_ground_truth()
    print("\nGround truth filtering complete!")
