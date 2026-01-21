import os
import csv

# Conversion factor: inches to meters
INCHES_TO_METERS = 0.0254

def convert_pos_txt_to_csv(txt_file_path, convert_to_meters=False):
    """Convert gt_skel_gbl_pos.txt to CSV format with grouped coordinates.
    
    Args:
        txt_file_path: Path to the txt file to convert
        convert_to_meters: If True, converts coordinates from inches to meters
    """
    csv_file_path = txt_file_path.replace('.txt', '.csv')
    
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()
    
    # First line is header with tab-separated joint names
    header = lines[0].strip().split('\t')
    header = [h for h in header if h]  # Remove empty strings
    
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        
        # Process data lines (starting from line 2)
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                # Split by tabs - each part contains x y z for one joint
                parts = line.strip().split('\t')
                row = []
                for part in parts:
                    if part.strip():
                        # Keep x, y, z together as "x, y, z"
                        coords = part.strip().split()
                        if len(coords) == 3:
                            if convert_to_meters:
                                # Convert from inches to meters
                                x = float(coords[0]) * INCHES_TO_METERS
                                y = float(coords[1]) * INCHES_TO_METERS
                                z = float(coords[2]) * INCHES_TO_METERS
                                row.append(f"{x}, {y}, {z}")
                            else:
                                row.append(f"{coords[0]}, {coords[1]}, {coords[2]}")
                
                writer.writerow(row)
    
    conversion_msg = " (converted to meters)" if convert_to_meters else ""
    print(f"Converted: {txt_file_path} -> {csv_file_path}{conversion_msg}")
    return csv_file_path

def convert_ori_txt_to_csv(txt_file_path):
    """Convert gt_skel_gbl_ori.txt to CSV format with grouped coordinates.
    
    Note: Orientation data (quaternions) are not converted as they are unitless.
    """
    csv_file_path = txt_file_path.replace('.txt', '.csv')
    
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()
    
    # First line is header with tab-separated joint names
    header = lines[0].strip().split('\t')
    header = [h for h in header if h]  # Remove empty strings
    
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        
        # Process data lines (starting from line 2)
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                # Split by tabs - each part contains quaternion values for one joint
                parts = line.strip().split('\t')
                row = []
                for part in parts:
                    if part.strip():
                        # Keep quaternion values together
                        values = part.strip().split()
                        if len(values) == 4:
                            row.append(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}")
                
                writer.writerow(row)
    
    print(f"Converted: {txt_file_path} -> {csv_file_path}")
    return csv_file_path


def batch_convert_positions(positions_dir, subjects=['S1', 'S2', 'S3'], convert_to_meters=False):
    """Convert all gt_skel_gbl_pos.txt and gt_skel_gbl_ori.txt files for specified subjects.
    
    Args:
        positions_dir: Base directory containing subject folders
        subjects: List of subject folders to process (e.g., ['S1', 'S2', 'S3'])
        convert_to_meters: If True, converts position coordinates from inches to meters
    """
    pos_count = 0
    ori_count = 0
    
    for subject in subjects:
        subject_dir = os.path.join(positions_dir, subject)
        if not os.path.exists(subject_dir):
            print(f"Warning: Subject directory {subject_dir} not found, skipping...")
            continue
        
        print(f"\n=== Processing {subject} ===")
        for root, dirs, files in os.walk(subject_dir):
            for file in files:
                if file == 'gt_skel_gbl_pos.txt':
                    txt_path = os.path.join(root, file)
                    convert_pos_txt_to_csv(txt_path, convert_to_meters=convert_to_meters)
                    pos_count += 1
                elif file == 'gt_skel_gbl_ori.txt':
                    txt_path = os.path.join(root, file)
                    convert_ori_txt_to_csv(txt_path)
                    ori_count += 1
    
    print(f"\n{'='*50}")
    print(f"Total position files converted: {pos_count}")
    print(f"Total orientation files converted: {ori_count}")
    if convert_to_meters:
        print(f"Position coordinates converted from inches to meters (×{INCHES_TO_METERS})")

if __name__ == "__main__":
    # Update this path to match your workspace
    positions_dir = r"d:\Ashok\_AI\_COMPUTER_VISION\____RESEARCH\___MOTION_T_LIGHTNING\totalcapture_dataset\positions"
    
    # Process S4 and S5 with inch-to-meter conversion
    print("Converting S4 and S5 data (inches to meters)...")
    batch_convert_positions(positions_dir, subjects=['S4', 'S5'], convert_to_meters=True)

