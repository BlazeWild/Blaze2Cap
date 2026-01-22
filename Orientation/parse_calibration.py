"""
Parse TotalCapture calibration.txt file and convert to structured format
Outputs: 
- calibration_params.npy (numpy structured array)
- calibration_params.json (JSON for easy reading)
- calibration_params_csv/ (separate CSV files for each component)
"""

import numpy as np
import json
import os
from pathlib import Path

def parse_calibration_file(filepath):
    """
    Parse TotalCapture calibration.txt file
    
    Format per camera:
        - min_row max_row min_col max_col (image bounds)
        - fx fy cx cy (intrinsic parameters)
        - distortion_param (single value)
        - 3x3 Rotation matrix R (3 rows)
        - 3x1 translation vector t (1 row)
    
    Args:
        filepath: path to calibration.txt
        
    Returns:
        list of dictionaries, one per camera
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First line: num_cameras distortion_order
    first_line = lines[0].strip().split()
    num_cameras = int(first_line[0])
    distortion_order = int(first_line[1])
    
    print(f"Number of cameras: {num_cameras}")
    print(f"Distortion order: {distortion_order}")
    
    cameras = []
    line_idx = 1
    
    for cam_id in range(num_cameras):
        camera_data = {}
        camera_data['camera_id'] = cam_id + 1  # 1-indexed
        
        # Image bounds: min_row max_row min_col max_col
        bounds = list(map(float, lines[line_idx].strip().split()))
        camera_data['min_row'] = bounds[0]
        camera_data['max_row'] = bounds[1]
        camera_data['min_col'] = bounds[2]
        camera_data['max_col'] = bounds[3]
        line_idx += 1
        
        # Intrinsic parameters: fx fy cx cy
        intrinsics = list(map(float, lines[line_idx].strip().split()))
        camera_data['fx'] = intrinsics[0]
        camera_data['fy'] = intrinsics[1]
        camera_data['cx'] = intrinsics[2]
        camera_data['cy'] = intrinsics[3]
        line_idx += 1
        
        # Distortion parameter
        camera_data['distortion'] = float(lines[line_idx].strip())
        line_idx += 1
        
        # Rotation matrix (3x3)
        rotation = []
        for i in range(3):
            row = list(map(float, lines[line_idx].strip().split()))
            rotation.append(row)
            line_idx += 1
        camera_data['rotation_matrix'] = np.array(rotation)
        
        # Translation vector (3x1)
        translation = list(map(float, lines[line_idx].strip().split()))
        camera_data['translation_vector'] = np.array(translation)
        line_idx += 1
        
        cameras.append(camera_data)
        print(f"  Parsed Camera {cam_id + 1}")
    
    return {
        'num_cameras': num_cameras,
        'distortion_order': distortion_order,
        'cameras': cameras
    }


def save_as_numpy(calibration_data, output_path):
    """
    Save calibration data as numpy .npz file
    """
    num_cameras = calibration_data['num_cameras']
    cameras = calibration_data['cameras']
    
    # Create structured arrays
    camera_ids = np.array([cam['camera_id'] for cam in cameras])
    
    # Image bounds
    bounds = np.array([[cam['min_row'], cam['max_row'], cam['min_col'], cam['max_col']] 
                       for cam in cameras])
    
    # Intrinsics
    intrinsics = np.array([[cam['fx'], cam['fy'], cam['cx'], cam['cy']] 
                           for cam in cameras])
    
    # Distortion
    distortion = np.array([cam['distortion'] for cam in cameras])
    
    # Rotation matrices (num_cameras x 3 x 3)
    rotation_matrices = np.array([cam['rotation_matrix'] for cam in cameras])
    
    # Translation vectors (num_cameras x 3)
    translation_vectors = np.array([cam['translation_vector'] for cam in cameras])
    
    # Save as .npz
    np.savez(
        output_path,
        num_cameras=num_cameras,
        distortion_order=calibration_data['distortion_order'],
        camera_ids=camera_ids,
        image_bounds=bounds,  # shape: (num_cameras, 4) [min_row, max_row, min_col, max_col]
        intrinsics=intrinsics,  # shape: (num_cameras, 4) [fx, fy, cx, cy]
        distortion=distortion,  # shape: (num_cameras,)
        rotation_matrices=rotation_matrices,  # shape: (num_cameras, 3, 3)
        translation_vectors=translation_vectors  # shape: (num_cameras, 3)
    )
    
    print(f"\nSaved numpy format: {output_path}")
    print(f"  - Arrays: camera_ids, image_bounds, intrinsics, distortion, rotation_matrices, translation_vectors")


def save_as_json(calibration_data, output_path):
    """
    Save calibration data as JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_data = {
        'num_cameras': calibration_data['num_cameras'],
        'distortion_order': calibration_data['distortion_order'],
        'cameras': []
    }
    
    for cam in calibration_data['cameras']:
        cam_json = {
            'camera_id': cam['camera_id'],
            'image_bounds': {
                'min_row': cam['min_row'],
                'max_row': cam['max_row'],
                'min_col': cam['min_col'],
                'max_col': cam['max_col']
            },
            'intrinsics': {
                'fx': cam['fx'],
                'fy': cam['fy'],
                'cx': cam['cx'],
                'cy': cam['cy']
            },
            'distortion': cam['distortion'],
            'rotation_matrix': cam['rotation_matrix'].tolist(),
            'translation_vector': cam['translation_vector'].tolist()
        }
        json_data['cameras'].append(cam_json)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved JSON format: {output_path}")


def save_as_csv(calibration_data, output_dir):
    """
    Save calibration data as CSV files (separate files for different components)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cameras = calibration_data['cameras']
    
    # 1. Image bounds CSV
    with open(output_dir / 'image_bounds.csv', 'w') as f:
        f.write('camera_id,min_row,max_row,min_col,max_col\n')
        for cam in cameras:
            f.write(f"{cam['camera_id']},{cam['min_row']},{cam['max_row']},{cam['min_col']},{cam['max_col']}\n")
    
    # 2. Intrinsics CSV
    with open(output_dir / 'intrinsics.csv', 'w') as f:
        f.write('camera_id,fx,fy,cx,cy\n')
        for cam in cameras:
            f.write(f"{cam['camera_id']},{cam['fx']},{cam['fy']},{cam['cx']},{cam['cy']}\n")
    
    # 3. Distortion CSV
    with open(output_dir / 'distortion.csv', 'w') as f:
        f.write('camera_id,distortion_param\n')
        for cam in cameras:
            f.write(f"{cam['camera_id']},{cam['distortion']}\n")
    
    # 4. Rotation matrices CSV (one file per camera)
    for cam in cameras:
        with open(output_dir / f'rotation_matrix_cam{cam["camera_id"]}.csv', 'w') as f:
            f.write('r11,r12,r13\n')
            f.write('r21,r22,r23\n')
            f.write('r31,r32,r33\n')
            for row in cam['rotation_matrix']:
                f.write(','.join(map(str, row)) + '\n')
    
    # 5. Translation vectors CSV
    with open(output_dir / 'translation_vectors.csv', 'w') as f:
        f.write('camera_id,tx,ty,tz\n')
        for cam in cameras:
            t = cam['translation_vector']
            f.write(f"{cam['camera_id']},{t[0]},{t[1]},{t[2]}\n")
    
    # 6. Combined CSV (all cameras, all params except rotation matrix)
    with open(output_dir / 'calibration_summary.csv', 'w') as f:
        f.write('camera_id,min_row,max_row,min_col,max_col,fx,fy,cx,cy,distortion,tx,ty,tz\n')
        for cam in cameras:
            t = cam['translation_vector']
            f.write(f"{cam['camera_id']},{cam['min_row']},{cam['max_row']},{cam['min_col']},{cam['max_col']},"
                   f"{cam['fx']},{cam['fy']},{cam['cx']},{cam['cy']},{cam['distortion']},"
                   f"{t[0]},{t[1]},{t[2]}\n")
    
    print(f"Saved CSV files in: {output_dir}")
    print(f"  - image_bounds.csv")
    print(f"  - intrinsics.csv")
    print(f"  - distortion.csv")
    print(f"  - rotation_matrix_cam*.csv (per camera)")
    print(f"  - translation_vectors.csv")
    print(f"  - calibration_summary.csv")


def main():
    """
    Main function
    """
    # Input file
    input_file = Path(r"D:\Ashok\_AI\_COMPUTER_VISION\____RESEARCH\___MOTION_T_LIGHTNING\Orientation\calibration.txt")
    output_dir = input_file.parent
    
    print("="*80)
    print("TotalCapture Calibration File Parser")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Parse calibration file
    print("\nParsing calibration file...")
    calibration_data = parse_calibration_file(input_file)
    
    # Save in different formats
    print("\n" + "="*80)
    print("Saving calibration data...")
    print("="*80)
    
    # 1. Numpy format
    numpy_output = output_dir / "calibration_params.npz"
    save_as_numpy(calibration_data, numpy_output)
    
    # 2. JSON format
    json_output = output_dir / "calibration_params.json"
    save_as_json(calibration_data, json_output)
    
    # 3. CSV format
    csv_output_dir = output_dir / "calibration_params_csv"
    save_as_csv(calibration_data, csv_output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {numpy_output.name} - NumPy format (use np.load)")
    print(f"  2. {json_output.name} - JSON format (human-readable)")
    print(f"  3. {csv_output_dir.name}/ - CSV files (Excel-compatible)")
    
    # Example usage
    print("\n" + "="*80)
    print("Usage Examples:")
    print("="*80)
    print("\nPython (NumPy):")
    print("  >>> data = np.load('calibration_params.npz')")
    print("  >>> intrinsics = data['intrinsics']  # shape: (8, 4)")
    print("  >>> rotation_matrices = data['rotation_matrices']  # shape: (8, 3, 3)")
    print("  >>> translation_vectors = data['translation_vectors']  # shape: (8, 3)")
    print("\nPython (JSON):")
    print("  >>> with open('calibration_params.json') as f:")
    print("  >>>     data = json.load(f)")
    print("  >>> cam1 = data['cameras'][0]")
    print("\nExcel/Pandas:")
    print("  >>> import pandas as pd")
    print("  >>> df = pd.read_csv('calibration_params_csv/calibration_summary.csv')")


if __name__ == "__main__":
    main()
