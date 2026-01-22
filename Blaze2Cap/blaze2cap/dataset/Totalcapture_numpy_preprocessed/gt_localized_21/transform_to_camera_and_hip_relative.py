"""
Transform Ground Truth data from global world coordinates to:
1. Camera-relative 3D positions (using calibration data)
2. Hip-relative positions (hip at 0,0,0)

Input: GT numpy files in global coordinates (already synced)
Output: Transformed GT files per camera with proper folder structure
"""

import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm


def load_calibration_data(calibration_json_path):
    """
    Load camera calibration data from JSON file.
    
    Returns:
        dict: Contains rotation_matrices (8, 3, 3) and translation_vectors (8, 3)
    """
    with open(calibration_json_path, 'r') as f:
        calib_data = json.load(f)
    
    num_cameras = calib_data['num_cameras']
    rotation_matrices = []
    translation_vectors = []
    
    for camera in calib_data['cameras']:
        R = np.array(camera['rotation_matrix'])
        t = np.array(camera['translation_vector'])
        rotation_matrices.append(R)
        translation_vectors.append(t)
    
    return {
        'num_cameras': num_cameras,
        'rotation_matrices': np.array(rotation_matrices),
        'translation_vectors': np.array(translation_vectors)
    }


def transform_to_camera_space(world_positions, camera_rotation, camera_translation):
    """
    Transform world positions to camera coordinate space.
    
    Camera transformation: p_camera = R * p_world + t
    
    Args:
        world_positions: (frames, joints, 3) array in world coordinates
        camera_rotation: (3, 3) rotation matrix
        camera_translation: (3,) translation vector
        
    Returns:
        (frames, joints, 3) array in camera coordinates
    """
    frames, joints, _ = world_positions.shape
    
    # Reshape to (frames*joints, 3) for batch transformation
    positions_flat = world_positions.reshape(-1, 3)
    
    # Apply transformation: R @ p + t
    camera_positions = (camera_rotation @ positions_flat.T).T + camera_translation
    
    # Reshape back to (frames, joints, 3)
    return camera_positions.reshape(frames, joints, 3)


def make_hip_relative(positions, hip_index=0):
    """
    Make all joint positions relative to hip (set hip to origin).
    
    Args:
        positions: (frames, joints, 3) array
        hip_index: Index of hip joint (default 0)
        
    Returns:
        (frames, joints, 3) array with hip at origin
    """
    # Extract hip positions (frames, 3)
    hip_positions = positions[:, hip_index:hip_index+1, :]
    
    # Subtract hip position from all joints
    relative_positions = positions - hip_positions
    
    return relative_positions


def process_single_file(input_path, output_base_dir, calibration_data, subject, action, camera_id):
    """
    Process a single GT file: 
    1. Change y to -y
    2. Transform to camera space
    3. Make hip-relative
    
    Args:
        input_path: Path to input .npy file (in world coordinates)
        output_base_dir: Base output directory
        calibration_data: Dictionary with calibration matrices
        subject: Subject name (e.g., 'S1')
        action: Action name (e.g., 'acting1')
        camera_id: Camera index (0-7)
    """
    # Load world positions
    world_positions = np.load(input_path)
    
    # Step 1: Transform y to -y
    world_positions[:, :, 1] = -world_positions[:, :, 1]
    
    # Step 2: Get camera calibration and transform to camera space
    R_cam = calibration_data['rotation_matrices'][camera_id]
    t_cam = calibration_data['translation_vectors'][camera_id]
    camera_positions = transform_to_camera_space(world_positions, R_cam, t_cam)
    
    # Step 3: Make hip-relative
    hip_relative_positions = make_hip_relative(camera_positions, hip_index=0)
    
    # Create output directory structure: output_base_dir/subject/action/
    output_dir = os.path.join(output_base_dir, subject, action)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename: gtl_{subject}_{action}_cam{camera_id+1}.npy
    output_filename = f"gtl_{subject}_{action}_cam{camera_id+1}.npy"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save
    np.save(output_path, hip_relative_positions)
    
    return output_path


def process_all_files(input_base_dir, output_base_dir, calibration_path):
    """
    Process all GT files from input directory.
    
    Args:
        input_base_dir: Directory containing synced GT files
        output_base_dir: Directory to save localized GT files
        calibration_path: Path to calibration JSON file
    """
    # Load calibration
    print("Loading calibration data...")
    calibration_data = load_calibration_data(calibration_path)
    print(f"Loaded calibration for {calibration_data['num_cameras']} cameras.")
    
    # Find all subjects
    input_path = Path(input_base_dir)
    subjects = sorted([d.name for d in input_path.iterdir() if d.is_dir()])
    
    print(f"\nFound subjects: {subjects}")
    
    total_files = 0
    processed_files = 0
    
    # Iterate through subjects and actions
    for subject in subjects:
        subject_dir = input_path / subject
        actions = sorted([d.name for d in subject_dir.iterdir() if d.is_dir()])
        
        print(f"\nProcessing {subject}...")
        print(f"  Actions: {actions}")
        
        for action in actions:
            action_dir = subject_dir / action
            
            # Find GT files for this action
            gt_files = sorted(list(action_dir.glob("gt_*.npy")))
            
            if not gt_files:
                print(f"  Warning: No GT files found for {subject}/{action}")
                continue
            
            # Since files are already per-camera, we process each
            # But the user wants to transform world->camera, so maybe the current files
            # are NOT yet camera-specific? Let me check the logic...
            
            # Actually, looking at the file listing, files are named:
            # gt_S1_acting1_cam1.npy, gt_S1_acting1_cam2.npy, etc.
            # This suggests they're already split by camera.
            
            # BUT the user said they want to transform FROM global world positions
            # TO camera-relative positions. So these current files might be in world coords
            # but just synced per camera (same frame count as blazepose for that camera).
            
            # Let's assume: current files are in WORLD coordinates, just synced.
            # We need to transform them to CAMERA coordinates.
            
            # So for each existing file (which is already camera-specific in terms of sync),
            # we apply the transformation for that specific camera.
            
            for gt_file in gt_files:
                # Extract camera number from filename
                # Format: gt_S1_acting1_cam1.npy -> camera 1 (index 0)
                filename = gt_file.name
                
                # Parse camera number
                if '_cam' in filename:
                    cam_str = filename.split('_cam')[1].split('.')[0]
                    camera_num = int(cam_str)  # 1-indexed
                    camera_idx = camera_num - 1  # 0-indexed
                else:
                    print(f"  Warning: Could not parse camera from {filename}")
                    continue
                
                # Process this file
                output_path = process_single_file(
                    input_path=str(gt_file),
                    output_base_dir=output_base_dir,
                    calibration_data=calibration_data,
                    subject=subject,
                    action=action,
                    camera_id=camera_idx
                )
                
                processed_files += 1
                total_files += 1
                
                if processed_files % 10 == 0:
                    print(f"  Processed {processed_files} files...")
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Total files processed: {total_files}")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}")


def verify_output(output_base_dir):
    """
    Verify the output by checking a sample file.
    """
    print("\nVerifying output...")
    
    # Find first output file
    output_path = Path(output_base_dir)
    sample_file = None
    
    for subject_dir in output_path.iterdir():
        if subject_dir.is_dir():
            for action_dir in subject_dir.iterdir():
                if action_dir.is_dir():
                    files = list(action_dir.glob("*.npy"))
                    if files:
                        sample_file = files[0]
                        break
            if sample_file:
                break
    
    if sample_file:
        data = np.load(sample_file)
        print(f"\nSample file: {sample_file.name}")
        print(f"  Shape: {data.shape}")
        print(f"  Hip position (frame 0, joint 0): {data[0, 0]}")
        print(f"  Expected: [0, 0, 0] (hip-relative)")
        
        if np.allclose(data[0, 0], [0, 0, 0], atol=1e-6):
            print("  ✓ Hip-relative transformation verified!")
        else:
            print("  ⚠ Warning: Hip is not at origin!")
    else:
        print("  No output files found for verification.")


def main():
    """Main execution function."""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # final_numpy_dataset
    
    input_dir = os.path.join(base_dir, "gt_numpy_synced_21")
    output_dir = os.path.join(base_dir, "gt_localized_21")
    calibration_file = os.path.join(script_dir, "calibration_params.json")
    
    print("="*80)
    print("GROUND TRUTH: WORLD → CAMERA-RELATIVE → HIP-RELATIVE TRANSFORMATION")
    print("="*80)
    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Calibration file: {calibration_file}")
    
    # Check if paths exist
    if not os.path.exists(input_dir):
        print(f"\nError: Input directory not found: {input_dir}")
        return
    
    if not os.path.exists(calibration_file):
        print(f"\nError: Calibration file not found: {calibration_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    process_all_files(input_dir, output_dir, calibration_file)
    
    # Verify
    verify_output(output_dir)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
