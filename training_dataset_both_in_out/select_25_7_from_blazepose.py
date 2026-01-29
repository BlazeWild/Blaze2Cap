"""
BlazePose Data Filtering Script
Filter BlazePose landmarks from (frames, 33, 10) to (frames, 25, 7)

Keypoints to Remove: 1, 2, 3, 4, 5, 6, 9, 10 (8 keypoints removed, 25 kept)
Channels to Keep: 0, 1, 2, 5, 6, 8, 9 (7 channels)
  - 0: world_x
  - 1: world_y
  - 2: world_z
  - 5: screen_x
  - 6: screen_y
  - 8: screen_visibility
  - 9: anchor_backup
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configuration
BLAZEPOSE_DIR = Path(__file__).parent.parent / "blazeposeall"
OUTPUT_DIR = Path(__file__).parent / "blazepose_25_7_nosync"

# Keypoints to keep (all except 1,2,3,4,5,6,9,10)
KEYPOINTS_TO_REMOVE = [1, 2, 3, 4, 5, 6, 9, 10]
KEYPOINTS_TO_KEEP = [i for i in range(33) if i not in KEYPOINTS_TO_REMOVE]  # Should be 25 keypoints

# Channels to keep
CHANNELS_TO_KEEP = [0, 1, 2, 5, 6, 8, 9]  # 7 channels

print(f"Keypoints to keep ({len(KEYPOINTS_TO_KEEP)}): {KEYPOINTS_TO_KEEP}")
print(f"Channels to keep ({len(CHANNELS_TO_KEEP)}): {CHANNELS_TO_KEEP}")
print()


def process_single_file(file_info):
    """Process a single NPY file"""
    input_file, relative_path = file_info
    
    try:
        # Load the data
        data = np.load(input_file)
        
        # Validate input shape
        if len(data.shape) != 3:
            return False, f"Invalid shape {data.shape} for {input_file}"
        
        frames, keypoints, channels = data.shape
        
        if keypoints != 33:
            return False, f"Expected 33 keypoints, got {keypoints} for {input_file}"
        
        if channels != 10:
            return False, f"Expected 10 channels, got {channels} for {input_file}"
        
        # Filter keypoints (remove indices 1,2,3,4,5,6,9,10)
        data_filtered_kp = data[:, KEYPOINTS_TO_KEEP, :]
        
        # Filter channels (keep indices 0,1,2,5,6,8,9)
        data_filtered = data_filtered_kp[:, :, CHANNELS_TO_KEEP]
        
        # Validate output shape
        expected_shape = (frames, 25, 7)
        if data_filtered.shape != expected_shape:
            return False, f"Output shape {data_filtered.shape} != expected {expected_shape}"
        
        # Create output directory
        output_file = OUTPUT_DIR / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered data
        np.save(output_file, data_filtered)
        
        return True, f"Processed {relative_path}: {data.shape} -> {data_filtered.shape}"
        
    except Exception as e:
        return False, f"Error processing {input_file}: {str(e)}"


def find_all_npy_files():
    """Find all NPY files in the blazeposeall directory"""
    npy_files = []
    
    for subject_dir in sorted(BLAZEPOSE_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        for action_dir in sorted(subject_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            
            for npy_file in sorted(action_dir.glob("*.npy")):
                relative_path = npy_file.relative_to(BLAZEPOSE_DIR)
                npy_files.append((npy_file, relative_path))
    
    return npy_files


def main():
    print(f"Source directory: {BLAZEPOSE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all NPY files
    print("Finding all NPY files...")
    npy_files = find_all_npy_files()
    print(f"Found {len(npy_files)} NPY files to process\n")
    
    if len(npy_files) == 0:
        print("No NPY files found. Exiting.")
        return
    
    # Process files in parallel
    num_workers = min(multiprocessing.cpu_count(), 16)
    print(f"Processing with {num_workers} workers...\n")
    
    success_count = 0
    error_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, file_info): file_info 
                   for file_info in npy_files}
        
        # Process results with progress bar
        with tqdm(total=len(npy_files), desc="Processing files") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(message)
                
                pbar.update(1)
                pbar.set_postfix({"Success": success_count, "Errors": error_count})
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total files: {len(npy_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\nFiltered data saved to: {OUTPUT_DIR}")
    print(f"Output shape for each file: (frames, 25, 7)")


if __name__ == "__main__":
    main()
