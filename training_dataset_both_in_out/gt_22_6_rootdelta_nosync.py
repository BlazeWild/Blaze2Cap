"""
Ground Truth 22x6 Root Delta Dataset Generator

Takes existing gt_22_6_nosync data and converts:
- Index 0 (root position): Frame 0 = zeros, Frame N = position[N] - position[N-1]
- Index 1 (root orientation 6D): Frame 0 = zeros, Frame N = orientation[N] - orientation[N-1]
- Indices 2-21: Local 6D rotations (unchanged)

Input: gt_22_6_nosync/{Subject}/{Action}/gt_{Subject}_{Action}_cam{N}.npy
Output: gt_22_6_rootdelta_nosync/{Subject}/{Action}/gt_{Subject}_{Action}_cam{N}.npy
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING'
INPUT_DIR = os.path.join(BASE_DIR, 'training_dataset_both_in_out/gt_22_6_nosync')
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_dataset_both_in_out/gt_22_6_rootdelta_nosync')


def process_file(input_path, output_path):
    """
    Process a single GT file to convert root motion/orientation to deltas.
    
    Frame 0: root position = (0,0,0,0,0,0), root orientation = (0,0,0,0,0,0)
    Frame N: root position = pos[N] - pos[N-1], root orientation = ori[N] - ori[N-1]
    """
    # Load data
    data = np.load(input_path)  # Shape: (frames, 22, 6)
    num_frames = data.shape[0]
    
    # Create output array
    output = data.copy()
    
    # Process root position (index 0)
    # Frame 0: set to zeros
    output[0, 0, :] = 0.0
    # Frame N onwards: delta = current - previous
    for i in range(1, num_frames):
        output[i, 0, :] = data[i, 0, :] - data[i-1, 0, :]
    
    # Process root orientation (index 1)
    # Frame 0: set to zeros
    output[0, 1, :] = 0.0
    # Frame N onwards: delta = current - previous
    for i in range(1, num_frames):
        output[i, 1, :] = data[i, 1, :] - data[i-1, 1, :]
    
    # Indices 2-21 (local 6D) remain unchanged
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    np.save(output_path, output)
    
    return num_frames


def main():
    print("=" * 60)
    print("Ground Truth 22x6 Root Delta Dataset Generator")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Find all .npy files
    input_files = list(Path(INPUT_DIR).rglob("*.npy"))
    print(f"Found {len(input_files)} files to process")
    print()
    
    total_frames = 0
    
    for input_path in tqdm(input_files, desc="Processing files"):
        # Compute output path (same relative structure)
        rel_path = input_path.relative_to(INPUT_DIR)
        output_path = Path(OUTPUT_DIR) / rel_path
        
        # Process file
        frames = process_file(str(input_path), str(output_path))
        total_frames += frames
    
    print()
    print("=" * 60)
    print(f"Completed! Processed {len(input_files)} files, {total_frames} total frames")
    print(f"Output saved to: {OUTPUT_DIR}")
    print()
    print("Output format (22 joints x 6 channels):")
    print("  Index 0: Root position delta (zeros at frame 0)")
    print("  Index 1: Root orientation 6D delta (zeros at frame 0)")
    print("  Indices 2-21: Local 6D for 20 child joints (unchanged)")


if __name__ == "__main__":
    main()
