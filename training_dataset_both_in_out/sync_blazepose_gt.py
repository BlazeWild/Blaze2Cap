"""
BlazePose and Ground Truth Synchronization Script

Synchronizes blazepose_coordinates_matched and gt_22_6_rootdelta_nosync:
1. Take min(blazepose_frames, gt_frames) to align frame counts
2. Remove frames where subject is not seen (world x,y,z = 0,0,0 for all keypoints)
3. Reset GT root delta (index 0 and 1) to zeros at anchor frames (flag==0)
4. Save synchronized files to blazepose_synced and gt_synced folders

Input:
    - blazepose_coordinates_matched/{S}/{action}/blaze_{S}_{action}_cam{N}.npy
    - gt_22_6_rootdelta_nosync/{S}/{action}/gt_{S}_{action}_cam{N}.npy

Output:
    - blazepose_synced/{S}/{action}/blaze_{S}_{action}_cam{N}.npy
    - gt_synced/{S}/{action}/gt_{S}_{action}_cam{N}.npy
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/training_dataset_both_in_out'

BLAZEPOSE_INPUT_DIR = os.path.join(BASE_DIR, 'blazepose_coordinates_matched')
GT_INPUT_DIR = os.path.join(BASE_DIR, 'gt_22_6_rootdelta_nosync')

BLAZEPOSE_OUTPUT_DIR = os.path.join(BASE_DIR, 'blazepose_synced')
GT_OUTPUT_DIR = os.path.join(BASE_DIR, 'gt_synced')


def is_subject_visible(blaze_frame):
    """
    Check if subject is visible in frame.
    Subject is NOT visible if world x,y,z (indices 0,1,2) are all 0 for ALL keypoints.
    
    Args:
        blaze_frame: (25, 7) - single frame of blazepose data
    
    Returns:
        True if subject is visible, False otherwise
    """
    world_coords = blaze_frame[:, 0:3]  # Shape: (25, 3)
    # Check if all world coordinates are zeros
    all_zeros = np.allclose(world_coords, 0.0, atol=1e-6)
    return not all_zeros


def process_pair(blaze_path, gt_path, blaze_out_path, gt_out_path):
    """
    Process a pair of blazepose and GT files.
    
    Steps:
    1. Truncate to min(blaze_frames, gt_frames)
    2. Find frames where subject is visible
    3. Reset GT root delta at anchor frames (flag==0)
    4. Filter both to only visible frames
    5. Save synchronized files
    
    Returns: (original_frames, synced_frames, removed_frames)
    """
    # Load data
    blaze_data = np.load(blaze_path)  # Shape: (frames, 25, 7)
    gt_data = np.load(gt_path)        # Shape: (frames, 22, 6)
    
    # Step 1: Truncate to same length
    min_frames = min(blaze_data.shape[0], gt_data.shape[0])
    blaze_data = blaze_data[:min_frames]
    gt_data = gt_data[:min_frames]
    
    # Step 2: Find visible frames (where subject is seen)
    visible_mask = np.array([is_subject_visible(blaze_data[i]) for i in range(min_frames)])
    
    # Step 3: Reset GT root delta at anchor frames (where flag==0)
    # Anchor flag is in blazepose channel 6
    for i in range(min_frames):
        anchor_flag = blaze_data[i, 0, 6]  # Check first keypoint's anchor flag
        if anchor_flag == 0:
            # Reset root position delta and root orientation delta to zeros
            gt_data[i, 0, :] = 0.0  # Root position delta
            gt_data[i, 1, :] = 0.0  # Root orientation delta
    
    # Step 4: Filter to only visible frames
    blaze_synced = blaze_data[visible_mask]
    gt_synced = gt_data[visible_mask]
    
    # Step 5: Save
    os.makedirs(os.path.dirname(blaze_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_out_path), exist_ok=True)
    
    np.save(blaze_out_path, blaze_synced)
    np.save(gt_out_path, gt_synced)
    
    removed_frames = min_frames - visible_mask.sum()
    return min_frames, blaze_synced.shape[0], removed_frames


def find_matching_files():
    """
    Find matching blazepose and GT files.
    Matches by subject/action/camera.
    
    Returns: list of (blaze_path, gt_path, subject, action, cam) tuples
    """
    pairs = []
    
    # Iterate through blazepose files
    for blaze_path in Path(BLAZEPOSE_INPUT_DIR).rglob("*.npy"):
        # Parse filename: blaze_S1_acting1_cam1.npy
        filename = blaze_path.stem  # blaze_S1_acting1_cam1
        parts = filename.split('_')
        
        if len(parts) >= 4 and parts[0] == 'blaze':
            subject = parts[1]      # S1
            action = parts[2]       # acting1
            cam = parts[3]          # cam1
            
            # Find corresponding GT file
            gt_filename = f"gt_{subject}_{action}_{cam}.npy"
            gt_path = Path(GT_INPUT_DIR) / subject / action / gt_filename
            
            if gt_path.exists():
                pairs.append((str(blaze_path), str(gt_path), subject, action, cam))
    
    return pairs


def main():
    print("=" * 60)
    print("BlazePose and Ground Truth Synchronization")
    print("=" * 60)
    print(f"BlazePose Input:  {BLAZEPOSE_INPUT_DIR}")
    print(f"GT Input:         {GT_INPUT_DIR}")
    print(f"BlazePose Output: {BLAZEPOSE_OUTPUT_DIR}")
    print(f"GT Output:        {GT_OUTPUT_DIR}")
    print()
    
    # Find matching files
    pairs = find_matching_files()
    print(f"Found {len(pairs)} matching file pairs")
    print()
    
    total_original_frames = 0
    total_synced_frames = 0
    total_removed_frames = 0
    
    for blaze_path, gt_path, subject, action, cam in tqdm(pairs, desc="Syncing files"):
        # Compute output paths
        blaze_out = os.path.join(BLAZEPOSE_OUTPUT_DIR, subject, action, f"blaze_{subject}_{action}_{cam}.npy")
        gt_out = os.path.join(GT_OUTPUT_DIR, subject, action, f"gt_{subject}_{action}_{cam}.npy")
        
        # Process pair
        orig, synced, removed = process_pair(blaze_path, gt_path, blaze_out, gt_out)
        
        total_original_frames += orig
        total_synced_frames += synced
        total_removed_frames += removed
    
    print()
    print("=" * 60)
    print(f"Completed! Synchronized {len(pairs)} file pairs")
    print(f"Original frames:  {total_original_frames}")
    print(f"Synced frames:    {total_synced_frames}")
    print(f"Removed frames:   {total_removed_frames} ({100*total_removed_frames/total_original_frames:.2f}%)")
    print()
    print("Output:")
    print(f"  BlazePose: {BLAZEPOSE_OUTPUT_DIR}")
    print(f"  GT:        {GT_OUTPUT_DIR}")
    print()
    print("Processing done:")
    print("  - Frame count aligned (min of blazepose and GT)")
    print("  - Invisible frames removed (world coords all zeros)")
    print("  - GT root delta reset at anchor frames (flag==0)")


if __name__ == "__main__":
    main()
