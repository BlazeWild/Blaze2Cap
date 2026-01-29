"""
BlazePose Coordinate Matched Dataset Generator

Transforms BlazePose 25x7 data into coordinate-matched format with:
- Channels 0,1,2: world_x, world_y, world_z (with 90° pitch rotation)
- Channels 3,4: delta_screen_x, delta_screen_y (transformed, hip-relative with anchor)
- Channel 5: visibility
- Channel 6: anchor flag

Input: blazepose_25_7_nosync (25 keypoints x 7 channels)
    - 0,1,2: world x,y,z
    - 3,4: screen x,y (0-1 range)
    - 5: visibility
    - 6: anchor flag

Output: blazepose_coordinates_matched (25 keypoints x 7 channels)
    - 0,1,2: world_x, world_y, world_z (rotated to match GT)
    - 3,4: delta_screen_x, delta_screen_y (transformed, hip-relative deltas)
    - 5: visibility
    - 6: anchor flag
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/training_dataset_both_in_out/blazepose_25_7_nosync'
OUTPUT_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/training_dataset_both_in_out/blazepose_ccordinate_matched_and_delta'

# Hip indices for center calculation
LEFT_HIP_IDX = 15
RIGHT_HIP_IDX = 16

# ==========================================
# TRANSFORMATION FUNCTIONS
# ==========================================

def transform_world_coords(world_xyz):
    """
    Apply 90-degree pitch rotation to world coordinates.
    BlazePose: X horizontal, Y vertical (down), Z depth
    Target: X horizontal, Y depth, Z vertical (up)
    
    Transform:
        new_x = x
        new_y = z  (depth)
        new_z = -y (flip vertical)
    
    Args:
        world_xyz: (frames, 25, 3) - world x,y,z coordinates
    Returns:
        transformed: (frames, 25, 3) - rotated coordinates
    """
    transformed = np.zeros_like(world_xyz)
    # Target: instead of x we get -x, instead of y we get -y for world coords
    # Original Input: x, y, z
    # Original Transform: x -> x, z -> y, -y -> z
    # New Transform: x -> -x, z -> -y (interpreted as negated planar coords?), -y -> z
    # "instead of x we get -x and instead of y we get -y for the x,y,z of world in the first 3 indexes"
    # Mapping to output channels 0, 1, 2:
    # 0 (was x) -> becomes -x
    # 1 (was z ie depth) -> becomes -z (assuming y refers to the second output channel)
    # 2 (was -y ie vertical) -> stays same?
    
    # Based on "x,y,z of world in the first 3 indexes":
    # Index 0: -X (from original X)
    # Index 1: -Z (from original Z, which maps to Y/depth channel)
    # Index 2: -Y (from original -Y? or original Y?)
    
    # Let's apply literally: 
    # Old Output 0 = x. New Output 0 = -x.
    # Old Output 1 = z. New Output 1 = -z.
    # Old Output 2 = -y. New Output 2 = -y (unchanged from previous transform logic, just copying z-axis logic?)
    
    transformed[:, :, 0] = -world_xyz[:, :, 0]   # X becomes -X
    transformed[:, :, 1] = -world_xyz[:, :, 2]   # Y(output) becomes -Z(input)
    transformed[:, :, 2] = -world_xyz[:, :, 1]   # Z(output) becomes -Y(input) (Vertical flip maintained)
    return transformed


def transform_screen_coords(screen_xy):
    """
    Transform screen coordinates from BlazePose to center-origin.
    BlazePose: (0,0) top-left, (1,1) bottom-right
    Target: (0,0) center, (-1,-1) bottom-left, (1,1) top-right
    
    Args:
        screen_xy: (frames, 25, 2) - screen x,y in 0-1 range
    Returns:
        transformed: (frames, 25, 2) - center-origin coordinates
    """
    transformed = np.zeros_like(screen_xy)
    transformed[:, :, 0] = (screen_xy[:, :, 0] - 0.5) * 2  # x: left(-1) to right(1)
    transformed[:, :, 1] = (0.5 - screen_xy[:, :, 1]) * 2  # y: bottom(-1) to top(1)
    return transformed


def compute_hip_center(keypoints):
    """Compute hip center as midpoint of left and right hip."""
    return (keypoints[:, LEFT_HIP_IDX] + keypoints[:, RIGHT_HIP_IDX]) / 2


def compute_delta_screen_with_anchors(screen_xy_transformed, anchor_flags):
    """
    Compute delta screen coordinates with anchor frame handling.
    
    Following the logic from blazepose2dscreen.py:
    
    At anchor frames (flag==0):
        - Delta = transformed_coords - hip_center (hip-relative position)
        - This establishes the reference with hip at origin
    
    At non-anchor frames (flag==1):
        - Delta = current - previous (in transformed space)
        - Pure frame-to-frame movement
    
    When reconstructing for visualization:
        - Accumulate these deltas: positions[i] = positions[i-1] + delta[i]
        - Frame 0 starts from hip-relative position
        - Subsequent frames add their deltas
    
    Args:
        screen_xy_transformed: (frames, 25, 2) - transformed screen coords
        anchor_flags: (frames, 25) - anchor flags per keypoint (0=anchor, 1=regular)
    
    Returns:
        delta_screen: (frames, 25, 2) - delta screen coordinates
    """
    num_frames = screen_xy_transformed.shape[0]
    delta_screen = np.zeros_like(screen_xy_transformed)
    
    prev_transformed = None
    
    for i in range(num_frames):
        current_transformed = screen_xy_transformed[i]
        
        # Check if this is an anchor frame (flag==0 means anchor/reference frame)
        is_anchor = np.any(anchor_flags[i] == 0)
        
        if is_anchor:
            # Anchor frame: store hip-relative position
            # Hip center = midpoint of left_hip (15) and right_hip (16)
            hip_center = (current_transformed[LEFT_HIP_IDX] + current_transformed[RIGHT_HIP_IDX]) / 2
            # All keypoints relative to hip center
            delta_screen[i] = current_transformed - hip_center
        else:
            # Non-anchor frame: compute delta from previous frame
            # Delta = current_transformed - prev_transformed
            if prev_transformed is not None:
                delta_screen[i] = current_transformed - prev_transformed
        
        prev_transformed = current_transformed.copy()
    
    return delta_screen


def process_file(input_path, output_path):
    """
    Process a single BlazePose file.
    
    Args:
        input_path: Path to input .npy file
        output_path: Path to save output .npy file
    
    For anchor frames (flag==0):
        - delta_screen_x, delta_screen_y = 0, 0
        - visibility = 0
    For non-anchor frames (flag==1):
        - delta = transformed(frame_n) - transformed(frame_n-1)
        - visibility = original visibility
    """
    # Load data
    data = np.load(input_path)  # Shape: (frames, 25, 7)
    
    # Extract channels
    world_xyz = data[:, :, 0:3]       # world x,y,z
    screen_xy = data[:, :, 3:5]       # screen x,y (0-1 range)
    visibility = data[:, :, 5:6].copy()  # visibility (will be modified)
    anchor_flag = data[:, :, 6:7]     # anchor flag
    
    # Transform world coordinates (90° pitch rotation)
    world_xyz_transformed = transform_world_coords(world_xyz)
    
    # Transform screen coordinates from 0-1 to -1,1 range
    # -1,-1 at bottom-left, 1,1 at top-right
    screen_xy_transformed = transform_screen_coords(screen_xy)
    
    # Compute delta screen with anchor handling
    delta_screen = compute_delta_screen_with_anchors(
        screen_xy_transformed, 
        data[:, :, 6]  # anchor flags
    )
    
    # Set visibility to 0 for anchor frames (flag==0)
    anchor_mask = (anchor_flag == 0)  # Shape: (frames, 25, 1)
    
    # "make index 3,4 = 0 where index 6 = 0"
    # Zero out delta screen coordinates at anchor frames
    # (Note: shape of delta_screen is (frames, 25, 2))
    # We need to broadcast anchor_mask (frames, 25, 1) to (frames, 25, 2)
    anchor_mask_broadcast = np.repeat(anchor_mask, 2, axis=2)
    delta_screen[anchor_mask_broadcast] = 0.0

    visibility[anchor_mask] = 0.0
    
    # Combine into output format
    # Channels: world_x, world_y, world_z, delta_screen_x, delta_screen_y, visibility, anchor_flag
    output_data = np.concatenate([
        world_xyz_transformed,  # 0,1,2
        delta_screen,           # 3,4
        visibility,             # 5
        anchor_flag             # 6
    ], axis=2)
    
    # Ensure output shape
    assert output_data.shape == data.shape, f"Shape mismatch: {output_data.shape} vs {data.shape}"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    np.save(output_path, output_data)
    
    return data.shape[0]  # Return frame count for logging


def main():
    print("=" * 60)
    print("BlazePose Coordinate Matched Dataset Generator")
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
    print("Output format (25 keypoints x 7 channels):")
    print("  0: world_x (rotated)")
    print("  1: world_y (rotated)")
    print("  2: world_z (rotated)")
    print("  3: delta_screen_x (transformed, hip-relative)")
    print("  4: delta_screen_y (transformed, hip-relative)")
    print("  5: visibility")
    print("  6: anchor_flag")


if __name__ == "__main__":
    main()
