import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(SCRIPT_DIR, "bvh_bone_structure.json")

def calculate_positions_from_hierarchy(structure):
    """
    Calculate 3D positions for each joint by traversing the hierarchy
    and accumulating offsets from root to each joint.
    """
    positions = {}
    
    # Start with root at origin
    root_name = None
    for name, info in structure.items():
        if info['parent'] is None:
            root_name = name
            positions[name] = np.array([0.0, 0.0, 0.0])
            break
    
    if root_name is None:
        print("Error: No root joint found!")
        return positions
    
    # Breadth-first traversal to compute positions
    processed = {root_name}
    changed = True
    
    while changed:
        changed = False
        for joint_name, info in structure.items():
            if joint_name in processed:
                continue
                
            parent_name = info['parent']
            if parent_name and parent_name in positions:
                # Parent position is known, compute this joint's position
                parent_pos = positions[parent_name]
                offset = np.array(info.get('offset_meters', [0, 0, 0]))
                positions[joint_name] = parent_pos + offset
                processed.add(joint_name)
                changed = True
    
    return positions

def plot_skeleton(positions, structure):
    """Plot the skeleton in 3D with joint names."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions as arrays
    joint_names = list(positions.keys())
    pos_array = np.array([positions[name] for name in joint_names])
    
    # Plot joints
    ax.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
               c='red', marker='o', s=100, alpha=0.8, label='Joints')
    
    # Plot bones (connections)
    for joint_name, info in structure.items():
        parent_name = info['parent']
        if parent_name and parent_name in positions and joint_name in positions:
            p1 = positions[parent_name]
            p2 = positions[joint_name]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'b-', linewidth=2, alpha=0.6)
    
    # Add labels for each joint
    for joint_name in joint_names:
        pos = positions[joint_name]
        # Only label main joints (not End sites) to reduce clutter
        if not joint_name.endswith('_End'):
            ax.text(pos[0], pos[1], pos[2], joint_name, 
                   fontsize=8, color='darkblue')
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('BVH Skeleton Structure (T-Pose)', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    all_pos = np.array(list(positions.values()))
    max_range = np.array([
        all_pos[:, 0].max() - all_pos[:, 0].min(),
        all_pos[:, 1].max() - all_pos[:, 1].min(),
        all_pos[:, 2].max() - all_pos[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_pos[:, 0].max() + all_pos[:, 0].min()) * 0.5
    mid_y = (all_pos[:, 1].max() + all_pos[:, 1].min()) * 0.5
    mid_z = (all_pos[:, 2].max() + all_pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=10, azim=45)
    
    plt.tight_layout()
    return fig, ax

def main():
    if not os.path.exists(JSON_FILE):
        print(f"Error: JSON file not found at {JSON_FILE}")
        return
    
    # Load structure
    print(f"Loading bone structure from: {JSON_FILE}")
    with open(JSON_FILE, 'r') as f:
        structure = json.load(f)
    
    print(f"Loaded {len(structure)} joints")
    
    # Calculate positions
    print("Calculating joint positions...")
    positions = calculate_positions_from_hierarchy(structure)
    
    print(f"Computed positions for {len(positions)} joints\n")
    
    # Print positions
    print("=" * 80)
    print(f"{'JOINT NAME':<25} | {'POSITION (meters)':<40}")
    print("-" * 80)
    for name, pos in positions.items():
        pos_str = f"[{pos[0]:>7.4f}, {pos[1]:>7.4f}, {pos[2]:>7.4f}]"
        print(f"{name:<25} | {pos_str:<40}")
    print("=" * 80)
    
    # Plot skeleton
    print("\nPlotting skeleton...")
    fig, ax = plot_skeleton(positions, structure)
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()
