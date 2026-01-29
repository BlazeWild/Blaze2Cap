"""
Quaternion Orientation Example: Simple Arm Skeleton
Demonstrates:
1. Joint positions (shoulder, elbow, wrist)
2. Calculating bone orientations from positions
3. Global vs Local quaternions
4. Visualization in 3D space with coordinate frames
5. Camera coordinate transformations with interactive selection
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from scipy.spatial.transform import Rotation as R
import os


# ============================================================================
# JOINT POSITIONS (in world/global coordinates)
# ============================================================================

# Define joint positions in world space
SHOULDER_POS_WORLD = np.array([0.0, 0.0, 0.0])  # Root/origin
ELBOW_POS_WORLD = np.array([0.0, 1.0, 1.0])     # Y=1, Z=1 (up and forward)
WRIST_POS_WORLD = np.array([0.0, 2.0, 1.0])     # Y=2, Z=1 (further up)


# ============================================================================
# LOAD CALIBRATION DATA
# ============================================================================

def load_calibration_data():
    """
    Load camera calibration data from the parsed npz file.
    """
    calibration_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'calibration_params.npz'
    )
    
    if not os.path.exists(calibration_path):
        print(f"Warning: Calibration file not found: {calibration_path}")
        print("Using identity matrices for all cameras.")
        # Return identity matrices for 8 cameras
        return {
            'num_cameras': 8,
            'rotation_matrices': np.array([np.eye(3) for _ in range(8)]),
            'translation_vectors': np.array([np.zeros(3) for _ in range(8)])
        }
    
    data = np.load(calibration_path)
    return {
        'num_cameras': int(data['num_cameras']),
        'rotation_matrices': data['rotation_matrices'],
        'translation_vectors': data['translation_vectors']
    }


# ============================================================================
# CAMERA TRANSFORMATIONS
# ============================================================================

def transform_to_camera_space(positions, camera_rotation, camera_translation):
    """
    Transform world positions to camera coordinate space.
    Camera transformation: p' = R*p + t
    
    Args:
        positions: Nx3 array of world positions
        camera_rotation: 3x3 rotation matrix
        camera_translation: 3x1 translation vector
        
    Returns:
        Nx3 array of camera-space positions
    """
    # Apply rotation and translation
    camera_positions = (camera_rotation @ positions.T).T + camera_translation
    return camera_positions


def transform_orientation_to_camera(quaternion, camera_rotation):
    """
    Transform orientation quaternion from world to camera space.
    
    Args:
        quaternion: orientation in world space (x,y,z,w format)
        camera_rotation: 3x3 camera rotation matrix
        
    Returns:
        quaternion in camera space
    """
    r_camera = R.from_matrix(camera_rotation)
    r_world = R.from_quat(quaternion)
    
    # Apply camera rotation to world orientation
    r_camera_space = r_camera * r_world
    return r_camera_space.as_quat()


# ============================================================================
# CALCULATING ORIENTATIONS FROM POSITIONS
# ============================================================================

def calculate_bone_orientation(bone_vector, up_hint=np.array([0, 1, 0])):
    """
    Calculate orientation quaternion from a bone vector (direction from parent to child).
    
    The bone's local coordinate system:
    - Z-axis: points along the bone (bone direction)
    - Y-axis: points "up" (perpendicular to bone, uses up_hint)
    - X-axis: computed from cross product (right-handed system)
    
    Args:
        bone_vector: 3D vector from parent joint to child joint
        up_hint: preferred "up" direction for the bone's Y-axis
        
    Returns:
        quaternion representing the bone's orientation (scipy format: x,y,z,w)
    """
    # Normalize bone vector
    bone_dir = bone_vector / np.linalg.norm(bone_vector)
    
    # Build rotation matrix
    # Z-axis = bone direction
    z_axis = bone_dir
    
    # X-axis = perpendicular to both up_hint and bone_dir
    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # Bone is parallel to up_hint, use a different perpendicular vector
        x_axis = np.cross(np.array([1, 0, 0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y-axis = perpendicular to both z and x (right-handed)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Construct rotation matrix [X, Y, Z] as columns
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Convert to quaternion
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_quat()  # Returns [x, y, z, w]


def quaternion_to_euler(quat, degrees=True):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    rotation = R.from_quat(quat)
    euler = rotation.as_euler('xyz', degrees=degrees)
    return euler  # [roll, pitch, yaw] in degrees if degrees=True


# ============================================================================
# GLOBAL VS LOCAL QUATERNIONS EXPLAINED
# ============================================================================

def explain_global_vs_local():
    """
    Explains the difference between global and local quaternions.
    """
    print("\n" + "="*80)
    print("UNDERSTANDING GLOBAL VS LOCAL QUATERNIONS")
    print("="*80)
    
    print("\n1. GLOBAL QUATERNION:")
    print("   - Orientation relative to the world/camera coordinate system")
    print("   - For shoulder (root joint): describes how the shoulder is oriented")
    print("     relative to the world origin")
    print("   - Even though shoulder is at (0,0,0), it can still have rotation!")
    print("   - Example: Person's shoulder can be rotated 45° even if centered at origin")
    
    print("\n2. LOCAL/RELATIVE QUATERNION:")
    print("   - Orientation relative to the PARENT joint")
    print("   - For elbow: orientation relative to shoulder's coordinate frame")
    print("   - For wrist: orientation relative to elbow's coordinate frame")
    print("   - This is what you typically store in animation/motion data")
    
    print("\n3. ROOT JOINT (Shoulder) QUATERNIONS:")
    print("   - Global quaternion: Describes shoulder orientation in world space")
    print("   - No 'local' quaternion (no parent), but can have a 'default pose' reference")
    print("   - If person faces forward, shoulder global quat = identity [0,0,0,1]")
    print("   - If person faces right (+90° Y), shoulder quat changes accordingly")
    
    print("\n4. COMPUTING FROM POSITIONS:")
    print("   - Calculate bone direction vector: child_pos - parent_pos")
    print("   - Construct coordinate frame aligned with bone")
    print("   - Convert frame to quaternion")
    print("="*80)


# ============================================================================
# CALCULATE ALL ORIENTATIONS
# ============================================================================

def calculate_skeleton_orientations(shoulder_pos, elbow_pos, wrist_pos):
    """
    Calculate orientations for all bones in the skeleton.
    Returns global orientations for each joint.
    
    Args:
        shoulder_pos, elbow_pos, wrist_pos: 3D positions of joints
    """
    bone_upper_arm = elbow_pos - shoulder_pos
    bone_forearm = wrist_pos - elbow_pos
    
    print("\n" + "="*80)
    print("CALCULATING BONE ORIENTATIONS")
    print("="*80)
    
    # Upper arm orientation (shoulder to elbow)
    upper_arm_quat = calculate_bone_orientation(bone_upper_arm)
    upper_arm_euler = quaternion_to_euler(upper_arm_quat)
    
    print(f"\nUPPER ARM (Shoulder → Elbow):")
    print(f"  Bone vector: {bone_upper_arm}")
    print(f"  Length: {np.linalg.norm(bone_upper_arm):.3f}")
    print(f"  Global quaternion: [{upper_arm_quat[0]:.4f}, {upper_arm_quat[1]:.4f}, "
          f"{upper_arm_quat[2]:.4f}, {upper_arm_quat[3]:.4f}]")
    print(f"  Euler angles (deg): Roll={upper_arm_euler[0]:.1f}°, "
          f"Pitch={upper_arm_euler[1]:.1f}°, Yaw={upper_arm_euler[2]:.1f}°")
    
    # Forearm orientation (elbow to wrist)
    forearm_quat = calculate_bone_orientation(bone_forearm)
    forearm_euler = quaternion_to_euler(forearm_quat)
    
    print(f"\nFOREARM (Elbow → Wrist):")
    print(f"  Bone vector: {bone_forearm}")
    print(f"  Length: {np.linalg.norm(bone_forearm):.3f}")
    print(f"  Global quaternion: [{forearm_quat[0]:.4f}, {forearm_quat[1]:.4f}, "
          f"{forearm_quat[2]:.4f}, {forearm_quat[3]:.4f}]")
    print(f"  Euler angles (deg): Roll={forearm_euler[0]:.1f}°, "
          f"Pitch={forearm_euler[1]:.1f}°, Yaw={forearm_euler[2]:.1f}°")
    
    # Calculate LOCAL/RELATIVE orientation (forearm relative to upper arm)
    # This is: inverse(upper_arm_quat) * forearm_quat
    upper_arm_rot = R.from_quat(upper_arm_quat)
    forearm_rot = R.from_quat(forearm_quat)
    
    relative_forearm_rot = upper_arm_rot.inv() * forearm_rot
    relative_forearm_quat = relative_forearm_rot.as_quat()
    relative_forearm_euler = quaternion_to_euler(relative_forearm_quat)
    
    print(f"\nFOREARM RELATIVE TO UPPER ARM (Local orientation):")
    print(f"  Local quaternion: [{relative_forearm_quat[0]:.4f}, {relative_forearm_quat[1]:.4f}, "
          f"{relative_forearm_quat[2]:.4f}, {relative_forearm_quat[3]:.4f}]")
    print(f"  Local Euler angles (deg): Roll={relative_forearm_euler[0]:.1f}°, "
          f"Pitch={relative_forearm_euler[1]:.1f}°, Yaw={relative_forearm_euler[2]:.1f}°")
    
    print("="*80)
    
    return {
        'upper_arm': {
            'global_quat': upper_arm_quat,
            'global_euler': upper_arm_euler
        },
        'forearm': {
            'global_quat': forearm_quat,
            'global_euler': forearm_euler,
            'local_quat': relative_forearm_quat,
            'local_euler': relative_forearm_euler
        }
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_coordinate_frame(ax, position, quaternion, scale=0.3, label=""):
    """
    Plot a 3D coordinate frame (X=red, Y=green, Z=blue) at given position and orientation.
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    
    # Local axes in the rotated frame
    x_axis = rotation_matrix[:, 0] * scale
    y_axis = rotation_matrix[:, 1] * scale
    z_axis = rotation_matrix[:, 2] * scale
    
    # Plot axes
    ax.quiver(position[0], position[1], position[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='red', arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(position[0], position[1], position[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='green', arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(position[0], position[1], position[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='blue', arrow_length_ratio=0.3, linewidth=2)
    
    # Label
    if label:
        ax.text(position[0], position[1], position[2] + 0.1, label,
                fontsize=10, fontweight='bold')


def update_and_visualize(camera_id, calibration_data=None):
    """
    Handle updates when camera selection changes.
    """
    if calibration_data is None:
        calibration_data = load_calibration_data()
        
    # Standard world positions
    shoulder_pos = SHOULDER_POS_WORLD
    elbow_pos = ELBOW_POS_WORLD
    wrist_pos = WRIST_POS_WORLD
    
    # Apply camera transformation if selected
    if camera_id is not None:
        R_cam = calibration_data['rotation_matrices'][camera_id]
        t_cam = calibration_data['translation_vectors'][camera_id]
        
        # Transform positions to camera space
        positions = np.array([shoulder_pos, elbow_pos, wrist_pos])
        cam_positions = transform_to_camera_space(positions, R_cam, t_cam)
        
        shoulder_pos = cam_positions[0]
        elbow_pos = cam_positions[1]
        wrist_pos = cam_positions[2]
        
    # Calculate orientations based on current positions (World or Camera space)
    # This ensures the orientations make sense in the current view
    orientations = calculate_skeleton_orientations(shoulder_pos, elbow_pos, wrist_pos)
    
    # Visualize
    visualize_skeleton_3d(shoulder_pos, elbow_pos, wrist_pos, orientations, camera_id, calibration_data)


def visualize_skeleton_3d(shoulder_pos, elbow_pos, wrist_pos, orientations, 
                          camera_id=None, calibration_data=None):
    """
    Visualize the arm skeleton in 3D with coordinate frames and camera selection.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert positions to numpy array for easier plotting
    joints = np.array([shoulder_pos, elbow_pos, wrist_pos])
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='blue', s=100, alpha=0.8, edgecolors='black', linewidths=1, label='Joints')
    
    # Plot bones as lines
    ax.plot([shoulder_pos[0], elbow_pos[0]],
            [shoulder_pos[1], elbow_pos[1]],
            [shoulder_pos[2], elbow_pos[2]],
            'k-', linewidth=3, label='Upper Arm')
    
    ax.plot([elbow_pos[0], wrist_pos[0]],
            [elbow_pos[1], wrist_pos[1]],
            [elbow_pos[2], wrist_pos[2]],
            'k--', linewidth=2, label='Forearm')
    
    # Upper arm frame at shoulder
    plot_coordinate_frame(ax, shoulder_pos, orientations['upper_arm']['global_quat'],
                         scale=0.5, label="Shoulder")
    
    # Forearm frame at elbow
    plot_coordinate_frame(ax, elbow_pos, orientations['forearm']['global_quat'],
                         scale=0.4, label="Elbow")
                         
    # Wrist frame (using forearm orientation)
    plot_coordinate_frame(ax, wrist_pos, orientations['forearm']['global_quat'],
                         scale=0.3, label="Wrist")
    
    # Joint labels with coordinates
    ax.text(shoulder_pos[0], shoulder_pos[1], shoulder_pos[2]-0.2,
            f'Shoulder\n({shoulder_pos[0]:.2f},{shoulder_pos[1]:.2f},{shoulder_pos[2]:.2f})', 
            fontsize=8, ha='center')
    ax.text(elbow_pos[0], elbow_pos[1], elbow_pos[2]-0.2,
            f'Elbow\n({elbow_pos[0]:.2f},{elbow_pos[1]:.2f},{elbow_pos[2]:.2f})', 
            fontsize=8, ha='center')
    ax.text(wrist_pos[0], wrist_pos[1], wrist_pos[2]-0.2,
            f'Wrist\n({wrist_pos[0]:.2f},{wrist_pos[1]:.2f},{wrist_pos[2]:.2f})', 
            fontsize=8, ha='center')
    
    # Title changes based on camera
    if camera_id is None:
        title = 'Arm Skeleton - World Coordinates\nRed=X, Green=Y, Blue=Z'
    else:
        title = f'Arm Skeleton - Camera {camera_id + 1} View\nRed=X, Green=Y, Blue=Z'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set axis limits to keep view consistent/centered
    mid_x, mid_y, mid_z = np.mean(joints, axis=0)
    max_range = 2.5
    
    ax.set_xlim([mid_x - max_range, mid_x + max_range])
    ax.set_ylim([mid_y - max_range, mid_y + max_range])
    ax.set_zlim([mid_z - max_range, mid_z + max_range])
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add camera selection buttons if calibration data is available
    if calibration_data is not None:
        # Create button axes at the bottom
        button_height = 0.04
        button_width = 0.08
        button_spacing = 0.01
        start_x = 0.1
        start_y = 0.02
        
        buttons = []
        button_axes = []
        
        # World button
        ax_world = plt.axes([start_x, start_y, button_width, button_height])
        btn_world = Button(ax_world, 'World', color='lightgreen' if camera_id is None else 'lightgray')
        button_axes.append(ax_world)
        buttons.append(btn_world)
        
        # Camera buttons (1-8)
        for i in range(calibration_data['num_cameras']):
            x_pos = start_x + (button_width + button_spacing) * (i + 1)
            # Simple wrap if too many
            if x_pos > 0.9: 
                # Just shift up slightly for overflow or stack? 
                # For 8 cams + world, 9 buttons fits in one row usually if width is small.
                # Let's just keep one row for now assuming full screen width is enough
                pass
                
            ax_btn = plt.axes([x_pos, start_y, button_width, button_height])
            color = 'lightgreen' if camera_id == i else 'lightgray'
            btn = Button(ax_btn, f'Cam {i+1}', color=color)
            button_axes.append(ax_btn)
            buttons.append(btn)
        
        # Button click handlers
        def make_camera_callback(cam_id):
            def callback(event):
                plt.close(fig)
                update_and_visualize(cam_id, calibration_data)
            return callback
        
        # World button callback
        btn_world.on_clicked(make_camera_callback(None))
        
        # Camera button callbacks
        for i, btn in enumerate(buttons[1:]):
            btn.on_clicked(make_camera_callback(i))
        
        # Store buttons to prevent garbage collection
        fig.buttons = buttons
        fig.button_axes = button_axes
    
    plt.subplots_adjust(bottom=0.15)  # Make room for buttons
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to demonstrate quaternion orientations.
    """
    
    print("\n" + "="*80)
    print("QUATERNION ORIENTATION EXAMPLE: ARM SKELETON")
    print("="*80)
    
    print("\nJoint Positions (World Coordinates):")
    print(f"  Shoulder: {SHOULDER_POS_WORLD}")
    print(f"  Elbow:    {ELBOW_POS_WORLD}")
    print(f"  Wrist:    {WRIST_POS_WORLD}")
    
    # Load calibration
    print("\nLoading camera calibration...")
    calibration_data = load_calibration_data()
    print(f"Loaded calibration for {calibration_data['num_cameras']} cameras.")
    
    # Explain concepts
    explain_global_vs_local()
    
    # Start visualization
    print("\nStarting 3D visualization...")
    print("Use the buttons at the bottom to switch between World and Camera views.")
    update_and_visualize(None, calibration_data)


if __name__ == "__main__":
    main()
