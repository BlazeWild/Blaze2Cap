import numpy as np

# Load sample data
blaze = np.load('final_numpy_dataset/blazepose_numpy/S1/acting1/blaze_S1_acting1_cam1.npy')
gt = np.load('final_numpy_dataset/gt_numpy/S1/acting1/gt_S1_acting1_cam1.npy')

print("=" * 70)
print("FINAL DATASET VERIFICATION")
print("=" * 70)

print(f"\nBlazePose shape: {blaze.shape}")
print(f"GT shape: {gt.shape}")
print(f"Both have same frames: {blaze.shape[0] == gt.shape[0]}")

print(f"\n--- GT Data Verification ---")
print(f"First frame, Hip (joint 0) position: {gt[0, 0, :]}")
print(f"  Should be close to [0, 0, 0] (hip-relative)")

print(f"\nFirst frame, Spine (joint 1) position: {gt[0, 1, :]}")

print(f"\n--- Coordinate System Check ---")
print(f"Y-axis values (should be flipped):")
print(f"  Frame 0, Joint 0, Y: {gt[0, 0, 1]:.6f}")
print(f"  Frame 0, Joint 1, Y: {gt[0, 1, 1]:.6f}")

print(f"\n--- Units Check (should be in meters) ---")
print(f"  Max coordinate value: {np.abs(gt).max():.3f} m")
print(f"  Min coordinate value: {np.abs(gt).min():.6f} m")

print("\n✓ Verification complete!")
