import os
import numpy as np
import pandas as pd
from pathlib import Path

def count_csv_rows(csv_path):
    """Count rows in CSV file excluding header."""
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        return f"Error: {e}"

def get_npy_size(npy_path):
    """Get the shape of numpy array."""
    try:
        data = np.load(npy_path)
        return data.shape
    except Exception as e:
        return f"Error: {e}"

def analyze_subject_activity(subject, activity, positions_base, blazepose_base):
    """Analyze size comparison for a specific subject and activity."""
    # Path to CSV file
    csv_path = os.path.join(positions_base, subject, activity, "gt_skel_gbl_pos.csv")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        return None
    
    # Count CSV rows
    csv_rows = count_csv_rows(csv_path)
    
    # Check all 8 camera numpy files
    npy_results = {}
    blazepose_activity_path = os.path.join(blazepose_base, subject, activity)
    
    if os.path.exists(blazepose_activity_path):
        for cam in range(1, 9):
            npy_file = f"blaze_{subject}_{activity}_cam{cam}.npy"
            npy_path = os.path.join(blazepose_activity_path, npy_file)
            
            if os.path.exists(npy_path):
                npy_results[f"cam{cam}"] = get_npy_size(npy_path)
            else:
                npy_results[f"cam{cam}"] = "Not found"
    
    return {
        "csv_rows": csv_rows,
        "npy_files": npy_results
    }

def main():
    # Base paths
    positions_base = r"d:\Ashok\_AI\_COMPUTER_VISION\____RESEARCH\___MOTION_T_LIGHTNING\totalcapture_dataset\positions"
    blazepose_base = r"d:\Ashok\_AI\_COMPUTER_VISION\____RESEARCH\___MOTION_T_LIGHTNING\main_all_keypoints\blazepose"
    
    # Subjects to analyze
    subjects = ['S1', 'S2', 'S3']
    
    # Activities to check
    activities = ['acting1', 'acting2', 'acting3', 'freestyle1', 'freestyle2', 
                  'freestyle3', 'rom1', 'rom2', 'rom3', 'walking1', 'walking2', 'walking3']
    
    print("="*100)
    print("SIZE COMPARISON ANALYSIS: CSV Ground Truth vs Blazepose NPY Arrays")
    print("="*100)
    
    # Track mismatches
    mismatches = []
    matches = []
    
    for subject in subjects:
        print(f"\n{'='*100}")
        print(f"SUBJECT: {subject}")
        print(f"{'='*100}")
        
        for activity in activities:
            result = analyze_subject_activity(subject, activity, positions_base, blazepose_base)
            
            if result is None:
                continue
            
            csv_rows = result['csv_rows']
            npy_files = result['npy_files']
            
            print(f"\n--- {activity} ---")
            print(f"CSV Rows (Ground Truth): {csv_rows}")
            
            # Check if all cameras have the same size
            all_match = True
            npy_sizes = []
            
            for cam, size in npy_files.items():
                print(f"  {cam}: {size}", end="")
                if isinstance(size, tuple) and len(size) > 0:
                    npy_sizes.append(size[0])
                    # Check if first dimension matches CSV rows
                    if size[0] == csv_rows:
                        print(" ✓")
                    else:
                        print(f" ✗ (MISMATCH: CSV={csv_rows}, NPY={size[0]})")
                        all_match = False
                        mismatches.append({
                            'subject': subject,
                            'activity': activity,
                            'camera': cam,
                            'csv_rows': csv_rows,
                            'npy_size': size[0]
                        })
                else:
                    print()
                    all_match = False
            
            # Summary for this activity
            if all_match and npy_sizes:
                print(f"  ✓ ALL MATCH: CSV={csv_rows}, All cameras={npy_sizes[0]} frames")
                matches.append(f"{subject}/{activity}")
            elif npy_sizes:
                unique_sizes = set(npy_sizes)
                if len(unique_sizes) == 1 and csv_rows == list(unique_sizes)[0]:
                    print(f"  ✓ ALL MATCH: CSV={csv_rows}, All cameras={list(unique_sizes)[0]} frames")
                    matches.append(f"{subject}/{activity}")
                else:
                    print(f"  ✗ MISMATCH DETECTED")
    
    # Final summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total matched activities: {len(matches)}")
    print(f"Total mismatches found: {len(mismatches)}")
    
    if mismatches:
        print(f"\n{'='*100}")
        print("DETAILED MISMATCH LIST:")
        print(f"{'='*100}")
        for mismatch in mismatches:
            print(f"{mismatch['subject']}/{mismatch['activity']}/{mismatch['camera']}: "
                  f"CSV={mismatch['csv_rows']} vs NPY={mismatch['npy_size']} "
                  f"(Difference: {mismatch['npy_size'] - mismatch['csv_rows']})")
    else:
        print("\n✓ ALL FILES MATCH PERFECTLY!")

if __name__ == "__main__":
    main()
