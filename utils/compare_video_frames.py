import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def count_video_frames(video_path):
    """Count total frames in a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Error: Cannot open video"
        
        # Try to get frame count from video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Verify by actually counting (more reliable)
        actual_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            actual_count += 1
        
        cap.release()
        return actual_count
    except Exception as e:
        return f"Error: {e}"

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
        return data.shape[0]  # Return first dimension (frames)
    except Exception as e:
        return f"Error: {e}"

def analyze_activity(subject, activity, base_dir):
    """Analyze video frames, NPY, and CSV for a specific activity."""
    
    # Paths
    video_base = os.path.join(base_dir, "totalcapture_dataset", "Videos", subject.lower(), activity)
    csv_path = os.path.join(base_dir, "totalcapture_dataset", "positions", subject.upper(), activity, "gt_skel_gbl_pos.csv")
    npy_base = os.path.join(base_dir, "main_all_keypoints", "blazepose", subject.upper(), activity)
    
    # Get CSV count
    csv_count = count_csv_rows(csv_path)
    
    print(f"\n{'='*80}")
    print(f"SUBJECT: {subject.upper()} | ACTIVITY: {activity}")
    print(f"{'='*80}")
    print(f"\nCSV Ground Truth: {csv_count} rows")
    print(f"\n{'Camera':<10} {'Video Frames':<15} {'NPY Frames':<15} {'Match Video':<15} {'Match CSV':<15}")
    print(f"{'-'*80}")
    
    results = []
    
    for cam in range(1, 9):
        video_file = f"TC_{subject.upper()}_{activity}_cam{cam}.mp4"
        video_path = os.path.join(video_base, video_file)
        
        npy_file = f"blaze_{subject.upper()}_{activity}_cam{cam}.npy"
        npy_path = os.path.join(npy_base, npy_file)
        
        # Get counts
        video_frames = "N/A"
        if os.path.exists(video_path):
            print(f"Counting frames for cam{cam}...", end=" ", flush=True)
            video_frames = count_video_frames(video_path)
            print(f"Done: {video_frames}")
        else:
            print(f"Video not found: {video_path}")
        
        npy_frames = "N/A"
        if os.path.exists(npy_path):
            npy_frames = get_npy_size(npy_path)
        
        # Check matches
        match_video = "✓" if isinstance(video_frames, int) and isinstance(npy_frames, int) and video_frames == npy_frames else "✗"
        match_csv = "✓" if isinstance(csv_count, int) and isinstance(npy_frames, int) and csv_count == npy_frames else "✗"
        
        results.append({
            'camera': f"cam{cam}",
            'video_frames': video_frames,
            'npy_frames': npy_frames,
            'csv_count': csv_count,
            'match_video': match_video,
            'match_csv': match_csv
        })
        
        print(f"cam{cam:<9} {str(video_frames):<15} {str(npy_frames):<15} {match_video:<15} {match_csv:<15}")
    
    return results

def main():
    base_dir = r"d:\Ashok\_AI\_COMPUTER_VISION\____RESEARCH\___MOTION_T_LIGHTNING"
    
    # Activities to check
    activities_to_check = [
        ("S1", "acting1"),
        ("S1", "acting3"),
        ("S2", "acting1")
    ]
    
    all_results = {}
    
    print("="*80)
    print("VIDEO FRAME COUNT COMPARISON ANALYSIS")
    print("="*80)
    
    for subject, activity in activities_to_check:
        results = analyze_activity(subject, activity, base_dir)
        all_results[f"{subject}/{activity}"] = results
    
    # Write detailed report
    output_file = os.path.join(base_dir, "video_frame_comparison_report.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VIDEO FRAME COUNT COMPARISON REPORT\n")
        f.write("Comparing: Actual Video Frames | Blazepose NPY | CSV Ground Truth\n")
        f.write("="*80 + "\n\n")
        
        for activity_key, results in all_results.items():
            subject, activity = activity_key.split('/')
            
            f.write(f"\n{'='*80}\n")
            f.write(f"{subject} / {activity}\n")
            f.write(f"{'='*80}\n\n")
            
            csv_count = results[0]['csv_count']
            f.write(f"CSV Ground Truth Rows: {csv_count}\n\n")
            
            f.write(f"{'Camera':<10} {'Video':<12} {'NPY':<12} {'CSV':<12} {'Video=NPY':<12} {'CSV=NPY':<12} {'Status':<20}\n")
            f.write(f"{'-'*80}\n")
            
            all_match = True
            video_match_count = 0
            csv_match_count = 0
            
            for result in results:
                cam = result['camera']
                vid_frames = result['video_frames']
                npy_frames = result['npy_frames']
                csv = result['csv_count']
                match_vid = result['match_video']
                match_csv = result['match_csv']
                
                # Determine status
                if isinstance(vid_frames, int) and isinstance(npy_frames, int) and isinstance(csv, int):
                    if vid_frames == npy_frames == csv:
                        status = "PERFECT MATCH"
                        video_match_count += 1
                        csv_match_count += 1
                    elif vid_frames == npy_frames and csv != npy_frames:
                        status = "NPY matches Video, not CSV"
                        video_match_count += 1
                        all_match = False
                    elif csv == npy_frames and vid_frames != npy_frames:
                        status = "NPY matches CSV, not Video"
                        csv_match_count += 1
                        all_match = False
                    else:
                        status = "ALL DIFFERENT"
                        all_match = False
                else:
                    status = "DATA MISSING"
                    all_match = False
                
                f.write(f"{cam:<10} {str(vid_frames):<12} {str(npy_frames):<12} {str(csv):<12} {match_vid:<12} {match_csv:<12} {status:<20}\n")
            
            f.write(f"\n")
            f.write(f"Summary:\n")
            f.write(f"  - Cameras matching Video: {video_match_count}/8\n")
            f.write(f"  - Cameras matching CSV: {csv_match_count}/8\n")
            if all_match and video_match_count == 8:
                f.write(f"  - Overall: ✓ ALL PERFECT (Video = NPY = CSV)\n")
            else:
                f.write(f"  - Overall: ✗ MISMATCHES DETECTED\n")
            
            # Analysis
            if isinstance(vid_frames, int) and isinstance(npy_frames, int) and isinstance(csv, int):
                f.write(f"\nAnalysis:\n")
                if video_match_count == 8 and csv_match_count == 0:
                    f.write(f"  → Blazepose extraction is correct (matches video)\n")
                    f.write(f"  → CSV ground truth has wrong frame count\n")
                elif video_match_count == 0 and csv_match_count == 8:
                    f.write(f"  → CSV ground truth is correct\n")
                    f.write(f"  → Blazepose extraction needs to be redone\n")
                elif video_match_count == 8 and csv_match_count == 8:
                    f.write(f"  → Everything is perfectly aligned! ✓\n")
                else:
                    f.write(f"  → Mixed results - needs investigation\n")
            
            f.write(f"\n")
    
    print(f"\n{'='*80}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
