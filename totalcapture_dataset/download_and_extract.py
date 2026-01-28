#!/usr/bin/env python3
import os
import tarfile
import requests
from requests.auth import HTTPBasicAuth
import time
from pathlib import Path

# Authentication credentials
USERNAME = "cvssp3d"
PASSWORD = "Roxanne"

# Base paths
base_dir = r"d:\Ashok\_AI\_COMPUTER_VISION\____RESEARCH\___MOTION_T_LIGHTNING\totalcapture_dataset"
videos_dir = os.path.join(base_dir, "Videos")

# All available activities per subject
ACTIVITIES = {
    "s1": ["acting1", "acting2", "acting3", "freestyle1", "freestyle2", "freestyle3",
           "rom1", "rom2", "rom3", "walking1", "walking2", "walking3"],
    "s2": ["acting1", "acting2", "acting3", "freestyle1", "freestyle2", "freestyle3",
           "rom1", "rom2", "rom3", "walking1", "walking2", "walking3"],
    "s3": ["acting1", "acting2", "acting3", "freestyle1", "freestyle2", "freestyle3",
           "rom1", "rom2", "rom3", "walking1", "walking2", "walking3"],
    "s4": ["acting3", "freestyle1", "freestyle3", "rom3", "walking2"],
    "s5": ["acting3", "freestyle1", "freestyle3", "rom3", "walking2"]
}

def check_existing_videos(subject, activity):
    """Check if activity directory already has all 8 videos"""
    activity_dir = os.path.join(videos_dir, subject, activity)
    if not os.path.exists(activity_dir):
        return False
    
    # Count mp4 files
    mp4_files = list(Path(activity_dir).glob("*.mp4"))
    return len(mp4_files) == 8


def download_file(url, filepath, chunk_size=8192):
    """Download file with progress indicator and authentication"""
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        total_mb = total_size / (1024 * 1024)
        
        # Download with progress
        downloaded = 0
        start_time = time.time()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            speed_mbps = (downloaded / (1024 * 1024)) / elapsed
                            print(f"\r  Progress: {percent:.1f}% | {downloaded/(1024*1024):.1f}/{total_mb:.1f} MB | Speed: {speed_mbps:.2f} MB/s", end='', flush=True)
        
        print()  # New line after progress
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n  ✗ Download failed: {e}")
        return False

def extract_tar_gz(filepath, extract_to):
    """Extract tar.gz file"""
    try:
        print(f"  Extracting...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def download_and_extract_activity(subject, activity):
    """Download and extract videos for a specific activity"""
    # Check if already exists
    if check_existing_videos(subject, activity):
        return "skipped"
    
    # Create subject directory
    subject_dir = os.path.join(videos_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Build URL
    filename = f"{subject}_{activity}.tar.gz"
    url = f"https://cvssp.org/data/totalcapture/data/dataset/video/{subject}/{filename}"
    filepath = os.path.join(subject_dir, filename)
    
    print(f"\n  → {activity}")
    
    # Download
    if not download_file(url, filepath):
        return "failed"
    
    # Extract
    if not extract_tar_gz(filepath, subject_dir):
        return "failed"
    
    # Clean up tar.gz
    try:
        os.remove(filepath)
        print(f"  ✓ Completed and cleaned up")
    except Exception as e:
        print(f"  ⚠ Cleanup failed: {e}")
    
    return "success"

def main():
    print("="*80)
    print("TotalCapture Video Downloader & Extractor")
    print("="*80)
    print(f"Output directory: {videos_dir}")
    print(f"Authentication: {USERNAME}")
    print("="*80)
    
    # Create videos directory
    os.makedirs(videos_dir, exist_ok=True)
    
    # Count total activities
    total_activities = sum(len(activities) for activities in ACTIVITIES.values())
    stats = {"success": 0, "failed": 0, "skipped": 0}
    current = 0
    
    # Process each subject
    for subject in sorted(ACTIVITIES.keys()):
        activities = ACTIVITIES[subject]
        print(f"\n{'='*80}")
        print(f"Subject: {subject.upper()} ({len(activities)} activities)")
        print(f"{'='*80}")
        
        for activity in activities:
            current += 1
            print(f"\n[{current}/{total_activities}] {subject}/{activity}", end="")
            
            # Check if exists first
            if check_existing_videos(subject, activity):
                print(" - Already exists (8 videos)")
                stats["skipped"] += 1
                continue
            
            result = download_and_extract_activity(subject, activity)
            stats[result] += 1
            
            # Small delay between downloads
            if result == "success":
                time.sleep(0.5)
    
    # Final summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Total activities:     {total_activities}")
    print(f"✓ Successful:         {stats['success']}")
    print(f"⊘ Skipped (existing): {stats['skipped']}")
    print(f"✗ Failed:             {stats['failed']}")
    print(f"{'='*80}")
    
    if stats['failed'] == 0:
        print("✓ All downloads completed successfully!")
    else:
        print("⚠ Some downloads failed. Check the log above for details.")
    
if __name__ == "__main__":
    main()
