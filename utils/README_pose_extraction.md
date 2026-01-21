# MediaPipe Pose Landmark Extraction

This script extracts 3D world landmarks from videos using MediaPipe BlazePose and saves them to CSV files.

## Features

- Extracts 33 3D world landmarks (x, y, z coordinates in meters)
- Tracks video frame numbers and MediaPipe prediction frame numbers
- Handles skipped frames (when MediaPipe fails to detect pose)
- Supports single video or batch processing

## Installation

```bash
pip install -r requirements_pose.txt
```

## Usage

### Single Video Processing

```bash
python extract_pose_landmarks.py --video /path/to/video.mp4 --output landmarks.csv
```

### Batch Processing (Multiple Videos)

```bash
python extract_pose_landmarks.py --video_dir /path/to/videos/ --output_dir /path/to/output/
```

## Output CSV Format

The CSV file contains:
- `video_frame_number`: Sequential frame number from the video (0, 1, 2, ...)
- `mediapipe_predicted_frame`: Count of frames where MediaPipe detected pose (-1 for skipped frames)
- `landmark_0_x`, `landmark_0_y`, `landmark_0_z`: 3D coordinates for landmark 0 (nose)
- ... (continues for all 33 landmarks)

### MediaPipe Pose Landmarks (33 points):

0. Nose
1. Left Eye Inner
2. Left Eye
3. Left Eye Outer
4. Right Eye Inner
5. Right Eye
6. Right Eye Outer
7. Left Ear
8. Right Ear
9. Mouth Left
10. Mouth Right
11. Left Shoulder
12. Right Shoulder
13. Left Elbow
14. Right Elbow
15. Left Wrist
16. Right Wrist
17. Left Pinky
18. Right Pinky
19. Left Index
20. Right Index
21. Left Thumb
22. Right Thumb
23. Left Hip
24. Right Hip
25. Left Knee
26. Right Knee
27. Left Ankle
28. Right Ankle
29. Left Heel
30. Right Heel
31. Left Foot Index
32. Right Foot Index

## Notes

- Skipped frames are marked with `mediapipe_predicted_frame = -1` and have `None` values for all landmarks
- World landmarks are in meters relative to the hip center
- For better accuracy, use `model_complexity=2` (default in script)
