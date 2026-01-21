#!/usr/bin/env python3
"""
Visualize 3D keypoints from ground truth and predicted data.
Shows GT in blue and predictions in red with skeleton connections.
Creates an interactive HTML file that can be opened in a browser.
"""

import csv
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

# Define skeleton connections
# Based on body structure: arms, torso, legs
CONNECTIONS = [
    (0, 1),   # LeftArm to RightArm (shoulders)
    (0, 2),   # LeftArm to LeftForeArm
    (1, 3),   # RightArm to RightForeArm
    (2, 4),   # LeftForeArm to LeftHand
    (3, 5),   # RightForeArm to RightHand
    (0, 6),   # LeftArm to LeftUpLeg (torso left)
    (1, 7),   # RightArm to RightUpLeg (torso right)
    (6, 7),   # LeftUpLeg to RightUpLeg (hips)
    (6, 8),   # LeftUpLeg to LeftLeg
    (7, 9),   # RightUpLeg to RightLeg
    (8, 10),  # LeftLeg to LeftFoot
    (9, 11)   # RightLeg to RightFoot
]

# Keypoint names (for reference)
KEYPOINT_NAMES = [
    'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
    'LeftHand', 'RightHand', 'LeftUpLeg', 'RightUpLeg',
    'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'
]

def parse_coord_string(coord_str):
    """Parse coordinate string 'x, y, z' to numpy array."""
    if not coord_str or coord_str.strip() == "":
        return None
    coords = [float(x.strip()) for x in coord_str.split(',')]
    return np.array(coords)

def load_data(input_csv, gt_csv):
    """
    Load input (predictions) and ground truth data.
    Returns lists of frames with keypoint coordinates.
    """
    input_data = []
    gt_data = []
    
    print(f"Loading data from:")
    print(f"  Input: {input_csv}")
    print(f"  GT: {gt_csv}")
    
    # Load input data
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_info = {
                'camera': int(row['camera']),
                'video_frame_number': int(row['video_frame_number']),
                'keypoints': []
            }
            for kp_name in KEYPOINT_NAMES:
                coord = parse_coord_string(row[kp_name])
                frame_info['keypoints'].append(coord)
            input_data.append(frame_info)
    
    # Load ground truth data
    with open(gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_info = {
                'camera': int(row['camera']),
                'video_frame_number': int(row['video_frame_number']),
                'keypoints': []
            }
            for kp_name in KEYPOINT_NAMES:
                coord = parse_coord_string(row[kp_name])
                frame_info['keypoints'].append(coord)
            gt_data.append(frame_info)
    
    print(f"\nLoaded {len(input_data)} frames successfully")
    return input_data, gt_data

def create_skeleton_traces(keypoints, color, name, showlegend=True):
    """
    Create plotly traces for skeleton (points and lines).
    
    Args:
        keypoints: List of 12 (x,y,z) coordinates
        color: Color for points and lines
        name: Name for legend
        showlegend: Whether to show in legend
    
    Returns:
        List of traces (scatter3d for points and lines)
    """
    traces = []
    
    # Extract valid coordinates
    valid_points = []
    for kp in keypoints:
        if kp is not None:
            valid_points.append(kp)
    
    if not valid_points:
        return traces
    
    points = np.array(valid_points)
    
    # Add keypoints as scatter
    traces.append(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=5, color=color),
        name=name,
        showlegend=showlegend,
        hovertext=[f"{KEYPOINT_NAMES[i]}" for i in range(len(valid_points))],
        hoverinfo='text'
    ))
    
    # Add connections as lines
    for conn in CONNECTIONS:
        i, j = conn
        if keypoints[i] is not None and keypoints[j] is not None:
            line_points = np.array([keypoints[i], keypoints[j]])
            traces.append(go.Scatter3d(
                x=line_points[:, 0],
                y=line_points[:, 1],
                z=line_points[:, 2],
                mode='lines',
                line=dict(width=3, color=color),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    return traces

def visualize_keypoints(input_csv, gt_csv, output_html='visualization.html'):
    """
    Create interactive HTML visualization with frame slider.
    """
    input_data, gt_data = load_data(input_csv, gt_csv)
    
    print("Creating interactive HTML visualization...")
    
    # Create frames for animation
    frames = []
    
    for frame_idx in range(len(input_data)):
        input_frame = input_data[frame_idx]
        gt_frame = gt_data[frame_idx]
        
        # Create traces for this frame
        frame_traces = []
        
        # Ground truth (blue)
        frame_traces.extend(create_skeleton_traces(
            gt_frame['keypoints'], 
            'blue', 
            'Ground Truth',
            showlegend=(frame_idx == 0)
        ))
        
        # Prediction (red)
        frame_traces.extend(create_skeleton_traces(
            input_frame['keypoints'], 
            'red', 
            'Prediction',
            showlegend=(frame_idx == 0)
        ))
        
        frames.append(go.Frame(
            data=frame_traces,
            name=str(frame_idx),
            layout=go.Layout(
                title_text=f'Frame {frame_idx} - Camera {input_frame["camera"]} - Video Frame {input_frame["video_frame_number"]}'
            )
        ))
    
    # Create initial frame
    initial_traces = []
    initial_traces.extend(create_skeleton_traces(
        gt_data[0]['keypoints'], 
        'blue', 
        'Ground Truth'
    ))
    initial_traces.extend(create_skeleton_traces(
        input_data[0]['keypoints'], 
        'red', 
        'Prediction'
    ))
    
    # Create figure
    fig = go.Figure(
        data=initial_traces,
        frames=frames,
        layout=go.Layout(
            title=f'Frame 0 - Camera {input_data[0]["camera"]} - Video Frame {input_data[0]["video_frame_number"]}',
            scene=dict(
                xaxis_title='X (meters)',
                yaxis_title='Y (meters)',
                zaxis_title='Z (meters)',
                aspectmode='data'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': str(k),
                        'method': 'animate'
                    }
                    for k, f in enumerate(frames)
                ],
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': 'Frame: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0
            }]
        )
    )
    
    # Save to HTML
    fig.write_html(output_html)
    print(f"\nVisualization saved to: {output_html}")
    print(f"Open this file in a web browser to view the interactive 3D visualization")
    print(f"Total frames: {len(input_data)}")
    print(f"\nFeatures:")
    print(f"  - Use the slider at the bottom to navigate frames")
    print(f"  - Click 'Play' to animate through frames")
    print(f"  - Drag to rotate, scroll to zoom")
    print(f"  - Blue = Ground Truth, Red = Prediction")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize 3D keypoints with ground truth and predictions (HTML output)'
    )
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input CSV file (predictions)')
    parser.add_argument('--gt', type=str, required=True,
                       help='Path to ground truth CSV file')
    parser.add_argument('--output', type=str, default='visualization.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    visualize_keypoints(args.input, args.gt, args.output)
