import os
import json
import numpy as np

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes the script is run from totalcapture_dataset folder
BVH_FILE = os.path.join(SCRIPT_DIR, "bvh", "acting1_BlenderZXY_YmZ.bvh")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "bvh_bone_structure.json")
INCHES_TO_METERS = 0.0254

def parse_bvh_structure(file_path):
    structure = {}
    joint_stack = []
    
    print(f"Parsing BVH file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        
    for line in lines:
        parts = line.split()
        if not parts: continue
        
        token = parts[0]
        
        if token == 'ROOT':
            joint_name = parts[1]
            joint_stack.append(joint_name)
            structure[joint_name] = {'parent': None, 'type': 'ROOT'}
            
        elif token == 'JOINT':
            joint_name = parts[1]
            parent_name = joint_stack[-1] if joint_stack else None
            joint_stack.append(joint_name)
            structure[joint_name] = {'parent': parent_name, 'type': 'JOINT'}
            
        elif token == 'End': # End Site
            if len(parts) > 1 and parts[1] == 'Site':
                parent_name = joint_stack[-1] if joint_stack else None
                joint_name = parent_name + '_End'
                joint_stack.append(joint_name)
                structure[joint_name] = {'parent': parent_name, 'type': 'End Site'}
            
        elif token == 'OFFSET':
            if not joint_stack:
                continue
            current_joint = joint_stack[-1]
            
            # Read Values (assuming inches in this dataset)
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            # Convert to meters
            offset_meters = [x * INCHES_TO_METERS, y * INCHES_TO_METERS, z * INCHES_TO_METERS]
            length_meters = (offset_meters[0]**2 + offset_meters[1]**2 + offset_meters[2]**2)**0.5
            
            structure[current_joint]['offset_raw'] = [x, y, z]
            structure[current_joint]['offset_meters'] = offset_meters
            structure[current_joint]['length_meters'] = length_meters
            
        elif token == '}':
            if joint_stack:
                joint_stack.pop()
                
    return structure

def main():
    if not os.path.exists(BVH_FILE):
        print(f"Error: BVH file not found at {BVH_FILE}")
        return

    data = parse_bvh_structure(BVH_FILE)
    
    # Print to terminal
    print("\n" + "="*90)
    print(f"{'JOINT NAME':<20} | {'PARENT':<15} | {'OFFSET (meters)':<30} | {'LENGTH (m)':<10}")
    print("-" * 90)
    
    # Sort for consistent display (or use insertion order if Python 3.7+)
    for joint_name, info in data.items():
        parent_str = str(info['parent']) if info['parent'] else "None"
        offset = info.get('offset_meters', [0,0,0])
        offset_str = f"[{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]"
        length = info.get('length_meters', 0.0)
        
        print(f"{joint_name:<20} | {parent_str:<15} | {offset_str:<30} | {length:.4f}")
        
    print("="*90 + "\n")
    
    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Structure analysis saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
