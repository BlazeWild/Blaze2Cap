import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define the bone hierarchy (Child: [OffsetX, OffsetY, OffsetZ, Parent])
# This maps directly to your provided BVH text.
hierarchy = {
    'Hips': [0, 0, 0, None],
    'Spine': [0.000000, 1.817770, -2.726650, 'Hips'],
    'Spine1': [0.000000, 0.631300, -3.580300, 'Spine'],
    'Spine2': [0.000000, 0.316860, -3.621700, 'Spine1'],
    'Spine3': [0.000000, 0.000000, -3.635530, 'Spine2'],
    'Neck': [0.000001, -1.363320, -9.543270, 'Spine3'],
    'Head': [0.000000, -0.868040, -4.922911, 'Neck'],
    'HeadEnd': [0.000001, 0.000000, -5.885021, 'Head'],
    'RightShoulder': [1.148680, -1.914470, -6.202890, 'Spine3'],
    'RightArm': [5.705120, 0.000000, 0.000000, 'RightShoulder'],
    'RightForeArm': [11.372740, 0.000000, 0.000002, 'RightArm'],
    'RightHand': [8.645779, 0.000000, 0.000000, 'RightForeArm'],
    'LeftShoulder': [-1.148679, -1.914470, -6.202890, 'Spine3'],
    'LeftArm': [-5.705120, 0.000000, 0.000000, 'LeftShoulder'],
    'LeftForeArm': [-11.372740, 0.000000, -0.000002, 'LeftArm'],
    'LeftHand': [-8.645779, 0.000000, 0.000000, 'LeftForeArm'],
    'RightUpLeg': [3.407760, 0.000000, 0.995530, 'Hips'],
    'RightLeg': [-0.000001, 0.000000, 14.916970, 'RightUpLeg'],
    'RightFoot': [-0.000001, 0.000000, 14.781019, 'RightLeg'],
    'LeftUpLeg': [-3.407760, 0.000000, 0.995530, 'Hips'],
    'LeftLeg': [-0.000001, 0.000000, 14.916970, 'LeftUpLeg'],
    'LeftFoot': [-0.000001, 0.000000, 14.781019, 'LeftLeg'],
}

def calculate_world_positions(hierarchy):
    world_pos = {}
    
    def get_pos(name):
        if name in world_pos:
            return world_pos[name]
        
        ox, oy, oz, parent = hierarchy[name]
        if parent is None:
            world_pos[name] = (ox, oy, oz)
        else:
            px, py, pz = get_pos(parent)
            world_pos[name] = (px + ox, py + oy, pz + oz)
        return world_pos[name]

    for joint in hierarchy:
        get_pos(joint)
    return world_pos

# Calculate the positions
positions = calculate_world_positions(hierarchy)

# --- Plotting ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for joint, info in hierarchy.items():
    parent_name = info[3]
    if parent_name:
        # Get start (parent) and end (child) coordinates
        start = positions[parent_name]
        end = positions[joint]
        
        # Plot the bone as a line
        ax.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]], 
                color='black', linewidth=2, marker='o', markersize=4)

# Total Capture data often has Z as the "vertical" or depth axis.
# Setting limits helps keep the aspect ratio sane.
ax.set_xlim([-30, 30])
ax.set_ylim([-30, 30])
ax.set_zlim([-10, 60])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Standard BVH view adjustment
ax.view_init(elev=-75, azim=-90)

plt.show()