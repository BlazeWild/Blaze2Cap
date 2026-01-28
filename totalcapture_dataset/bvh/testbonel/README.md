# BVH Motion Reconstruction Notes (testbonel)

This document explains **exactly** how the pose is reconstructed in the viewer, including formulas and numeric examples taken from the BVH/motion data.

## Files
- Viewer script: [eulerdata_length.py](eulerdata_length.py)
- Motion data: [motion.txt](motion.txt)
- Source BVH: [../acting1_BlenderZXY_YmZ.bvh](../acting1_BlenderZXY_YmZ.bvh)
- Hip trajectory plot: [plot_hip_trajectory.py](plot_hip_trajectory.py)

## What is extracted from the BVH
- **Hierarchy and parent links** are parsed from the BVH HIERARCHY section.
- **Bone offsets** (the OFFSET lines) are used as fixed bone lengths and directions.
- **Per‑joint channel order** is read from the CHANNELS line and stored for each joint.

Example from [../acting1_BlenderZXY_YmZ.bvh](../acting1_BlenderZXY_YmZ.bvh):

- `Spine` OFFSET: `(0.000000, 1.817770, -2.726650)`
- `RightArm` OFFSET: `(5.705120, 0.000000, 0.000000)`
- `RightUpLeg` OFFSET: `(3.407760, 0.000000, 0.995530)`

Channels for each joint are read verbatim. For example:

```
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
```

This means the **rotation order is Z → X → Y** for that joint, and that same order is used in the rotation matrix multiplication.

## Bone lengths
- Bone lengths are **not computed from motion**; they come directly from BVH OFFSETS.
- The OFFSET vector is applied as the local translation from parent to child.

Example:

- If `RightArm` offset is `(5.705120, 0, 0)`, then the local translation for that joint is:

$$
T_{RightArm} = \begin{bmatrix}
1 & 0 & 0 & 5.705120 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

## How joint orientations are computed
- Euler rotations are used **directly** from the motion channels.
- The rotation order is **taken from each joint’s CHANNELS**, not assumed.
- No 6D rotation conversion is used.
- The local rotation matrix is built by multiplying axis rotations in the exact channel order.

If a joint’s rotation channels are `Zrotation, Xrotation, Yrotation`, the local rotation is:

$$
R = R_z(\theta_z) \cdot R_x(\theta_x) \cdot R_y(\theta_y)
$$

Where:

$$
R_x(\theta) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix},
\quad
R_y(\theta) = \begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix},
\quad
R_z(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### Complete numeric walkthrough: Frame 0

The first frame from [motion.txt](motion.txt) starts with:
```
0.963698 33.280895 5.105301 -39.879443 105.325007 43.561137 0.000000 1.817772 -2.726647 -0.295321 -9.474851 -3.079455 ...
```

This is read as 6 channels per joint in order: `Xposition Yposition Zposition Zrotation Xrotation Yrotation`

#### Joint breakdown (first 3 joints):

**Hips (root):**
- Channels 0-5: `0.963698 33.280895 5.105301 -39.879443 105.325007 43.561137`
- Position: `(0.963698, 33.280895, 5.105301)`
- Rotation: `Z=-39.879443°, X=105.325007°, Y=43.561137°`

**Spine:**
- Channels 6-11: `0.000000 1.817772 -2.726647 -0.295321 -9.474851 -3.079455`
- Position: `(0.000000, 1.817772, -2.726647)` ← **ignored**, use BVH offset instead
- Rotation: `Z=-0.295321°, X=-9.474851°, Y=-3.079455°`

**Spine1:**
- Channels 12-17: `0.000000 0.631300 -3.580301 -0.267627 -3.257609 -0.391612`
- Position: `(0.000000, 0.631300, -3.580301)` ← **ignored**, use BVH offset instead
- Rotation: `Z=-0.267627°, X=-3.257609°, Y=-0.391612°`

#### Step 1: Process Hips (root joint)

**Raw position from motion:**
```
p_raw = (0.963698, 33.280895, 5.105301)
```

**Apply axis mapping** (`GLOBAL_POS_MAP = {'X': 'X', 'Y': 'Z', 'Z': 'Y'}`):
```
p_mapped = (p_raw[X], p_raw[Z], p_raw[Y])
         = (0.963698, 5.105301, 33.280895)
```

**Apply sign flips** (`GLOBAL_POS_FLIP = {'X': -1, 'Y': 1, 'Z': -1}`):
```
p_final = (-0.963698, 5.105301, -33.280895)
```

**Translation matrix:**
$$
T_{hip} = \begin{bmatrix}
1 & 0 & 0 & -0.963698 \\
0 & 1 & 0 & 5.105301 \\
0 & 0 & 1 & -33.280895 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Rotation from motion** (Z → X → Y order):
```
θ_z = -39.879443° = -0.6961 rad
θ_x = 105.325007° = 1.8382 rad
θ_y = 43.561137° = 0.7602 rad
```

$$
R_{motion} = R_z(-39.879443°) \cdot R_x(105.325007°) \cdot R_y(43.561137°)
$$

**Correction rotation** (`HIP_CORRECTION_X=90°, HIP_CORRECTION_Y=180°, HIP_CORRECTION_Z=0°`):
$$
R_{corr} = R_z(0°) \cdot R_x(90°) \cdot R_y(180°)
$$

Breaking this down:
$$
R_x(90°) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
R_y(180°) = \begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

$$
R_{corr} = R_z(0) \cdot R_x(90°) \cdot R_y(180°) = \begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Final hip transform:**
$$
M_{Hips} = T_{hip} \cdot R_{corr} \cdot R_{motion}
$$

**Hip position** (extract from $M_{Hips}$):
```
Position_Hips = (-0.963698, 5.105301, -33.280895)
```

#### Step 2: Process Spine (child of Hips)

**Offset from BVH** (from HIERARCHY section):
```
OFFSET_Spine = (0.000000, 1.817770, -2.726650)
```

**Translation matrix:**
$$
T_{Spine} = \begin{bmatrix}
1 & 0 & 0 & 0.000000 \\
0 & 1 & 0 & 1.817770 \\
0 & 0 & 1 & -2.726650 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Rotation from motion** (channels 6-11):
```
θ_z = -0.295321° = -0.00515 rad
θ_x = -9.474851° = -0.1654 rad
θ_y = -3.079455° = -0.0537 rad
```

$$
R_{Spine} = R_z(-0.295321°) \cdot R_x(-9.474851°) \cdot R_y(-3.079455°)
$$

**Local transform:**
$$
M_{Spine}^{local} = T_{Spine} \cdot R_{Spine}
$$

**Global transform:**
$$
M_{Spine}^{global} = M_{Hips} \cdot M_{Spine}^{local}
$$

**Spine position** (extract from $M_{Spine}^{global}$):
```
Position_Spine = M_{Spine}^{global} \times (0, 0, 0, 1)^T
```

#### Step 3: Process Spine1 (child of Spine)

**Offset from BVH:**
```
OFFSET_Spine1 = (0.000000, 0.631300, -3.580300)
```

**Translation matrix:**
$$
T_{Spine1} = \begin{bmatrix}
1 & 0 & 0 & 0.000000 \\
0 & 1 & 0 & 0.631300 \\
0 & 0 & 1 & -3.580300 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

**Rotation from motion** (channels 12-17):
```
θ_z = -0.267627° = -0.00467 rad
θ_x = -3.257609° = -0.05686 rad
θ_y = -0.391612° = -0.00683 rad
```

$$
R_{Spine1} = R_z(-0.267627°) \cdot R_x(-3.257609°) \cdot R_y(-0.391612°)
$$

**Local transform:**
$$
M_{Spine1}^{local} = T_{Spine1} \cdot R_{Spine1}
$$

**Global transform:**
$$
M_{Spine1}^{global} = M_{Spine}^{global} \cdot M_{Spine1}^{local}
$$

**Spine1 position:**
```
Position_Spine1 = M_{Spine1}^{global} \times (0, 0, 0, 1)^T
```

### Key observations from this example:

1. **Position channels are ignored for all child joints** — only the BVH OFFSET is used
2. **Rotation channels are read for all joints** and applied in the exact channel order
3. **The root hip undergoes:**
   - Axis remapping: Y↔Z swap
   - Sign flips: X and Z negated
   - Base correction rotation: 90° around X, then 180° around Y
4. **All child joints simply use:** OFFSET translation + local rotation
5. **Hierarchy is built by matrix multiplication:** $M_{child} = M_{parent} \cdot T_{offset} \cdot R_{local}$

## How positions are computed
### Root joint (Hips)
- Uses the **global translation** from the root position channels.
- Uses the **global orientation** from the root rotation channels.
- A base correction rotation can be applied if needed for coordinate alignment.

For the root (Hips), the local transform is:

$$
M_{root} = T(\mathbf{p}) \cdot R_{corr} \cdot R_{root}
$$

Where $\mathbf{p} = (x, y, z)$ from motion channels (or mapped/flipped version). Example from the first frame:

- Position: `(0.963698, 33.280895, 5.105301)`
- If we apply axis mapping and flips, we use:

$$
(x', y', z') = (x, z, y) \odot (s_x, s_y, s_z)
$$

Current mapping/flip settings are defined at the top of [eulerdata_length.py](eulerdata_length.py).

### Child joints
- **Position channels are ignored** for all non‑root joints.
- Local transform is:
  1) Translate by the joint’s OFFSET (bone length)
  2) Apply the joint’s local rotation
- Global transform is built by multiplying the parent’s global transform by the joint’s local transform.

Mathematically for a child joint $j$ with parent $p$:

$$
M_j = M_p \cdot T(\text{OFFSET}_j) \cdot R_j
$$

Example for `RightArm` (first frame):

- OFFSET: `(5.705120, 0, 0)`
- Rotations from motion channels (order from BVH): `Z`, `X`, `Y`

So:

$$
M_{RightArm} = M_{RightShoulder} \cdot T(5.705120, 0, 0) \cdot R_{RightArm}
$$

## End Sites
- End Sites are handled by applying their OFFSET from the parent’s global transform.

If an End Site offset is $(dx, dy, dz)$:

$$
M_{end} = M_{parent} \cdot T(dx, dy, dz)
$$

## Orientation and sign conventions used
- Axis conventions and sign fixes are controlled at the top of [eulerdata_length.py](eulerdata_length.py):
  - `APPLY_HIP_CORRECTION` and `HIP_CORRECTION_X/Y/Z`
  - `GLOBAL_POS_MAP` for axis swapping of root position
  - `GLOBAL_POS_FLIP` for sign flips of root position

These are used to align the motion to the ground plane and correct facing direction **without touching child joint rotations**.

Current values (as set in code):

- `APPLY_HIP_CORRECTION = True`
- `HIP_CORRECTION_X = 90`
- `HIP_CORRECTION_Y = 180`
- `HIP_CORRECTION_Z = 0`
- `GLOBAL_POS_MAP = {'X': 'X', 'Y': 'Z', 'Z': 'Y'}`
- `GLOBAL_POS_FLIP = {'X': -1, 'Y': 1, 'Z': -1}`

## Where rotations are applied in the pipeline
- For each joint:
  - Build local rotation from channel order.
  - For root: apply global translation, then optional correction rotation, then local rotation.
  - For children: apply OFFSET translation, then local rotation.

## Notes on debug modes
- The root can be pinned at the origin by replacing the root translation with $(0,0,0)$. This is only a visualization/debug step.

## Summary
- Bone lengths come from BVH OFFSETS.
- Child joints use **rotation only** (their position channels are ignored).
- Root uses **global translation + rotation**.
- Rotation order is taken from the BVH channel list, not assumed.
- No 6D rotation is used.
