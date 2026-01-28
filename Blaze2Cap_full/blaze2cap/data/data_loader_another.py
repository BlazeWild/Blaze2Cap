import os
import json
import numpy as np
import torch
from torch.utils import data

class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root: str, window_size: int, split: str = "train"):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.split = split
        
        # 1. Load JSON
        json_path = os.path.join(dataset_root, "dataset_map.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON map not found at {json_path}")
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        self.samples = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Vectorized Loader. Samples: {len(self.samples)} | Window: {window_size}")

        # ------------------------------------------------------------------
        # HIERARCHY DEFINITION (Indices 0-24 based on your 25-joint subset)
        # ------------------------------------------------------------------
        # Assumption: Your 25 joints are ordered: 
        # 0:Nose, 1-2:Ears, 3-4:Shoulders, 5-6:Elbows, 7-8:Wrists, 
        # 9-14:Hands(P,I,T), 15-16:Hips, 17-18:Knees, 19-20:Ankles, 21-24:Feet
        
        # We create index maps to calculate vectors instantly:
        # parent_map[i] = index of i's parent
        # child_map[i] = index of i's child
        
        # Initialize with self-reference (vector=0)
        self.parents = np.arange(25) 
        self.children = np.arange(25) 

        # -- Define Hierarchy (Manual Map) --
        # Head (0,1,2) -> Parent is Virtual Neck (Handle separately), Child None
        
        # Shoulders(3,4) -> Child: Elbows(5,6)
        self.children[3], self.children[4] = 5, 6
        
        # Elbows(5,6) -> Parent: Shoulders(3,4) | Child: Wrists(7,8)
        self.parents[5], self.parents[6] = 3, 4
        self.children[5], self.children[6] = 7, 8
        
        # Wrists(7,8) -> Parent: Elbows(5,6) | Child: Index Fingers (11,12)
        self.parents[7], self.parents[8] = 5, 6
        self.children[7], self.children[8] = 11, 12 # Point to Index finger as direction
        
        # Hand Leaves (9-14) -> Parent: Wrists(7,8)
        # L_Pinky(9), L_Index(11), L_Thumb(13) -> Parent L_Wrist(7)
        self.parents[[9, 11, 13]] = 7
        # R_Pinky(10), R_Index(12), R_Thumb(14) -> Parent R_Wrist(8)
        self.parents[[10, 12, 14]] = 8
        
        # Hips(15,16) -> Parent: Virtual Hip (Handle separately) | Child: Knees(17,18)
        self.children[15], self.children[16] = 17, 18
        
        # Knees(17,18) -> Parent: Hips(15,16) | Child: Ankles(19,20)
        self.parents[17], self.parents[18] = 15, 16
        self.children[17], self.children[18] = 19, 20
        
        # Ankles(19,20) -> Parent: Knees(17,18) | Child: Toes(23,24)
        self.parents[19], self.parents[20] = 17, 18
        self.children[19], self.children[20] = 23, 24
        
        # Feet Leaves (21-24) -> Parent: Ankles(19,20)
        # L_Heel(21), L_Toe(23) -> Parent L_Ankle(19)
        self.parents[[21, 23]] = 19
        # R_Heel(22), R_Toe(24) -> Parent R_Ankle(20)
        self.parents[[22, 24]] = 20

    def _process_vectorized(self, raw_data):
        """
        Input: raw_data (F, 25, 4)
        Output: X_windows (F, N, 300), M_masks (F, N)
        """
        N = self.window_size
        F, J, _ = raw_data.shape # F, 25, 4
        
        # 1. Structural Padding
        # Replicate first frame N-1 times
        padding = np.repeat(raw_data[0:1], N-1, axis=0)
        padding[:, :, 3] = 0 # Mask padded frames
        full_data = np.concatenate([padding, raw_data], axis=0) # (F_pad, 25, 4)
        F_pad = full_data.shape[0]

        # 2. Extract Components
        pos = full_data[:, :, :3] # (F_pad, 25, 3)
        flags = full_data[:, 0, 3] # (F_pad,)
        
        # 3. Calculate Virtual Anchors (Vectorized across F_pad)
        # Hips are indices 15, 16
        hip_center = (pos[:, 15] + pos[:, 16]) / 2.0 # (F_pad, 3)
        # Shoulders are indices 3, 4 (Virtual Neck)
        neck_center = (pos[:, 3] + pos[:, 4]) / 2.0 # (F_pad, 3)
        
        # 4. Normalize Position (Center to Hip)
        # Broadcast hip_center: (F_pad, 1, 3)
        centered_pos = pos - hip_center[:, None, :] 

        # 5. Compute Velocity (Vectorized Diff)
        # vel[t] = pos[t] - pos[t-1]
        vel = np.zeros_like(centered_pos)
        vel[1:] = centered_pos[1:] - centered_pos[:-1]
        
        # 6. Compute Structural Vectors (Using Index Maps)
        # Bone Vector: Child - Current
        # Parent Vector: Current - Parent
        
        # Standard lookups
        bone_vecs = centered_pos[:, self.children] - centered_pos # (F_pad, 25, 3)
        parent_vecs = centered_pos - centered_pos[:, self.parents] # (F_pad, 25, 3)
        
        # -- FIX SPECIAL CASES (Roots & Leaves) --
        
        # Fix Leaves (Fingers/Toes/Head): They have no child, bone_vec should be 0
        leaf_mask = (self.children == np.arange(25)) # Indices pointing to self
        bone_vecs[:, leaf_mask] = 0 
        
        # Fix Roots (Head, Shoulders): Parent is Virtual Neck
        # Indices: 0(Nose), 1,2(Ears), 3,4(Shoulders)
        neck_rel = neck_center - hip_center # (F_pad, 3) relative to new origin
        for i in [0, 1, 2, 3, 4]:
            parent_vecs[:, i] = centered_pos[:, i] - (neck_rel - 0) # 0 is virtual hip

        # Fix Roots (Hips): Parent is Virtual Hip (0,0,0)
        # Indices: 15, 16. Parent vector is just their position
        for i in [15, 16]:
            parent_vecs[:, i] = centered_pos[:, i] # - 0
            
        # 7. Stack Features (F_pad, 25, 12)
        # [Pos(3), Vel(3), Bone(3), Parent(3)]
        features = np.concatenate([centered_pos, vel, bone_vecs, parent_vecs], axis=2)
        
        # 8. Flatten Joints (F_pad, 300)
        features_flat = features.reshape(F_pad, -1)
        
        # 9. Apply Validity Mask (Zero out invalid frames)
        valid_mask = (flags > 0.5)[:, None] # (F_pad, 1)
        features_flat *= valid_mask
        
        # 10. Sliding Window (Stride Tricks)
        D = 300
        strides = (features_flat.strides[0], features_flat.strides[0], features_flat.strides[1])
        X_windows = np.lib.stride_tricks.as_strided(
            features_flat, 
            shape=(F, N, D), 
            strides=strides
        )
        
        # 11. Mask Window
        idx_matrix = np.arange(F)[:, None] + np.arange(N)
        M_masks = (flags[idx_matrix] > 0.5)
        
        return X_windows, M_masks

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load (Already 25 joints)
        # Source Shape: (F, 25, 4)
        input_data = np.load(os.path.join(self.dataset_root, item["source"])).astype(np.float32)
        gt_data = np.load(os.path.join(self.dataset_root, item["target"])).astype(np.float32)
        
        # Alignment
        min_len = min(len(input_data), len(gt_data))
        input_data = input_data[:min_len]
        gt_data = gt_data[:min_len]
        
        # Vectorized Processing
        X, M = self._process_vectorized(input_data)
        
        # Flatten GT: (F, 21, 3) -> (F, 63)
        Y = gt_data.reshape(min_len, -1)
        
        return {
            "source": torch.from_numpy(X),
            "mask": torch.from_numpy(M),
            "target": torch.from_numpy(Y)
        }