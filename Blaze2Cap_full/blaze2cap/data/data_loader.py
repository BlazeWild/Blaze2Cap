import os
import json
import numpy as np
import torch
from torch.utils import data

class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root:str, window_size:int, split:str = "train"):
        """
        Args:
            dataset_root (str): The folder containing 'dataset_map.json' and the data subfolders.
            window_size (int): N for the sliding window.
            split (str): 'train' or 'test'.
        """
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.split = split
        
        #locate JSON
        json_path = os.path.join(dataset_root, "dataset_map.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON map not found at {json_path}")
        
        # Load JSON
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        # Filter based on split
        self.samples = []
        for item in full_data:
            if split == "train" and item["split_train"]:
                self.samples.append(item)
            elif split == "test" and item["split_test"]:
                self.samples.append(item)
                
        print(f"[{split.upper()}] Initialized. Samples: {len(self.samples)} | Window: {window_size}")
        
    def _get_fragmented_window(self, raw_data):
        """
        Converts raw sequence -> Windows with Velocity & Masking.
        Input: (F, 33, 4) -> (x, y, z, visibility)
        Output: X=(F, N, 198), M=(F, N)
        """
        N = self.window_size
        F,J,C = raw_data.shape # frames, joints(33), channels(4)
        D = J*6 # 3 coords (pos) + 3 coords (vel) = 6 dims per joint 
        # Shape becomes (F_pad, 33, 6) -> flattened to (F_pad, 198)
        
        # structural padding (only at start)
        padding = np.repeat(raw_data[0:1], N-1, axis=0) # (N-1, 33, 4)
        padding[:,:,3] = 0 # force initial padding to 0
        full_timeline = np.concatenate([padding, raw_data], axis=0) # (F+N-1, 33, 4)
        
        # extract master flags (channel 3 is flag)
        # this array tells us where 'anchor' is located
        all_flags = full_timeline[:,0,3] # (F+N-1,)
        
        # calculate velocities with 'anchor zeroing'
        # we only take the first 3 channels (x,y,z) for velocity
        pos = full_timeline[:,:,:3] # (F+N-1, 33, 3)
        vel = np.zeros_like(pos) # (F+N-1, 33, 3)
        
        # standard velocity = current - previous
        vel[1:] = pos[1:] - pos[:-1]
        
        # the reappearance correction
        # if flag is 0, velocity must be 0 (no teleporting)
        vel[all_flags ==0]=0
        
        # concatenate position+velocity and reshape to D (flattened)
        features = np.concatenate([pos, vel], axis=-1)
        features = features.reshape(features.shape[0], D) # (F+N-1, 198)
        
        # efficient striding
        # features.strdes[1] = D dimension stride and features.strides[0] = frame dimension stride
        row_size = features.strides[0]
        X = np.lib.stride_tricks.as_strided(
            features, 
            shape=(F, N, D), 
            strides=(row_size, row_size, features.strides[1]) 
        )
        
        # virtual wall mask
        window_indices = np.arange(F)[:, None] + np.arange(N)
        M = (all_flags[window_indices] == 0) # (F, N) True = valid, False = invalid
        
        return X, M
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        #get metadata from memory
        item = self.samples[idx]
        
        #construct absolute paths
        source_path = os.path.join(self.dataset_root, item["source"])
        target_path = os.path.join(self.dataset_root, item["target"])
        
        #load heavy data
        # using .astype(np.float32) to ensure float32 dtype
        input_data = np.load(source_path).astype(np.float32) # (F, 33, 4)
        gt_data = np.load(target_path).astype(np.float32)# (F, 17,3)
        
        # frame alignment(sanity check)
        #this is already processed as sometimes source and gt might differ by 1-2 frames due to preprocessing
        min_len = min(len(input_data), len(gt_data))
        input_data = input_data[:min_len]
        gt_data = gt_data[:min_len]
        
        #process windows(input side)
        # X_windows shape: (F, N, 198), masks shape: (F, N)
        X_windows, masks = self._get_fragmented_window(input_data)
        
        # process target (gt side)
        # we predict the pose at current frame t (end of window)
        # so we flatten the gt to (frames, joints*3), so shape = (F, 51)
        Y_target = gt_data.reshape(min_len, -1)
        
        # return tensors
        return{
            "source" : torch.from_numpy(X_windows),   # input features, (F, N, 198)
            "mask"   : torch.from_numpy(masks),       # validity mask, (F, N)
            "target" : torch.from_numpy(Y_target),      # ground truth, (F, 51)
            "meta_subject": item["subject"],            # metadata
            "meta_path": source_path   # debugging info
        }