# Blaze2Cap

> **⚠️ Ongoing Research Project**

**3D Human Pose Estimation: BlazePose to TotalCapture Motion Dataset Pipeline**

A PyTorch dataset loader for 3D human pose estimation research, bridging MediaPipe BlazePose predictions with TotalCapture motion capture ground truth.

---

## 📋 Overview

Blaze2Cap provides a PyTorch-ready dataset pipeline for motion capture research:

- **PyTorch DataLoader** with sliding window sequences and temporal features
- **Synchronized data** between BlazePose (33 landmarks) and TotalCapture ground truth
- **12 key body joints** filtered for motion analysis
- **Dual masking** strategy for frame-level and joint-level validity
- **Velocity features** computed frame-to-frame for temporal modeling

**Data Format:**

- Input: `(frames, 12, 4)` → `[x, y, z, prediction_flag]`
- The 4th channel indicates if the previous frame was predicted (used for preprocessing and masking)
- Features: Position + Velocity = 6 channels per joint → 72 total features

---

## 📁 Project Structure

```
Blaze2Cap/
├── Blaze2Cap/                          # Main package
│   ├── data/                           # Dataset loaders and JSON generation
│   │   ├── data_loader.py             # PyTorch Dataset with windowing & masking
│   │   └── generate_json.py           # Create dataset_map.json for splits
│   ├── dataset/                        # Preprocessed dataset storage
│   │   └── Totalcapture_numpy_preprocessed/
│   ├── modules/                        # Model modules (placeholder)
│   └── learn/                          # Jupyter notebooks for experimentation
│
├── pyproject.toml                     # Project dependencies (uv)
└── README.md                          # This file
```

> **Note:** Data processing utilities, test scripts, and preprocessing tools are in the [private repository](https://github.com/BlazeWild/Blaze2cap-all-data-exceptcode).

---

## 📊 Usage

### **Load Data in PyTorch**

```python
from Blaze2Cap.data.data_loader import PoseSequenceDataset

# Initialize dataset
train_dataset = PoseSequenceDataset(
    dataset_root='final_numpy_dataset',
    window_size=64,  # Sliding window of 64 frames
    split='train'
)

# Create DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Iterate through batches
for X, GT, M_frame, M_joint in train_loader:
    # X: Input features (batch, windows, features)
    # GT: Ground truth (batch, windows, features)
    # M_frame: Frame-level mask (batch, windows)
    # M_joint: Joint-level mask (batch, windows, joints)
    pass
```

---

## 📈 Dataset Details

**12 Key Body Joints:** Nose, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hip, L/R Knee, L/R Ankle

**Splits:** Train (S1-S3) | Test (S1-S5)

**TotalCapture:** 5 subjects × 12 actions × 8 camera views

- **BlazePose extraction** (33 landmarks from videos)
- **Data filtering** (12 key joints)
- **Frame synchronization** between BlazePose and ground truth
- **Verification scripts** and quality analysis
- **Visualization tools** for 3D keypoints

---

## 📚 TotalCapture Dataset

This project processes the **TotalCapture** dataset:

- **5 subjects** (S1-S5)
- **12 actions** (acting, freestyle, rom, walking)
- **8 camera views** per action
- **Marker-based motion capture** ground truth

**Download:** [TotalCapture Dataset](http://cvssp.org/data/totalcapture/)  
⚠️ Videos (~50GB) are **NOT included** in this repository.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## � Data Processing

Data extraction, preprocessing, and analysis tools are in the [private repository](https://github.com/BlazeWild/Blaze2cap-all-data-exceptcode)� References

- **TotalCapture Dataset:** [Official Page](http://cvssp.org/data/totalcapture/)
- **MediaPipe BlazePose:** [Documentation](https://google.github.io/mediapipe/solutions/pose.html)

**License:** MIT
