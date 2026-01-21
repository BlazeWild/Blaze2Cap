# Blaze2Cap

**3D Human Pose Estimation: BlazePose to TotalCapture Motion Dataset Pipeline**

A comprehensive pipeline for extracting 3D human pose landmarks from the TotalCapture dataset using MediaPipe BlazePose, with data preprocessing, synchronization, and PyTorch dataset loaders for motion capture research and machine learning.

---

## 📋 Overview

Blaze2Cap bridges the gap between marker-based motion capture systems (TotalCapture) and markerless pose estimation (MediaPipe BlazePose). This project provides a complete pipeline to:

- **Extract 33 BlazePose 3D landmarks** from TotalCapture videos (8 cameras × 12 actions × 5 subjects)
- **Synchronize** BlazePose predictions with ground truth motion capture data
- **Process & filter** data to match 12 key body joints for motion analysis
- **Generate training-ready datasets** with sliding window sequences, velocity features, and masking
- **Provide PyTorch DataLoader** for easy integration into deep learning workflows

---

## 🚀 Key Features

### 1. **Pose Extraction**

- Extracts all 33 MediaPipe BlazePose 3D world landmarks (x, y, z in meters)
- Sequential video processing with GPU acceleration support
- Handles frame skipping when pose detection fails
- Outputs NumPy arrays `(frames, 33, 4)` with `[x, y, z, prediction_flag]`
  - The 4th channel is a flag indicating if the previous frame was predicted, used for preprocessing and masking

### 2. **Data Synchronization**

- Aligns BlazePose predictions with TotalCapture ground truth (GT)
- Handles frame count mismatches between videos and GT data
- Filters to 12 key body joints for consistency
- Exports synchronized NumPy datasets

### 3. **Dataset Preparation**

- Sliding window sequences with configurable window size
- Computes frame-to-frame velocity for each joint
- Dual masking strategy: frame-level and joint-level
- Train/test split metadata generation (JSON mapping)
- PyTorch Dataset class with automatic data loading

### 4. **Utilities**

- Frame count verification across videos, BlazePose outputs, and GT data
- Interactive HTML visualizations of keypoints
- Comprehensive analysis scripts for data quality checks

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

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for video processing)
- TotalCapture dataset videos (~50GB, not included in repo)

### Setup with `uv`

```bash
# Clone the repository
git clone https://github.com/BlazeWild/Blaze2Cap.git
cd Blaze2Cap

# Install dependencies with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Optional Development Dependencies

```bash
uv pip install -e ".[dev]"  # Includes Jupyter, Matplotlib, Plotly
```

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

### **12 Key Body Joints**

Selected from BlazePose 33 landmarks for compatibility with motion capture:

| Index | Joint Name     | BlazePose ID |
| ----- | -------------- | ------------ |
| 0     | Nose           | 0            |
| 1     | Left Shoulder  | 11           |
| 2     | Right Shoulder | 12           |
| 3     | Left Elbow     | 13           |
| 4     | Right Elbow    | 14           |
| 5     | Left Wrist     | 15           |
| 6     | Right Wrist    | 16           |
| 7     | Left Hip       | 23           |
| 8     | Right Hip      | 24           |
| 9     | Left Knee      | 25           |
| 10    | Right Knee     | 26           |
| 11    | Left Ankle     | 27           |
| 12    | Right Ankle    | 28           |

### **Data Format**

- **Input:** `(frames, 12, 4)` → `[x, y, z, prediction_flag]`
  - The 4th channel indicates if the previous frame was predicted (used for preprocessing and masking)
- **Features:** Position + Velocity = 6 channels per joint → 72 total features
- **Windows:** Sliding window sequences for temporal context
- **Masking:**
  - Frame mask: Valid/invalid entire frames
  - Joint mask: Individual joint prediction status per frame

### **Splits**

- **Train:** S1, S2, S3
- **Test:** S1, S2, S3, S4, S5

---

## 🔍 Data Processing

For data extraction, preprocessing, and analysis tools, see the [private repository](https://github.com/BlazeWild/Blaze2cap-all-data-exceptcode) which includes:

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

## 📄 License

This project is licensed under the MIT License.

---

## 🔗 Links

- **GitHub:** [BlazeWild/Blaze2Cap](https://github.com/BlazeWild/Blaze2Cap)
- **Issues:** [Report bugs or request features](https://github.com/BlazeWild/Blaze2Cap/issues)
- **TotalCapture:** [Official Dataset Page](http://cvssp.org/data/totalcapture/)
- **MediaPipe:** [BlazePose Documentation](https://google.github.io/mediapipe/solutions/pose.html)

---

## 📧 Contact

For questions or collaboration:

- Open an issue on GitHub
- Repository: [github.com/BlazeWild/Blaze2Cap](https://github.com/BlazeWild/Blaze2Cap)

---

## 🙏 Acknowledgments

- **TotalCapture Dataset:** University of Surrey
- **MediaPipe BlazePose:** Google Research
- Built with PyTorch, NumPy, and OpenCV

---

**Happy Pose Estimating! 🏃‍♂️💡**
