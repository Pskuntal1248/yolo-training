# YOLOv8s Football Detection Training

Complete setup for training YOLOv8s (11M parameters) on football/soccer detection dataset.

## 🎯 Dataset Overview

- **Total Images**: 1,981
  - **Training**: 1,056 images
  - **Validation**: 587 images
  - **Testing**: 338 images

- **Classes** (4):
  - `0: ball`
  - `1: goalkeeper`
  - `2: player`
  - `3: referee`

- **Format**: YOLO format (normalized coordinates)
- **License**: CC BY 4.0

## 📦 Dataset Structure

```
.
├── data.yaml                 # Dataset configuration
├── train/
│   ├── images/              # 1,056 training images
│   └── labels/              # 1,056 YOLO format labels
├── valid/
│   ├── images/              # 587 validation images
│   └── labels/              # 587 YOLO format labels
└── test/
    ├── images/              # 338 test images
    └── labels/              # 338 YOLO format labels
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install ultralytics opencv-python pillow pyyaml

# Or use the provided requirements
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Basic training (auto-detects GPU/CPU)
python train.py

# The script automatically:
# - Detects GPU (MPS/CUDA) or falls back to CPU
# - Uses YOLOv8s (11M parameters)
# - Trains for 50 epochs
# - Saves results to runs/detect/train/
```

### 3. Validate Model

```bash
python validate_model.py
```

### 4. Run Inference

```bash
python inference.py
```

## 🎮 Training Configuration

**Model**: YOLOv8s (11M parameters)
- Best balance of speed and accuracy
- Suitable for real-time detection
- Good for systems with limited GPU

**Hyperparameters**:
- Epochs: 50
- Image size: 640x640
- Batch size: Auto (based on available memory)
- Optimizer: Auto
- Device: Auto-detect (MPS/CUDA/CPU)

## 💻 Hardware Requirements

### Minimum:
- **CPU**: Any modern CPU
- **RAM**: 8GB
- **Storage**: 2GB free space

### Recommended:
- **GPU**: NVIDIA GPU with CUDA or Apple Silicon with MPS
- **RAM**: 16GB+
- **Storage**: 5GB+ free space

## 📊 Model Performance

After training, find results in:
- `runs/detect/train/weights/best.pt` - Best model
- `runs/detect/train/weights/last.pt` - Last checkpoint
- `runs/detect/train/results.png` - Training metrics
- `runs/detect/train/confusion_matrix.png` - Confusion matrix

## 🔧 Files Description

- **train.py**: Main training script with GPU auto-detection
- **validate_model.py**: Validate trained model on test set
- **inference.py**: Run inference on images/videos
- **data.yaml**: Dataset configuration for YOLO
- **requirements.txt**: Python dependencies

## 📝 Label Format

YOLO format (one .txt file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]

Example:
```
2 0.5494 0.3513 0.0104 0.0379
1 0.2345 0.6789 0.0521 0.0893
```

## 🎯 Usage Examples

### Train with custom epochs
```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='data.yaml', epochs=100)
```

### Inference on image
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model('path/to/image.jpg')
results[0].show()
```

### Inference on video
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
results = model('path/to/video.mp4', stream=True)

for r in results:
    r.show()
```

## 🔄 For Your Friend (Dedicated GPU Setup)

This repository is ready to clone and train on any system with dedicated GPU:

```bash
# Clone the repository
git clone https://github.com/Pskuntal1248/yolo-training.git
cd yolo-training

# Install dependencies
pip install ultralytics

# Start training (auto-detects GPU)
python train.py
```

The training script automatically detects:
- ✅ NVIDIA CUDA GPUs
- ✅ Apple Silicon MPS
- ✅ CPU fallback

## 📈 Expected Training Time

- **NVIDIA RTX 3090**: ~15-20 minutes (50 epochs)
- **Apple M1/M2**: ~30-40 minutes (50 epochs)
- **CPU only**: ~2-3 hours (50 epochs)

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in train.py:
```python
model.train(data='data.yaml', epochs=50, batch=8)
```

### MPS Issues (Mac)
Force CPU if MPS has issues:
```python
model.train(data='data.yaml', epochs=50, device='cpu')
```

## 📄 License

Dataset: CC BY 4.0  
Code: MIT License

## 🔗 Source

Dataset from Roboflow Universe:
- Workspace: glider-umpjg
- Project: properlyannotatedsoccer-icg4z-ige6w
- Version: 1

## ✅ Verification

All image-label pairs verified:
- ✅ 1,056 train pairs matched
- ✅ 587 validation pairs matched
- ✅ 338 test pairs matched
- ✅ YOLO format validated
- ✅ Ready for YOLOv8s training

---

**Ready to train!** Just run `python train.py` 🚀
