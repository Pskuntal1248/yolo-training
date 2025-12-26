# Football Object Detection - YOLO Training Dataset

## Dataset Overview
- **Total Images**: 8,639
- **Classes**: 4 (ball, goalkeeper, player, referee)
- **Format**: YOLOv8 (YOLO format)
- **Splits**:
  - Training: 6,429 images
  - Validation: 1,470 images  
  - Test: 740 images

## Quick Start for GPU Training

### 1. Install Requirements
```bash
pip install ultralytics torch torchvision
```

For NVIDIA GPU, also install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Training

**Simple (Recommended)**:
```bash
python train_simple.py
```

**Advanced with full control**:
```bash
python train_yolo.py
```

Both scripts automatically:
- ✅ Detect NVIDIA GPU (CUDA) or Apple Silicon (MPS)
- ✅ Optimize batch size and workers for your hardware
- ✅ Use appropriate model size (yolo11s for GPU)
- ✅ Configure best settings for performance

### Expected Performance

**With Dedicated NVIDIA GPU** (RTX 3060/3070/3080/4090):
- Batch size: 32+
- Speed: ~0.2-0.5s per iteration
- Training time: ~1-3 hours for 100 epochs


## Training Configuration

### Model Sizes (change in script):
```python
model = YOLO('yolo11n.pt')  # Nano - fastest (3M params)
model = YOLO('yolo11s.pt')  # Small - recommended (11M params) ⭐
model = YOLO('yolo11m.pt')  # Medium - better accuracy (25M params)
model = YOLO('yolo11l.pt')  # Large - high accuracy (43M params)
model = YOLO('yolo11x.pt')  # XLarge - best accuracy (68M params)
```

### GPU Memory Requirements:
- **4GB**: yolo11n, batch=16
- **6GB**: yolo11s, batch=24-32
- **8GB**: yolo11m, batch=24-32
- **12GB+**: yolo11l/x, batch=32-64

### Adjusting Batch Size:
If you get CUDA out of memory errors, reduce batch size in the script:
```python
batch=16  # or batch=8, batch=4
```

## After Training

### Results Location
```
runs/detect/powerfoot_gpu/
├── weights/
│   ├── best.pt          # Best model (use this!)
│   └── last.pt          # Last epoch
├── results.png          # Training metrics graphs
├── confusion_matrix.png # Confusion matrix
├── val_batch0_pred.jpg  # Validation predictions
└── ...
```

### Validate Model
```bash
python validate_model.py
```

### Run Inference
```bash
python predict.py
```

## Files Included

```
.
├── data.yaml              # Dataset configuration
├── train_simple.py        # Simple training script 
├── train_yolo.py         # Advanced training script
├── validate_model.py     # Model validation
├── predict.py            # Run predictions
├── TRAINING_GUIDE.md     # Detailed guide
├── README.txt            # This file
├── train/                # Training images & labels
├── valid/                # Validation images & labels
└── test/                 # Test images & labels
```

## Tips for Best Results

1. **Use GPU**: 10-50x faster than CPU
2. **Start with yolo11s**: Best balance of speed/accuracy
3. **Monitor training**: Watch for overfitting (train loss << val loss)
4. **Increase epochs if needed**: If loss still decreasing at epoch 100
5. **Try different models**: Compare yolo11s vs yolo11m for your use case

## Troubleshooting

### CUDA Out of Memory
```python
# In training script, reduce:
batch=16  # or 8, or 4
imgsz=416  # from 640
```

### Slow Training
- Make sure CUDA/GPU is detected
- Increase workers: `workers=8`
- Use smaller model: `yolo11n.pt` or `yolo11s.pt`

### Poor Accuracy
- Train longer: `epochs=150` or `epochs=200`
- Use bigger model: `yolo11m.pt` or `yolo11l.pt`
- Check data quality and annotations


