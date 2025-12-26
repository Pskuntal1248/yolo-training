from ultralytics import YOLO
import torch

# Quick training script optimized for dedicated GPU
# Auto-detects and configures for best performance

# Auto-detect and configure for available hardware
if torch.cuda.is_available():
    device = 0  # NVIDIA GPU
    batch = 32  # Large batch for dedicated GPU
    workers = 8
    print(f" Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon
    batch = 16
    workers = 4
    print(" Using Apple Silicon GPU (MPS)")
else:
    device = 'cpu'
    batch = 8
    workers = 4
    print("⚠️  Using CPU - training will be slow!")

# Initialize model
# yolov8s.pt recommended for dedicated GPU (11M params - good balance)
model = YOLO('yolov8s.pt')

# Train with optimized settings
results = model.train(
    data='data.yaml',
    epochs=100,            # More epochs for better training
    imgsz=640,
    batch=batch,           # Auto-adjusted for your hardware
    device=device,
    workers=workers,
    name='powerfoot_yolov8',
    patience=30,
)

print(f"\nTraining complete! Best model: {results.save_dir}/weights/best.pt")
