from ultralytics import YOLO
import torch

# Auto-detect best available device
if torch.cuda.is_available():
    device = 0  
    batch = 32  
    workers = 8
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif torch.backends.mps.is_available():
    device = 'mps'  
    batch = 16
    workers = 4
    print("Using Apple Silicon GPU (MPS)")
else:
    device = 'cpu'
    batch = 8
    workers = 4
    print("Using CPU - training will be slow!")


model = YOLO('yolov8s.pt')  


results = model.train(
    data='data.yaml',           
    epochs=100,                  # Number of training epochs
    imgsz=640,                   # Image size
    batch=batch,                 # Batch size (auto-adjusted based on GPU)
    name='powerfoot_yolov8',     # Name of the training run
    patience=50,                 # Early stopping patience
    save=True,                   # Save checkpoints
    device=device,               # Auto-detected device
    workers=workers,             # Number of worker threads (auto-adjusted)
    project='runs/detect',       # Project directory
    exist_ok=True,               # Overwrite existing project
    pretrained=True,             # Use pretrained weights
    optimizer='auto',            # Optimizer (auto, SGD, Adam, AdamW, etc.)
    verbose=True,                # Verbose output
    seed=0,                      # Random seed for reproducibility
    deterministic=True,          # Deterministic mode
    single_cls=False,            # Train as single-class dataset
    rect=False,                  # Rectangular training
    cos_lr=False,                # Cosine learning rate scheduler
    close_mosaic=10,             # Disable mosaic augmentation for final epochs
    resume=False,                # Resume from last checkpoint
    amp=True,                    # Automatic Mixed Precision training
    fraction=1.0,                # Fraction of dataset to train on
    profile=False,               # Profile ONNX and TensorRT speeds
    freeze=None,                 # Freeze layers (None, int, or list)
    # Learning rate settings
    lr0=0.01,                    # Initial learning rate
    lrf=0.01,                    # Final learning rate (lr0 * lrf)
    momentum=0.937,              # SGD momentum/Adam beta1
    weight_decay=0.0005,         # Optimizer weight decay
    warmup_epochs=3.0,           # Warmup epochs
    warmup_momentum=0.8,         # Warmup initial momentum
    warmup_bias_lr=0.1,          # Warmup initial bias lr
    # Augmentation settings
    hsv_h=0.015,                 # HSV-Hue augmentation
    hsv_s=0.7,                   # HSV-Saturation augmentation
    hsv_v=0.4,                   # HSV-Value augmentation
    degrees=0.0,                 # Rotation (+/- deg)
    translate=0.1,               # Translation (+/- fraction)
    scale=0.5,                   # Image scale (+/- gain)
    shear=0.0,                   # Shear (+/- deg)
    perspective=0.0,             # Perspective (+/- fraction)
    flipud=0.0,                  # Flip up-down (probability)
    fliplr=0.5,                  # Flip left-right (probability)
    mosaic=1.0,                  # Mosaic augmentation (probability)
    mixup=0.0,                   # Mixup augmentation (probability)
    copy_paste=0.0,              # Copy-paste augmentation (probability)
)


print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
print(f"Last model saved at: {results.save_dir}/weights/last.pt")
print("\nTo validate the model, run:")
print(f"  model = YOLO('{results.save_dir}/weights/best.pt')")
print("  metrics = model.val()")
print("\nTo run inference:")
print(f"  model = YOLO('{results.save_dir}/weights/best.pt')")
print("  results = model.predict(source='path/to/image.jpg', save=True)")
