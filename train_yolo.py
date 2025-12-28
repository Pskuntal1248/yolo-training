from ultralytics import YOLO
import torch
import os

print("="*60)
print("YOLO TRAINING SETUP - OPTIMIZED FOR WINDOWS GPU")
print("="*60)

# GPU Detection and Configuration
print("\n[1] Checking GPU availability...")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA Available: YES")
    print(f"   GPU Count: {gpu_count}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    device = 0  # Use first GPU
    batch = 16  # Safe batch size for most GPUs
    workers = 4  # Optimal for Windows
    
    # Set CUDA optimizations
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    print(f"\n[2] Training Configuration:")
    print(f"   Device: GPU (cuda:0)")
    print(f"   Batch Size: {batch}")
    print(f"   Workers: {workers}")
    
else:
    print("❌ CUDA NOT Available!")
    print("\n   To fix this:")
    print("   1. Install CUDA-enabled PyTorch:")
    print("      pip uninstall torch torchvision")
    print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("   2. Verify: python -c \"import torch; print(torch.cuda.is_available())\"")
    
    device = 'cpu'
    batch = 8
    workers = 2
    print(f"\n   Falling back to CPU (SLOW!)")

print("\n[3] Loading YOLOv8s model...")
model = YOLO('yolov8s.pt')

print("\n[4] Starting training...")
print("="*60)

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=batch,
    device=device,
    workers=workers,
    name='powerfoot_yolov8',
    project='runs/detect',
    exist_ok=True,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    seed=0,
    deterministic=True,
    patience=50,
    save=True,
    plots=True,
    amp=True,  # Automatic Mixed Precision for faster training
    cache=False,  # Don't cache images (saves RAM)
    # Learning rate
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
)

# Print training results
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
