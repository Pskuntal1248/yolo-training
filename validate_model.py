from ultralytics import YOLO

model = YOLO('runs/detect/powerfoot_yolov8/weights/best.pt')

metrics = model.val(
    data='data.yaml',
    split='test',
    imgsz=640,
    batch=16,
    device=0,
    save_json=True,
    save_hybrid=False,
    conf=0.001,
    iou=0.6,
    max_det=300,
    plots=True,
)

# Print metrics
print("\n" + "="*50)
print("Validation Metrics")
print("="*50)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")

# Per-class metrics
print("\nPer-class mAP50:")
for i, name in enumerate(metrics.names.values()):
    print(f"  {name}: {metrics.box.maps[i]:.4f}")
