from ultralytics import YOLO
import os

# Inference script - run predictions on new images

# Load your trained model
model = YOLO('runs/detect/powerfoot_yolov8/weights/best.pt')

# Single image prediction
results = model.predict(
    source='test/images',  
    conf=0.25,  
    iou=0.7,  
    imgsz=640,  
    save=True,  
    save_txt=True,  
    save_conf=True,  
    show_labels=True,  
    show_conf=True,  
    line_width=2, 
    device='mps',  
    project='runs/predict',  
    name='powerfoot_inference',
    exist_ok=True,
)

print("\n" + "="*50)
print("Inference Complete!")
print("="*50)
print(f"Results saved to: runs/predict/powerfoot_inference")
print(f"\nDetected objects:")

# Print detections for each image
for result in results:
    boxes = result.boxes
    print(f"\n{os.path.basename(result.path)}:")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls]
        print(f"  - {class_name}: {conf:.2f}")
