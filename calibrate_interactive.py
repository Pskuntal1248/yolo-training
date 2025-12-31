"""
Interactive Field Calibration Tool
Run this to click on field markings and generate accurate pixel coordinates
"""
import cv2
import numpy as np
import json

class InteractiveCalibrator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.points = []
        self.point_labels = [
            "1. BOTTOM-LEFT: Click near-camera left corner/touchline",
            "2. TOP-LEFT: Click far-side left corner/touchline", 
            "3. TOP-RIGHT: Click far-side right corner/touchline",
            "4. BOTTOM-RIGHT: Click near-camera right corner/touchline"
        ]
        self.colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
        self.frame = None
        self.display = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            print(f"Point {len(self.points)}: [{x}, {y}]")
            self.update_display()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show crosshair
            temp = self.display.copy()
            cv2.line(temp, (x, 0), (x, temp.shape[0]), (255,255,255), 1)
            cv2.line(temp, (0, y), (temp.shape[1], y), (255,255,255), 1)
            cv2.putText(temp, f"({x}, {y})", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow('Calibration', temp)

    def update_display(self):
        self.display = self.frame.copy()
        
        # Draw points
        for i, pt in enumerate(self.points):
            cv2.circle(self.display, tuple(pt), 10, self.colors[i], -1)
            cv2.circle(self.display, tuple(pt), 12, (255,255,255), 2)
            cv2.putText(self.display, f"P{i+1}", (pt[0]+15, pt[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[i], 2)
        
        # Draw polygon if 4 points
        if len(self.points) == 4:
            pts = np.array(self.points, np.int32).reshape((-1,1,2))
            cv2.polylines(self.display, [pts], True, (0,255,255), 2)
        elif len(self.points) > 1:
            for i in range(len(self.points)-1):
                cv2.line(self.display, tuple(self.points[i]), 
                        tuple(self.points[i+1]), (0,255,255), 2)
        
        # Instructions
        if len(self.points) < 4:
            label = self.point_labels[len(self.points)]
        else:
            label = "Press 's' to SAVE | 'r' to RESET | 'q' to QUIT"
        
        # Draw instruction box
        cv2.rectangle(self.display, (0, 0), (700, 35), (0,0,0), -1)
        cv2.putText(self.display, label, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.imshow('Calibration', self.display)

    def calculate_scale_info(self):
        """Calculate and display scale information"""
        if len(self.points) != 4:
            return
            
        # Pixel distances
        bottom_px = np.linalg.norm(np.array(self.points[3]) - np.array(self.points[0]))
        top_px = np.linalg.norm(np.array(self.points[2]) - np.array(self.points[1]))
        left_px = np.linalg.norm(np.array(self.points[1]) - np.array(self.points[0]))
        right_px = np.linalg.norm(np.array(self.points[2]) - np.array(self.points[3]))
        
        print("\n" + "="*60)
        print("SCALE ANALYSIS")
        print("="*60)
        print(f"Bottom edge: {bottom_px:.0f} px = 68m width → {bottom_px/68:.1f} px/m")
        print(f"Top edge: {top_px:.0f} px = 68m width → {top_px/68:.1f} px/m")
        print(f"Left edge: {left_px:.0f} px = 105m length → {left_px/105:.1f} px/m")
        print(f"Right edge: {right_px:.0f} px = 105m length → {right_px/105:.1f} px/m")
        print(f"Perspective ratio (bottom/top): {bottom_px/top_px:.2f}x")
        print("="*60)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        # Skip to frame 50 for stable image
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read video")
            return None
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', self.mouse_callback)
        
        self.update_display()
        
        print("\n" + "="*60)
        print("FIELD CALIBRATION TOOL")
        print("="*60)
        print("Click the 4 corners of the visible pitch area")
        print("Order: Bottom-Left → Top-Left → Top-Right → Bottom-Right")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(self.points) == 4:
                self.calculate_scale_info()
                self.save_calibration()
                break
            elif key == ord('r'):
                self.points = []
                self.update_display()
                print("Reset - click 4 corners again")
            elif key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        return self.points if len(self.points) == 4 else None

    def save_calibration(self):
        config = {
            "pixel_vertices": self.points,
            "pitch_length": 105.0,
            "pitch_width": 68.0,
            "target_vertices": [
                [0, 68.0],
                [0, 0],
                [105.0, 0],
                [105.0, 68.0]
            ]
        }
        
        with open('field_calibration.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "="*60)
        print("CALIBRATION SAVED!")
        print("="*60)
        print("\nCopy this to view_transformer.py:\n")
        print("self.pixel_vertices = np.array([")
        for p in self.points:
            print(f"    [{p[0]}, {p[1]}],")
        print("], dtype=np.float32)")
        print("\n" + "="*60)


if __name__ == '__main__':
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'input_videos/test.mp4'
    
    calibrator = InteractiveCalibrator(video_path)
    points = calibrator.run()
