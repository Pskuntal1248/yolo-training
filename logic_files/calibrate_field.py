
import cv2
import numpy as np


points = []
image = None
image_copy = None

def mouse_callback(event, x, y, flags, param):
   
    global points, image, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: [{x}, {y}]")
            
          
            cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(image, f"P{len(points)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
          
            if len(points) > 1:
                cv2.line(image, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            

            if len(points) == 4:
                cv2.line(image, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                print("\n All 4 corners selected!")
                print("\nCopy this array to view_transformer.py:")
                print("self.pixel_vertices = np.array([")
                for p in points:
                    print(f"    {p},")
                print("])")
            
            cv2.imshow('Calibrate Field Corners', image)


def main():
    """Main calibration function"""
    global points, image, image_copy
    
    print("=" * 70)
    print(" FOOTBALL FIELD CALIBRATION TOOL")
    print("=" * 70)
    print("\nINSTRUCTIONS:")
    print("1. Click on the 4 corners of the football field in this order:")
    print("   - Point 1: Bottom-left corner")
    print("   - Point 2: Top-left corner")
    print("   - Point 3: Top-right corner")
    print("   - Point 4: Bottom-right corner")
    print("\n2. Press 'r' to reset if you make a mistake")
    print("3. Press 'q' or ESC when done to exit")
    print("=" * 70)
    
   
    cap = cv2.VideoCapture('test.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read test.mp4")
        return
    
    image = frame.copy()
    image_copy = frame.copy()
    
   
    cv2.namedWindow('Calibrate Field Corners', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Calibrate Field Corners', mouse_callback)
    
    # Show current points from code
    print("\n📍 Current points in view_transformer.py:")
    current_points = np.array([
        [110, 1035],
        [265, 275],
        [910, 260],
        [1640, 915]
    ])
    for i, p in enumerate(current_points):
        print(f"   Point {i+1}: {p}")
        cv2.circle(image, tuple(p), 8, (0, 0, 255), -1)
        cv2.putText(image, f"OLD-{i+1}", (p[0]+10, p[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Draw old polygon
    for i in range(4):
        cv2.line(image, tuple(current_points[i]), 
                tuple(current_points[(i+1)%4]), (0, 0, 255), 2)
    
    print("\n🔴 Red points = Current configuration (may be wrong)")
    print("🟢 Green points = Your new selection (click to add)")
    
    cv2.imshow('Calibrate Field Corners', image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Reset
        if key == ord('r'):
            print("\n🔄 Resetting points...")
            points = []
            image = image_copy.copy()
            
            # Redraw old points
            for i, p in enumerate(current_points):
                cv2.circle(image, tuple(p), 8, (0, 0, 255), -1)
                cv2.putText(image, f"OLD-{i+1}", (p[0]+10, p[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            for i in range(4):
                cv2.line(image, tuple(current_points[i]), 
                        tuple(current_points[(i+1)%4]), (0, 0, 255), 2)
            
            cv2.imshow('Calibrate Field Corners', image)
        
        # Quit
        elif key == ord('q') or key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()
    
    if len(points) == 4:
        print("\n✅ Calibration complete!")
        print("\n📋 UPDATE YOUR view_transformer.py with these values:")
        print("-" * 70)
        print("self.pixel_vertices = np.array([")
        for p in points:
            print(f"    {p},")
        print("])")
        print("-" * 70)
    else:
        print(f"\n⚠️  Only {len(points)}/4 points selected. Run again to complete.")


if __name__ == '__main__':
    main()
