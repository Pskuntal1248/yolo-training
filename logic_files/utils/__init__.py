"""
Utility functions for Tactical Ghost football analysis
"""
import cv2
import pickle


def get_center_of_bbox(bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)


def get_bbox_width(bbox):
    """Get width of bounding box"""
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def measure_xy_distance(p1, p2):
    """Calculate x and y distance components"""
    return p1[0]-p2[0], p1[1]-p2[1]


def get_foot_position(bbox):
    """Get foot position (bottom center) of bounding box"""
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)


def read_video(video_path):
    """Read video file and return frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=24):
    """Save frames to video file"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), 
                         (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def get_video_fps(video_path, default_fps=24.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return float(default_fps)
    return float(fps)


def read_stub(stub_path):
    """Read pickled stub data"""
    with open(stub_path, 'rb') as f:
        return pickle.load(f)


def save_stub(data, stub_path):
    """Save data as pickle stub"""
    with open(stub_path, 'wb') as f:
        pickle.dump(data, f)
