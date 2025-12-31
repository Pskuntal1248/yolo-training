import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, calibration_file=None):
       
        self.pitch_length = 40.0   # Visible length in meters
        self.pitch_width = 28.0    # Visible width in meters
        
       
        self.pixel_vertices = np.array([
            [110, 1000],   # Bottom-left (near camera, left touchline)
            [350, 280],    # Top-left - moved RIGHT to reduce scaling
            [1550, 280],   # Top-right - moved LEFT to reduce scaling
            [1810, 900]    # Bottom-right (near camera, right touchline)
        ], dtype=np.float32)

        
        self.target_vertices = np.array([
            [0, self.pitch_width],
            [0, 0],
            [self.pitch_length, 0],
            [self.pitch_length, self.pitch_width]
        ], dtype=np.float32)
        
        # Load calibration if provided
        if calibration_file:
            self.load_calibration(calibration_file)

        # Compute transformation matrix
        self.perspective_matrix = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )
        
        # Compute inverse for validation
        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.target_vertices, self.pixel_vertices
        )
        
        # Validate calibration on init
        self.validate_calibration()

    def load_calibration(self, filepath):
        import json
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            self.pixel_vertices = np.array(config['pixel_vertices'], dtype=np.float32)
            if 'pitch_length' in config:
                self.pitch_length = config['pitch_length']
            if 'pitch_width' in config:
                self.pitch_width = config['pitch_width']
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")

    def validate_calibration(self):
        """Check if calibration produces realistic scale factors"""
        # Test points at different field positions
        test_points_meters = np.array([
            [0, 34],      # Left goal line, center
            [52.5, 34],   # Center spot
            [105, 34],    # Right goal line, center
            [52.5, 0],    # Center, top touchline
            [52.5, 68],   # Center, bottom touchline
        ], dtype=np.float32)
        
        # Transform to pixels and back
        test_pixels = self.meters_to_pixels(test_points_meters)
        back_to_meters = self.transform_points(test_pixels)
        
        # Check round-trip error
        error = np.mean(np.abs(back_to_meters - test_points_meters))
        if error > 0.5:
            print(f"Warning: Calibration round-trip error: {error:.2f}m")

    def meters_to_pixels(self, points_meters):
        """Convert meter coordinates back to pixels (for validation)"""
        if len(points_meters) == 0:
            return np.array([])
        points = np.array(points_meters, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.inverse_matrix)
        return transformed.reshape(-1, 2)

    def transform_points(self, points):
        """Transform pixel coordinates to meters"""
        if len(points) == 0:
            return np.array([])
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.perspective_matrix)
        return transformed.reshape(-1, 2)

    def get_scale_at_position(self, pixel_pos):
        """Get local scale factor (pixels per meter) at a position"""
        # Small offset for numerical differentiation
        eps = 5.0
        p = np.array(pixel_pos)
        
        # Transform nearby points
        points = np.array([
            p,
            p + [eps, 0],
            p + [0, eps]
        ], dtype=np.float32)
        
        meters = self.transform_points(points)
        
        # Calculate local scale
        dx_meters = np.linalg.norm(meters[1] - meters[0])
        dy_meters = np.linalg.norm(meters[2] - meters[0])
        
        scale_x = eps / dx_meters if dx_meters > 0 else 0
        scale_y = eps / dy_meters if dy_meters > 0 else 0
        
        return (scale_x + scale_y) / 2  # Average scale

    def add_transformed_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_dict in enumerate(object_tracks):
                if not frame_dict:
                    continue
                
                track_ids = list(frame_dict.keys())
                pixel_positions = [
                    frame_dict[tid]['position_adjusted'] for tid in track_ids
                ]

                transformed_points = self.transform_points(pixel_positions)

                for i, track_id in enumerate(track_ids):
                    pos_meters = transformed_points[i].tolist()
                    
                    # Validate position is within reasonable bounds
                    x, y = pos_meters
                    if x < -10 or x > 115 or y < -10 or y > 78:
                        # Position outside field + margin, likely tracking error
                        pos_meters = [max(0, min(105, x)), max(0, min(68, y))]
                    
                    tracks[object_type][frame_num][track_id]['position_transformed'] = pos_meters