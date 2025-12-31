"""
kalman filter we are calculating speed
"""
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from logic_files.utils import measure_distance, get_foot_position


class AdaptivePlayerKalmanFilter:
    
    def __init__(self, initial_pos, dt=1/24):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
     
        self.kf.x = np.array([initial_pos[0], initial_pos[1], 0., 0.])
        
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
       
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
      
        self.base_R = 2.0  # Balanced - responsive but still smooth
        self.kf.R = np.eye(2) * self.base_R
        
        # Process noise - higher = allow more velocity changes  
        self.base_Q = 0.1  # Allow velocity to change faster
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=self.base_Q, block_size=2)
        
        # Initial covariance
        self.kf.P = np.eye(4) * 5
        
        self.total_distance = 0.0
        self.last_filtered_pos = np.array(initial_pos)
        self.velocity_history = []
        self.position_history = []
        self.max_history = 15
        
       
        self.max_acceleration = 3.0  
        self.max_speed_ms = 7.0     
        self.min_movement = 0.02     

    def adapt_noise(self):
    
        if len(self.velocity_history) < 3:
            return
            
        recent_velocities = np.array(self.velocity_history[-5:])
        speeds = np.linalg.norm(recent_velocities, axis=1)
        speed_variance = np.var(speeds)
        mean_speed = np.mean(speeds)
        
      
        if mean_speed < 0.3:  # < 1 km/h
            self.kf.R = np.eye(2) * (self.base_R * 1.5)  # Slight smoothing increase
            self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.base_Q * 0.5, block_size=2)
    
        elif speed_variance > 1.0:
            self.kf.R = np.eye(2) * (self.base_R * 0.7)  # More responsive
            self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.base_Q * 1.5, block_size=2)
        else:
            self.kf.R = np.eye(2) * self.base_R
            self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.base_Q, block_size=2)

    def apply_physics_constraints(self):
       
        velocity = self.kf.x[2:4]
        speed = np.linalg.norm(velocity)

        if speed > self.max_speed_ms:
            scale = self.max_speed_ms / speed
            self.kf.x[2:4] *= scale
        
       
        if len(self.velocity_history) > 0:
            prev_velocity = self.velocity_history[-1]
            acceleration = np.linalg.norm(velocity - prev_velocity) / self.dt
            if acceleration > self.max_acceleration:
              
                direction = (velocity - prev_velocity)
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    max_change = self.max_acceleration * self.dt
                    self.kf.x[2:4] = prev_velocity + direction * max_change

    def update(self, measured_pos):
      
        self.adapt_noise()
        
       
        self.kf.predict()
        
      
        self.apply_physics_constraints()
        
      
        self.kf.update(np.array(measured_pos))
        
       
        self.apply_physics_constraints()
        
        filtered_pos = self.kf.x[:2].copy()
        velocity = self.kf.x[2:4].copy()
        
       
        self.velocity_history.append(velocity.copy())
        self.position_history.append(filtered_pos.copy())
        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)
            self.position_history.pop(0)
        
       
        dist_inc = np.linalg.norm(filtered_pos - self.last_filtered_pos)
        if dist_inc > self.min_movement:
            self.total_distance += dist_inc
        
        self.last_filtered_pos = filtered_pos.copy()
        
     
        if len(self.velocity_history) >= 3:   
            recent_speeds = [np.linalg.norm(v) for v in self.velocity_history[-3:]]
        
            speed_ms = np.median(recent_speeds)
        else:
            speed_ms = np.linalg.norm(velocity) * 0.7  # Less dampening
        
        speed_kmh = speed_ms * 3.6
        
        return filtered_pos, speed_kmh, self.total_distance


class SpeedAndDistance_Estimator:
    def __init__(self, frame_window=15, frame_rate=24):
        self.frame_rate = float(frame_rate)
        self.dt = 1.0 / self.frame_rate
        self.max_speed = 20.0  # km/h - strict cap
        self.min_speed_display = 1.5  # km/h - below this show 0
        self.kalman_filters = {}
        
        # Outlier detection
        self.global_speed_history = []
        self.outlier_threshold = 1.8  # Very strict outlier detection

    def detect_outlier_speed(self, speed):
        """Check if speed is an outlier compared to recent history"""
        if len(self.global_speed_history) < 20:
            return False
        
        mean_speed = np.mean(self.global_speed_history[-50:])
        std_speed = np.std(self.global_speed_history[-50:])
        
        if std_speed > 0:
            z_score = abs(speed - mean_speed) / std_speed
            return z_score > self.outlier_threshold
        return False

    def add_speed_and_distance_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            if object_type in {"ball", "referees"}:
                continue

            if object_type not in self.kalman_filters:
                self.kalman_filters[object_type] = {}

            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    pos = track_info.get("position_transformed")
                    
                    if pos is None:
                        track_info["speed"] = 0.0
                        track_info["distance"] = 0.0
                        continue

                    # Initialize filter for new tracks
                    if track_id not in self.kalman_filters[object_type]:
                        self.kalman_filters[object_type][track_id] = AdaptivePlayerKalmanFilter(pos, self.dt)
                        track_info["speed"] = 0.0
                        track_info["distance"] = 0.0
                        track_info["position_filtered"] = pos
                        continue

                    # Update existing filter
                    kf = self.kalman_filters[object_type][track_id]
                    filtered_pos, speed_kmh, total_dist = kf.update(pos)
                    
                    # Apply speed constraints
                    if speed_kmh > self.max_speed:
                        speed_kmh = self.max_speed * 0.9  # Soft cap
                    
                    if speed_kmh < self.min_speed_display:
                        speed_kmh = 0.0
                    
                    # Outlier detection
                    if self.detect_outlier_speed(speed_kmh) and speed_kmh > 25:
                        speed_kmh = np.mean(self.global_speed_history[-10:]) if self.global_speed_history else 0.0
                    
                    # Track global speed for outlier detection
                    if speed_kmh > 0:
                        self.global_speed_history.append(speed_kmh)
                        if len(self.global_speed_history) > 100:
                            self.global_speed_history.pop(0)

                    track_info["speed"] = round(float(speed_kmh), 1)
                    track_info["distance"] = round(float(total_dist), 1)
                    track_info["position_filtered"] = filtered_pos.tolist()

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type == "ball" or object_type == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')
                    if speed is None or distance is None:
                        continue

                    bbox = track_info['bbox']
                    position = get_foot_position(bbox)
                    position = list(position)
                    position[1] += 40
                    position = tuple(map(int, position))
                    
                    cv2.putText(frame, f"{speed:.1f} km/h", position, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.1f} m", (position[0], position[1]+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)

        return output_frames
