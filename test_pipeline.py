import json
import numpy as np
import cv2
from logic_files.utils import read_video, save_video, get_video_fps
from trackers import Tracker
from logic_files.team_assigner import TeamAssigner
from logic_files.player_ball_assigner import PlayerBallAssigner
from logic_files.camera_movement_estimator import CameraMovementEstimator
from logic_files.view_transformer import ViewTransformer
from logic_files.speed_and_distance_estimator import SpeedAndDistance_Estimator
from logic_files.event_logger import EventLogger

class TacticalTestPipeline:
    def __init__(self, video_path, test_limit=200):
        self.video_path = video_path
        self.test_limit = test_limit
        self.fps = get_video_fps(video_path)
        
        
        self.tracker = Tracker('models/best.pt')
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = ViewTransformer()
        self.speed_estimator = SpeedAndDistance_Estimator(frame_window=10, frame_rate=self.fps)

    def run(self):
        # 1. Load and slice video
        print(f"--- Loading Video (Limit: {self.test_limit} frames) ---")
        video_frames = read_video(self.video_path)
        video_frames = video_frames[:self.test_limit]
        num_frames = len(video_frames)

        # 2. Object Tracking
        print("--- Tracking Objects ---")
        tracks = self.tracker.get_object_tracks(
            video_frames, 
            read_from_stub=True, 
            stub_path='stubs/track_stubs.pkl'
        )
        
        # Sync tracks with sliced frames
        for obj in tracks:
            tracks[obj] = tracks[obj][:num_frames]

        # 3. Handle Positions & Camera
        print("--- Adjusting for Camera Movement ---")
        self.tracker.add_position_to_tracks(tracks)
        
        cam_estimator = CameraMovementEstimator(video_frames[0])
        cam_movements = cam_estimator.get_camera_movement(
            video_frames, 
            read_from_stub=True, 
            stub_path='stubs/camera_movement_stub.pkl'
        )
        cam_movements = cam_movements[:num_frames]
        
        cam_estimator.add_adjust_positions_to_tracks(tracks, cam_movements)

        # 4. Transform to Real World (Meters)
        print("--- Transforming Perspective ---")
        self.view_transformer.add_transformed_position_to_tracks(tracks)

        # 5. Physics & Logistics
        print("--- Calculating Physics & Teams ---")
        tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
        self.speed_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # Team Assignment (Frame 0 reference)
        self.team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        self._process_teams_and_ball(video_frames, tracks)

        # 6. Visualization
        print("--- Drawing and Saving ---")
        team_control_array = np.array(self.team_control)
        output_frames = self.tracker.draw_annotations(video_frames, tracks, team_control_array)
        output_frames = cam_estimator.draw_camera_movement(output_frames, cam_movements)
        output_frames = self.speed_estimator.draw_speed_and_distance(output_frames, tracks)

        save_video(output_frames, 'output_videos/test_output.avi', fps=self.fps)
        print("Done! Check 'output_videos/test_output.avi'")

    def _process_teams_and_ball(self, frames, tracks):
        """Internal helper to handle frame-by-frame assignments"""
        self.team_control = []
        for i, frame in enumerate(frames):
            # Team Assignment
            for pid, track in tracks['players'][i].items():
                team = self.team_assigner.get_player_team(frame, track['bbox'], pid)
                tracks['players'][i][pid]['team'] = team
                tracks['players'][i][pid]['team_color'] = self.team_assigner.team_colors[team]
            
            # Ball Possession
            ball_bbox = tracks['ball'][i][1]['bbox']
            assigned_player = self.player_assigner.assign_ball_to_player(tracks['players'][i], ball_bbox)
            
            if assigned_player != -1:
                tracks['players'][i][assigned_player]['has_ball'] = True
                self.team_control.append(tracks['players'][i][assigned_player]['team'])
            else:
                last_team = self.team_control[-1] if self.team_control else 1
                self.team_control.append(last_team)

if __name__ == '__main__':
    pipeline = TacticalTestPipeline(
        video_path='input_videos/test.mp4', 
        test_limit=200
    )
    pipeline.run()