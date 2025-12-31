
from logic_files.utils import read_video, save_video, get_video_fps
from trackers import Tracker
import cv2
import numpy as np
from logic_files.team_assigner import TeamAssigner
from logic_files.player_ball_assigner import PlayerBallAssigner
from logic_files.camera_movement_estimator import CameraMovementEstimator
from logic_files.view_transformer import ViewTransformer
from logic_files.speed_and_distance_estimator import SpeedAndDistance_Estimator
from logic_files.event_logger import EventLogger
import json


def validate_tracks(tracks):
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            assert 'bbox' in track, f"Missing bbox for player {player_id} at frame {frame_num}"
            assert 'position' in track, f"Missing position for player {player_id} at frame {frame_num}"


def export_tactical_data(tracks, output_path='tactical_data.json'):
  
    tactical_data = []
    
    for frame_num in range(len(tracks['players'])):
        # Find possession player (player with ball)
        possession_player_id = None
        for player_id, track in tracks['players'][frame_num].items():
            if track.get('has_ball', False):
                possession_player_id = player_id
                break
        
        # Format timestamp as HH:MM:SS.mmm
        total_seconds = frame_num / 24.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
        frame_data = {
            'frame': int(frame_num),
            'timestamp': timestamp,
            'ball': None,
            'possession_player_id': int(possession_player_id) if possession_player_id is not None else None,
            'entities': []
        }
        
        # Ball data
        if 1 in tracks['ball'][frame_num]:
            ball_track = tracks['ball'][frame_num][1]
            position_meters = ball_track.get('position_transformed', None)
            if position_meters:
                frame_data['ball'] = {
                    'x': round(float(position_meters[0]), 1),
                    'y': round(float(position_meters[1]), 1),
                    'z': 0.2  # Assuming ball is on ground
                }
        
        # Player/Referee/Goalkeeper data
        for player_id, track in tracks['players'][frame_num].items():
            position_meters = track.get('position_transformed', None)
            if position_meters:
                entity_data = {
                    'id': int(player_id),
                    'team': int(track.get('team')) if track.get('team') is not None else None,
                    'type': 'PLAYER',  # Default type
                    'x': round(float(position_meters[0]), 1),
                    'y': round(float(position_meters[1]), 1),
                    'speed_kmh': round(float(track.get('speed', 0)), 1),
                    'distance_m': round(float(track.get('distance', 0)), 1)
                }
                frame_data['entities'].append(entity_data)
        
        # Referee data
        for referee_id, track in tracks['referees'][frame_num].items():
            position_meters = track.get('position_transformed', None)
            if position_meters:
                entity_data = {
                    'id': int(referee_id) + 1000,  # Offset to avoid conflict with player IDs
                    'team': None,
                    'type': 'REFEREE',
                    'x': round(float(position_meters[0]), 1),
                    'y': round(float(position_meters[1]), 1),
                    'speed_kmh': round(float(track.get('speed', 0)), 1),
                    'distance_m': round(float(track.get('distance', 0)), 1)
                }
                frame_data['entities'].append(entity_data)
        
        tactical_data.append(frame_data)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(tactical_data, f, indent=2)


def generate_tactical_state_vector(tracks, frame_num):
   
    state_vector = []
    
    # Add all player positions (sorted by player_id for consistency)
    for player_id in sorted(tracks['players'][frame_num].keys()):
        pos = tracks['players'][frame_num][player_id].get('position_transformed')
        if pos is not None:
            state_vector.extend([pos[0], pos[1]])
        else:
            state_vector.extend([0, 0])  # Placeholder for missing data
    
    # Add ball position
    if 1 in tracks['ball'][frame_num]:
        ball_pos = tracks['ball'][frame_num][1].get('position_transformed')
        if ball_pos is not None:
            state_vector.extend([ball_pos[0], ball_pos[1]])
        else:
            state_vector.extend([0, 0])
    else:
        state_vector.extend([0, 0])
    
    return state_vector


def main():
    """
    Main pipeline for Tactical Ghost Phase 1
    Processes football match video and extracts structured tactical data
    """
    video_path = 'input_videos/test.mp4'
    video_fps = get_video_fps(video_path, default_fps=24.0)
    video_frames = read_video(video_path)

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Trim tracks to match video frames
    num_frames = len(video_frames)
    for obj_type in tracks:
        if len(tracks[obj_type]) > num_frames:
            tracks[obj_type] = tracks[obj_type][:num_frames]
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator - Uses transformed coordinates
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_window=5, frame_rate=video_fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1
    
    team_ball_control = np.array(team_ball_control)

    # Validate data integrity
    validate_tracks(tracks)

    # Export tactical data
    export_tactical_data(tracks, 'tactical_data.json')

    # Detect and log pass events
    event_logger = EventLogger(fps=24)
    events = event_logger.detect_pass_events(tracks)
    event_logger.export_events('pass_events.json')
    event_logger.export_failed_passes_for_ghost('fail_passes.json')

    # Generate sample state vector
    state_vector = generate_tactical_state_vector(tracks, 0)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi', fps=video_fps)


if __name__ == '__main__':
    main()
