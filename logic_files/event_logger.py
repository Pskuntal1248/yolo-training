
import numpy as np
from logic_files.utils import measure_distance


class EventLogger:
    def __init__(self, fps=25):
        self.fps = fps
        self.events = []
        self.min_pass_frames = 3 
       
        self.min_pass_distance = 1.5  # 1.5 meters minimum
        
    def detect_pass_events(self, tracks):
       
        events = []
        current_event = None
        
        num_frames = len(tracks['players'])
        
        for frame_num in range(num_frames):
           
            ball_in_possession = False
            current_possessor = None
            current_team = None
          
            for player_id, track in tracks['players'][frame_num].items():
                if track.get('has_ball', False):
                    ball_in_possession = True
                    current_possessor = player_id
                    current_team = track.get('team')
                    break
            
          
            ball_pos = None
            ball_speed = 0
            if 1 in tracks['ball'][frame_num]:
                ball_pos = tracks['ball'][frame_num][1].get('position_transformed')
                ball_speed = tracks['ball'][frame_num][1].get('speed', 0)
            
           
            if current_event is None and not ball_in_possession:
                passer_id = None
                passer_team = None
                for prev_frame in range(max(0, frame_num - 5), frame_num):
                    for player_id, track in tracks['players'][prev_frame].items():
                        if track.get('has_ball', False):
                            passer_id = player_id
                            passer_team = track.get('team')
                            break
                    if passer_id:
                        break
                
               
                if passer_id:
                    current_event = {
                        'type': 'pass_attempt',
                        'start_frame': frame_num,
                        'passer_id': passer_id,
                        'passer_team': passer_team,
                        'ball_initial_speed': ball_speed,
                        'ball_trajectory': [],
                        'start_ball_pos': ball_pos
                    }
            
           
            if current_event and not ball_in_possession:
                if ball_pos:
                    current_event['ball_trajectory'].append({
                        'frame': frame_num,
                        'position': ball_pos,
                        'speed': ball_speed
                    })
            
        
            if current_event and ball_in_possession:
                duration_frames = frame_num - current_event['start_frame']
                
              
                pass_distance = 0
                if len(current_event['ball_trajectory']) > 0 and current_event.get('start_ball_pos'):
                    start_pos = current_event['start_ball_pos']
                    end_pos = current_event['ball_trajectory'][-1]['position']
                    pass_distance = measure_distance(start_pos, end_pos)
                
             
                is_valid_pass = (duration_frames >= self.min_pass_frames and 
                                pass_distance >= self.min_pass_distance)
                
                if is_valid_pass:
                    current_event['end_frame'] = frame_num
                    current_event['receiver_id'] = current_possessor
                    current_event['receiver_team'] = current_team
                    current_event['duration_seconds'] = duration_frames / self.fps
                    current_event['pass_distance'] = pass_distance
                    
                    # Classify the pass - same team = successful, different = interception
                    if current_team == current_event['passer_team']:
                        current_event['result'] = 'successful'
                    else:
                        current_event['result'] = 'failed_interception'
                        current_event['interceptor_id'] = current_possessor
                        current_event['interceptor_team'] = current_team
                    
                  
                    current_event['tactical_snapshot'] = self._get_tactical_snapshot(
                        tracks, current_event['start_frame']
                    )
                    
                    events.append(current_event)
                
               
                current_event = None
            
           
            if current_event and frame_num - current_event['start_frame'] > 75: 

                pass_distance = 0
                if len(current_event['ball_trajectory']) > 0 and current_event.get('start_ball_pos'):
                    start_pos = current_event['start_ball_pos']
                    end_pos = current_event['ball_trajectory'][-1]['position']
                    pass_distance = measure_distance(start_pos, end_pos)
                
                # Only record if ball actually traveled
                if pass_distance >= self.min_pass_distance:
                    current_event['end_frame'] = frame_num
                    current_event['result'] = 'failed_out_of_bounds'
                    current_event['duration_seconds'] = (frame_num - current_event['start_frame']) / self.fps
                    current_event['pass_distance'] = pass_distance
                    
                    # Get tactical snapshot
                    current_event['tactical_snapshot'] = self._get_tactical_snapshot(
                        tracks, current_event['start_frame']
                    )
                    
                    events.append(current_event)
                
                current_event = None
        
        self.events = events
        return events
    
    def _get_tactical_snapshot(self, tracks, frame_num):
        """
        Capture positions of all players and ball at a specific frame
        
        Args:
            tracks: Dictionary of tracked objects
            frame_num: Frame number to capture
            
        Returns:
            Dictionary with all entity positions
        """
        snapshot = {
            'frame': frame_num,
            'timestamp': f"{int(frame_num / self.fps // 3600):02d}:{int((frame_num / self.fps % 3600) // 60):02d}:{(frame_num / self.fps % 60):06.3f}",
            'players': [],
            'ball': None
        }
        
        for player_id, track in tracks['players'][frame_num].items():
            pos = track.get('position_transformed')
            if pos:
                player_data = {
                    'id': int(player_id),
                    'team': int(track.get('team', 0)),
                    'position': [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                    'speed': round(float(track.get('speed', 0)), 2),
                    'has_ball': track.get('has_ball', False)
                }
                snapshot['players'].append(player_data)
       
        if 1 in tracks['ball'][frame_num]:
            ball_pos = tracks['ball'][frame_num][1].get('position_transformed')
            if ball_pos:
                snapshot['ball'] = {
                    'position': [round(float(ball_pos[0]), 2), round(float(ball_pos[1]), 2)],
                    'speed': round(float(tracks['ball'][frame_num][1].get('speed', 0)), 2)
                }
        
        return snapshot
    
    def get_failed_passes(self):
        """
        Filter and return only failed pass events
        
        Returns:
            List of failed pass events (interceptions and out of bounds)
        """
        return [event for event in self.events 
                if event.get('result') in ['failed_interception', 'failed_out_of_bounds']]
    
    def export_events(self, output_path='pass_events.json'):
        """
        Export all detected events to JSON file
        
        Args:
            output_path: Path to save the JSON file
        """
        import json
        import numpy as np

        def _json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        
        # Prepare export data
        export_data = {
            'total_events': len(self.events),
            'successful_passes': len([e for e in self.events if e.get('result') == 'successful']),
            'failed_interceptions': len([e for e in self.events if e.get('result') == 'failed_interception']),
            'failed_out_of_bounds': len([e for e in self.events if e.get('result') == 'failed_out_of_bounds']),
            'events': self.events
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=_json_default)
        
        print(f"\n📊 Event Analysis:")
        print(f"   Total pass attempts: {export_data['total_events']}")
        print(f"   Successful: {export_data['successful_passes']}")
        print(f"   Failed (Interceptions): {export_data['failed_interceptions']}")
        print(f"   Failed (Out of Bounds): {export_data['failed_out_of_bounds']}")
        print(f"   Exported to: {output_path}")
    
    def export_failed_passes_for_ghost(self, output_path='failed_passes_for_ghost.json'):
        
        import json
        import numpy as np

        def _json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        
        failed_passes = self.get_failed_passes()
        
       
        ghost_data = {
            'video_info': {
                'fps': self.fps,
                'total_failed_passes': len(failed_passes)
            },
            'failed_passes': []
        }
        
        for idx, event in enumerate(failed_passes):
            ghost_event = {
                'event_id': idx + 1,
                'type': event['result'],
                'start_frame': int(event['start_frame']),
                'end_frame': int(event['end_frame']),
                'duration_seconds': float(event['duration_seconds']),
                'passer': {
                    'id': int(event['passer_id']) if event.get('passer_id') is not None else None,
                    'team': int(event['passer_team']) if event.get('passer_team') is not None else None
                },
                'tactical_snapshot': event['tactical_snapshot'],
                'pass_distance': float(event.get('pass_distance', 0) or 0),
                'ball_speed': float(event.get('ball_initial_speed', 0) or 0)
            }
            
            if event['result'] == 'failed_interception':
                ghost_event['interceptor'] = {
                    'id': event.get('interceptor_id'),
                    'team': event.get('interceptor_team')
                }
            
            ghost_data['failed_passes'].append(ghost_event)
        
        with open(output_path, 'w') as f:
            json.dump(ghost_data, f, indent=2, default=_json_default)
        
        print(f"\n👻 Ghost AI Data:")
        print(f"   Failed passes for analysis: {len(failed_passes)}")
        print(f"   💾 Exported to: {output_path}")
    
        for event in ghost_data['failed_passes']:
            print(f"\n   Event #{event['event_id']}:")
            print(f"      Type: {event['type']}")
            print(f"      Frame: {event['start_frame']} → {event['end_frame']}")
            print(f"      Passer: Player {event['passer']['id']} (Team {event['passer']['team']})")
            if 'interceptor' in event:
                print(f"      Interceptor: Player {event['interceptor']['id']} (Team {event['interceptor']['team']})")
            print(f"      Distance: {event['pass_distance']:.1f}m at {event['ball_speed']:.1f} km/h")
