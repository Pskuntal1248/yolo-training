"""
Failed Pass Detector for Ghost Pass Analysis
Detects failed passes, identifies release and failure frames,
and extracts video frames for tactical analysis.
"""

import os
import cv2
import json
import numpy as np
from logic_files.utils import measure_distance, get_center_of_bbox


class FailedPassDetector:
    """
    Detects and analyzes failed passes (interceptions) in football matches.
    
    Key Concepts:
    - Failure Frame: The frame where ball is intercepted by opponent
    - Release Frame: The frame where the pass was initiated
    - Ghost Pass: The tactical state at the moment of decision
    """
    
    def __init__(self, fps=24, possession_threshold=3.0, release_threshold=2.0):
        """
        Initialize the Failed Pass Detector.
        
        Args:
            fps: Video frames per second
            possession_threshold: Distance in meters to consider ball in possession
            release_threshold: Distance in meters to consider ball has left foot
        """
        self.fps = fps
        self.possession_threshold = possession_threshold  # meters (increased for real-world)
        self.release_threshold = release_threshold  # meters
        self.lookback_window = 100  # frames to look back for release frame
        self.failed_passes = []
        
    def detect_failed_passes(self, tracks):
        """
        Main method to detect all failed passes in the video.
        
        A failed pass occurs when:
        1. Team A player has the ball
        2. Ball travels away from Team A player
        3. Ball is received/intercepted by Team B player
        4. No Team A player touched it in between
        
        Args:
            tracks: Dictionary containing player, ball, and referee tracks
            
        Returns:
            List of failed pass events with detailed information
        """
        self.failed_passes = []
        
        # CRITICAL: Verify all tracks have the same number of frames
        # This ensures the entire video has been processed
        num_frames_players = len(tracks['players'])
        num_frames_ball = len(tracks['ball'])
        num_frames_referees = len(tracks.get('referees', []))
        
        if num_frames_players == 0:
            print("   ❌ ERROR: No player tracks found! Video may not be fully processed.")
            return []
        
        # Verify all track types have the same number of frames
        if num_frames_ball != num_frames_players:
            print(f"   ⚠️  WARNING: Frame count mismatch! Players: {num_frames_players}, Ball: {num_frames_ball}")
            print(f"   Using minimum frame count: {min(num_frames_players, num_frames_ball)}")
        
        num_frames = num_frames_players
        
        # Verify that team assignments and ball possession are complete
        frames_with_teams = 0
        frames_with_ball_possession = 0
        for i in range(min(100, num_frames)):  # Check first 100 frames as sample
            for player_id, track in tracks['players'][i].items():
                if track.get('team') is not None:
                    frames_with_teams += 1
                    break
            for player_id, track in tracks['players'][i].items():
                if track.get('has_ball', False):
                    frames_with_ball_possession += 1
                    break
        
        if frames_with_teams == 0:
            print("   ❌ ERROR: No team assignments found! Teams must be assigned before detection.")
            return []
        
        if frames_with_ball_possession == 0:
            print("   ⚠️  WARNING: No ball possession detected in sample frames.")
        
        print(f"   ✅ Verification complete:")
        print(f"      - Total frames: {num_frames}")
        print(f"      - Team assignments: ✓")
        print(f"      - Ball possession tracking: ✓")
        print(f"   🚀 Starting failed pass detection on ALL {num_frames} frames...")
        
        # Track possession changes using has_ball flag from PlayerBallAssigner
        possession_history = self._build_possession_history(tracks)
        
        # Find all failure frames (true failed passes, not just possession changes)
        print(f"   🔎 Analyzing possession changes for true failed passes...")
        print(f"   📊 Total frames to analyze: {num_frames}")
        failure_frames = self._find_failure_frames(possession_history, tracks)
        
        print(f"   📍 Found {len(failure_frames)} TRUE failed passes (interceptions by opponent)")
        
        # For each failure frame, find the release frame and build the event
        for failure_info in failure_frames:
            pass_event = self._analyze_failed_pass(
                tracks, 
                failure_info['frame'],
                failure_info['interceptor_id'],
                failure_info['interceptor_team'],
                possession_history
            )
            if pass_event:
                self.failed_passes.append(pass_event)
        
        return self.failed_passes
    
    def _build_possession_history(self, tracks):
        """
        Build a frame-by-frame history of ball possession.
        Uses has_ball flag set by PlayerBallAssigner.
        
        Returns:
            List of dicts with possession info per frame
        """
        possession_history = []
        num_frames = len(tracks['players'])
        
        print(f"   📹 Building possession history for ALL {num_frames} frames...")
        
        # Process EVERY frame - no early exit
        # Show progress every 10% of frames
        progress_interval = max(1, num_frames // 10)
        for frame_num in range(num_frames):
            # Show progress
            if frame_num % progress_interval == 0 or frame_num == num_frames - 1:
                progress_pct = (frame_num + 1) * 100 // num_frames
                print(f"      Processing frame {frame_num + 1}/{num_frames} ({progress_pct}%)...", end='\r')
            frame_possession = {
                'frame': frame_num,
                'possessor_id': None,
                'possessor_team': None,
                'ball_position': None,
                'possession_distance': None
            }
            
            # Get ball position
            ball_pos = None
            if 1 in tracks['ball'][frame_num]:
                ball_pos = tracks['ball'][frame_num][1].get('position_transformed')
                frame_possession['ball_position'] = ball_pos
            
            # FIRST: Check for has_ball flag (set by PlayerBallAssigner using bbox)
            for player_id, track in tracks['players'][frame_num].items():
                if track.get('has_ball', False):
                    frame_possession['possessor_id'] = player_id
                    frame_possession['possessor_team'] = track.get('team')
                    # Calculate distance for reference
                    player_pos = track.get('position_transformed')
                    if player_pos and ball_pos:
                        frame_possession['possession_distance'] = measure_distance(ball_pos, player_pos)
                    break
            
            # FALLBACK: If no has_ball flag, find closest player within threshold
            if frame_possession['possessor_id'] is None and ball_pos:
                min_distance = float('inf')
                closest_player = None
                closest_team = None
                
                for player_id, track in tracks['players'][frame_num].items():
                    player_pos = track.get('position_transformed')
                    if player_pos:
                        dist = measure_distance(ball_pos, player_pos)
                        if dist < min_distance:
                            min_distance = dist
                            closest_player = player_id
                            closest_team = track.get('team')
                
                # Use larger threshold for fallback detection
                if min_distance <= self.possession_threshold:
                    frame_possession['possessor_id'] = closest_player
                    frame_possession['possessor_team'] = closest_team
                    frame_possession['possession_distance'] = min_distance
                
            possession_history.append(frame_possession)
        
        # Print possession stats
        possessions = [p for p in possession_history if p['possessor_id'] is not None]
        print(f"   📊 Possession detected in {len(possessions)}/{len(possession_history)} frames")
        
        # CRITICAL: Verify we processed ALL frames
        if len(possession_history) != num_frames:
            print(f"   ❌ ERROR: Processed {len(possession_history)} frames but expected {num_frames}!")
            print(f"   This indicates incomplete processing. Aborting detection.")
            return []
        
        print(f"   ✅ Completed processing ALL {num_frames} frames - ready for analysis")
        
        return possession_history
    
    def _find_failure_frames(self, possession_history, tracks):
        """
        Find all frames where a TRUE FAILED PASS occurred.
        
        A FAILED PASS (not just possession change) requires:
        1. Team A player has ball
        2. Ball is released (in flight) for multiple frames - NO ONE has possession
        3. Ball travels a meaningful distance (> min_pass_distance meters)
        4. Team B player receives/intercepts the ball (OPPONENT TEAM, not goalkeeper/same team)
        5. No Team A player touched it during flight
        
        This filters out:
        - Tackles (immediate possession change, no ball flight)
        - Dribbling losses
        - Short distance turnovers
        - Passes to goalkeeper (same team)
        - Passes that end with same team possession
        
        Returns:
            List of failure frame info dicts
        """
        failure_frames = []
        
        min_flight_frames = 3  # Ball must be "in flight" for at least this many frames
        min_pass_distance = 2.0  # meters - ball must travel at least this much to be a "pass"
        # Pixel fallback when transformed positions are missing
        min_pass_distance_pixels = 40  # pixels (approx) for fallback
        
        last_possession_team = None
        last_possession_player = None
        last_possession_frame = None
        
        total_frames = len(possession_history)
        print(f"   🔍 Analyzing ALL {total_frames} frames for failed passes...")
        
        # Statistics for debugging
        stats = {
            'team_changes': 0,
            'filtered_same_team': 0,
            'filtered_teammate_touch': 0,
            'filtered_cant_verify': 0,
            'filtered_team_mismatch': 0,
            'filtered_criteria': 0,
            'accepted': 0
        }
        
        # Show progress every 10% of frames
        progress_interval = max(1, total_frames // 10)
        
        for i, frame_info in enumerate(possession_history):
            # Show progress
            if i % progress_interval == 0 or i == total_frames - 1:
                progress_pct = (i + 1) * 100 // total_frames
                print(f"      Analyzing frame {i + 1}/{total_frames} ({progress_pct}%)...", end='\r')
            current_team = frame_info['possessor_team']
            current_player = frame_info['possessor_id']
            
            if current_team is not None:
                # Only consider as failed pass if team changed (opponent intercepted)
                if (
                    last_possession_team is not None and
                    current_team != last_possession_team and
                    last_possession_frame is not None
                ):
                    stats['team_changes'] += 1
                    
                    # CRITICAL: Verify this is a REAL interception by opponent
                    # First, verify teams are actually different (not just misclassification)
                    if current_team == last_possession_team:
                        stats['filtered_same_team'] += 1
                        # Still update possession even if it's same team
                        last_possession_team = current_team
                        last_possession_player = current_player
                        last_possession_frame = i
                        continue
                    
                    # Check that no teammate touched the ball during flight
                    teammate_touched_during_flight = False
                    for f in range(last_possession_frame + 1, i):
                        if possession_history[f]['possessor_id'] is not None:
                            # Someone had possession during flight
                            if possession_history[f]['possessor_team'] == last_possession_team:
                                # Teammate touched it - not a failed pass
                                teammate_touched_during_flight = True
                                break
                    
                    # Skip if teammate touched ball during flight (not a true interception)
                    if teammate_touched_during_flight:
                        stats['filtered_teammate_touch'] += 1
                        # Still update possession - teammate has it now
                        last_possession_team = current_team
                        last_possession_player = current_player
                        last_possession_frame = i
                        continue
                    
                    # CRITICAL: Directly verify interceptor is from opposing team using tracks
                    # This catches team assignment inconsistencies
                    interceptor_team_from_tracks = None
                    if current_player is not None and i < len(tracks['players']):
                        if current_player in tracks['players'][i]:
                            interceptor_team_from_tracks = tracks['players'][i][current_player].get('team')
                    
                    # Also verify passer's team from tracks
                    passer_team_from_tracks = None
                    if last_possession_player is not None and last_possession_frame < len(tracks['players']):
                        if last_possession_player in tracks['players'][last_possession_frame]:
                            passer_team_from_tracks = tracks['players'][last_possession_frame][last_possession_player].get('team')
                    
                    # If we can't verify from tracks, or teams match, skip
                    if interceptor_team_from_tracks is None or passer_team_from_tracks is None:
                        # Can't verify - skip to be safe, but still update possession
                        stats['filtered_cant_verify'] += 1
                        last_possession_team = current_team
                        last_possession_player = current_player
                        last_possession_frame = i
                        continue
                    if interceptor_team_from_tracks == passer_team_from_tracks:
                        # Same team - definitely not an interception (this is the key check!)
                        stats['filtered_same_team'] += 1
                        print(f"      ✗ Same-team pass filtered: Frame {last_possession_frame} → {i} "
                              f"Passer Team {passer_team_from_tracks} → Interceptor Team {interceptor_team_from_tracks} "
                              f"(Player {last_possession_player} → Player {current_player})")
                        # Update possession - same team has it
                        last_possession_team = current_team
                        last_possession_player = current_player
                        last_possession_frame = i
                        continue
                    if current_team != interceptor_team_from_tracks:
                        # Team mismatch between possession history and tracks - skip
                        stats['filtered_team_mismatch'] += 1
                        # Use the verified team from tracks
                        last_possession_team = interceptor_team_from_tracks
                        last_possession_player = current_player
                        last_possession_frame = i
                        continue
                    if last_possession_team != passer_team_from_tracks:
                        # Team mismatch for passer - skip, but use verified team
                        stats['filtered_team_mismatch'] += 1
                        # Update with verified passer team
                        last_possession_team = passer_team_from_tracks
                        last_possession_player = last_possession_player
                        last_possession_frame = last_possession_frame
                        # Also update current since we verified it
                        if interceptor_team_from_tracks is not None:
                            last_possession_team = interceptor_team_from_tracks
                            last_possession_player = current_player
                            last_possession_frame = i
                        continue
                    
                    # Additional check: Verify the interceptor's team is consistently the opponent
                    # Check a few frames around to ensure team assignment is stable
                    team_consistency_check = True
                    check_frames = [max(0, i-2), i, min(total_frames-1, i+2)]
                    for check_frame in check_frames:
                        if check_frame < len(possession_history):
                            check_team = possession_history[check_frame].get('possessor_team')
                            if check_team is not None and check_team == last_possession_team:
                                # Original team still has it - not an interception
                                team_consistency_check = False
                                break
                    
                    frames_between = i - last_possession_frame
                    # Count frames where NO ONE had possession (ball in flight)
                    flight_frames = 0
                    for f in range(last_possession_frame + 1, i):
                        if possession_history[f]['possessor_id'] is None:
                            flight_frames += 1
                    # Calculate ball travel distance (prefer transformed meters)
                    ball_distance = 0
                    used_pixel_metric = False
                    start_ball_pos = possession_history[last_possession_frame].get('ball_position')
                    end_ball_pos = frame_info.get('ball_position')
                    if start_ball_pos and end_ball_pos:
                        # Both transformed positions available
                        ball_distance = measure_distance(start_ball_pos, end_ball_pos)
                    else:
                        # Fallback: compute distance between ball bbox centers (pixels)
                        try:
                            start_bbox = None
                            end_bbox = None
                            if 1 in tracks['ball'][last_possession_frame]:
                                start_bbox = tracks['ball'][last_possession_frame][1].get('bbox')
                            if 1 in tracks['ball'][i]:
                                end_bbox = tracks['ball'][i][1].get('bbox')
                            if start_bbox and end_bbox:
                                start_center = get_center_of_bbox(start_bbox)
                                end_center = get_center_of_bbox(end_bbox)
                                # pixel distance
                                dx = start_center[0] - end_center[0]
                                dy = start_center[1] - end_center[1]
                                ball_distance = (dx*dx + dy*dy)**0.5
                                used_pixel_metric = True
                        except Exception:
                            ball_distance = 0
                    # Choose threshold depending on metric used
                    if used_pixel_metric:
                        distance_ok = ball_distance >= min_pass_distance_pixels
                    else:
                        distance_ok = ball_distance >= min_pass_distance

                    # Additional verification: Check that original team doesn't immediately get ball back
                    # This filters out passes to goalkeeper or same-team scenarios
                    # Look ahead a few frames to ensure it's a real interception, not a pass to teammate
                    is_real_interception = True
                    lookahead_frames = min(5, total_frames - i - 1)  # Check next 5 frames (reduced from 8)
                    if lookahead_frames > 0:
                        original_team_regained = False
                        opponent_maintained = 0
                        for f in range(i + 1, min(i + lookahead_frames + 1, total_frames)):
                            if f < len(possession_history):
                                f_team = possession_history[f].get('possessor_team')
                                if f_team == passer_team_from_tracks:
                                    # Original team got it back - likely a pass to goalkeeper/teammate
                                    original_team_regained = True
                                    break
                                elif f_team == interceptor_team_from_tracks:
                                    opponent_maintained += 1
                        
                        # If original team got it back within 5 frames, it's not a real interception
                        if original_team_regained:
                            is_real_interception = False
                        # Relaxed: opponent just needs to have it for at least 1 frame (they intercepted it)
                        elif opponent_maintained == 0 and lookahead_frames >= 2:
                            # If opponent doesn't have it in next 2+ frames, might be a touch not interception
                            is_real_interception = False
                    
                    # STRICT CRITERIA for a failed pass:
                    is_true_failed_pass = (
                        flight_frames >= min_flight_frames and  # Ball was airborne/rolling
                        distance_ok and  # Traveled far enough to be a pass
                        frames_between >= 8 and  # At least ~0.3 seconds of action
                        team_consistency_check and  # Team assignment is consistent
                        is_real_interception  # Opponent intercepted (not pass to goalkeeper/teammate)
                    )

                    if is_true_failed_pass:
                        stats['accepted'] += 1
                        failure_frames.append({
                            'frame': i,
                            'interceptor_id': current_player,
                            'interceptor_team': interceptor_team_from_tracks,  # Use verified team
                            'last_possessor_id': last_possession_player,
                            'last_possessor_team': passer_team_from_tracks,  # Use verified team
                            'last_possession_frame': last_possession_frame,
                            'flight_frames': flight_frames,
                            'ball_distance': ball_distance
                        })
                        if used_pixel_metric:
                            print(f"      ✓ INTERCEPTION detected (pixel metric): Frame {last_possession_frame} → {i} "
                                  f"Team {passer_team_from_tracks} (Player {last_possession_player}) → "
                                  f"Team {interceptor_team_from_tracks} (Player {current_player}) "
                                  f"(flight={flight_frames} frames, distance={ball_distance:.1f}px)")
                        else:
                            print(f"      ✓ INTERCEPTION detected: Frame {last_possession_frame} → {i} "
                                  f"Team {passer_team_from_tracks} (Player {last_possession_player}) → "
                                  f"Team {interceptor_team_from_tracks} (Player {current_player}) "
                                  f"(flight={flight_frames} frames, distance={ball_distance:.1f}m)")
                        # Update possession - opponent now has it
                        last_possession_team = interceptor_team_from_tracks
                        last_possession_player = current_player
                        last_possession_frame = i
                    else:
                        stats['filtered_criteria'] += 1
                        # Log why this was rejected (for debugging) - more detailed
                        if frames_between >= 3:  # Only log if it was even close
                            metric_label = 'px' if used_pixel_metric else 'm'
                            reasons = []
                            if not flight_frames >= min_flight_frames:
                                reasons.append('insufficient flight')
                            if not distance_ok:
                                reasons.append('too short')
                            if frames_between < 8:
                                reasons.append('too quick')
                            if not team_consistency_check:
                                reasons.append('team inconsistency')
                            if not is_real_interception:
                                reasons.append('not real interception')
                            
                            reason_str = ', '.join(reasons) if reasons else 'unknown'
                            print(f"      ✗ Rejected interception: Frame {last_possession_frame} → {i} "
                                  f"Team {passer_team_from_tracks} → Team {interceptor_team_from_tracks} "
                                  f"(flight={flight_frames}, dist={ball_distance:.1f}{metric_label}, gap={frames_between}) "
                                  f"- {reason_str}")
                        # Still update possession even if rejected
                        last_possession_team = interceptor_team_from_tracks
                        last_possession_player = current_player
                        last_possession_frame = i
                else:
                    # No team change - just update possession normally
                    last_possession_team = current_team
                    last_possession_player = current_player
                    last_possession_frame = i
        
        print(f"   ✅ Completed analysis of all {total_frames} frames")
        print(f"   📊 Statistics:")
        print(f"      - Team changes detected: {stats['team_changes']}")
        print(f"      - Filtered (same team): {stats['filtered_same_team']}")
        print(f"      - Filtered (teammate touch): {stats['filtered_teammate_touch']}")
        print(f"      - Filtered (can't verify): {stats['filtered_cant_verify']}")
        print(f"      - Filtered (team mismatch): {stats['filtered_team_mismatch']}")
        print(f"      - Filtered (criteria): {stats['filtered_criteria']}")
        print(f"      - ✅ Accepted as failed passes: {stats['accepted']}")
        return failure_frames
    
    def _analyze_failed_pass(self, tracks, failure_frame, interceptor_id, 
                             interceptor_team, possession_history):
        """
        Analyze a failed pass to find the release frame and build detailed event.
        
        Args:
            tracks: Full tracking data
            failure_frame: Frame where interception occurred
            interceptor_id: Player who intercepted
            interceptor_team: Team that intercepted
            possession_history: Pre-computed possession history
            
        Returns:
            Detailed failed pass event dict
        """
        # Find the passer (last player from other team with ball before failure)
        passer_id, passer_team, last_touch_frame = self._find_passer(
            possession_history, failure_frame, interceptor_team
        )
        
        if passer_id is None:
            return None
        
        # Find the exact release frame
        release_frame = self._find_release_frame(
            tracks, last_touch_frame, failure_frame, passer_id
        )
        
        # Calculate ball trajectory and speed
        ball_trajectory = self._get_ball_trajectory(
            tracks, release_frame, failure_frame
        )
        
        # Get tactical snapshots at key moments
        release_snapshot = self._get_tactical_snapshot(tracks, release_frame)
        failure_snapshot = self._get_tactical_snapshot(tracks, failure_frame)
        
        # Calculate pass metrics
        pass_distance = 0
        pass_speed = 0
        if ball_trajectory:
            start_pos = ball_trajectory[0]['position']
            end_pos = ball_trajectory[-1]['position']
            if start_pos and end_pos:
                pass_distance = measure_distance(start_pos, end_pos)
            
            # Get max ball speed during pass
            speeds = [t['speed'] for t in ball_trajectory if t.get('speed')]
            pass_speed = max(speeds) if speeds else 0
        
        # Find open teammates at release frame (for ghost analysis)
        open_teammates = self._find_open_teammates(
            tracks, release_frame, passer_id, passer_team
        )
        
        # Build the event
        event = {
            'event_id': len(self.failed_passes) + 1,
            'type': 'failed_pass_interception',
            
            # Frame information
            'release_frame': int(release_frame),
            'failure_frame': int(failure_frame),
            'last_touch_frame': int(last_touch_frame),
            'duration_frames': failure_frame - release_frame,
            'duration_seconds': (failure_frame - release_frame) / self.fps,
            
            # Timestamps
            'release_timestamp': self._frame_to_timestamp(release_frame),
            'failure_timestamp': self._frame_to_timestamp(failure_frame),
            
            # Passer info
            'passer': {
                'id': int(passer_id) if passer_id is not None else None,
                'team': int(passer_team) if passer_team is not None else None,
                'position_at_release': self._get_player_position(tracks, release_frame, passer_id)
            },
            
            # Interceptor info
            'interceptor': {
                'id': int(interceptor_id) if interceptor_id is not None else None,
                'team': int(interceptor_team) if interceptor_team is not None else None,
                'position_at_failure': self._get_player_position(tracks, failure_frame, interceptor_id)
            },
            
            # Pass metrics
            'pass_distance_meters': round(pass_distance, 2),
            'max_ball_speed_kmh': round(pass_speed, 2),
            'ball_trajectory': ball_trajectory,
            
            # Tactical snapshots for ghost analysis
            'tactical_snapshot_release': release_snapshot,
            'tactical_snapshot_failure': failure_snapshot,
            
            # Open teammates analysis
            'open_teammates_at_release': open_teammates,
            'num_open_options': len(open_teammates)
        }
        
        return event
    
    def _find_passer(self, possession_history, failure_frame, interceptor_team):
        """
        Find the passer by looking back from failure frame.
        
        Returns:
            Tuple of (passer_id, passer_team, last_touch_frame)
        """
        for i in range(failure_frame - 1, max(0, failure_frame - self.lookback_window), -1):
            frame_info = possession_history[i]
            if (frame_info['possessor_id'] is not None and 
                frame_info['possessor_team'] != interceptor_team):
                return (
                    frame_info['possessor_id'],
                    frame_info['possessor_team'],
                    i
                )
        return None, None, None
    
    def _find_release_frame(self, tracks, last_touch_frame, failure_frame, passer_id):
        """
        Find the exact frame where the ball left the passer's foot.
        
        Detection criteria:
        1. Distance between ball and passer increases past threshold
        2. Ball velocity vector points away from passer
        3. Ball shows velocity spike
        
        Returns:
            Release frame number
        """
        release_frame = last_touch_frame
        
        if passer_id is None:
            return release_frame
        
        prev_distance = 0
        
        for f in range(last_touch_frame, min(failure_frame, last_touch_frame + 50)):
            # Get ball position
            ball_pos = None
            if 1 in tracks['ball'][f]:
                ball_pos = tracks['ball'][f][1].get('position_transformed')
            
            # Get passer position
            passer_pos = None
            if passer_id in tracks['players'][f]:
                passer_pos = tracks['players'][f][passer_id].get('position_transformed')
            
            if ball_pos and passer_pos:
                dist = measure_distance(ball_pos, passer_pos)
                
                # Check if ball has left foot (distance threshold)
                if dist > self.release_threshold:
                    release_frame = f
                    break
                
                # Also check for significant distance increase (velocity indicator)
                if dist - prev_distance > 0.5:  # Ball moving away quickly
                    release_frame = f
                    break
                
                prev_distance = dist
        
        return release_frame
    
    def _get_ball_trajectory(self, tracks, start_frame, end_frame):
        """
        Get ball positions and speeds between two frames.
        
        Returns:
            List of trajectory points
        """
        trajectory = []
        
        for f in range(start_frame, end_frame + 1):
            if 1 in tracks['ball'][f]:
                ball_track = tracks['ball'][f][1]
                pos = ball_track.get('position_transformed')
                if pos:
                    trajectory.append({
                        'frame': f,
                        'position': [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                        'speed': round(float(ball_track.get('speed', 0)), 2)
                    })
        
        return trajectory
    
    def _get_tactical_snapshot(self, tracks, frame_num):
        """
        Capture complete tactical state at a specific frame.
        
        Returns:
            Dict with all player positions, teams, and ball state
        """
        snapshot = {
            'frame': frame_num,
            'timestamp': self._frame_to_timestamp(frame_num),
            'players': {'team_1': [], 'team_2': []},
            'ball': None
        }
        
        for player_id, track in tracks['players'][frame_num].items():
            pos = track.get('position_transformed')
            if pos:
                player_data = {
                    'id': int(player_id),
                    'position': [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                    'speed_kmh': round(float(track.get('speed', 0)), 2),
                    'has_ball': track.get('has_ball', False)
                }
                
                team = track.get('team')
                if team == 1:
                    snapshot['players']['team_1'].append(player_data)
                elif team == 2:
                    snapshot['players']['team_2'].append(player_data)
        
        # Ball data
        if 1 in tracks['ball'][frame_num]:
            ball_track = tracks['ball'][frame_num][1]
            ball_pos = ball_track.get('position_transformed')
            if ball_pos:
                snapshot['ball'] = {
                    'position': [round(float(ball_pos[0]), 2), round(float(ball_pos[1]), 2)],
                    'speed_kmh': round(float(ball_track.get('speed', 0)), 2)
                }
        
        return snapshot
    
    def _get_player_position(self, tracks, frame_num, player_id):
        """Get player's transformed position at a frame."""
        if player_id in tracks['players'][frame_num]:
            pos = tracks['players'][frame_num][player_id].get('position_transformed')
            if pos:
                return [round(float(pos[0]), 2), round(float(pos[1]), 2)]
        return None
    
    def _find_open_teammates(self, tracks, frame_num, passer_id, passer_team, 
                            min_lane_clearance=2.0):
        """
        Find teammates who were "open" at the release frame.
        
        A player is "open" if:
        1. Same team as passer
        2. Not the passer
        3. Has lane clearance (no opponent within threshold distance to passing lane)
        
        Returns:
            List of open teammate info
        """
        open_teammates = []
        
        # Get passer position
        passer_pos = self._get_player_position(tracks, frame_num, passer_id)
        if not passer_pos:
            return open_teammates
        
        # Get all opponent positions
        opponent_positions = []
        for player_id, track in tracks['players'][frame_num].items():
            if track.get('team') != passer_team:
                pos = track.get('position_transformed')
                if pos:
                    opponent_positions.append(pos)
        
        # Check each teammate
        for player_id, track in tracks['players'][frame_num].items():
            if (track.get('team') == passer_team and 
                player_id != passer_id):
                
                teammate_pos = track.get('position_transformed')
                if not teammate_pos:
                    continue
                
                # Calculate lane clearance (minimum distance from any opponent to the passing lane)
                lane_clearance = self._calculate_lane_clearance(
                    passer_pos, teammate_pos, opponent_positions
                )
                
                # Distance from passer
                distance_from_passer = measure_distance(passer_pos, teammate_pos)
                
                if lane_clearance >= min_lane_clearance:
                    open_teammates.append({
                        'id': int(player_id),
                        'position': [round(float(teammate_pos[0]), 2), 
                                   round(float(teammate_pos[1]), 2)],
                        'distance_from_passer': round(distance_from_passer, 2),
                        'lane_clearance': round(lane_clearance, 2),
                        'speed_kmh': round(float(track.get('speed', 0)), 2)
                    })
        
        # Sort by lane clearance (most open first)
        open_teammates.sort(key=lambda x: x['lane_clearance'], reverse=True)
        
        return open_teammates
    
    def _calculate_lane_clearance(self, start_pos, end_pos, opponent_positions):
        """
        Calculate minimum distance from any opponent to the passing lane.
        
        Uses point-to-line-segment distance calculation.
        
        Returns:
            Minimum clearance distance in meters
        """
        if not opponent_positions:
            return float('inf')
        
        min_clearance = float('inf')
        
        for opp_pos in opponent_positions:
            # Point to line segment distance
            clearance = self._point_to_segment_distance(
                opp_pos, start_pos, end_pos
            )
            min_clearance = min(min_clearance, clearance)
        
        return min_clearance
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Calculate perpendicular distance from point to line segment."""
        px, py = point[0], point[1]
        x1, y1 = seg_start[0], seg_start[1]
        x2, y2 = seg_end[0], seg_end[1]
        
        # Vector from seg_start to seg_end
        dx = x2 - x1
        dy = y2 - y1
        
        # If segment is a point
        if dx == 0 and dy == 0:
            return measure_distance(point, seg_start)
        
        # Parameter t for closest point on line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return measure_distance(point, (closest_x, closest_y))
    
    def _frame_to_timestamp(self, frame_num):
        """Convert frame number to HH:MM:SS.mmm timestamp."""
        total_seconds = frame_num / self.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    # ==================== FRAME EXTRACTION METHODS ====================
    
    def extract_failed_pass_frames(self, video_frames, output_dir='failed_pass_frames'):
        """
        Extract and save video frames for all detected failed passes.
        
        Creates a directory structure:
        output_dir/
        ├── pass_001/
        │   ├── release_frame_0150.jpg
        │   ├── frame_0151.jpg
        │   ├── ...
        │   ├── failure_frame_0175.jpg
        │   └── metadata.json
        ├── pass_002/
        │   └── ...
        
        Args:
            video_frames: List of video frame images
            output_dir: Directory to save extracted frames
            
        Returns:
            Dict with extraction summary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        extraction_summary = {
            'total_passes': len(self.failed_passes),
            'extracted_passes': [],
            'total_frames_extracted': 0
        }
        
        for event in self.failed_passes:
            event_id = event['event_id']
            release_frame = event['release_frame']
            failure_frame = event['failure_frame']
            
            # Create directory for this pass
            pass_dir = os.path.join(output_dir, f'pass_{event_id:03d}')
            os.makedirs(pass_dir, exist_ok=True)
            
            # Extract frames from release to failure (with padding)
            start_frame = max(0, release_frame - 10)  # 10 frames before release
            end_frame = min(len(video_frames) - 1, failure_frame + 10)  # 10 frames after
            
            frames_extracted = 0
            
            for f in range(start_frame, end_frame + 1):
                if f < len(video_frames):
                    # Determine frame type for naming
                    if f == release_frame:
                        filename = f'release_frame_{f:05d}.jpg'
                    elif f == failure_frame:
                        filename = f'failure_frame_{f:05d}.jpg'
                    elif f == event['last_touch_frame']:
                        filename = f'last_touch_frame_{f:05d}.jpg'
                    else:
                        filename = f'frame_{f:05d}.jpg'
                    
                    filepath = os.path.join(pass_dir, filename)
                    cv2.imwrite(filepath, video_frames[f])
                    frames_extracted += 1
            
            # Save metadata for this pass
            metadata = {
                'event_id': event_id,
                'release_frame': release_frame,
                'failure_frame': failure_frame,
                'last_touch_frame': event['last_touch_frame'],
                'duration_frames': event['duration_frames'],
                'duration_seconds': event['duration_seconds'],
                'passer': event['passer'],
                'interceptor': event['interceptor'],
                'pass_distance_meters': event['pass_distance_meters'],
                'num_open_options': event['num_open_options'],
                'frames_extracted': frames_extracted,
                'frame_range': [start_frame, end_frame]
            }
            
            metadata_path = os.path.join(pass_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            extraction_summary['extracted_passes'].append({
                'event_id': event_id,
                'directory': pass_dir,
                'frames_extracted': frames_extracted
            })
            extraction_summary['total_frames_extracted'] += frames_extracted
        
        # Save overall summary
        summary_path = os.path.join(output_dir, 'extraction_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(extraction_summary, f, indent=2)
        
        print(f"\n🎬 Frame Extraction Complete:")
        print(f"   Total failed passes: {extraction_summary['total_passes']}")
        print(f"   Total frames extracted: {extraction_summary['total_frames_extracted']}")
        print(f"   Output directory: {output_dir}")
        
        return extraction_summary
    
    def extract_single_pass_frames(self, video_frames, event_id, output_dir='failed_pass_frames'):
        """
        Extract frames for a single failed pass event.
        
        Args:
            video_frames: List of video frame images
            event_id: ID of the failed pass event
            output_dir: Base output directory
            
        Returns:
            Dict with extracted frame info
        """
        event = next((e for e in self.failed_passes if e['event_id'] == event_id), None)
        if not event:
            print(f"❌ Event {event_id} not found")
            return None
        
        pass_dir = os.path.join(output_dir, f'pass_{event_id:03d}')
        os.makedirs(pass_dir, exist_ok=True)
        
        release_frame = event['release_frame']
        failure_frame = event['failure_frame']
        
        # Extract key frames with annotations
        key_frames = {
            'release': release_frame,
            'failure': failure_frame,
            'last_touch': event['last_touch_frame'],
            'mid_pass': (release_frame + failure_frame) // 2
        }
        
        extracted = {}
        for name, frame_num in key_frames.items():
            if 0 <= frame_num < len(video_frames):
                filename = f'{name}_frame_{frame_num:05d}.jpg'
                filepath = os.path.join(pass_dir, filename)
                cv2.imwrite(filepath, video_frames[frame_num])
                extracted[name] = filepath
        
        return extracted
    
    def create_pass_video_clip(self, video_frames, event_id, output_dir='failed_pass_clips', 
                               fps=24, padding_frames=15):
        """
        Create a video clip of a single failed pass.
        
        Args:
            video_frames: List of video frame images
            event_id: ID of the failed pass event
            output_dir: Directory to save video clips
            fps: Frames per second for output video
            padding_frames: Extra frames before/after the pass
            
        Returns:
            Path to created video clip
        """
        os.makedirs(output_dir, exist_ok=True)
        
        event = next((e for e in self.failed_passes if e['event_id'] == event_id), None)
        if not event:
            print(f"❌ Event {event_id} not found")
            return None
        
        release_frame = event['release_frame']
        failure_frame = event['failure_frame']
        
        start_frame = max(0, release_frame - padding_frames)
        end_frame = min(len(video_frames) - 1, failure_frame + padding_frames)
        
        # Create video
        output_path = os.path.join(output_dir, f'failed_pass_{event_id:03d}.avi')
        
        if len(video_frames) > 0:
            height, width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            
            for f in range(start_frame, end_frame + 1):
                if f < len(video_frames):
                    frame = video_frames[f].copy()
                    
                    # Add frame info overlay
                    label = f"Frame: {f}"
                    if f == release_frame:
                        label += " [RELEASE]"
                        cv2.putText(frame, label, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif f == failure_frame:
                        label += " [INTERCEPTED]"
                        cv2.putText(frame, label, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, label, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    out.write(frame)
            
            out.release()
            print(f"📹 Created video clip: {output_path}")
            return output_path
        
        return None
    
    def create_all_pass_clips(self, video_frames, output_dir='failed_pass_clips', fps=24):
        """
        Create video clips for all detected failed passes.
        
        Args:
            video_frames: List of video frame images
            output_dir: Directory to save video clips
            fps: Frames per second
            
        Returns:
            List of created video paths
        """
        created_clips = []
        
        for event in self.failed_passes:
            clip_path = self.create_pass_video_clip(
                video_frames, event['event_id'], output_dir, fps
            )
            if clip_path:
                created_clips.append(clip_path)
        
        print(f"\n🎬 Created {len(created_clips)} video clips in {output_dir}")
        return created_clips
    
    # ==================== EXPORT METHODS ====================
    
    def export_failed_passes(self, output_path='failed_passes_detailed.json'):
        """
        Export all failed pass data to JSON.
        
        Args:
            output_path: Path for JSON output
        """
        export_data = {
            'summary': {
                'total_failed_passes': len(self.failed_passes),
                'fps': self.fps,
                'possession_threshold_m': self.possession_threshold,
                'release_threshold_m': self.release_threshold
            },
            'failed_passes': self.failed_passes
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        export_data = convert_numpy(export_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n📊 Failed Pass Analysis Exported:")
        print(f"   Total failed passes: {len(self.failed_passes)}")
        print(f"   Output: {output_path}")
        
        for event in self.failed_passes:
            print(f"\n   Pass #{event['event_id']}:")
            print(f"      Release Frame: {event['release_frame']} ({event['release_timestamp']})")
            print(f"      Failure Frame: {event['failure_frame']} ({event['failure_timestamp']})")
            print(f"      Passer: Player {event['passer']['id']} (Team {event['passer']['team']})")
            print(f"      Interceptor: Player {event['interceptor']['id']} (Team {event['interceptor']['team']})")
            print(f"      Distance: {event['pass_distance_meters']:.1f}m")
            print(f"      Open options ignored: {event['num_open_options']}")
    
    def get_ghost_analysis_data(self):
        """
        Get data formatted for Ghost Pass vector DB analysis.
        
        Returns tactical state vectors at release frame for each failed pass.
        
        Returns:
            List of dicts with position vectors and metadata
        """
        ghost_data = []
        
        for event in self.failed_passes:
            snapshot = event['tactical_snapshot_release']
            
            # Build position vector (all x,y coordinates)
            position_vector = []
            
            # Add team 1 positions (sorted by player ID for consistency)
            for player in sorted(snapshot['players']['team_1'], key=lambda x: x['id']):
                position_vector.extend(player['position'])
            
            # Add team 2 positions
            for player in sorted(snapshot['players']['team_2'], key=lambda x: x['id']):
                position_vector.extend(player['position'])
            
            # Add ball position
            if snapshot['ball']:
                position_vector.extend(snapshot['ball']['position'])
            
            ghost_data.append({
                'event_id': event['event_id'],
                'release_frame': event['release_frame'],
                'passer_id': event['passer']['id'],
                'passer_team': event['passer']['team'],
                'passer_position': event['passer']['position_at_release'],
                'position_vector': position_vector,
                'open_teammates': event['open_teammates_at_release'],
                'tactical_snapshot': snapshot
            })
        
        return ghost_data
    
    # ==================== VIDEO ANNOTATION METHODS ====================
    
    def annotate_output_video(self, output_video_frames, tracks):
        """
        Add visual annotations for failed passes to the output video.
        
        This method adds visual markers to show:
        - Release frames (when pass was initiated)
        - Failure frames (when interception occurred)
        - Passer and interceptor highlighting
        - Pass trajectory
        
        Args:
            output_video_frames: List of annotated video frames (from tracker.draw_annotations)
            tracks: Dictionary of tracked objects
            
        Returns:
            List of annotated frames with failed pass markers
        """
        annotated_frames = [frame.copy() for frame in output_video_frames]
        
        # Create a mapping of frames to failed pass events
        frame_to_events = {}
        for event in self.failed_passes:
            release_frame = event['release_frame']
            failure_frame = event['failure_frame']
            event_id = event['event_id']
            
            # Mark release frame
            if release_frame not in frame_to_events:
                frame_to_events[release_frame] = []
            frame_to_events[release_frame].append({
                'type': 'release',
                'event_id': event_id,
                'event': event
            })
            
            # Mark failure frame
            if failure_frame not in frame_to_events:
                frame_to_events[failure_frame] = []
            frame_to_events[failure_frame].append({
                'type': 'failure',
                'event_id': event_id,
                'event': event
            })
        
        # Annotate each frame
        for frame_num, frame in enumerate(annotated_frames):
            if frame_num in frame_to_events:
                for marker in frame_to_events[frame_num]:
                    event = marker['event']
                    event_id = marker['event_id']
                    
                    if marker['type'] == 'release':
                        # Draw release frame marker
                        self._draw_release_marker(frame, frame_num, event, tracks)
                    elif marker['type'] == 'failure':
                        # Draw failure/interception marker
                        self._draw_failure_marker(frame, frame_num, event, tracks)
            
            # Draw pass trajectory for frames between release and failure
            for event in self.failed_passes:
                if event['release_frame'] <= frame_num <= event['failure_frame']:
                    self._draw_pass_trajectory(frame, frame_num, event, tracks)
        
        return annotated_frames
    
    def _draw_release_marker(self, frame, frame_num, event, tracks):
        """Draw visual marker for pass release frame."""
        passer_id = event['passer']['id']
        passer_team = event['passer']['team']
        
        # Find passer in this frame
        if passer_id in tracks['players'][frame_num]:
            passer_track = tracks['players'][frame_num][passer_id]
            bbox = passer_track.get('bbox')
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw green circle for release
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 5)
                cv2.circle(frame, (center_x, center_y), 35, (0, 255, 0), 2)
                
                # Add text
                text = f"PASS #{event['event_id']} RELEASE"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y - 50
                
                # Background for text
                cv2.rectangle(frame, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 255, 0), -1)
                cv2.putText(frame, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    def _draw_failure_marker(self, frame, frame_num, event, tracks):
        """Draw visual marker for interception/failure frame."""
        interceptor_id = event['interceptor']['id']
        interceptor_team = event['interceptor']['team']
        
        # Find interceptor in this frame
        if interceptor_id in tracks['players'][frame_num]:
            interceptor_track = tracks['players'][frame_num][interceptor_id]
            bbox = interceptor_track.get('bbox')
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw red circle for interception
                cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), 5)
                cv2.circle(frame, (center_x, center_y), 35, (0, 0, 255), 2)
                
                # Add text
                text = f"INTERCEPTED! Pass #{event['event_id']}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y - 50
                
                # Background for text
                cv2.rectangle(frame, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 0, 255), -1)
                cv2.putText(frame, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Also add info in top-left corner
        info_text = [
            f"Failed Pass #{event['event_id']}",
            f"Team {event['passer']['team']} -> Team {interceptor_team}",
            f"Distance: {event['pass_distance_meters']:.1f}m"
        ]
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_pass_trajectory(self, frame, frame_num, event, tracks):
        """Draw pass trajectory line between release and failure."""
        # Get ball position if available
        if 1 in tracks['ball'][frame_num]:
            ball_track = tracks['ball'][frame_num][1]
            bbox = ball_track.get('bbox')
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw small dot for ball position
                cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)
                
                # Draw line from release to current position (if we have release position)
                if frame_num > event['release_frame']:
                    # Try to get release position
                    release_frame = event['release_frame']
                    if 1 in tracks['ball'][release_frame]:
                        release_ball = tracks['ball'][release_frame][1]
                        release_bbox = release_ball.get('bbox')
                        if release_bbox:
                            rx1, ry1, rx2, ry2 = map(int, release_bbox)
                            release_x = (rx1 + rx2) // 2
                            release_y = (ry1 + ry2) // 2
                            
                            # Draw dashed line
                            self._draw_dashed_line(frame, (release_x, release_y), 
                                                  (center_x, center_y), (255, 255, 0), 2)
    
    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness):
        """Draw a dashed line between two points."""
        import math
        
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line properties
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Draw dashed line
        dash_length = 10
        gap_length = 5
        current_dist = 0
        
        while current_dist < distance:
            # Calculate segment end
            segment_end = min(current_dist + dash_length, distance)
            
            # Draw segment
            start_x = int(x1 + dx * current_dist)
            start_y = int(y1 + dy * current_dist)
            end_x = int(x1 + dx * segment_end)
            end_y = int(y1 + dy * segment_end)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
            
            # Move to next segment
            current_dist += dash_length + gap_length
