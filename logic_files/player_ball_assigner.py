
from logic_files.utils import get_center_of_bbox, measure_distance, get_foot_position


class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 100  # Increased from 70 to capture more possessions

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assign ball to the closest player within threshold distance.
        
        Args:
            players: Dictionary of player tracks with bbox
            ball_bbox: Ball bounding box [x1, y1, x2, y2]
            
        Returns:
            Player ID of assigned player, or -1 if no assignment
        """
        if ball_bbox is None:
            return -1
            
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            foot_position = get_foot_position(player_bbox)
            distance = measure_distance(foot_position, ball_position)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
