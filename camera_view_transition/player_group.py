from typing import List

import numpy as np

from camera_view_transition.player import Player


class PlayerGroup:
    """A group of detected players in different frames which correspond to the same player,
    according to the matching algorithm.

    Args:
        players: Players in the group.
    """

    def __init__(self, players: List[Player]):
        if len(players) == 0:
            raise ValueError("Player group must contain at least one player.")

        self.players = players

    def get_position(self, t: float = 0.5) -> np.ndarray:
        """Get the position of the player group at a given interpolation step.

        Args:
            t: Interpolation step.

        Returns:
            np.ndarray: Interpolated position of the player group.
        """
        if len(self.players) == 1:
            return self.players[0].foot_position_world
        return (1 - t) * self.players[0].foot_position_world + t * self.players[1].foot_position_world


    def select_player_from_best_view(self, target_billboard: np.ndarray) -> Player:
        """Select the player from the group that best matches the target camera view.
        It is the player with the highest cosine similarity between its billboard
        and the target billboard.

        Args:
            target_billboard: Target billboard in the target camera view.

        Returns:
            Player: Selected player.
        """
        if len(self.players) == 1:
            return self.players[0]

        best_similarity = -1
        best_player = None
        billboard_width = self.players[0].billboard_width

        target_bottom_vector = target_billboard[0] - target_billboard[1]
        assert np.isclose(np.linalg.norm(target_bottom_vector), billboard_width)

        for candidate_player in self.players:
            player_bottom_vector = candidate_player.billboard_corners_world[0] - \
                                   candidate_player.billboard_corners_world[1]
            assert np.isclose(np.linalg.norm(player_bottom_vector), billboard_width)

            similarity = np.dot(target_bottom_vector, player_bottom_vector) / (billboard_width ** 2)

            if similarity > best_similarity:
                best_similarity = similarity
                best_player = candidate_player

        return best_player
