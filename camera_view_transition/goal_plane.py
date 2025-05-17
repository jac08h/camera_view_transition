from enum import auto, Enum
from typing import List, Tuple

import numpy as np

from camera_view_transition.layered_texture import LayeredTexture


class GoalPlanePosition(Enum):
    """Position of the goal plane relative to the goalkeeper."""
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BACK = auto()


class GoalPlaneTeamSide(Enum):
    """Side of the team that the goal plane belongs to. The side with negative x-coordinates is the left side."""
    LEFT = auto()
    RIGHT = auto()


class GoalPlane(LayeredTexture):
    """Class to represent a goal plane.

    Args:
        plane_position: Position of the goal plane.
        plane_side: Side of the team that the goal plane belongs to.
        image_size: Size of the image in pixels.
        pitch_width: Width of the pitch in meters.
        goal_width: Width of the goal in meters.
        goal_height: Height of the goal in meters.
        goal_depth: Depth of the goal in meters. This refers to how far the net extends behind the goal line.
        goal_line_shift: Shift of the goal line from the edge of the pitch in meters.
    """

    def __init__(self,
                 plane_position: GoalPlanePosition,
                 plane_side: GoalPlaneTeamSide,
                 image_size: Tuple[int, int],
                 pitch_width: float,
                 goal_width: float,
                 goal_height: float,
                 goal_depth: float,
                 goal_line_shift: float = 0.1
                 ) -> None:
        self.plane_position = plane_position
        self.plane_side = plane_side
        self.width, self.height = image_size
        self.pitch_width = pitch_width
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.goal_depth = goal_depth
        self.goal_line_shift = goal_line_shift

        super().__init__(self.width, self.height)

    def get_world_coordinates(self) -> np.ndarray:
        line_x = -self.pitch_width / 2 + self.goal_line_shift
        coordinates_dict = {
            GoalPlanePosition.LEFT:
                [[line_x - self.goal_depth, self.goal_width / 2, self.goal_height],
                 [line_x, self.goal_width / 2, self.goal_height]],
            GoalPlanePosition.RIGHT:
                [[line_x, -self.goal_width / 2, self.goal_height],
                 [line_x - self.goal_depth, -self.goal_width / 2, self.goal_height]],
            GoalPlanePosition.BACK:
                [[line_x - self.goal_depth, -self.goal_width / 2, self.goal_height],
                 [line_x - self.goal_depth, self.goal_width / 2, self.goal_height]],
            GoalPlanePosition.TOP:
                [[line_x - self.goal_depth, -self.goal_width / 2, self.goal_height],
                 [line_x - self.goal_depth, self.goal_width / 2, self.goal_height],
                 [line_x, -self.goal_width / 2, self.goal_height],
                 [line_x, self.goal_width / 2, self.goal_height]]
        }

        coords = coordinates_dict[self.plane_position]
        if self.plane_position != GoalPlanePosition.TOP:
            coords = self.extend_top_corners_to_ground(coords)
        coords = np.array(coords)
        if self.plane_side == GoalPlaneTeamSide.RIGHT:
            coords *= (-1, -1, 1)

        return coords


def create_goal_planes(image_size: Tuple[int, int],
                       pitch_width: float,
                       goal_width: float,
                       goal_height: float,
                       goal_depth: float
                       ) -> List[GoalPlane]:
    """Create all goal planes in the scene.

    Args:
        image_size: Size of the image in pixels.
        pitch_width: Width of the pitch in meters.
        goal_width: Width of the goal in meters.
        goal_height: Height of the goal in meters.
        goal_depth: Depth of the goal in meters. This refers to how far the net extends behind the goal line.

    Returns:
        List[GoalPlane]: Goal planes in the scene of length 8: 4 planes for each of the two goals.
    """
    planes = []
    for plane_side in GoalPlaneTeamSide:
        for plane_position in GoalPlanePosition:
            planes.append(GoalPlane(
                plane_position, plane_side, image_size, pitch_width, goal_width, goal_height, goal_depth
            ))
    return planes
