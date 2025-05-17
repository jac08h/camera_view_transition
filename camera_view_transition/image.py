from typing import List, Tuple

import cv2
import numpy as np

from camera_view_transition.frame import Frame
from camera_view_transition.player_group import PlayerGroup
from camera_view_transition.soccer_pitch import SoccerPitch


def visualize_player_positions(frames: Tuple[Frame, Frame],
                               player_groups: List[PlayerGroup],
                               pitch: SoccerPitch,
                               color_a: Tuple[int, int, int] = (0, 0, 255),
                               color_b: Tuple[int, int, int] = (0, 255, 0),
                               matched_color: Tuple[int, int, int] = (0, 255, 255),
                               initial_img: np.ndarray = None,
                               draw_points: bool = True,
                               draw_lines: bool = True,
                               draw_cameras: bool = True
                               ) -> np.ndarray:
    """Visualize player positions on the pitch.

    Args:
        frames: Two frames to visualize player positions.
        player_groups: Player groups to visualize.
        matched_color: Color for matched players.
        color_a: Color for players in the first frame.
        color_b: Color for players in the second frame.
        pitch: Pitch object to draw player positions.
        initial_img: Initial image to draw player positions. If None, use pitch template image.
        draw_points: If True, draw player positions as points.
        draw_lines: If True, draw player bottom corners as lines.
        draw_cameras: If True, draw camera positions.

    Returns:
        np.ndarray: Image with player positions.
    """
    img = initial_img.copy() if initial_img is not None else None
    colors = [color_a, color_b]

    for player_group in player_groups:
        if len(player_group.players) == 2:
            if draw_points:
                img = pitch.draw_points(player_group.get_position().reshape(1, 3), img, color=matched_color, radius=2,
                                        thickness=-1)
        else:
            player = player_group.players[0]
            frame_index = 0 if player in frames[0].players else 1
            if draw_points:
                img = pitch.draw_points(player.foot_position_world.reshape(1, 3), img,
                                        color=colors[frame_index], radius=2, thickness=-1)

        if draw_lines:
            for player in player_group.players:
                frame_index = 0 if player in frames[0].players else 1
                a, b = pitch.world_to_image(player.billboard_corners_world[[2, 3]])
                img = cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color=colors[frame_index], thickness=1)

    if draw_cameras:
        for i, frame in enumerate(frames):
            img = pitch.draw_points(frame.camera.position.reshape(1, 3), img, radius=10, color=colors[i], thickness=-1)

    return img


def get_bounding_box_relative_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Return normalized coordinates of the bounding box,
    such that top-left corner = (0, 0), bottom-right corner = (bbox_width, bbox_height).

    Args:
        coordinates: Bounding box coordinates in shape (4, 2) ordered as top-left, top-right, bottom-left, bottom-right.

    Returns:
        np.ndarray: Normalized bounding box coordinates in shape (4, 2).
    """
    target_x = max(coordinates[:, 0]) - min(coordinates[:, 0])
    target_y = max(coordinates[:, 1]) - min(coordinates[:, 1])
    return np.array([[0, 0], [target_x, 0], [0, target_y], [target_x, target_y]])


def overlay_image(source_image: np.ndarray,
                  destination_image: np.ndarray,
                  destination_top_left: Tuple[int, int],
                  allow_partial: bool = True,
                  alpha: float = 1.0
                  ) -> None:
    """Overlay the source image on the destination image at the specified coordinates.

    Args:
        source_image: Source image to overlay, in (H, W, 3) shape.
        destination_image: Destination image to overlay on, in (H, W, 3) shape. Will be modified in-place.
        destination_top_left: A coordinates of the destination image where the top-left
            corner of the source image will be placed.
        allow_partial: If True, overlay the source image even if it partially goes out of the destination image.
        alpha: Alpha value for the source image. The part of the destination image overlapping with the source
            image is calculated as source_image * alpha + destination_image * (1 - alpha).
    """
    source_height, source_width = source_image.shape[:2]
    destination_height, destination_width = destination_image.shape[:2]
    destination_x, destination_y = destination_top_left

    y1 = max(0, destination_y)
    y2 = min(destination_height, destination_y + source_height)
    x1 = max(0, destination_x)
    x2 = min(destination_width, destination_x + source_width)

    if x1 >= x2 or y1 >= y2:
        return

    if (y2 - y1 < source_height or x2 - x1 < source_width) and not allow_partial:
        return

    source_y1 = max(0, -destination_y)
    source_x1 = max(0, -destination_x)
    source_y2 = source_y1 + (y2 - y1)
    source_x2 = source_x1 + (x2 - x1)

    mask = np.sum(source_image[source_y1:source_y2, source_x1:source_x2], axis=2) > 0
    destination_image[y1:y2, x1:x2][mask] = (source_image[source_y1:source_y2, source_x1:source_x2][mask] * alpha +
                                             destination_image[y1:y2, x1:x2][mask] * (1 - alpha))
