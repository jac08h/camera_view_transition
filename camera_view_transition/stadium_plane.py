from enum import Enum
from typing import List, Tuple

import numpy as np

from camera_view_transition.camera import Camera
from camera_view_transition.layered_texture import LayeredTexture


class PlanePosition(Enum):
    """Position of the background plane relative to the center of the pitch.
    LEFT: Behind the left goal,
    RIGHT: Behind the right goal,
    TOP: Behind the top side of the pitch,
    BOTTOM: Behind the bottom side of the pitch.
    """
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


class StadiumPlane(LayeredTexture):
    """Class to represent a stadium background plane.
    The plane faces the center of the pitch.

    Args:
        position: Position of the plane.
        image_size: Size of the image in pixels.
        pitch_width: Width of the pitch in meters.
        width_behind_pitch: Width of the area behind the pitch in meters.
        pitch_height: Height of the pitch in meters.
        height_behind_pitch: Height of the area behind the pitch in meters.
        plane_height: Height of the background plane in meters.
    """

    def __init__(self,
                 position: PlanePosition,
                 image_size: Tuple[int, int],
                 pitch_width: float,
                 width_behind_pitch: float,
                 pitch_height: float,
                 height_behind_pitch: float,
                 plane_height: float
                 ) -> None:
        self.position = position
        self.width = image_size[0]
        self.height = image_size[1]
        self.pitch_width = pitch_width
        self.width_behind_pitch = width_behind_pitch
        self.pitch_height = pitch_height
        self.height_behind_pitch = height_behind_pitch
        self.plane_height = plane_height

        super().__init__(self.width, self.height)

    def get_world_coordinates(self) -> np.ndarray:
        left_x = -self.pitch_width / 2 - self.width_behind_pitch
        right_x = self.pitch_width / 2 + self.width_behind_pitch
        top_y = self.pitch_height / 2 + self.height_behind_pitch
        bottom_y = -self.pitch_height / 2 - self.height_behind_pitch
        top_coordinates = {
            PlanePosition.LEFT: [[left_x, bottom_y, self.plane_height], [left_x, top_y, self.plane_height]],
            PlanePosition.RIGHT: [[right_x, top_y, self.plane_height], [right_x, bottom_y, self.plane_height]],
            PlanePosition.TOP: [[left_x, top_y, self.plane_height], [right_x, top_y, self.plane_height]],
            PlanePosition.BOTTOM: [[right_x, bottom_y, self.plane_height], [left_x, bottom_y, self.plane_height]]
        }

        return np.array(self.extend_top_corners_to_ground(top_coordinates[self.position]))

    def is_camera_in_front_of_plane(self, camera: Camera) -> bool:
        """Check if camera is physically in front of the plane based on position.
        Each plane is facing the center of the pitch.
        For example, the left plane is in front of the camera if the camera is to the right of the plane.

        Args:
            camera: Camera object.

        Returns:
            bool: True if camera is physically in front of plane, False otherwise.
        """
        world_coords = self.get_world_coordinates()
        if self.position == PlanePosition.LEFT:
            in_front = camera.position[0] > world_coords[0, 0]
        elif self.position == PlanePosition.RIGHT:
            in_front = camera.position[0] < world_coords[0, 0]
        elif self.position == PlanePosition.TOP:
            in_front = camera.position[1] < world_coords[0, 1]
        else:
            in_front = camera.position[1] > world_coords[0, 1]
        return in_front.item()

    def is_camera_facing_plane(self, camera: Camera, max_angle: float) -> bool:
        """Check if camera is facing towards the plane based on camera rotation.
        The camera is facing the plane if the vector from the camera to the plane center
        is in the same direction (i.e. dot product > 0) as the camera direction.

        Args:
            camera: Camera object.
            max_angle: Maximum allowed angle in degrees between camera direction and plane center
                to classify the camera as facing the plane.

        Returns:
            bool: True if camera is facing towards plane, False otherwise.
        """
        world_coords = self.get_world_coordinates()[:, :2]
        plane_center = np.mean(world_coords, axis=0)
        direction = np.array([
            np.sin(camera.pan) * np.cos(camera.tilt),
            np.cos(camera.pan) * np.cos(camera.tilt)
        ])

        ray_to_plane = plane_center - camera.position[:2]
        ray_to_plane /= np.linalg.norm(ray_to_plane)

        return np.dot(direction, ray_to_plane) > np.cos(np.deg2rad(max_angle))

    def is_visible(self, camera: Camera, max_angle: float) -> bool:
        """Check if camera can see the plane.
        This requires the camera to be in front of the plane and face towards it.

        Args:
            camera: Camera object.
            max_angle: Maximum allowed angle in degrees between camera direction and plane center
                to classify the camera as facing the plane.

        Returns:
            bool: True if plane is visible to camera, False otherwise.
        """
        return self.is_camera_in_front_of_plane(camera) and self.is_camera_facing_plane(camera, max_angle)


def create_stadium_planes(image_size: Tuple[int, int],
                          pitch_width: float,
                          width_behind_pitch: float,
                          pitch_height: float,
                          height_behind_pitch: float,
                          plane_height: float
                          ) -> List[StadiumPlane]:
    """Create stadium planes behind the goals and lines.

    Args:
        image_size: Size of the image in pixels.
        pitch_width: Width of the pitch in meters.
        width_behind_pitch: Width of the area behind the pitch in meters.
        pitch_height: Height of the pitch in meters.
        height_behind_pitch: Height of the area behind the pitch in meters.
        plane_height: Depth of the background planes in meters.

    Returns:
        List[StadiumPlane]: Background planes in the top, right, bottom, and left positions.
    """
    positions = [PlanePosition.TOP, PlanePosition.RIGHT, PlanePosition.BOTTOM, PlanePosition.LEFT]
    return [StadiumPlane(position, image_size, pitch_width, width_behind_pitch, pitch_height,
                         height_behind_pitch, plane_height) for position in positions]
