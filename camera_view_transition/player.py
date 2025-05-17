import cv2
import numpy as np

from camera_view_transition.camera import Camera
from camera_view_transition.coordinate_transform import image_to_world, world_to_image
from camera_view_transition.utils import get_billboard


class Player:
    """A class to represent a player in a calibrated frame.

    Args:
        mask: Player mask in (H, W) shape.
        frame: Frame image in (H, W, 3), with the other players removed.
        camera: Camera object.
        billboard_width: Width of the player billboard in meters.
        billboard_height: Height of the player billboard in meters.
    """

    def __init__(self,
                 mask: np.ndarray,
                 frame: np.ndarray,
                 camera: Camera,
                 billboard_width: int,
                 billboard_height: int
                 ) -> None:
        self.mask = mask
        self.height, self.width = mask.shape[:2]
        self.bbox = cv2.boundingRect(mask)
        self.x, self.y, self.w, self.h = self.bbox
        self.billboard_width = billboard_width
        self.billboard_height = billboard_height

        foot_position_image = np.array((self.x + (self.w // 2), self.y + self.h))
        self.foot_position_world = image_to_world(foot_position_image.reshape(1, 2),
                                                  camera.projection_matrix,
                                                  self.width,
                                                  self.height)[0]
        self.billboard_corners_world = get_billboard(self.foot_position_world,
                                                     camera,
                                                     billboard_width,
                                                     billboard_height)

        self.image_billboard_coordinates = world_to_image(self.billboard_corners_world,
                                                          camera.projection_matrix,
                                                          self.width,
                                                          self.height)
        self.image_billboard = self.get_image_billboard(frame, mask, self.image_billboard_coordinates)


    @staticmethod
    def get_image_billboard(frame: np.ndarray,
                            mask: np.ndarray,
                            image_billboard_coordinates: np.ndarray
                            ) -> np.ndarray:
        """Extract the player billboard from the frame.
        If the billboard is partially outside the frame, it is padded with zeros to keep the
        correspondence between the billboard and the world coordinates.

        Args:
            frame: Frame image in (H, W, 3).
            mask: Player mask in (H, W).
            image_billboard_coordinates: Image coordinates of the player billboard corners.

        Returns:
            np.ndarray: Player billboard image in (H, W, 3).
        """
        h, w = frame.shape[:2]

        corners_min = np.clip(np.min(image_billboard_coordinates, axis=0).astype(int), 0, None)
        corners_max = np.clip(np.max(image_billboard_coordinates, axis=0).astype(int), None, (w, h))

        x_start, y_start = corners_min
        x_end, y_end = corners_max
        billboard = frame[y_start:y_end, x_start:x_end].copy()
        billboard[mask[y_start:y_end, x_start:x_end] == 0] = 0

        orig_corners_min = np.min(image_billboard_coordinates, axis=0).astype(int)
        orig_corners_max = np.max(image_billboard_coordinates, axis=0).astype(int)
        padding = (
            (abs(min(orig_corners_min[1], 0)), max(orig_corners_max[1] - h, 0)),
            (abs(min(orig_corners_min[0], 0)), max(orig_corners_max[0] - w, 0)),
            (0, 0)
        )

        if any(sum(p) for p in padding):
            billboard = np.pad(billboard, padding, mode="constant")
        return billboard
