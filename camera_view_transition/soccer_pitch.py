from typing import Optional

import cv2
import numpy as np

from camera_view_transition.layered_texture import LayeredTexture


class SoccerPitch(LayeredTexture):
    """A class to represent a soccer pitch.
    Origin is at the center of the pitch.
    Positive x-axis points to the right side of the pitch,
    positive y-axis points to the top side of the pitch,
    and positive z-axis points upwards.

    Args:
        template_image: BGR image of the pitch template in (H, W, 3) shape. Used as a background image
            for drawing points.
        width_meters: Width of the pitch in meters.
        height_meters: Height of the pitch in meters.
        width_pixels: Target width of the pitch template after resizing.
        height_pixels: Target height of the pitch template after resizing.
        width_behind_pitch: Width of the area behind the pitch in meters.
        height_behind_pitch: Height of the area behind the pitch in meters.
    """

    def __init__(self,
                 template_image: np.ndarray,
                 width_meters: int = 105,
                 height_meters: int = 68,
                 width_pixels: int = 1280,
                 height_pixels: int = 720,
                 width_behind_pitch: int = 5,
                 height_behind_pitch: int = 4
                 ) -> None:
        super().__init__(width_pixels, height_pixels)
        self.width_meters = width_meters
        self.height_meters = height_meters
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
        self.width_behind_pitch = width_behind_pitch
        self.height_behind_pitch = height_behind_pitch

        self.width_pixel_per_meter = self.width_pixels / (self.width_meters + 2 * width_behind_pitch)
        self.height_pixel_per_meter = self.height_pixels / (self.height_meters + 2 * height_behind_pitch)

        self.template_image = self._create_template(template_image, width_behind_pitch, height_behind_pitch)

    def _create_template(self,
                         template_image: np.ndarray,
                         delta_width_meters: int,
                         delta_height_meters: int
                         ) -> np.ndarray:
        """Create a pitch template image with the specified dimensions.

        Args:
            template_image: BGR image of the pitch template in (H, W, 3) shape.
            delta_width_meters: Width of the area behind the pitch in meters.
            delta_height_meters: Height of the area behind the pitch in meters.

        Returns:
            np.ndarray: Pitch template image of shape (self.height_pixels, self.width_pixels, 3).
        """
        original_height_pixels, original_width_pixels = template_image.shape[:2]

        delta_width_pixels = int(delta_width_meters * (original_width_pixels / self.width_meters))
        delta_height_pixels = int(delta_height_meters * (original_height_pixels / self.height_meters))

        template_image = cv2.copyMakeBorder(template_image,
                                            top=delta_height_pixels,
                                            bottom=delta_height_pixels,
                                            left=delta_width_pixels,
                                            right=delta_width_pixels,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))

        return cv2.resize(template_image, (self.width_pixels, self.height_pixels))

    def get_world_coordinates(self) -> np.ndarray:
        return np.array([
            [-self.width_meters / 2 - self.width_behind_pitch, self.height_meters / 2 + self.height_behind_pitch, 0],
            [self.width_meters / 2 + self.width_behind_pitch, self.height_meters / 2 + self.height_behind_pitch, 0],
            [-self.width_meters / 2 - self.width_behind_pitch, -self.height_meters / 2 - self.height_behind_pitch, 0],
            [self.width_meters / 2 + self.width_behind_pitch, -self.height_meters / 2 - self.height_behind_pitch, 0],
        ])

    def draw_points(self, points: np.ndarray, image: Optional[np.ndarray] = None, **draw_kwargs):
        """Draw points on the pitch template image.

        Args:
            points: Points to draw in world coordinates in shape (N, 3).
            image: Image to draw points on. If None, the pitch template image is used.
            **draw_kwargs: Keyword arguments for cv2.circle.

        Returns:
            np.ndarray: Image with points drawn.
        """
        if image is None:
            image = self.template_image.copy()

        for image_point in self.world_to_image(points):
            image_point = tuple(map(int, image_point))

            new_image_point = (
                max(0, min(self.width_pixels, image_point[0])),
                max(0, min(self.height_pixels, image_point[1]))
            )
            if new_image_point != image_point:
                print(f"Point {image_point} is outside the image. Clipped to {new_image_point}")

            cv2.circle(image, new_image_point, **draw_kwargs)

        return image

    def world_to_image(self, points_world: np.ndarray) -> np.ndarray:
        """Convert world coordinates to image coordinates.
        Since the pitch image is a top-down view centered at the origin, we can convert directly
        without the camera matrix.

        Args:
            points_world: Points in world coordinates in shape (N, 3). Z-coordinate is ignored and
                assumed to be 0.

        Returns:
            np.ndarray: Points in image coordinates in shape (N, 2).
        """
        center_x_pixels = self.template_image.shape[1] // 2
        center_y_pixels = self.template_image.shape[0] // 2

        return np.column_stack([
            center_x_pixels + (points_world[:, 0] * self.width_pixel_per_meter),
            center_y_pixels - (points_world[:, 1] * self.height_pixel_per_meter)
        ])
