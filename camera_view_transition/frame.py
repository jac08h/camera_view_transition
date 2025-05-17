from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera_view_transition.camera import Camera
from camera_view_transition.coordinate_transform import image_to_world, world_to_image
from camera_view_transition.player import Player


class Frame:
    """A class to represent a calibrated frame with players.

    Args:
        frame: Loaded BGR frame in (H, W, 3) shape.
        frame_with_graphics: Loaded BGR frame before inpainting graphics in (H, W, 3) shape.
        player_masks: List of player masks in (H, W) shape.
        ball_mask: Ball mask in (H, W) shape. None if no ball is detected.
        camera: Camera object.
        billboard_dimensions: Dimensions of the billboards in meters. Format:
            {"player": {"width": width, "height": height}, "ball": {"width": width, "height": height}}
    """

    def __init__(self,
                 frame: np.ndarray,
                 frame_with_graphics: np.ndarray,
                 player_masks: List[np.ndarray],
                 ball_mask: Optional[np.ndarray],
                 camera: Camera,
                 billboard_dimensions: Dict[str, Dict[str, int]]
                 ) -> None:
        self.frame = frame
        self.frame_with_graphics = frame_with_graphics
        self.height, self.width = frame.shape[:2]
        object_masks = player_masks + [ball_mask] if ball_mask is not None else player_masks
        self.background = Frame.inpaint_image(frame, object_masks)
        self.players = []
        self.camera = camera

        self.players = [self.initialize_player(mask,
                                               billboard_dimensions["player"]["width"],
                                               billboard_dimensions["player"]["height"]
                                               ) for mask in player_masks]
        self.ball = self.initialize_player(ball_mask,
                                           billboard_dimensions["ball"]["width"],
                                           billboard_dimensions["ball"]["height"]
                                           ) if ball_mask is not None else None

        self.corners_image = Frame.get_image_corners(self.width, self.height)
        self.corners_world = image_to_world(self.corners_image, self.camera.projection_matrix, self.width, self.height)

        if len(self.players) > 0:
            self.foot_positions_world = np.stack([player.foot_position_world for player in self.players])
        else:
            self.foot_positions_world = np.empty((0, 3))

    def initialize_player(self, mask: np.ndarray, width: float, height: float) -> Player:
        """Initialize a player object from the mask.

        Args:
            mask: Player mask in (H, W) shape.
            width: Width of the player billboard in meters.
            height: Height of the player billboard in meters.

        Returns:
            Player: Player object.
        """
        background_with_player = self.background.copy()
        background_with_player[mask != 0] = self.frame[mask != 0]
        return Player(mask, background_with_player, self.camera, width, height)

    @staticmethod
    def inpaint_image(image: np.ndarray,
                      masks: List[np.ndarray],
                      dilate_kernel_size: Optional[Tuple[int, int]] = (11, 11),
                      inpaint_radius: int = 3,
                      method: int = cv2.INPAINT_TELEA
                      ) -> np.ndarray:
        """Inpaint the masked regions of the image.

        Args:
            image: Input image of shape (H, W, 3) in BGR format.
            masks: List of masks to inpaint.
            dilate_kernel_size: Kernel size for dilating the masks. If None, no dilation is applied.
            inpaint_radius: Radius of the inpainting.
            method: Inpainting method.

        Returns:
            np.ndarray: Inpainted image of shape (H, W, 3) in BGR format.
        """
        if dilate_kernel_size is not None:
            kernel = np.ones(dilate_kernel_size, np.uint8)
            masks = [cv2.dilate(mask, kernel, iterations=1) for mask in masks]
        combined_mask = np.max(np.stack(masks), axis=0) if masks else np.zeros_like(image)
        if combined_mask.any():
            return cv2.inpaint(image, combined_mask, inpaint_radius, method)
        return image

    @staticmethod
    def inpaint_image_coordinates(image: np.ndarray,
                                  objects_corners: List[np.ndarray],
                                  camera: Camera,
                                  image_width: int,
                                  image_height: int,
                                  **kwargs
                                  ) -> np.ndarray:
        """Inpaint the masked regions of the image defined by the world corners.

        Args:
            image: Input image of shape (H, W, 3) in BGR format.
            objects_corners: Corners of objects to inpaint in world coordinates. Each element is an array of
                shape (4, 3) in the order top-left, top-right, bottom-left, bottom-right.
            camera: Camera object.
            image_width: Image width.
            image_height: Image height.
            **kwargs: Keyword arguments for inpainting. See Frame.inpaint_image.

        Returns:
            np.ndarray: Inpainted image of shape (H, W, 3) in BGR format.
        """
        masks = []
        for world_corners in objects_corners:
            image_corners = world_to_image(world_corners,
                                           camera.projection_matrix,
                                           image_width,
                                           image_height)
            corners_clockwise = image_corners[[0, 1, 3, 2]]
            mask = np.zeros((image_height, image_width), np.uint8)
            cv2.fillPoly(mask, [corners_clockwise.astype(int)], 1)
            masks.append(mask)

        return Frame.inpaint_image(image, masks, **kwargs)


    @staticmethod
    def get_image_corners(width: int, height: int) -> np.ndarray:
        """Get image corners in image coordinates, ordered as top-left, top-right, bottom-left, bottom-right.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            np.ndarray: Image corners in image coordinates in shape (4, 2).
        """
        return np.array([[0, 0], [width, 0], [0, height], [width, height]])
