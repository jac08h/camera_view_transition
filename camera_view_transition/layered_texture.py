from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import numpy as np


class LayeredTexture(ABC):
    """A class to represent a texture composed of multiple layers.

    Args:
        width: Width of the texture in pixels.
        height: Height of the texture in pixels.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.image_coordinates = np.array([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
        self.composite_img = self.multiple_bg_mask = self.overlapping_pixels = self.textures = None

    def load_textures(self, textures: List[np.ndarray]) -> None:
        """Load textures.
        Parts of the textures that don't overlap are directly added to the composite image,
        while overlapping parts are stored separately and will be added later during rendering.

        Args:
            textures: Textures to load. Each texture is a (H, W, 3) array, where
                height and width are dimension specified upon initialization.

        Raises:
            ValueError: If the texture shape doesn't match the specified height and width.
        """
        for texture in textures:
            if texture.shape != (self.height, self.width, 3):
                raise ValueError("The texture shape doesn't match the specified height and width.")

        self.textures = np.stack(textures) if textures else np.zeros((0, self.height, self.width, 3),
                                                                     dtype=np.uint8)
        self.composite_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        non_zero_pixels = self.textures.any(axis=-1)
        pixel_counts = non_zero_pixels.sum(axis=0)
        frame_count_mask = np.repeat(pixel_counts[..., np.newaxis], 3, axis=-1)

        single_bg_mask = frame_count_mask == 1
        self.composite_img[single_bg_mask] = self.textures.sum(axis=0)[single_bg_mask]

        self.multiple_bg_mask = frame_count_mask > 1
        self.overlapping_pixels = self.textures[:, self.multiple_bg_mask]

    def is_empty(self) -> bool:
        """Check if the texture is empty.

        Returns:
            bool: True if the texture is empty, False otherwise.
        """
        return not self.textures.any()

    def render(self, alphas: List[float]) -> np.ndarray:
        """Render the texture.
        The texture is rendered by combining the non-overlapping parts of the textures
        and adding the overlapping parts with the specified alpha values.

        Args:
            alphas: Alpha values for each layer for overlapping texture parts.
                There must be one alpha value for each layer, and the sum of alphas must be 1.

        Returns:
            np.ndarray: Rendered texture.

        Raises:
            RuntimeError: If textures are not loaded before rendering.
            ValueError: If the number of alphas doesn't match the number of layers
                or the sum of alphas is not equal to 1.
        """
        if len(self.textures) == 0:
            raise RuntimeError("Textures must be loaded before rendering.")

        if len(alphas) != len(self.textures):
            raise ValueError("Number of alphas must be equal to the number of layers.")

        if sum(alphas) != 1:
            raise ValueError("Sum of alphas must be equal to 1.")

        self.composite_img[self.multiple_bg_mask] = 0.0
        for texture, alpha in zip(self.textures, alphas):
            self.composite_img[self.multiple_bg_mask] += (alpha * texture[self.multiple_bg_mask]).astype(np.uint8)

        return self.composite_img

    @abstractmethod
    def get_world_coordinates(self) -> np.ndarray:
        """Get world coordinates of the texture corners.

        Returns:
            np.ndarray: World coordinates of the texture corners in shape (4, 3), ordered as
                top-left, top-right, bottom-left, bottom-right.
        """
        pass

    def extend_top_corners_to_ground(self, top_corners: List[List[float]]) -> List[List[float]]:
        """Extend the top corners to the ground plane.

        Args:
            top_corners: Top-left and top-right corners in world coordinates.

        Returns:
            List[float]: World coordinates in the order top-left, top-right, bottom-left, bottom-right,
                where the bottom corners have the same x and y coordinates as the top corners but z = 0.
        """
        bottom_corners = deepcopy(top_corners)
        bottom_corners[0][2] = bottom_corners[1][2] = 0
        return top_corners + bottom_corners
