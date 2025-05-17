import numpy as np


def world_to_image(world_points: np.ndarray,
                   camera_matrix: np.ndarray,
                   width: int,
                   height: int
                   ) -> np.ndarray:
    """Convert world coordinates to image coordinates.

    Args:
        world_points: Array of 3D points in meters of shape (N, 3).
        camera_matrix: Camera matrix of shape (4, 4).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        np.ndarray: Image coordinates of shape (N, 2).
    """
    world_points_homogeneous = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
    camera_points = camera_matrix @ world_points_homogeneous.T

    # Normalize to [-1, 1]
    x, y, z, w = camera_points
    x, y = x / w, y / w

    # Shift and scale from [-1, 1] to [0, W] and [0, H]
    image_points = np.vstack([(x + 1) * (width / 2),
                              (1 - y) * (height / 2)]).T
    return image_points


def image_to_world(image_points: np.ndarray,
                   camera_matrix: np.ndarray,
                   width: int,
                   height: int
                   ) -> np.ndarray:
    """Convert image coordinates to world coordinates with Z=0.

    Args:
        image_points: Points in image coordinates of shape (N, 2).
        camera_matrix: Camera matrix of shape (4, 4).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        np.ndarray: World coordinates of shape (N, 3) with Z=0, in meters.

    Raises:
        ValueError: If the camera matrix is not invertible.
    """
    # Shift and scale from [0, W] and [0, H] to [-1, 1].
    x, y = (image_points[:, 0] / (width / 2) - 1,
            1 - image_points[:, 1] / (height / 2))
    try:
        inv_camera_matrix = np.linalg.inv(camera_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Camera matrix is not invertible.")

    # Transform image points to world coordinates. In range [-1, 1].
    homogeneous_points = np.vstack([x, y, np.zeros_like(x), np.ones_like(x)])
    world_points = inv_camera_matrix @ homogeneous_points

    # Normalize to 3D points in range [-W/2, W/2], [-H/2, H/2], [0].
    world_points = world_points[:3, :] / world_points[3, :]
    world_points[2, :] = 0
    return world_points.T
