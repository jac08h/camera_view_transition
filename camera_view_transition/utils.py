from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from camera_view_transition.camera import Camera


def match_points(points1: np.ndarray,
                 points2: np.ndarray,
                 threshold: float,
                 filter_far_points: bool = True
                 ) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Match points from two sets based on the Euclidean distance using the Hungarian algorithm.

    Args:
        points1: Points of shape (N, 2) to match.
        points2: Points of shape (M, 2) to match.
        threshold: Maximum distance between points to consider a match.
        filter_far_points: Remove points which are farther than the threshold from all other points before matching.

    Returns:
        Tuple[List[Tuple[int, int]], List[float]]: Indices of matched points and their distances.
    """
    distances = cdist(points1, points2)

    if filter_far_points:
        valid_rows = np.any(distances <= threshold, axis=1)
        valid_cols = np.any(distances <= threshold, axis=0)
    else:
        valid_rows = np.ones(len(points1), dtype=bool)
        valid_cols = np.ones(len(points2), dtype=bool)

    filtered_distances = distances[valid_rows][:, valid_cols]
    row_ind, col_ind = linear_sum_assignment(filtered_distances)

    matches = []
    match_distances = []
    for i, j in zip(row_ind, col_ind):
        orig_i = np.where(valid_rows)[0][i]
        orig_j = np.where(valid_cols)[0][j]
        if distances[orig_i, orig_j] < threshold:
            matches.append((orig_i.item(), orig_j.item()))
            match_distances.append(distances[orig_i, orig_j].item())

    return matches, match_distances


def get_billboard(world_point: np.ndarray, camera: Camera, width: int, height: int) -> np.ndarray:
    """Create a billboard for a player at the given world point perpendicular to the camera.

    Args:
        world_point: World point of the player. Shape: (3,).
        camera: Camera object.
        width: Width of the billboard.
        height: Height of the billboard.

    Returns:
        np.ndarray: World points of the billboard. Shape: (4, 3).
    """
    vector = world_point[:2] - camera.position[:2]
    perpendicular_vector = np.array([-vector[1], vector[0], 0])
    perpendicular_vector[:2] /= np.linalg.norm(perpendicular_vector[:2])

    offset = perpendicular_vector * width / 2

    lower_corners = np.array([world_point + offset, world_point - offset])
    assert np.isclose(np.linalg.norm(lower_corners[0][:2] - lower_corners[1][:2]), width)

    upper_corners = lower_corners + [0, 0, height]
    return np.vstack([upper_corners, lower_corners])
