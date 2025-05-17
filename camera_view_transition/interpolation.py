from enum import auto, Enum
from typing import Iterable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from camera_view_transition.camera import Camera


class RotationInterpolation(Enum):
    """Types of rotation interpolation.

    LINEAR: Linear interpolation of the rotation angles.
    SPHERICAL: Spherical linear interpolation of the rotation angles.
    TOWARD_FOCUS_POINT: Calculate intersections I^A and I^B of the input cameras with the pitch plane. Then interpolate
        the positions linearly, and calculate the rotations toward point lying on the line between I^A and I^B,
        depending on the interpolation parameter.

    """
    LINEAR = auto()
    SPHERICAL = auto()
    TOWARD_FOCUS_POINT = auto()


def interpolate_between_cameras(start_camera: Camera,
                                end_camera: Camera,
                                steps: int,
                                time_step_exponent: int = 2,
                                rotation_type: RotationInterpolation = RotationInterpolation.TOWARD_FOCUS_POINT,
                                ) -> Iterable[Tuple[float, Camera]]:
    """Interpolate camera parameters between two cameras.

    Args:
        start_camera: Start camera.
        end_camera: End camera.
        steps: Number of steps to interpolate.
        time_step_exponent: Exponent of the time steps. Values higher than 1 will make the interpolation
            faster in the middle of the interval.
        rotation_type: Type of rotation interpolation.

    Yields:
        Tuple[float, Camera]: Interpolation parameter in [0, 1] and the interpolated camera.
    """
    ts = exponential_steps(time_step_exponent, steps)

    positions = multidimensional_interpolation(ts, (0, 1), start_camera.position, end_camera.position)
    rotations = get_rotations(start_camera, end_camera, ts, rotation_type)
    focals = np.interp(ts, [0, 1], [start_camera.focal, end_camera.focal])
    leans = np.interp(ts, [0, 1], [start_camera.lean, end_camera.lean])

    for t, position, rotation, focal, lean in zip(ts, positions, rotations, focals, leans):
        yield t, Camera(position=position,
                        rotation=rotation,
                        focal=float(focal),
                        lean=float(lean),
                        width_to_height_ratio=start_camera.width_to_height_ratio)


def exponential_steps(exponent: int, steps: int) -> np.ndarray:
    """Generate exponential steps in [0, 1] with the given exponent.
     For exponent > 1, the steps are bigger in the middle of the interval.
     For exponent < 1, the steps are bigger at the ends of the interval.
     For exponent = 1, the steps are linear.

    The steps are generated with the following function:
    f(t) = 2 * t^exponent            for t < 0.5
         = 1 - 2 * (1 - t)^exponent  for t >= 0.5

    Args:
        exponent: Exponent of the function.
        steps: Number of steps.

    Returns:
        np.ndarray: Exponential steps in [0, 1] of shape (steps + 1,).
    """
    ts = np.linspace(0, 1, steps + 1)
    return np.where(ts < 0.5,
                    np.power(2 * ts, exponent) / 2,
                    1 - np.power(2 * (1 - ts), exponent) / 2)


def get_rotations(start_camera: Camera,
                  end_camera: Camera,
                  ts: np.ndarray,
                  rotation_type: RotationInterpolation,
                  ) -> np.ndarray:
    """Interpolate rotations between two cameras.

    Args:
        start_camera: Start camera.
        end_camera: End camera.
        ts: Interpolation parameters of shape (N,) in [0, 1].
        rotation_type: Type of rotation interpolation.

    Returns:
        np.ndarray: Interpolated rotations of shape (N, 3) in radians.
    """
    if rotation_type == RotationInterpolation.LINEAR:
        return multidimensional_interpolation(ts, (0, 1), start_camera.rotation, end_camera.rotation)

    if rotation_type == RotationInterpolation.SPHERICAL:
        return spherical_linear_rotations(start_camera, end_camera, ts)

    if rotation_type == RotationInterpolation.TOWARD_FOCUS_POINT:
        positions = multidimensional_interpolation(ts, (0, 1), start_camera.position, end_camera.position)
        rotations = []
        start_camera_intersection = start_camera.pitch_intersection()
        end_camera_intersection = end_camera.pitch_intersection()
        for t, pos in zip(ts, positions):
            focus_point = start_camera_intersection + t * (end_camera_intersection - start_camera_intersection)
            rotations.append(rotations_facing_focus_point(pos.reshape(1, 3), focus_point)[0])

        rotations = np.array(rotations)
        rotations[:, 1] = multidimensional_interpolation(ts,
                                                         (0, 1),
                                                         start_camera.rotation,
                                                         end_camera.rotation)[:, 1]
        return rotations


def spherical_linear_rotations(start_camera: Camera, end_camera: Camera, ts: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation (SLERP) of rotations between two cameras.

    Args:
        start_camera: Start camera.
        end_camera: End camera.
        ts: Interpolation parameters of shape (N,) in [0, 1].

    Returns:
        np.ndarray: Interpolated rotations of shape (N, 3) in radians.
    """
    rotation_order = start_camera.rotation_order
    r1 = R.from_euler(rotation_order, start_camera.rotation)
    r2 = R.from_euler(rotation_order, end_camera.rotation)
    key_rots = R.from_quat([r1.as_quat(), r2.as_quat()])
    slerp = Slerp([0, 1], key_rots)
    return slerp(ts).as_euler(rotation_order)


def rotations_facing_focus_point(positions: np.ndarray, focus_point: np.ndarray) -> np.ndarray:
    """Calculate the camera rotations towards a point from the given positions.

    Args:
        positions: Positions of the cameras of shape (N, 3).
        focus_point: Point which the cameras should face of shape (3,).

    Returns:
        np.ndarray: Rotations towards the focus point of shape (N, 3) in radians.
    """
    vectors = focus_point - positions
    rotations = np.zeros_like(positions)

    rotations[:, 0] = np.arctan2(vectors[:, 0], vectors[:, 1])
    rotations[:, 1] = 0
    horizontal_distance = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
    rotations[:, 2] = np.arctan2(-vectors[:, 2], horizontal_distance)

    return rotations


def multidimensional_interpolation(x: np.ndarray,
                                   xp: Tuple[float, float],
                                   start: np.ndarray,
                                   end: np.ndarray,
                                   ) -> np.ndarray:
    """Interpolate multiple values along the last axis of the input array.

    Args:
        x: Interpolation parameters of shape (N,) in [xp[0], xp[1]].
        xp: Tuple of the interpolation parameter range.
        start: Start values of shape (M,).
        end: End values of shape (M,).

    Returns:
        np.ndarray: Interpolated values of shape (N, M).
    """
    fp = np.array([start, end])
    return np.array([np.interp(x, xp, f) for f in fp.T]).T
