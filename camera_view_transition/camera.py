from pathlib import Path
from typing import Optional

from lxml import objectify
import numpy as np


class Camera:
    """A class for representing a camera in 3D space.

    Args:
        position: The position of the camera in 3D space of shape (3,).
        rotation: The rotation of the camera in 3D space in radians of shape (3,),
            ordered as (pan, roll, tilt).
        focal: The focal length of the camera.
        lean: The lean of the camera in radians.
        projection_matrix: The matrix of the camera of shape (4, 4). If None, it will be computed from the parameters.
        width_to_height_ratio: The width to height ratio of the camera.
        rotation_order: The order of the rotation angles.
    """

    def __init__(self,
                 position: np.ndarray,
                 rotation: np.ndarray,
                 focal: float,
                 lean: float,
                 projection_matrix: Optional[np.ndarray] = None,
                 width_to_height_ratio: float = 16 / 9,
                 rotation_order: str = "zyx"
                 ) -> None:
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
        self.pan, self.roll, self.tilt = self.rotation
        self.focal = focal
        self.lean = lean

        self.width_to_height_ratio = width_to_height_ratio
        self.height_to_width_ratio = 1 / self.width_to_height_ratio

        self._width = 720
        self._height = self._width / self.width_to_height_ratio

        self.projection_matrix = projection_matrix if projection_matrix is not None \
            else self.compute_projection_matrix()

    @classmethod
    def load_from_xml(cls, path: Path) -> "Camera":
        """Load a Camera object from an XML file. The file should have the following structure:
            <Camera>
                <Position x="0" y="0" z="0"/>
                <Rotation pan="0" roll="0" tilt="0"/>
                <Focal value="0"/>
                <Lean value="0"/>
                <Matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</Matrix>
                ...
            </Camera>

        Args:
            path: Path to the XML file.

        Returns:
            Camera: Initialized camera object.
        """
        with open(path, "r") as file:
            camera_data = objectify.fromstring(file.read())

        matrix_values = list(map(float, camera_data["Matrix"].text.split()))
        matrix = np.array(matrix_values).reshape(4, 4)

        position = np.array([float(camera_data.Position.attrib[key]) for key in "xyz"])

        rotation = np.array([float(camera_data.Rotation.attrib[key]) for key in ["pan", "roll", "tilt"]])
        rotation = np.radians(rotation)

        focal = float(camera_data.Focal.attrib["value"])
        lean = np.radians(float(camera_data.Lean.attrib["value"]))

        return cls(position, rotation, focal, lean, matrix)

    def __str__(self) -> str:
        return f"Position: {self.position}, Rotation: {self.rotation}, Focal: {self.focal}, Lean: {self.lean}"

    def compute_projection_matrix(self) -> np.ndarray:
        """Compute the projection matrix for transforming 3D points to the camera's image plane coordinates.

        Returns:
            np.ndarray: The camera matrix of shape (4, 4).
        """
        K = self.get_intrinsic_matrix()
        R = self.get_rotation_matrix()
        T = self.get_translation_matrix()

        KRT = self.convert_to_homogenous_4x4_matrix(K @ R) @ T

        KRT[[2, 3]] = KRT[[3, 2]]
        KRT[2, 2] = 1
        KRT[2, 3] = 0

        return self.get_normalization_matrix() @ KRT

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get the intrinsic matrix of the camera.
        The last column is negative, because the camera is looking in the negative z-direction.

        Returns:
            np.ndarray: The intrinsic matrix of the camera of shape (3, 3).
        """
        return np.array([
            [(self._width / 2) * self.focal * self.height_to_width_ratio, 0, -(self._width / 2)],
            [0, (self._height / 2) * self.focal, -(self._height / 2)],
            [0, 0, -1]
        ])

    def get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix of the camera.

        Returns:
            np.ndarray: The rotation matrix of the camera of shape (3, 3).
        """

        def Rx(angle):
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])

        def Ry(angle):
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

        def Rz(angle):
            return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])

        ground_plane_to_xy_rotation = Rx(-np.pi / 2)
        return Rz(self.roll) @ Rx(self.tilt) @ Ry(self.pan) @ ground_plane_to_xy_rotation @ Rx(self.lean)

    def get_translation_matrix(self) -> np.ndarray:
        """Get the translation matrix of the camera.
        Camera at (x, y, z) means world shifted by (-x, -y, -z)

        Returns:
            np.ndarray: The translation matrix of the camera of shape (4, 4),
                with the translation values in the last column.
        """
        return np.array([
            [1, 0, 0, -self.position[0]],
            [0, 1, 0, -self.position[1]],
            [0, 0, 1, -self.position[2]],
            [0, 0, 0, 1]])

    def get_normalization_matrix(self) -> np.ndarray:
        """Get the normalization matrix of the camera.

        Returns:
            np.ndarray: The normalization matrix of the camera of shape (4, 4).
        """
        norm_matrix = np.linalg.inv(np.array([
            [self._width / 2, 0, self._width / 2],
            [0, self._height / 2, self._height / 2],
            [0, 0, 1]]
        ))

        norm_matrix = np.array([
            [norm_matrix[0, 0], norm_matrix[0, 1], 0, norm_matrix[0, 2]],
            [norm_matrix[1, 0], norm_matrix[1, 1], 0, norm_matrix[1, 2]],
            [0, 0, 1, 0],
            [norm_matrix[2, 0], norm_matrix[2, 1], 0, norm_matrix[2, 2]]
        ])

        return norm_matrix

    @staticmethod
    def convert_to_homogenous_4x4_matrix(matrix: np.ndarray) -> np.ndarray:
        """Convert a 3x3 matrix to a 4x4 homogenous matrix.
        The last row and column are filled with zeros and ones respectively,
        except for the last element which is set to 1.

        Args:
            matrix: The 3x3 matrix to convert.

        Returns:
            np.ndarray: The 4x4 homogenous matrix.
        """
        homogenous = np.eye(4)
        homogenous[:3, :3] = matrix
        return homogenous

    def pitch_intersection(self) -> np.ndarray:
        """Calculate the intersection of the camera's view with the pitch (Z=0).

        Returns:
            np.ndarray: Intersection point of shape (3,) with the z-coordinate set to 0.

        Raises:
            ValueError: If the camera is not facing the pitch.
        """
        direction = np.array([
            np.sin(self.pan) * np.cos(self.tilt),
            np.cos(self.pan) * np.cos(self.tilt),
            -np.sin(self.tilt)
        ])

        if direction[2] >= 0:
            raise ValueError("The camera is not facing the pitch.")

        t = -self.position[2] / direction[2]
        return self.position + t * direction
