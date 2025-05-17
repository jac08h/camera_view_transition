from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO
import yaml

from camera_view_transition.camera import Camera
from camera_view_transition.coordinate_transform import world_to_image
from camera_view_transition.data.masks import parse_masks, segment_players
from camera_view_transition.frame import Frame
from camera_view_transition.goal_plane import create_goal_planes
from camera_view_transition.image import get_bounding_box_relative_coordinates, overlay_image
from camera_view_transition.player import Player
from camera_view_transition.player_group import PlayerGroup
from camera_view_transition.soccer_pitch import SoccerPitch
from camera_view_transition.stadium_plane import create_stadium_planes, StadiumPlane
from camera_view_transition.utils import get_billboard, match_points


class CameraViewTransition:
    """Class to generate new views from existing frames.

    Args:
        config: Path to the configuration file.

    Raises:
        ValueError: If the number of loaded frames is not exactly two.
    """

    def __init__(self, config: Path, root_dir: Path, input_event_ids: List[str]) -> None:
        with open(config, "r") as file:
            self.config = yaml.safe_load(file)

        self.yolo = YOLO(self.config["segmentation"]["model"]) if self.config["segmentation"]["use_yolo"] else None
        self.frames = self.load_frames(root_dir, input_event_ids)
        if len(self.frames) != 2:
            raise ValueError("Currently, only two input frames are supported.")

        self.image_width, self.image_height = self.frames[0].width, self.frames[0].height

        self.pitch = SoccerPitch(
            cv2.imread(self.config["pitch"]["template"]),
            width_meters=self.config["pitch"]["width"],
            height_meters=self.config["pitch"]["height"],
            width_pixels=self.image_width * self.config["pitch"]["texture_scale"],
            height_pixels=self.image_height * self.config["pitch"]["texture_scale"],
            width_behind_pitch=int(self.config["pitch"]["width_behind"] + self.config["pitch"]["padding"]),
            height_behind_pitch=int(self.config["pitch"]["height_behind"] + self.config["pitch"]["padding"])
        )

        self.stadium_planes = create_stadium_planes(
            (int(self.image_width * self.config["stadium"]["texture_scale"]),
             int(self.image_height * self.config["stadium"]["texture_scale"])),
            self.config["pitch"]["width"],
            self.config["pitch"]["width_behind"],
            self.config["pitch"]["height"],
            self.config["pitch"]["height_behind"],
            self.config["stadium"]["height"]
        )

        self.goal_planes = create_goal_planes(
            (int(self.image_width * self.config["goal"]["texture_scale"]),
             int(self.image_height * self.config["goal"]["texture_scale"])),
            self.config["pitch"]["width"],
            self.config["goal"]["width"],
            self.config["goal"]["height"],
            self.config["goal"]["depth"]
        )

        self.backgrounds_without_goals = {}
        goal_plane_coordinates = [goal_plane.get_world_coordinates() for goal_plane in self.goal_planes]
        for frame in self.frames:
            self.backgrounds_without_goals[frame] = Frame.inpaint_image_coordinates(
                frame.background, goal_plane_coordinates, frame.camera,
                self.image_width, self.image_height)

        self.background_planes = [self.pitch] + self.stadium_planes + self.goal_planes

        balls = [frame.ball for frame in self.frames if frame.ball is not None]
        self.player_groups = self.match_players(self.frames, self.config["rendering"]["max_player_distance"])
        if len(balls) > 0 and not self.config["rendering"]["treat_ball_as_background"]:
            self.player_groups.append(PlayerGroup(balls))
        self.player_group_switch_ts: Dict[PlayerGroup, float] = defaultdict(None)

        self.load_textures()

    def load_frames(self, root_dir: Path, event_ids: List[str]) -> List[Frame]:
        """Load frames from the given event IDs.

        Args:
            root_dir: Root directory of the frames.
            event_ids: List of event IDs to load frames. For each event ID, there should be a
                frame image {event_id}.png, a labelme annotation {event_id}.json, and a camera
                calibration file {event_id}_camera.xml.

        Returns:
            List[Frame]: List of loaded frames.
        """
        frames = []
        for event_id in event_ids:
            frame_labelme_path = root_dir / f"{event_id}.json"

            loaded_frame = cv2.imread(str(root_dir / f"{event_id}.png"))
            loaded_frame = Frame.inpaint_image(loaded_frame, parse_masks(frame_labelme_path, label="graphics"))
            camera = Camera.load_from_xml(root_dir / f"{event_id}_camera.xml")
            ball_masks = parse_masks(frame_labelme_path, label="ball")
            if len(ball_masks) > 1:
                logging.warning("More than one ball detected in %s.png.", event_id)

            if self.yolo is None:
                player_masks = parse_masks(frame_labelme_path, label="person")
            else:
                player_masks = segment_players(loaded_frame, self.yolo, **self.config["segmentation"]["kwargs"])

            frames.append(Frame(frame=loaded_frame,
                                frame_with_graphics=cv2.imread(str(root_dir / f"{event_id}.png")),
                                player_masks=player_masks,
                                ball_mask=ball_masks[0] if ball_masks else None,
                                camera=camera,
                                billboard_dimensions=self.config["billboards"]))
        return frames

    def load_textures(self) -> None:
        """Load textures from input frames for the pitch and stadium planes."""
        plane_to_textures = defaultdict(list)

        for frame in self.frames:
            for plane in self.background_planes:
                if (isinstance(plane, StadiumPlane) and
                       not plane.is_visible(frame.camera, self.config["rendering"]["max_camera_plane_angle"])):
                    continue

                texture_corners_in_frame = world_to_image(plane.get_world_coordinates(),
                                                          frame.camera.projection_matrix,
                                                          self.image_width,
                                                          self.image_height)
                homography, _ = cv2.findHomography(texture_corners_in_frame,
                                                   plane.image_coordinates)
                texture = self.backgrounds_without_goals[frame] if isinstance(plane, SoccerPitch) else frame.background
                plane_to_textures[plane].append(
                    cv2.warpPerspective(texture, homography, (plane.width, plane.height), flags=cv2.INTER_NEAREST))

        for plane in self.background_planes:
            plane.load_textures(plane_to_textures[plane])

        self.background_planes = [plane for plane in self.background_planes if not plane.is_empty()]

    @staticmethod
    def match_players(frames: List[Frame], max_distance: float) -> List[PlayerGroup]:
        """Match players between two frames using the Hungarian algorithm based on the Euclidean distance.

        Args:
            frames: List containing the two frames between which to match players.
            max_distance: Maximum distance for two players which can be considered as the same player.

        Returns:
            List[PlayerGroup]: List of matched player groups.
        """
        matching_indices, _ = match_points(frames[0].foot_positions_world,
                                           frames[1].foot_positions_world,
                                           threshold=max_distance)
        matched_players = [PlayerGroup([frames[0].players[i], frames[1].players[j]])
                           for i, j in matching_indices]

        for frame_index, frame in enumerate(frames):
            matched_indices = set(m[frame_index] for m in matching_indices)
            unmatched_indices = [i for i in range(len(frame.players)) if i not in matched_indices]
            matched_players.extend(PlayerGroup([frame.players[i]]) for i in unmatched_indices)

        return matched_players

    def create_target_view(self, target_camera: Camera, t: float = 0.5) -> np.ndarray:
        """Create an image which simulates the view from the target camera.

        Args:
            target_camera: Camera object for the target view.
            t: Interpolation parameter between the two input frames (in [0, 1]).

        Returns:
            np.ndarray: Simulated view from the target camera.
        """
        background = self.create_target_background(target_camera, t)
        if self.config["rendering"]["inpaint_missing_background"]:
            missing_area_mask = cv2.inRange(background, np.array([0, 0, 0]), np.array([0, 0, 0]))
            background = Frame.inpaint_image(background, [missing_area_mask])

        background_with_players = self.add_players_to_target_view(target_camera, background, t)

        if t == 1:
            self.on_interpolation_end()

        return background_with_players

    def create_target_background(self, target_camera: Camera, t: float) -> np.ndarray:
        """Create the background for the target view.

        Args:
            target_camera: Camera object for the target view.
            t: Interpolation parameter between the two input frames (in [0, 1]).

        Returns:
            np.ndarray: Background image for the target view.
        """
        background = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        for plane in self.background_planes:
            if (isinstance(plane, StadiumPlane) and
                   not plane.is_visible(target_camera, self.config["rendering"]["max_camera_plane_angle"])):
                continue
            corners_in_frame = world_to_image(plane.get_world_coordinates(),
                                              target_camera.projection_matrix,
                                              self.image_width,
                                              self.image_height)
            homography, _ = cv2.findHomography(plane.image_coordinates, corners_in_frame)
            alphas = [1 - t, t] if len(plane.textures) == 2 else [1]
            warped_plane = cv2.warpPerspective(plane.render(alphas),
                                               homography,
                                               (self.image_width, self.image_height),
                                               flags=cv2.INTER_NEAREST)
            non_empty_plane = warped_plane != 0
            background[non_empty_plane] = warped_plane[non_empty_plane]

        return background

    def add_players_to_target_view(self, target_camera: Camera, background_image: np.ndarray, t: float) -> np.ndarray:
        """Add players to the target view. Players are rendered in the order of their distance from the camera,
        so that closer players are rendered on top of farther players.

        Args:
            target_camera: Camera object for the target view.
            background_image: Background image for the target view.
            t: Interpolation parameter between the two input frames (in [0, 1]).

        Returns:
            np.ndarray: Image with players added to the target view.
        """
        player_groups_sorted_by_distance_from_camera = sorted(
            self.player_groups,
            key=lambda g: np.linalg.norm(g.get_position(t)[:2] - target_camera.position[:2]),
            reverse=True
        )

        output_image = background_image.copy()
        for player_group in player_groups_sorted_by_distance_from_camera:
            self.add_player_to_target_view(player_group,
                                           target_camera,
                                           output_image,
                                           t)

        return output_image

    def add_player_to_target_view(self,
                                  player_group: PlayerGroup,
                                  target_camera: Camera,
                                  output_image: np.ndarray,
                                  t: float = 0.5,
                                  ) -> None:
        """Add the player to the target view.

        Args:
            player_group: Player group to add to the target view.
            target_camera: Camera object for the target view.
            output_image: Current output image. Will be modified in-place.
            t: Interpolation parameter between the two input frames (in [0, 1]) to determine the alpha
                values of the players for blending.
        """
        target_billboard = get_billboard(player_group.get_position(t),
                                         target_camera,
                                         player_group.players[0].billboard_width,
                                         player_group.players[0].billboard_height)

        target_frame_coordinates = world_to_image(target_billboard,
                                                  target_camera.projection_matrix,
                                                  width=output_image.shape[1],
                                                  height=output_image.shape[0])

        best_player = player_group.select_player_from_best_view(target_billboard)
        best_player_alpha = self.get_best_player_alpha(player_group, best_player, t)

        for player in player_group.players:
            source_normalized = get_bounding_box_relative_coordinates(player.image_billboard_coordinates)
            target_normalized = get_bounding_box_relative_coordinates(target_frame_coordinates)

            homography, _ = cv2.findHomography(source_normalized, target_normalized)

            target_width, target_height = tuple(map(lambda x: int(np.ceil(x)), target_normalized[3]))
            if target_width > output_image.shape[1] or target_height > output_image.shape[0]:
                logging.warning("Player dimensions too large: %dx%d. Skipping.", target_width, target_height)
                continue

            warped_player = cv2.warpPerspective(player.image_billboard, homography, (target_width, target_height),
                                                flags=cv2.INTER_NEAREST)

            target_x, target_y = map(int, target_frame_coordinates[0])
            alpha = best_player_alpha if player == best_player else 1 - best_player_alpha
            overlay_image(warped_player, output_image, (target_x, target_y), alpha=alpha)

    def get_best_player_alpha(self,
                              player_group: PlayerGroup,
                              best_player: Player,
                              t: float,
                              ) -> float:
        """Get the alpha value for the best player in the player group.

        For players present in both frames:
        The alpha decreases linearly from 1 to 0.5 where it stays fixed until there is a switch,
        increases from the value at the switch to 1 linearly.

        For players present in only one frame:
        The alpha decreases linearly from 1 to 0 if the player is present in the first frame,
        and increases from 0 to 1 if the player is present in the second frame.

        Args:
            player_group: Player group.
            best_player: Best player in the group.
            t: Interpolation parameter between the two input frames (in [0, 1]).

        Returns:
            float: Alpha value for the best player.
        """
        if not self.config["rendering"]["blend_players"]:
            return 1.0

        if len(player_group.players) == 1:
            is_from_first_frame = best_player in self.frames[0].players
            return 1 - t if is_from_first_frame else t

        if best_player == player_group.players[0]:
            return max(0.5, 1 - t)

        switch_t = self.player_group_switch_ts.setdefault(player_group, t)
        start_alpha = max(0.5, 1 - switch_t)
        final_alpha = 1.0
        final_t = min(1, switch_t * 2)
        result = start_alpha + (final_alpha - start_alpha) * ((t - switch_t) / (final_t - switch_t))
        return min(1.0, result)

    def on_interpolation_end(self) -> None:
        """Clean the state after the interpolation ends."""
        self.player_group_switch_ts.clear()
