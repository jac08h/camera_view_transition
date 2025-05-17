import argparse
from pathlib import Path

import cv2

from camera_view_transition.camera import Camera
from camera_view_transition.camera_view_transition import CameraViewTransition
from camera_view_transition.image import visualize_player_positions


def build_argparser() -> argparse.ArgumentParser:
    """Build an argument parser for the script.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate a target view for a given camera based on input frames."
    )
    parser.add_argument("--root_dir", type=Path, required=True, help="Root directory of the dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to the configuration file")
    parser.add_argument("--input_event_ids", nargs='+', required=True, help="List of input event IDs")
    parser.add_argument("--target_event_id", required=True, help="Target event ID")
    parser.add_argument("--output_dir", type=Path, help="Output directory", default="output")
    parser.add_argument("--name_prefix", type=str, help="Prefix for the output files", default="")
    return parser


def main():
    args = build_argparser().parse_args()

    cvt = CameraViewTransition(config=args.config,
                               root_dir=args.root_dir,
                               input_event_ids=args.input_event_ids)

    pitch_template = visualize_player_positions((cvt.frames[0], cvt.frames[1]),
                                                cvt.player_groups,
                                                cvt.pitch)

    target_frame_path = args.root_dir / f"{args.target_event_id}.png"
    target_camera_path = args.root_dir / f"{args.target_event_id}_camera.xml"
    target_actual = cv2.imread(str(target_frame_path))

    target_simulated = cvt.create_target_view(Camera.load_from_xml(target_camera_path))
    combined = cv2.vconcat([cvt.frames[0].frame, cvt.frames[1].frame, target_simulated, target_actual])

    args.output_dir.mkdir(exist_ok=True)
    prefix = "" if not args.name_prefix else f"{args.name_prefix}_"
    for title, img in zip(["pitch_template", "simulated_view", "actual_frame", "combined"],
                          [pitch_template, target_simulated, target_actual, combined]):
        cv2.imwrite(args.output_dir / f"{prefix}{title}.png", img)


if __name__ == "__main__":
    main()
