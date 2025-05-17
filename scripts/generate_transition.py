import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from camera_view_transition.camera_view_transition import CameraViewTransition
from camera_view_transition.interpolation import interpolate_between_cameras, RotationInterpolation


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
    parser.add_argument("--input_event_ids", nargs="+", required=True, help="List of input event IDs")
    parser.add_argument("--duration", type=float, help="Duration of the transition in seconds", default=2.0)
    parser.add_argument("--output_file", type=Path, help="Path to the output video file",
                        default="transition.mp4")
    parser.add_argument("--padding", type=float, default=0,
                        help="How long to display the start and end frames in seconds")
    parser.add_argument("--padding_source", type=str, default="with_graphics",
                        help="Source of the padding frame: with_graphics or without_graphics")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    video_writer = None
    video_fps = 25
    for (start_id, end_id) in zip(args.input_event_ids, args.input_event_ids[1:]):
        cvt = CameraViewTransition(config=args.config,
                                   root_dir=args.root_dir,
                                   input_event_ids=[start_id, end_id])
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width, _ = cvt.frames[0].frame.shape
            video_writer = cv2.VideoWriter(str(args.output_file), fourcc, video_fps, (width, height))

        steps = int(args.duration * video_fps)
        rotation_type = RotationInterpolation[cvt.config["rendering"]["rotation_interpolation"].upper()]
        interpolated_cameras = interpolate_between_cameras(
            cvt.frames[0].camera,
            cvt.frames[1].camera,
            steps=steps,
            time_step_exponent=cvt.config["rendering"]["time_step_exponent"],
            rotation_type=rotation_type
            )

        if args.padding > 0:
            for _ in range(int(args.padding * video_fps)):
                frame = cvt.frames[0].frame_with_graphics if args.padding_source == "with_graphics" \
                    else cvt.frames[0].frame
                video_writer.write(frame)

        for (t, camera) in tqdm(interpolated_cameras, total=steps, desc="Generating transition"):
            target_simulated = cvt.create_target_view(camera, t)
            video_writer.write(target_simulated)

        if args.padding > 0:
            for _ in range(int(args.padding * video_fps)):
                frame = cvt.frames[1].frame_with_graphics if args.padding_source == "with_graphics" \
                    else cvt.frames[1].frame
                video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":
    main()
