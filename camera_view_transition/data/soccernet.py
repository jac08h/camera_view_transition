from collections import defaultdict
import json
from pathlib import Path
import tempfile
from typing import List
from zipfile import ZipFile

import cv2
import numpy as np


def save_actions_by_replay_count(soccernet_dir: Path,
                                 min_replays: int = 2,
                                 output_filename: str = "replay_count_map.json"
                                 ) -> None:
    """Save the replays with a minimum number of occurrences.
    The output is a JSON file with the following structure:
    {
        "2": {
            "league": {
                "season": {
                    "match": [action_id1, action_id2, ...]
                } } } }

    Args:
        soccernet_dir: Path to the directory with the SoccerNet-v3 dataset.
        min_replays: Minimum number of replays for an action.
        output_filename: Output JSON filename.
    """
    replay_count_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for annotation_file in soccernet_dir.glob("**/Labels-v3.json"):
        with open(annotation_file) as f:
            annotations = json.load(f)

        replays = annotations["GameMetadata"]["list_replays"]

        parts = annotation_file.parts
        league, season, match = parts[-4], parts[-3], parts[-2]

        action_replays = defaultdict(int)
        for replay in replays:
            action_id = int(replay.split("_")[0])
            action_replays[action_id] += 1

        for action_id, replay_count in action_replays.items():
            replay_count_map[replay_count][league][season][match].append(action_id)

    replay_count_map = {k: v for k, v in replay_count_map.items() if k >= min_replays}

    sorted_replay_count_map = dict(sorted(replay_count_map.items(), key=lambda x: int(x[0]), reverse=False))

    with open(output_filename, "w") as f:
        json.dump(sorted_replay_count_map, f, indent=4)


def extract_all_event_frames(soccernet_dir: Path,
                             annotations_path: Path,
                             output_dir: Path,
                             visualize: bool = False
                             ) -> None:
    """Extract event frames for all events in an annotations file from their ZIP files.

    Args:
        soccernet_dir: Path to the directory with the SoccerNet-v3 dataset.
        annotations_path: Path to the annotations file.
        output_dir: Path to the output directory.
        visualize: Whether to visualize the extracted frames in a combined image.
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    for league, season_data in annotations.items():
        for season, match_data in season_data.items():
            for match, event_groups in match_data.items():
                frames_zip = soccernet_dir / league / season / match / f"Frames-v3.zip"
                for event_ids in event_groups:
                    print(f"Extracting {len(event_ids)} events from {league}/{season}/{match}")
                    images = extract_event_frames_from_zip(frames_zip, event_ids, output_dir)
                    if visualize:
                        visualize_group(images, event_ids, output_dir / league / season / match)


def extract_event_frames_from_zip(frames_zip: Path,
                                  event_ids: List[str],
                                  output_dir: Path
                                  ) -> List[np.ndarray]:
    """Extract frames from a ZIP file and copy them to the new directory.
    The images are saved in a directory structure that mirrors the soccernet folder structure.

    Args:
        frames_zip: Path to the ZIP file containing the frames.
        event_ids: List of event IDs to extract.
        output_dir: Path to the output directory.

    Returns:
        List[np.ndarray]: List of loaded images, each frame is (H, W, 3) in BGR format.
    """
    parts = frames_zip.parts
    league, season, match = parts[-4], parts[-3], parts[-2]

    if not frames_zip.exists():
        print(f"{frames_zip} not found.")
        return []

    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(frames_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        temp_dir = Path(temp_dir)

        images = [img for event_id in event_ids if
                  (img_path := temp_dir / f"{event_id}.png").exists()
                  and (img := cv2.imread(str(img_path))) is not None]

        if len(images) == 0:
            print(f"No images found for {match}")
            return []

        match_output_dir = output_dir / league / season / match
        match_output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            output_path = match_output_dir / f"{event_ids[i]}.png"
            cv2.imwrite(str(output_path), img)

    return images


def visualize_group(images: List[np.ndarray],
                    event_ids: List[str],
                    output_dir: Path,
                    name_suffix: str = "combined"
                    ) -> None:
    """Combine images into a single image and save it with the event ID as the filename.

    Args:
        images: List of images to combine. Each image is (H, W, 3) in BGR format.
        event_ids: List of event IDs corresponding to the images.
        output_dir: Path to the output directory.
        name_suffix: Suffix to append to the output filename.
    """
    if len(images) == 0:
        return

    max_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images)

    combined_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    y_offset = 0
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        combined_image[y_offset:y_offset + h, :w] = img
        cv2.putText(combined_image, f"{event_ids[i]}",
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        y_offset += h

    output_path = output_dir / f"{event_ids[0].split('_')[0]}_{name_suffix}.png"
    cv2.imwrite(str(output_path), combined_image)
