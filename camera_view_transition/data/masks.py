import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


def parse_masks(file: Path, label: str = "person") -> List[np.ndarray]:
    """Parse masks from a LabelMe JSON file.
    Supports the following shape types: mask, polygon, rectangle.

    Args:
        file: Path to the LabelMe JSON file.
        label: Label to filter the shapes.

    Returns:
        List[np.ndarray]: Binary masks of shape (H, W), where H and W are the image dimensions.
    """
    with open(file, "r") as f:
        labelme_data = json.load(f)
    height, width = labelme_data["imageHeight"], labelme_data["imageWidth"]
    masks = []

    for shape in labelme_data["shapes"]:
        if shape["label"] != label:
            continue
        masks.append(parse_shape_mask(shape, height, width))

    return masks


def parse_shape_mask(shape_dict: Dict[str, Any], height: int, width: int) -> np.ndarray:
    """Parse a single shape mask from LabelMe shape data.

    Args:
        shape_dict: LabelMe shape data.
        height: Height of the image.
        width: Width of the image.

    Returns:
        np.ndarray: Binary mask of shape (H, W), where H and W are the image dimensions.

    Raises:
        ValueError: Unsupported shape type.
    """
    if shape_dict["shape_type"] == "mask":
        mask_data = base64.b64decode(shape_dict["mask"])
        person_mask_in_bounding_box = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        x1, y1 = map(int, shape_dict["points"][0])
        x2, y2 = map(int, shape_dict["points"][1])

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2 + 1, x1:x2 + 1] = person_mask_in_bounding_box
        return mask

    if shape_dict["shape_type"] == "polygon":
        points = np.array(shape_dict["points"], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        return mask

    if shape_dict["shape_type"] == "rectangle":
        x1, y1 = map(int, shape_dict["points"][0])
        x2, y2 = map(int, shape_dict["points"][1])
        mask = np.zeros((height, width), dtype=np.uint8)

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        mask[y1:y2 + 1, x1:x2 + 1] = 1
        return mask

    raise ValueError(f"Unsupported shape type: {shape_dict['shape_type']}")


def segment_players(frame: np.ndarray, yolo: YOLO, **yolo_kwargs) -> List[np.ndarray]:
    """Get player masks from a frame with YOLO.

    Args:
        frame: BGR frame in (H, W, 3) shape.
        yolo: YOLO object.
        yolo_kwargs: Keyword arguments for the inference.

    Returns:
        List[np.ndarray]: Binary player masks of shape (H, W), where H and W are the image dimensions.
    """
    results = yolo(frame, classes=[0], **yolo_kwargs)[0]
    masks = []
    for yolo_mask in results.masks.data.cpu().numpy():
        masks.append(cv2.resize(yolo_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST))
    return masks
