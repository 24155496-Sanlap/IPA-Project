"""Image and JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def get_json(file_name):
    """Load UTF-8 JSON and report malformed files clearly."""
    path = Path(file_name)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def img_to_np_array(file_path):
    """Read an image from disk and return a contiguous RGB uint8 array."""
    path = str(file_path)
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not decode image file: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image_rgb)


def save_rgb_image(file_path, image):
    """Save an RGB uint8 NumPy image, converting to OpenCV's BGR order."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected a NumPy image array.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image with shape (H, W, 3).")
    if image.size == 0:
        raise ValueError("Cannot save an empty image.")
    path = str(file_path)
    success = cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f"Could not write image file: {path}")
