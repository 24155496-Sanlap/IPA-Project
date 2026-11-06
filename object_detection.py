"""Local pretrained object detection for PIXELFORGE.

The module never downloads model weights. Install weights manually and pass a
local file path. All public image inputs/outputs use contiguous RGB uint8 arrays.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import cv2
import numpy as np


DEFAULT_COMMON_MODEL = "models/yolo26m.pt"
DEFAULT_OPEN_VOCAB_MODEL = "models/yolov8l-worldv2.pt"


def validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """Validate and return an RGB uint8 image without mutating the caller's data."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError("Image must have dtype np.uint8.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image cannot be empty.")
    return np.ascontiguousarray(image)


def parse_class_prompts(prompts: str) -> list[str]:
    """Parse comma/newline-separated open-vocabulary labels with de-duplication."""
    if not isinstance(prompts, str):
        raise TypeError("Object prompts must be text.")
    labels = []
    seen = set()
    for raw in re.split(r"[,\n;]+", prompts):
        label = " ".join(raw.strip().split())
        key = label.casefold()
        if label and key not in seen:
            labels.append(label)
            seen.add(key)
    if not labels:
        raise ValueError("Enter at least one object category or text description.")
    if len(labels) > 80:
        raise ValueError("Use at most 80 object prompts per detection run.")
    return labels


@lru_cache(maxsize=2)
def _load_model(mode: str, model_path: str):
    """Load a local model once per path. Network access is not used here."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights were not found:\n{path}\n\n"
            "Download the pretrained weights separately and select the local .pt file. "
            "PIXELFORGE does not download weights automatically."
        )

    try:
        if mode == "common":
            from ultralytics import YOLO
            return YOLO(str(path), verbose=False)
        if mode == "open-vocabulary":
            from ultralytics import YOLOWorld
            return YOLOWorld(str(path), verbose=False)
    except ImportError as exc:
        raise RuntimeError(
            "The AI dependencies are missing. Install requirements.txt, including ultralytics."
        ) from exc
    raise ValueError(f"Unsupported detection mode: {mode!r}")


def detect_objects(
    image: np.ndarray,
    *,
    mode: str = "common",
    model_path: str = DEFAULT_COMMON_MODEL,
    prompts: str = "person, car, dog, cat, bicycle",
    confidence: float = 0.35,
    image_size: int = 960,
    max_detections: int = 300,
) -> tuple[list[dict], np.ndarray]:
    """Detect objects locally and return ``(detections, annotated_rgb_image)``.

    ``mode='common'`` uses the model's pretrained classes. ``mode='open-vocabulary'``
    uses user-provided text labels with YOLO-World. The original image is never
    modified. Confidence is in [0, 1].
    """
    rgb = validate_rgb_image(image)
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if not 320 <= int(image_size) <= 1536:
        raise ValueError("Image size must be between 320 and 1536 pixels.")
    if int(max_detections) < 1:
        raise ValueError("Maximum detections must be at least 1.")
    if mode not in {"common", "open-vocabulary"}:
        raise ValueError(f"Unsupported detection mode: {mode!r}")

    requested_path = Path(model_path).expanduser()
    if not requested_path.is_absolute() and not requested_path.is_file():
        project_relative = Path(__file__).resolve().parent / requested_path
        if project_relative.is_file():
            requested_path = project_relative
    model = _load_model(mode, str(requested_path.resolve()))
    if mode == "open-vocabulary":
        model.set_classes(parse_class_prompts(prompts))

    # Ultralytics accepts NumPy images in BGR order. Convert only at this boundary.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(
        source=bgr,
        conf=float(confidence),
        imgsz=int(image_size),
        max_det=int(max_detections),
        verbose=False,
        device="cpu",
    )[0]

    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        names = result.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            score = float(box.conf[0].detach().cpu().item())
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append({
                "label": str(label),
                "confidence": score,
                "box": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            })

    # result.plot() produces BGR pixels; convert back to the project's RGB contract.
    annotated_bgr = result.plot()
    annotated_rgb = np.ascontiguousarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    return detections, annotated_rgb
