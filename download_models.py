"""Explicit one-time downloader for the optional local AI model weights.

Run this script only when you want to download pretrained checkpoints and have
an internet connection. The main application never downloads model files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS = (
    ("common", "yolo26m.pt"),
    ("open-vocabulary", "yolov8l-worldv2.pt"),
)


def main():
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for mode, filename in MODELS:
        target = MODELS_DIR / filename
        if target.exists():
            print(f"Already present: {target}")
            continue
        print(f"Downloading pretrained weights for {filename}…")
        model = YOLO(filename, verbose=False) if mode == "common" else YOLOWorld(filename, verbose=False)
        source_value = getattr(model, "ckpt_path", None)
        source = Path(source_value) if source_value else None
        if source is None or not source.is_file():
            raise RuntimeError(
                f"Ultralytics loaded {filename}, but its local checkpoint path could not be located. "
                "Download the checkpoint manually from the official Ultralytics model documentation."
            )
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        print(f"Saved: {target}")

    print("\nDone. Inference uses these local files and does not need network access.")


if __name__ == "__main__":
    main()
