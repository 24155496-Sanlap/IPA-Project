"""Qt worker thread for keeping local AI inference off the GUI thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from object_detection import detect_objects


class DetectionWorker(QThread):
    succeeded = Signal(object, object)  # detections, annotated RGB image
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, image, options, parent=None):
        super().__init__(parent)
        self._image = image.copy()
        self._options = dict(options)

    def run(self):
        try:
            self.progress.emit("Running local object detection…")
            detections, annotated = detect_objects(self._image, **self._options)
            self.succeeded.emit(detections, annotated)
        except Exception as exc:  # surface model/runtime errors to the UI
            self.failed.emit(str(exc))
