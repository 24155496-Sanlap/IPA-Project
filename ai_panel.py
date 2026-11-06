"""User interface controls for local pretrained object detection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QDoubleSpinBox, QFileDialog, QPlainTextEdit,
    QFormLayout,
)

from object_detection import DEFAULT_COMMON_MODEL, DEFAULT_OPEN_VOCAB_MODEL


class AIPanel(QWidget):
    run_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(QLabel("Local AI Object Detection"))
        intro = QLabel(
            "Runs on your computer using local pretrained weights. "
            "No image is uploaded and this panel never downloads weights."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Common objects", "common")
        self.mode.addItem("Open-vocabulary (text prompts)", "open-vocabulary")
        form.addRow("Detection mode", self.mode)

        self.model_path = QLineEdit(DEFAULT_COMMON_MODEL)
        self.model_browse = QPushButton("Browse…")
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_path, 1)
        model_layout.addWidget(self.model_browse)
        form.addRow("Local model file", model_row)

        self.prompts = QPlainTextEdit("person, car, dog, cat, bicycle, chair")
        self.prompts.setPlaceholderText(
            "Comma-separated categories, e.g. red backpack, wooden chair, helmet"
        )
        self.prompts.setMaximumHeight(100)
        form.addRow("Object prompts", self.prompts)

        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.confidence.setValue(0.35)
        form.addRow("Confidence threshold", self.confidence)

        self.image_size = QComboBox()
        for size in (640, 800, 960, 1280, 1536):
            self.image_size.addItem(str(size), size)
        self.image_size.setCurrentIndex(2)
        form.addRow("Inference resolution", self.image_size)
        root.addLayout(form)

        self.model_note = QLabel(
            "Common model: models/yolo26m.pt\n"
            "Open-vocabulary model: models/yolov8l-worldv2.pt"
        )
        self.model_note.setWordWrap(True)
        root.addWidget(self.model_note)

        self.run_button = QPushButton("Detect objects")
        self.run_button.setEnabled(False)
        root.addWidget(self.run_button)
        root.addWidget(QLabel("Detections"))
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setPlaceholderText("Detected objects will appear here.")
        root.addWidget(self.results, 1)
        root.addStretch()

        self.mode.currentIndexChanged.connect(self._update_model_default)
        self.model_browse.clicked.connect(self._browse_model)
        self.run_button.clicked.connect(self._emit_request)
        self._update_model_default()

    def set_image_available(self, available: bool):
        self.run_button.setEnabled(bool(available))

    def set_busy(self, busy: bool):
        self.run_button.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.model_path.setEnabled(not busy)
        self.model_browse.setEnabled(not busy)
        self.prompts.setEnabled(not busy)
        self.confidence.setEnabled(not busy)
        self.image_size.setEnabled(not busy)
        if busy:
            self.run_button.setText("Detecting…")
        else:
            self.run_button.setText("Detect objects")

    def set_results(self, detections):
        if not detections:
            self.results.setPlainText("No objects detected at this confidence threshold.")
            return
        lines = [f"Detected {len(detections)} object(s):"]
        for index, item in enumerate(detections, 1):
            lines.append(
                f"{index}. {item['label']} — {item['confidence']:.1%} "
                f"at {item['box']}"
            )
        self.results.setPlainText("\n".join(lines))

    def _update_model_default(self):
        mode = self.mode.currentData()
        default = DEFAULT_COMMON_MODEL if mode == "common" else DEFAULT_OPEN_VOCAB_MODEL
        current = self.model_path.text().strip()
        # Switch the suggested path only if it is blank or still one of the defaults.
        known_defaults = {DEFAULT_COMMON_MODEL, DEFAULT_OPEN_VOCAB_MODEL}
        if not current or current in known_defaults:
            self.model_path.setText(default)
        self.prompts.setEnabled(mode == "open-vocabulary")

    def _browse_model(self):
        current = self.model_path.text().strip()
        start = str(Path(current).expanduser().parent) if current else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select local pretrained weights", start,
            "Model weights (*.pt *.onnx *.engine);;All files (*)"
        )
        if path:
            self.model_path.setText(path)

    def _emit_request(self):
        options = {
            "mode": self.mode.currentData(),
            "model_path": self.model_path.text().strip(),
            "prompts": self.prompts.toPlainText(),
            "confidence": self.confidence.value(),
            "image_size": int(self.image_size.currentData()),
        }
        self.run_requested.emit(options)
