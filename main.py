"""PIXELFORGE image-filter studio entry point."""

from __future__ import annotations

import sys
import cv2
from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QButtonGroup, QWidget, QHBoxLayout,
    QVBoxLayout, QLabel, QFileDialog, QMessageBox, QScrollArea,
)

from filters import FILTER_CATEGORIES, name_function_map
from helper import img_to_np_array, save_rgb_image
from image_viewer import ImageViewer
from param_ui_parser import ParserUI
from ai_panel import AIPanel
from ai_worker import DetectionWorker


def build_filter_kwargs(filter_name: str, values: dict) -> dict:
    """Convert individual UI controls into the grouped arguments filters expect."""
    values = dict(values)
    if filter_name == "vignette":
        values["center"] = (values.pop("center_x"), values.pop("center_y"))
    elif filter_name == "monochrome":
        values["tint"] = tuple(values.pop(f"tint_{channel}") for channel in "rgb")
    elif filter_name == "cinematic":
        values["shadow_tint"] = tuple(
            values.pop(f"shadow_tint_{channel}") for channel in "rgb"
        )
        values["highlight_tint"] = tuple(
            values.pop(f"highlight_tint_{channel}") for channel in "rgb"
        )
    return values


class LeftPanel(QWidget):
    filter_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        self.filter_button_group = QButtonGroup(self)
        self.filter_button_group.setExclusive(True)

        ai_title = QLabel("AI tools")
        ai_title.setWordWrap(True)
        ai_title.setStyleSheet("font-weight: 600; margin-top: 8px;")
        layout.addWidget(ai_title)
        ai_button = QPushButton("AI Object Detection")
        ai_button.setCheckable(True)
        ai_button.setToolTip("Run local pretrained common-object or open-vocabulary detection")
        self.filter_button_group.addButton(ai_button)
        ai_button.clicked.connect(
            lambda checked=False: self.filter_selected.emit("__ai_object_detection__")
        )
        layout.addWidget(ai_button)

        for category, filter_names in FILTER_CATEGORIES.items():
            title = QLabel(category)
            title.setWordWrap(True)
            title.setStyleSheet("font-weight: 600; margin-top: 8px;")
            layout.addWidget(title)
            for filter_name in filter_names:
                if filter_name not in name_function_map:
                    continue
                button = QPushButton(filter_name.replace("/", " / ").title())
                button.setCheckable(True)
                button.setToolTip(filter_name.replace("/", " / ").title())
                self.filter_button_group.addButton(button)
                button.clicked.connect(
                    lambda checked=False, name=filter_name: self.filter_selected.emit(name)
                )
                layout.addWidget(button)
        layout.addStretch()


class MiddlePanel(QWidget):
    image_loaded = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        viewers = QHBoxLayout()
        original_column = QVBoxLayout()
        result_column = QVBoxLayout()
        original_column.addWidget(QLabel("Original"))
        result_column.addWidget(QLabel("Preview"))
        self.old_image_view = ImageViewer()
        self.new_image_view = ImageViewer()
        original_column.addWidget(self.old_image_view)
        result_column.addWidget(self.new_image_view)
        viewers.addLayout(original_column, 1)
        viewers.addLayout(result_column, 1)
        layout.addLayout(viewers, 1)

        actions = QHBoxLayout()
        self.open_button = QPushButton("Open image")
        self.save_button = QPushButton("Export result")
        self.save_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_image)
        self.save_button.clicked.connect(self.save_image)
        actions.addWidget(self.open_button)
        actions.addWidget(self.save_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.original_image = None
        self.result_image = None

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not path:
            return
        try:
            image = img_to_np_array(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Unable to open image", str(exc))
            return
        self.original_image = image
        self.result_image = image.copy()
        self.old_image_view.set_image(image)
        self.new_image_view.set_image(image)
        self.save_button.setEnabled(True)
        self.image_loaded.emit(image)

    def update_preview(self, image):
        self.result_image = image
        self.new_image_view.set_image(image)

    def save_image(self):
        if self.result_image is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export image", "edited.png",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;BMP image (*.bmp)"
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".jpg" if "JPEG" in selected_filter else ".png"
        try:
            save_rgb_image(path, self.result_image)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Unable to export image", str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PIXELFORGE — Image Processing Studio")
        self.setMinimumSize(QSize(1000, 650))
        self.original_image = None
        self.current_filter = None
        self._image_revision = 0
        self._detection_revision = None

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        self.left_panel = LeftPanel()
        self.left_panel.setMinimumWidth(175)
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setWidget(self.left_panel)
        self.left_scroll.setMinimumWidth(195)
        self.left_scroll.setMaximumWidth(245)
        root.addWidget(self.left_scroll)

        self.middle_panel = MiddlePanel()
        root.addWidget(self.middle_panel, 1)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.parser = ParserUI()
        self.parser.setMinimumWidth(270)
        self.parser_scroll = QScrollArea()
        self.parser_scroll.setWidgetResizable(True)
        self.parser_scroll.setWidget(self.parser)
        self.parser_scroll.setMinimumWidth(285)
        right_layout.addWidget(self.parser_scroll)

        self.ai_panel = AIPanel()
        self.ai_scroll = QScrollArea()
        self.ai_scroll.setWidgetResizable(True)
        self.ai_scroll.setWidget(self.ai_panel)
        self.ai_scroll.setMinimumWidth(285)
        self.ai_scroll.hide()
        right_layout.addWidget(self.ai_scroll)
        root.addWidget(self.right_panel)

        self._detection_worker = None

        self.setCentralWidget(central)
        self.left_panel.filter_selected.connect(self.select_filter)
        self.middle_panel.image_loaded.connect(self.set_original_image)
        self.parser.value_changed.connect(self.update_preview)
        self.ai_panel.run_requested.connect(self.run_object_detection)

        self.statusBar().showMessage("Open an image, then choose a filter or AI Object Detection.")

    def set_original_image(self, image):
        self._image_revision += 1
        self.original_image = image
        self.ai_panel.set_image_available(True)
        self.update_preview()

    def select_filter(self, filter_name):
        if filter_name == "__ai_object_detection__":
            self.current_filter = None
            self.parser_scroll.hide()
            self.ai_scroll.show()
            self.ai_panel.set_image_available(self.original_image is not None)
            self.statusBar().showMessage(
                "AI mode: choose a local model and run detection. First inference may take a while."
            )
            return

        self.ai_scroll.hide()
        self.parser_scroll.show()
        self.current_filter = filter_name
        try:
            self.parser.initialize(filter_name)
        except (KeyError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Filter configuration error", str(exc))
            self.current_filter = None
            return
        self.statusBar().showMessage(f"Selected: {filter_name}")
        self.update_preview()

    def run_object_detection(self, options):
        if self.original_image is None:
            QMessageBox.information(self, "No image", "Open an image before running object detection.")
            return
        if self._detection_worker is not None and self._detection_worker.isRunning():
            return

        self.ai_panel.results.setPlainText("Starting local model…")
        self.ai_panel.set_busy(True)
        self._detection_revision = self._image_revision
        self._detection_worker = DetectionWorker(self.original_image, options, self)
        self._detection_worker.progress.connect(self.statusBar().showMessage)
        self._detection_worker.succeeded.connect(
            lambda detections, image, revision=self._detection_revision:
                self._on_detection_success(detections, image, revision)
        )
        self._detection_worker.failed.connect(self._on_detection_failed)
        self._detection_worker.finished.connect(self._on_detection_finished)
        self._detection_worker.start()

    def _on_detection_success(self, detections, annotated_image, revision):
        if revision != self._image_revision:
            self.statusBar().showMessage(
                "Detection finished for the previous image; its result was discarded."
            )
            return
        self.ai_panel.set_results(detections)
        self.middle_panel.update_preview(annotated_image)
        self.statusBar().showMessage(
            f"Local detection complete: {len(detections)} object(s) found."
        )

    def _on_detection_failed(self, message):
        self.ai_panel.results.setPlainText("Detection failed. See the message for details.")
        QMessageBox.warning(self, "Object detection failed", message)
        self.statusBar().showMessage("Object detection failed.")

    def _on_detection_finished(self):
        self.ai_panel.set_busy(False)
        self._detection_worker = None

    def update_preview(self, *_):
        if self.original_image is None or self.current_filter is None:
            return
        try:
            values = self.parser.get_values()
            kwargs = build_filter_kwargs(self.current_filter, values)
            result = name_function_map[self.current_filter](
                self.original_image, **kwargs
            )
            self.middle_panel.update_preview(result)
        except (TypeError, ValueError, KeyError, cv2.error) as exc:
            QMessageBox.warning(self, "Filter error", str(exc))
            self.statusBar().showMessage(f"Could not apply {self.current_filter}: {exc}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
