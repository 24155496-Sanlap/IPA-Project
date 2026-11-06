import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QFrame,
)


class ImageViewer(QGraphicsView):
    """
    Image viewer supporting:
    - NumPy RGB images
    - QImage and QPixmap
    - Mouse-wheel zoom
    - Middle-mouse-button panning
    - Fit-to-window
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Graphics scene and image item
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)

        # Viewer state
        self._empty = True
        self._zoom = 0
        self._min_zoom = -10
        self._max_zoom = 30
        self._panning = False
        self._pan_start = None

        # Rendering settings
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setFrameShape(QFrame.Shape.NoFrame)

    # --------------------------------------------------
    # Load and display an image
    # --------------------------------------------------

    def set_image(self, image):
        """
        Accepts:
            - NumPy RGB uint8 array: (height, width, 3)
            - QImage
            - QPixmap
        """

        if isinstance(image, QPixmap):
            pixmap = image

        elif isinstance(image, QImage):
            pixmap = QPixmap.fromImage(image)

        elif isinstance(image, np.ndarray):
            pixmap = self._numpy_to_pixmap(image)

        else:
            raise TypeError(
                "Expected a NumPy array, QImage, or QPixmap."
            )

        self._image_item.setPixmap(pixmap)
        self._empty = pixmap.isNull()

        self._zoom = 0

        if self._empty:
            self._scene.setSceneRect(0, 0, 0, 0)
            return

        self._scene.setSceneRect(
            self._image_item.boundingRect()
        )

        self.fit_image()

    # --------------------------------------------------
    # Convert NumPy RGB image to QPixmap
    # --------------------------------------------------

    @staticmethod
    def _numpy_to_pixmap(image):
        if image.dtype != np.uint8:
            raise TypeError("Image must have dtype np.uint8.")

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError(
                "Expected an RGB image with shape (H, W, 3)."
            )

        image = np.ascontiguousarray(image)

        height, width, _ = image.shape

        qimage = QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()

        return QPixmap.fromImage(qimage)

    # --------------------------------------------------
    # Fit image to viewer
    # --------------------------------------------------

    def fit_image(self):
        if self._empty:
            return

        self.resetTransform()

        self.fitInView(
            self._image_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        self._zoom = 0

    # --------------------------------------------------
    # Zoom controls
    # --------------------------------------------------

    def zoom_in(self):
        if self._empty or self._zoom >= self._max_zoom:
            return

        self.scale(1.25, 1.25)
        self._zoom += 1

    def zoom_out(self):
        if self._empty or self._zoom <= self._min_zoom:
            return

        self.scale(0.8, 0.8)
        self._zoom -= 1

    def wheelEvent(self, event):
        if self._empty:
            return

        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()
        else:
            event.ignore()
            return
        event.accept()

    # --------------------------------------------------
    # Pan using middle mouse button
    # --------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()

            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            current_pos = event.position().toPoint()
            delta = current_pos - self._pan_start

            self._pan_start = current_pos

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )

            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_start = None

            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    # --------------------------------------------------
    # Resize handling
    # --------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self._zoom == 0:
            self.fit_image()