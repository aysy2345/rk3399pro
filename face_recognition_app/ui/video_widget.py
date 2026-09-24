"""Aspect-ratio preserving BGR video display with face overlays."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from face_recognition_app.domain.runtime import FaceOverlay, FrameResult


class VideoWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("videoWidget")
        self.setMinimumSize(480, 300)
        self._image = QImage()
        self._source_size = QSize()
        self._overlays: Sequence[FaceOverlay] = ()

    @property
    def has_frame(self) -> bool:
        return not self._image.isNull()

    @property
    def overlay_count(self) -> int:
        return len(self._overlays)

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        )
        self._image = image.copy()
        self._source_size = QSize(width, height)
        self.update()

    def set_result(self, result: FrameResult) -> None:
        self._overlays = tuple(result.faces)
        self.set_frame(result.frame_bgr)

    def clear(self) -> None:
        self._image = QImage()
        self._source_size = QSize()
        self._overlays = ()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101820"))
        if self._image.isNull():
            painter.setPen(QColor("#91a4b7"))
            painter.setFont(QFont("Microsoft YaHei", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待摄像头画面")
            return

        scaled = self._image.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        origin = QPoint(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
        )
        painter.drawImage(origin, scaled)
        scale_x = scaled.width() / float(self._source_size.width())
        scale_y = scaled.height() / float(self._source_size.height())
        painter.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        for face in self._overlays:
            color = QColor("#22c55e" if face.is_known else "#f59e0b")
            painter.setPen(QPen(color, 3))
            x1, y1, x2, y2 = face.box
            rect = QRectF(
                origin.x() + x1 * scale_x,
                origin.y() + y1 * scale_y,
                max(1.0, (x2 - x1) * scale_x),
                max(1.0, (y2 - y1) * scale_y),
            )
            painter.drawRect(rect)
            label = face.name
            if face.is_known:
                label = "{} {:.1f}%".format(label, face.similarity * 100.0)
            text_rect = painter.fontMetrics().boundingRect(label)
            background = QRectF(
                rect.left(),
                max(0.0, rect.top() - text_rect.height() - 8),
                text_rect.width() + 12,
                text_rect.height() + 6,
            )
            painter.fillRect(background, color)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                background.adjusted(6, 0, -4, 0),
                Qt.AlignVCenter | Qt.AlignLeft,
                label,
            )
