"""USB camera boundary and OpenCV implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or used."""


class CameraReadError(CameraError):
    """Raised when an opened camera does not return a valid frame."""


class Camera(ABC):
    @property
    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class OpenCVCamera(Camera):
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        target_fps: int,
        retry_count: int,
        capture_factory: Optional[Callable[[int], Any]] = None,
    ) -> None:
        if index < 0:
            raise ValueError("camera index must be non-negative")
        if width < 1 or height < 1 or target_fps < 1:
            raise ValueError("camera dimensions and target_fps must be positive")
        if retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        self._index = index
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._retry_count = retry_count
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture: Optional[Any] = None

    @property
    def is_open(self) -> bool:
        if self._capture is None:
            return False
        try:
            return bool(self._capture.isOpened())
        except Exception:
            return False

    def open(self) -> None:
        if self.is_open:
            return
        self.release()
        attempts = self._retry_count + 1
        last_error: Optional[Exception] = None
        for _ in range(attempts):
            capture = None
            try:
                capture = self._capture_factory(self._index)
                if capture is not None and capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    capture.set(cv2.CAP_PROP_FPS, self._target_fps)
                    self._capture = capture
                    return
            except Exception as exc:
                last_error = exc
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    last_error = last_error or exc
        message = "unable to open camera {} after {} attempts".format(
            self._index, attempts
        )
        if last_error is not None:
            message = "{}: {}".format(message, last_error)
        raise CameraError(message)

    def read(self) -> np.ndarray:
        if not self.is_open:
            raise CameraError("camera is not open")
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            raise CameraReadError("unable to read camera frame: {}".format(exc)) from exc
        if (
            not ok
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise CameraReadError("unable to read a valid BGR camera frame")
        return frame.copy()

    def release(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def __enter__(self) -> "OpenCVCamera":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
