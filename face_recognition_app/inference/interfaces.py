"""Backend-independent inference contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


class InferenceError(RuntimeError):
    """Raised when a model backend returns unusable data."""


@dataclass(frozen=True)
class FaceDetection:
    """One detected face in source-image coordinates."""

    box: Tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray

    def __post_init__(self) -> None:
        if len(self.box) != 4:
            raise ValueError("box must contain x1, y1, x2, y2")
        if self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            raise ValueError("box must have positive width and height")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        points = np.asarray(self.landmarks, dtype=np.float32)
        if points.shape != (5, 2):
            raise ValueError("landmarks must have shape 5x2")
        if not np.all(np.isfinite(points)):
            raise ValueError("landmarks must contain finite values")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "landmarks", points)


class FaceDetector(ABC):
    """Detect faces and five-point landmarks from a BGR image."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        raise NotImplementedError


class FaceEmbedder(ABC):
    """Extract a normalized identity embedding from an aligned BGR face."""

    @abstractmethod
    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError
