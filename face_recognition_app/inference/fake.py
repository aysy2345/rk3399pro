"""Deterministic inference doubles used by tests and UI development."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from face_recognition_app.core.matcher import normalize_vector

from .interfaces import FaceDetection, FaceDetector, FaceEmbedder


class FakeFaceDetector(FaceDetector):
    """Return a predefined sequence of detection batches."""

    def __init__(self, batches: Iterable[Sequence[FaceDetection]]) -> None:
        self._batches = [list(batch) for batch in batches]
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        index = self.calls
        self.calls += 1
        if index >= len(self._batches):
            return []
        return list(self._batches[index])


class FakeFaceEmbedder(FaceEmbedder):
    """Return predefined normalized embeddings in order."""

    def __init__(self, embeddings: Iterable[np.ndarray]) -> None:
        self._embeddings = [normalize_vector(item) for item in embeddings]
        if not self._embeddings:
            raise ValueError("at least one embedding is required")
        self.calls = 0

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        if aligned_face_bgr.ndim != 3 or aligned_face_bgr.shape[2] != 3:
            raise ValueError("aligned face must be a BGR image")
        index = min(self.calls, len(self._embeddings) - 1)
        self.calls += 1
        return self._embeddings[index].copy()
