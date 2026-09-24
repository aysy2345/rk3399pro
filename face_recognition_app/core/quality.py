"""Deterministic quality checks for enrollment frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


BoundingBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceQuality:
    accepted: bool
    reason: str
    face_size: int
    sharpness: float


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        return (
            array[:, :, 0] * 0.114
            + array[:, :, 1] * 0.587
            + array[:, :, 2] * 0.299
        ).astype(np.float32)
    raise ValueError("image must be grayscale or BGR/RGB")


def gradient_sharpness(image: np.ndarray) -> float:
    gray = _gray(image)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    return float(np.mean(horizontal * horizontal) + np.mean(vertical * vertical))


def assess_face(
    image: np.ndarray,
    bounding_box: BoundingBox,
    face_count: int,
    min_face_size: int,
    min_sharpness: float,
) -> FaceQuality:
    if face_count != 1:
        return FaceQuality(False, "登记时画面中必须只有一张人脸", 0, 0.0)
    x1, y1, x2, y2 = bounding_box
    width = max(0, int(round(x2 - x1)))
    height = max(0, int(round(y2 - y1)))
    face_size = min(width, height)
    if face_size < min_face_size:
        return FaceQuality(False, "请靠近摄像头", face_size, 0.0)

    array = np.asarray(image)
    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(array.shape[1], int(np.ceil(x2)))
    bottom = min(array.shape[0], int(np.ceil(y2)))
    if right <= left or bottom <= top:
        return FaceQuality(False, "人脸区域无效", face_size, 0.0)
    sharpness = gradient_sharpness(array[top:bottom, left:right])
    if sharpness < min_sharpness:
        return FaceQuality(False, "画面模糊，请保持稳定", face_size, sharpness)
    return FaceQuality(True, "有效样本", face_size, sharpness)


def classify_pose(landmarks: np.ndarray, yaw_threshold: float = 0.12) -> str:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise ValueError("landmarks must have shape 5x2 and finite values")
    if yaw_threshold <= 0.0:
        raise ValueError("yaw_threshold must be positive")
    eye_distance = float(abs(points[1, 0] - points[0, 0]))
    if eye_distance <= 1e-6:
        raise ValueError("eye landmarks must be separated")
    eye_center_x = float((points[0, 0] + points[1, 0]) * 0.5)
    offset = float(points[2, 0] - eye_center_x) / eye_distance
    if offset < -yaw_threshold:
        return "left"
    if offset > yaw_threshold:
        return "right"
    return "front"
