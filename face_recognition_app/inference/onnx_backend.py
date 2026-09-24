"""ONNX Runtime adapters for RetinaFace and MobileFaceNet."""

from __future__ import annotations

from itertools import product
from math import ceil
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from face_recognition_app.core.matcher import normalize_vector

from .interfaces import FaceDetection, FaceDetector, FaceEmbedder, InferenceError


def _create_session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise InferenceError(
            "onnxruntime is required for the ONNX backend"
        ) from exc
    path = Path(model_path)
    if not path.is_file():
        raise InferenceError("model file not found: {}".format(path))
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _validate_bgr_image(image: np.ndarray, field: str) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("{} must be a numpy array".format(field))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("{} must be a BGR image".format(field))
    if image.size == 0:
        raise ValueError("{} cannot be empty".format(field))


class OnnxMobileFaceNetEmbedder(FaceEmbedder):
    """MobileFaceNet adapter for the selected 112x112 BGR model contract."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        session: Optional[Any] = None,
        input_size: Tuple[int, int] = (112, 112),
    ) -> None:
        if session is None:
            if model_path is None:
                raise ValueError("model_path or session is required")
            session = _create_session(model_path)
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [item.name for item in session.get_outputs()]
        self._input_size = input_size

    def preprocess(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        _validate_bgr_image(aligned_face_bgr, "aligned face")
        resized = cv2.resize(
            aligned_face_bgr, self._input_size, interpolation=cv2.INTER_LINEAR
        )
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(aligned_face_bgr)
        outputs = self._session.run(
            self._output_names, {self._input_name: tensor}
        )
        if not outputs:
            raise InferenceError("MobileFaceNet returned no outputs")
        feature = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(feature)):
            raise InferenceError("MobileFaceNet returned non-finite values")
        try:
            return normalize_vector(feature)
        except ValueError as exc:
            raise InferenceError(
                "MobileFaceNet returned an invalid embedding"
            ) from exc


def _generate_priors(height: int, width: int) -> np.ndarray:
    min_sizes = ((16, 32), (64, 128), (256, 512))
    steps = (8, 16, 32)
    anchors = []
    for sizes, step in zip(min_sizes, steps):
        feature_h = int(ceil(float(height) / step))
        feature_w = int(ceil(float(width) / step))
        for row, col in product(range(feature_h), range(feature_w)):
            for size in sizes:
                anchors.append(
                    (
                        (col + 0.5) * step / width,
                        (row + 0.5) * step / height,
                        size / width,
                        size / height,
                    )
                )
    return np.asarray(anchors, dtype=np.float32)


def _decode_boxes(locations: np.ndarray, priors: np.ndarray) -> np.ndarray:
    centers = priors[:, :2] + locations[:, :2] * 0.1 * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(locations[:, 2:] * 0.2)
    return np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)


def _decode_landmarks(values: np.ndarray, priors: np.ndarray) -> np.ndarray:
    points = values.reshape((-1, 5, 2))
    centers = priors[:, np.newaxis, :2]
    sizes = priors[:, np.newaxis, 2:]
    return centers + points * 0.1 * sizes


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(
            0.0, yy2 - yy1
        )
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-12)
        order = rest[iou <= threshold]
    return keep


class OnnxRetinaFaceDetector(FaceDetector):
    """RetinaFace MobileNet0.25 adapter with NumPy decoding and NMS."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        session: Optional[Any] = None,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
    ) -> None:
        if session is None:
            if model_path is None:
                raise ValueError("model_path or session is required")
            session = _create_session(model_path)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [item.name for item in session.get_outputs()]
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._keep_top_k = keep_top_k
        width, height = input_size
        self._priors = _generate_priors(height, width)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        _validate_bgr_image(frame_bgr, "frame")
        resized = cv2.resize(
            frame_bgr, self._input_size, interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        resized -= np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
        return np.transpose(resized, (2, 0, 1))[np.newaxis, ...]

    def _map_outputs(
        self, outputs: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mapped = {}
        for output in outputs:
            array = np.asarray(output, dtype=np.float32)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 2:
                raise InferenceError(
                    "RetinaFace output must have shape [1,N,C] or [N,C]"
                )
            mapped[array.shape[1]] = array
        if not all(size in mapped for size in (2, 4, 10)):
            raise InferenceError(
                "RetinaFace outputs must end in 4, 2 and 10 values"
            )
        loc, conf, landmarks = mapped[4], mapped[2], mapped[10]
        if not (len(loc) == len(conf) == len(landmarks) == len(self._priors)):
            raise InferenceError("RetinaFace output count does not match priors")
        return loc, conf, landmarks

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        tensor = self.preprocess(frame_bgr)
        source_h, source_w = frame_bgr.shape[:2]
        raw_outputs = self._session.run(
            self._output_names, {self._input_name: tensor}
        )
        loc, conf, landmark_values = self._map_outputs(raw_outputs)
        scores = conf[:, 1]
        selected = np.flatnonzero(scores >= self._confidence_threshold)
        if not selected.size:
            return []
        order = selected[np.argsort(scores[selected])[::-1]][: self._top_k]
        boxes = _decode_boxes(loc[order], self._priors[order])
        landmarks = _decode_landmarks(
            landmark_values[order], self._priors[order]
        )
        kept = _nms(boxes, scores[order], self._nms_threshold)[
            : self._keep_top_k
        ]
        box_scale = np.asarray(
            (source_w, source_h, source_w, source_h), dtype=np.float32
        )
        point_scale = np.asarray((source_w, source_h), dtype=np.float32)
        detections = []
        for index in kept:
            box = np.clip(
                boxes[index] * box_scale,
                (0.0, 0.0, 0.0, 0.0),
                (
                    float(source_w - 1),
                    float(source_h - 1),
                    float(source_w - 1),
                    float(source_h - 1),
                ),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            points = landmarks[index] * point_scale
            points[:, 0] = np.clip(points[:, 0], 0, source_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, source_h - 1)
            detections.append(
                FaceDetection(
                    box=tuple(float(value) for value in box),
                    score=float(scores[order[index]]),
                    landmarks=points,
                )
            )
        return detections
