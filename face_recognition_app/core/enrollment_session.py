"""Stateful automatic sample collection for one member enrollment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from face_recognition_app.core.alignment import align_face
from face_recognition_app.core.enrollment import (
    EnrollmentError,
    EnrollmentTemplate,
    build_enrollment_template,
)
from face_recognition_app.core.quality import assess_face, classify_pose
from face_recognition_app.inference.interfaces import FaceDetector, FaceEmbedder


POSE_LABELS = {"front": "正视", "left": "左转", "right": "右转"}


@dataclass(frozen=True)
class EnrollmentProgress:
    accepted: bool
    reason: str
    accepted_count: int
    target_count: int
    required_pose: Optional[str]
    complete: bool


def _default_pose_plan(target_samples: int) -> Sequence[str]:
    side_count = target_samples // 4
    front_count = target_samples - side_count * 2
    return (
        ("front",) * front_count
        + ("left",) * side_count
        + ("right",) * side_count
    )


class EnrollmentSession:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        target_samples: int,
        min_face_size: int,
        min_sharpness: float,
        enrollment_interval_ms: int,
        clock: Callable[[], float] = time.monotonic,
        pose_plan: Optional[Sequence[str]] = None,
        duplicate_threshold: float = 0.9995,
    ) -> None:
        if target_samples < 1:
            raise ValueError("target_samples must be positive")
        plan = tuple(pose_plan or _default_pose_plan(target_samples))
        if len(plan) != target_samples:
            raise ValueError("pose_plan length must equal target_samples")
        if any(pose not in POSE_LABELS for pose in plan):
            raise ValueError("pose_plan contains an unsupported pose")
        self._detector = detector
        self._embedder = embedder
        self._target_samples = target_samples
        self._min_face_size = min_face_size
        self._min_sharpness = min_sharpness
        self._interval_seconds = enrollment_interval_ms / 1000.0
        self._clock = clock
        self._pose_plan = plan
        self._duplicate_threshold = duplicate_threshold
        self._features: List[np.ndarray] = []
        self._last_sample_time: Optional[float] = None

    @property
    def complete(self) -> bool:
        return len(self._features) >= self._target_samples

    @property
    def required_pose(self) -> Optional[str]:
        if self.complete:
            return None
        return self._pose_plan[len(self._features)]

    def _progress(self, accepted: bool, reason: str) -> EnrollmentProgress:
        return EnrollmentProgress(
            accepted=accepted,
            reason=reason,
            accepted_count=len(self._features),
            target_count=self._target_samples,
            required_pose=self.required_pose,
            complete=self.complete,
        )

    def process(self, frame_bgr: np.ndarray) -> EnrollmentProgress:
        if self.complete:
            return self._progress(False, "采集已完成")
        detections = self._detector.detect(frame_bgr)
        if len(detections) != 1:
            return self._progress(False, "登记时画面中必须只有一张人脸")
        detection = detections[0]
        quality = assess_face(
            frame_bgr,
            detection.box,
            len(detections),
            self._min_face_size,
            self._min_sharpness,
        )
        if not quality.accepted:
            return self._progress(False, quality.reason)
        pose = classify_pose(detection.landmarks)
        if pose != self.required_pose:
            return self._progress(
                False, "请{}并保持稳定".format(POSE_LABELS[self.required_pose])
            )
        now = self._clock()
        if (
            self._last_sample_time is not None
            and now - self._last_sample_time < self._interval_seconds
        ):
            return self._progress(False, "请保持当前姿态")
        feature = self._embedder.embed(
            align_face(frame_bgr, detection.landmarks)
        )
        if self._features:
            similarities = np.asarray(self._features).dot(feature)
            if float(np.max(similarities)) >= self._duplicate_threshold:
                return self._progress(False, "请轻微调整姿态")
        self._features.append(feature.copy())
        self._last_sample_time = now
        return self._progress(True, "有效样本")

    def build_template(self) -> EnrollmentTemplate:
        if not self.complete:
            raise EnrollmentError("enrollment is not complete")
        return build_enrollment_template(
            self._features,
            minimum_samples=self._target_samples,
            duplicate_threshold=self._duplicate_threshold,
        )
