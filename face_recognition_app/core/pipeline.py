"""Backend-independent per-frame recognition pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from face_recognition_app.core.alignment import align_face
from face_recognition_app.core.matcher import FaceMatcher
from face_recognition_app.core.stabilizer import IdentityStabilizer
from face_recognition_app.core.tracking import IoUTracker
from face_recognition_app.domain.member import Member
from face_recognition_app.domain.runtime import FaceOverlay, FrameResult
from face_recognition_app.inference.interfaces import FaceDetector, FaceEmbedder
from face_recognition_app.storage.face_store import FaceStoreSnapshot


class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        snapshot: FaceStoreSnapshot,
        threshold: float,
        window_size: int,
        votes_required: int,
        tracker: Optional[IoUTracker] = None,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._threshold = threshold
        self._stabilizer = IdentityStabilizer(window_size, votes_required)
        self._tracker = tracker or IoUTracker()
        self._members: Dict[str, Member] = {}
        self._matcher = FaceMatcher([], np.empty((0, 0)), threshold)
        self.refresh_store(snapshot)

    def refresh_store(self, snapshot: FaceStoreSnapshot) -> None:
        self._matcher = FaceMatcher(
            snapshot.members, snapshot.embeddings, self._threshold
        )
        self._members = {member.member_id: member for member in snapshot.members}
        self._stabilizer.clear()

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        detections = self._detector.detect(frame_bgr)
        track_ids = self._tracker.update([item.box for item in detections])
        overlays = []
        for detection, track_id in zip(detections, track_ids):
            aligned = align_face(frame_bgr, detection.landmarks)
            feature = self._embedder.embed(aligned)
            raw = self._matcher.match(feature)
            confirmed_id = self._stabilizer.update(
                track_id, raw.member_id if raw.is_known else None
            )
            member = self._members.get(confirmed_id) if confirmed_id else None
            overlays.append(
                FaceOverlay(
                    track_id=track_id,
                    box=detection.box,
                    member_id=member.member_id if member else None,
                    name=member.name if member else "陌生人",
                    similarity=raw.similarity,
                    is_known=member is not None,
                )
            )
        return FrameResult(frame_bgr, overlays)

    def clear(self) -> None:
        self._tracker.clear()
        self._stabilizer.clear()
