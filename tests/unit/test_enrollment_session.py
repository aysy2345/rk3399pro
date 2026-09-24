import numpy as np
import pytest

from face_recognition_app.core.enrollment_session import EnrollmentSession
from face_recognition_app.core.quality import classify_pose
from face_recognition_app.inference.fake import FakeFaceDetector, FakeFaceEmbedder
from face_recognition_app.inference.interfaces import FaceDetection


def landmarks(nose_x):
    return np.asarray(
        [[30, 40], [70, 40], [nose_x, 58], [35, 75], [65, 75]],
        dtype=np.float32,
    )


def detection(nose_x=50):
    return FaceDetection(
        box=(10.0, 10.0, 100.0, 100.0),
        score=0.95,
        landmarks=landmarks(nose_x),
    )


def sharp_frame():
    checker = (np.indices((120, 120)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return np.stack([checker, checker, checker], axis=2)


def test_classify_pose_uses_normalized_nose_offset():
    assert classify_pose(landmarks(50)) == "front"
    assert classify_pose(landmarks(42)) == "left"
    assert classify_pose(landmarks(58)) == "right"


def test_enrollment_collects_pose_plan_and_builds_template():
    detector = FakeFaceDetector(
        [[detection(50)], [detection(42)], [detection(58)]]
    )
    embedder = FakeFaceEmbedder(
        [
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.98, 0.15, 0.0]),
            np.asarray([0.98, 0.0, 0.15]),
        ]
    )
    times = iter([0.0, 0.4, 0.8])
    session = EnrollmentSession(
        detector,
        embedder,
        target_samples=3,
        min_face_size=40,
        min_sharpness=100.0,
        enrollment_interval_ms=300,
        clock=lambda: next(times),
        pose_plan=("front", "left", "right"),
    )

    progress = [session.process(sharp_frame()) for _ in range(3)]
    template = session.build_template()

    assert [item.accepted_count for item in progress] == [1, 2, 3]
    assert all(item.accepted for item in progress)
    assert session.complete
    assert template.accepted_samples == 3
    assert np.linalg.norm(template.embedding) == pytest.approx(1.0)


def test_enrollment_rejects_frame_before_sample_interval():
    detector = FakeFaceDetector([[detection()], [detection()]])
    embedder = FakeFaceEmbedder(
        [np.asarray([1.0, 0.0]), np.asarray([0.98, 0.2])]
    )
    times = iter([0.0, 0.1])
    session = EnrollmentSession(
        detector,
        embedder,
        target_samples=2,
        min_face_size=40,
        min_sharpness=100.0,
        enrollment_interval_ms=300,
        clock=lambda: next(times),
        pose_plan=("front", "front"),
    )

    assert session.process(sharp_frame()).accepted
    rejected = session.process(sharp_frame())

    assert not rejected.accepted
    assert "保持" in rejected.reason
    assert rejected.accepted_count == 1
