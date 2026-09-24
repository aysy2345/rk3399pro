import numpy as np
import pytest

from face_recognition_app.core.alignment import REFERENCE_LANDMARKS_112, align_face
from face_recognition_app.core.enrollment import (
    EnrollmentError,
    build_enrollment_template,
)
from face_recognition_app.core.matcher import FaceMatcher, FeatureError, normalize_vector
from face_recognition_app.core.quality import assess_face, gradient_sharpness
from face_recognition_app.core.stabilizer import IdentityStabilizer
from face_recognition_app.domain.member import Member


def test_normalize_vector_rejects_zero_vector():
    with pytest.raises(FeatureError, match="greater than zero"):
        normalize_vector(np.zeros(3, dtype=np.float32))


def test_matcher_returns_best_member_and_unknown_below_threshold():
    members = [Member.create("001", "张三"), Member.create("002", "李四")]
    matcher = FaceMatcher(
        members,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        threshold=0.8,
    )

    known = matcher.match(np.asarray([0.99, 0.01], dtype=np.float32))
    unknown = matcher.match(np.asarray([0.7, 0.7], dtype=np.float32))

    assert known.is_known
    assert known.member_id == "001"
    assert not unknown.is_known
    assert unknown.name == "陌生人"


def test_empty_matcher_always_returns_unknown():
    matcher = FaceMatcher([], np.empty((0, 0), dtype=np.float32), threshold=0.6)

    assert not matcher.match(np.asarray([1.0, 0.0])).is_known


def test_stabilizer_requires_three_votes_in_five_frames():
    stabilizer = IdentityStabilizer(window_size=5, votes_required=3)

    assert stabilizer.update("face-1", "001") is None
    assert stabilizer.update("face-1", None) is None
    assert stabilizer.update("face-1", "001") is None
    assert stabilizer.update("face-1", "001") == "001"


def test_enrollment_builds_normalized_template_and_rejects_outlier():
    features = [
        np.asarray([1.0, 0.00, 0.00]),
        np.asarray([0.98, 0.15, 0.00]),
        np.asarray([0.98, -0.15, 0.00]),
        np.asarray([0.97, 0.00, 0.20]),
        np.asarray([0.97, 0.00, -0.20]),
        np.asarray([-1.0, 0.00, 0.00]),
    ]

    result = build_enrollment_template(
        features,
        minimum_samples=5,
        duplicate_threshold=0.9999,
        center_threshold=0.7,
    )

    assert result.accepted_samples == 5
    assert result.rejected_samples == 1
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)
    assert result.embedding[0] > 0.99


def test_enrollment_rejects_insufficient_samples():
    with pytest.raises(EnrollmentError, match="not enough"):
        build_enrollment_template([np.asarray([1.0, 0.0])], minimum_samples=2)


def test_quality_rejects_multiple_faces_and_small_face():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    multiple = assess_face(image, (0, 0, 90, 90), 2, 40, 1.0)
    small = assess_face(image, (0, 0, 20, 20), 1, 40, 1.0)

    assert not multiple.accepted
    assert not small.accepted


def test_quality_accepts_sharp_checkerboard():
    checker = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.stack([checker, checker, checker], axis=2)

    result = assess_face(image, (10, 10, 90, 90), 1, 40, 100.0)

    assert result.accepted
    assert gradient_sharpness(image) > 100.0


def test_quality_rejection_reports_measured_and_required_sharpness():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = assess_face(image, (10, 10, 90, 90), 1, 40, 40.0)

    assert not result.accepted
    assert result.sharpness == pytest.approx(0.0)
    assert "清晰度 0.0/40.0" in result.reason


def test_alignment_accepts_five_landmarks_and_returns_expected_size():
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[40:80, 40:80] = 255

    aligned = align_face(image, REFERENCE_LANDMARKS_112)

    assert aligned.shape == (112, 112, 3)


def test_alignment_rejects_invalid_landmark_shape():
    image = np.zeros((112, 112, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="5x2"):
        align_face(image, [[1.0, 2.0]])
