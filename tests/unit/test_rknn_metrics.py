import numpy as np
import pytest

from face_recognition_app.inference.onnx_backend import OnnxRetinaFaceDetector
from tools.rknn.metrics import (
    MetricError,
    compare_mobilefacenet,
    compare_retinaface_detections,
    cosine_similarity,
    decode_retinaface,
    map_retinaface_outputs,
)


class _TensorInfo(object):
    def __init__(self, name):
        self.name = name


class _Session(object):
    def __init__(self, outputs):
        self.outputs = outputs

    def get_inputs(self):
        return [_TensorInfo("input")]

    def get_outputs(self):
        return [_TensorInfo("output_{}".format(index)) for index in range(3)]

    def run(self, output_names, feeds):
        return self.outputs


def test_cosine_similarity_accepts_identical_finite_vectors():
    vector = np.arange(1, 513, dtype=np.float32)

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (np.ones(512), np.ones(511), "same size"),
        (np.full(512, np.nan), np.ones(512), "finite"),
        (np.zeros(512), np.ones(512), "zero-length"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(left, right, message):
    with pytest.raises(MetricError, match=message):
        cosine_similarity(left, right)


def test_mobilefacenet_comparison_requires_512_values_and_threshold():
    reference = np.ones((1, 512), dtype=np.float32)
    close = reference.copy()
    opposite = -reference

    passing = compare_mobilefacenet(reference, close, minimum_cosine=0.99)
    failing = compare_mobilefacenet(reference, opposite, minimum_cosine=0.99)

    assert passing == {
        "embedding_size": 512,
        "cosine_similarity": pytest.approx(1.0),
        "minimum_cosine": 0.99,
        "passed": True,
    }
    assert failing["passed"] is False
    with pytest.raises(MetricError, match="512"):
        compare_mobilefacenet(np.ones(511), np.ones(511))


def test_retinaface_outputs_are_mapped_by_last_dimension():
    boxes = np.zeros((1, 4, 4), dtype=np.float32)
    scores = np.zeros((1, 4, 2), dtype=np.float32)
    landmarks = np.zeros((1, 4, 10), dtype=np.float32)

    actual_boxes, actual_scores, actual_landmarks = map_retinaface_outputs(
        [landmarks, boxes, scores]
    )

    np.testing.assert_array_equal(actual_boxes, boxes[0])
    np.testing.assert_array_equal(actual_scores, scores[0])
    np.testing.assert_array_equal(actual_landmarks, landmarks[0])


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([np.zeros((1, 2, 3))], "4, 2 and 10"),
        (
            [
                np.zeros((1, 4, 4)),
                np.zeros((1, 5, 2)),
                np.zeros((1, 4, 10)),
            ],
            "same anchor count",
        ),
        (
            [
                np.zeros((1, 4, 4)),
                np.zeros((1, 4, 2)),
                np.full((1, 4, 10), np.nan),
            ],
            "finite",
        ),
    ],
)
def test_retinaface_output_mapping_rejects_invalid_data(outputs, message):
    with pytest.raises(MetricError, match=message):
        map_retinaface_outputs(outputs)


def test_retinaface_decode_uses_existing_prior_and_output_contract():
    anchor_count = 42
    locations = np.zeros((1, anchor_count, 4), dtype=np.float32)
    scores = np.zeros((1, anchor_count, 2), dtype=np.float32)
    scores[0, 0] = (0.01, 0.99)
    landmarks = np.zeros((1, anchor_count, 10), dtype=np.float32)

    boxes, points = decode_retinaface(
        [locations, scores, landmarks],
        source_shape=(64, 64),
        input_size=(32, 32),
        confidence_threshold=0.8,
        nms_threshold=0.4,
    )

    assert boxes.shape == (1, 4)
    assert points.shape == (1, 5, 2)
    assert np.all(np.isfinite(boxes))
    assert np.all(np.isfinite(points))


def test_retinaface_decode_matches_application_backend():
    anchor_count = 42
    locations = np.zeros((1, anchor_count, 4), dtype=np.float32)
    locations[0, 0] = (0.05, -0.03, 0.01, -0.02)
    scores = np.zeros((1, anchor_count, 2), dtype=np.float32)
    scores[0, 0] = (0.01, 0.99)
    landmarks = np.zeros((1, anchor_count, 10), dtype=np.float32)
    landmarks[0, 0] = np.linspace(-0.1, 0.1, 10)
    outputs = [locations, scores, landmarks]
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    detector = OnnxRetinaFaceDetector(
        session=_Session(outputs),
        input_size=(32, 32),
        confidence_threshold=0.8,
        nms_threshold=0.4,
    )

    expected = detector.detect(image)
    boxes, points = decode_retinaface(
        outputs,
        source_shape=image.shape[:2],
        input_size=(32, 32),
        confidence_threshold=0.8,
        nms_threshold=0.4,
    )

    assert len(expected) == 1
    np.testing.assert_allclose(boxes[0], expected[0].box, atol=1e-6)
    np.testing.assert_allclose(points[0], expected[0].landmarks, atol=1e-6)


def test_retinaface_detection_comparison_matches_by_iou():
    reference_boxes = np.asarray(((10, 10, 30, 30), (40, 40, 60, 60)), dtype=np.float32)
    candidate_boxes = np.asarray(((41, 40, 61, 60), (10, 11, 30, 31)), dtype=np.float32)
    reference_points = np.asarray(
        [np.full((5, 2), 20), np.full((5, 2), 50)], dtype=np.float32
    )
    candidate_points = np.asarray(
        [np.full((5, 2), 51), np.full((5, 2), 21)], dtype=np.float32
    )

    result = compare_retinaface_detections(
        reference_boxes,
        reference_points,
        candidate_boxes,
        candidate_points,
        box_tolerance=1.0,
        landmark_tolerance=1.0,
    )

    assert result["reference_count"] == 2
    assert result["candidate_count"] == 2
    assert result["matched_count"] == 2
    assert result["max_box_error"] == pytest.approx(1.0)
    assert result["max_landmark_error"] == pytest.approx(1.0)
    assert result["passed"] is True


def test_retinaface_detection_count_mismatch_fails():
    result = compare_retinaface_detections(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 5, 2), dtype=np.float32),
        np.asarray(((1, 1, 2, 2),), dtype=np.float32),
        np.ones((1, 5, 2), dtype=np.float32),
    )

    assert result["passed"] is False
    assert result["matched_count"] == 0
