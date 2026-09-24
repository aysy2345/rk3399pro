from types import SimpleNamespace

import numpy as np
import pytest

from face_recognition_app.inference.fake import FakeFaceDetector, FakeFaceEmbedder
from face_recognition_app.inference.interfaces import FaceDetection, InferenceError
from face_recognition_app.inference.onnx_backend import (
    OnnxMobileFaceNetEmbedder,
    OnnxRetinaFaceDetector,
    _generate_priors,
)


class StubSession:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs_seen = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [
            SimpleNamespace(name="output-{}".format(index))
            for index in range(len(self.outputs))
        ]

    def run(self, output_names, inputs):
        self.inputs_seen.append(inputs["input"])
        return self.outputs


def make_detection():
    return FaceDetection(
        box=(10.0, 20.0, 80.0, 100.0),
        score=0.9,
        landmarks=np.asarray(
            [[25, 45], [60, 45], [43, 62], [30, 82], [57, 82]],
            dtype=np.float32,
        ),
    )


def test_face_detection_validates_geometry():
    with pytest.raises(ValueError, match="positive"):
        FaceDetection(
            box=(1.0, 1.0, 1.0, 2.0),
            score=0.9,
            landmarks=np.zeros((5, 2), dtype=np.float32),
        )


def test_fake_backends_are_deterministic():
    detector = FakeFaceDetector([[make_detection()]])
    embedder = FakeFaceEmbedder([np.asarray([3.0, 4.0])])
    image = np.zeros((112, 112, 3), dtype=np.uint8)

    assert len(detector.detect(image)) == 1
    assert detector.detect(image) == []
    assert embedder.embed(image) == pytest.approx([0.6, 0.8])
    assert embedder.embed(image) == pytest.approx([0.6, 0.8])


def test_mobilefacenet_preprocesses_bgr_and_normalizes_output():
    session = StubSession([np.asarray([[3.0, 4.0]], dtype=np.float32)])
    embedder = OnnxMobileFaceNetEmbedder(session=session)
    blue_bgr = np.zeros((112, 112, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255

    result = embedder.embed(blue_bgr)

    assert result == pytest.approx([0.6, 0.8])
    tensor = session.inputs_seen[0]
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor[0, 0, 0, 0] == pytest.approx(1.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(-1.0)


def test_mobilefacenet_rejects_zero_embedding():
    session = StubSession([np.zeros((1, 4), dtype=np.float32)])
    embedder = OnnxMobileFaceNetEmbedder(session=session)

    with pytest.raises(InferenceError, match="invalid embedding"):
        embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))


def test_retinaface_decodes_one_high_confidence_detection():
    count = len(_generate_priors(32, 32))
    locations = np.zeros((1, count, 4), dtype=np.float32)
    confidence = np.zeros((1, count, 2), dtype=np.float32)
    confidence[0, 0] = (0.01, 0.99)
    landmarks = np.zeros((1, count, 10), dtype=np.float32)
    session = StubSession([locations, confidence, landmarks])
    detector = OnnxRetinaFaceDetector(
        session=session, input_size=(32, 32), confidence_threshold=0.8
    )

    result = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.99)
    assert result[0].landmarks.shape == (5, 2)
    assert session.inputs_seen[0].shape == (1, 3, 32, 32)


def test_retinaface_rejects_wrong_output_shapes():
    session = StubSession([np.zeros((1, 1, 3), dtype=np.float32)])
    detector = OnnxRetinaFaceDetector(session=session, input_size=(32, 32))

    with pytest.raises(InferenceError, match="4, 2 and 10"):
        detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
