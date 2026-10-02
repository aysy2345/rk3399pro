from pathlib import Path

import numpy as np
import pytest

from face_recognition_app.app.bootstrap import BackendFactory
from face_recognition_app.app.config import parse_config
from face_recognition_app.inference.interfaces import InferenceError
from face_recognition_app.inference.rknn_backend import (
    RknnMobileFaceNetEmbedder,
    RknnRetinaFaceDetector,
)


class FakeRknnLite:
    def __init__(self, outputs=None, load_result=0, init_result=0):
        self.outputs = outputs or [np.ones((1, 512), dtype=np.float32)]
        self.load_result = load_result
        self.init_result = init_result
        self.loaded_path = None
        self.inputs = []
        self.release_calls = 0

    def load_rknn(self, path):
        self.loaded_path = path
        return self.load_result

    def init_runtime(self):
        return self.init_result

    def inference(self, inputs):
        self.inputs.append(inputs[0])
        return self.outputs

    def release(self):
        self.release_calls += 1


def _config_data():
    return {
        "runtime": {"backend": "rknn", "inference_interval_ms": 0},
        "camera": {
            "index": 0,
            "width": 640,
            "height": 480,
            "target_fps": 30,
            "retry_count": 1,
        },
        "models": {
            "detector_path": "models/retinaface.rknn",
            "recognizer_path": "models/mobilefacenet.rknn",
        },
        "recognition": {
            "detection_threshold": 0.8,
            "recognition_threshold": 0.6,
            "min_face_size": 80,
            "min_sharpness": 40.0,
            "window_size": 5,
            "votes_required": 3,
            "enrollment_samples": 15,
            "enrollment_interval_ms": 300,
            "save_photos": False,
        },
        "storage": {"data_dir": "face_data"},
    }


def test_mobilefacenet_uses_hwc_bgr_input_and_normalizes_embedding(tmp_path):
    model_path = tmp_path / "mobilefacenet.rknn"
    model_path.write_bytes(b"model")
    runtime = FakeRknnLite(
        outputs=[np.arange(1, 513, dtype=np.float32)[np.newaxis, :]]
    )
    embedder = RknnMobileFaceNetEmbedder(
        model_path, runtime_factory=lambda: runtime
    )

    embedding = embedder.embed(np.full((40, 50, 3), 127, dtype=np.uint8))

    assert runtime.loaded_path == str(model_path)
    assert runtime.inputs[0].shape == (112, 112, 3)
    assert runtime.inputs[0].dtype == np.uint8
    assert embedding.shape == (512,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    embedder.release()
    embedder.release()
    assert runtime.release_calls == 1


def test_retinaface_reuses_existing_decoder_with_rknn_outputs(tmp_path):
    model_path = tmp_path / "retinaface.rknn"
    model_path.write_bytes(b"model")
    count = 16800
    locations = np.zeros((1, count, 4), dtype=np.float32)
    scores = np.zeros((1, count, 2), dtype=np.float32)
    landmarks = np.zeros((1, count, 10), dtype=np.float32)
    scores[0, 0, 1] = 0.95
    runtime = FakeRknnLite(outputs=[landmarks, scores, locations])
    detector = RknnRetinaFaceDetector(
        model_path, runtime_factory=lambda: runtime
    )

    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert runtime.inputs[0].shape == (640, 640, 3)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.95)
    assert detections[0].landmarks.shape == (5, 2)
    detector.release()


def test_runtime_failure_releases_resources(tmp_path):
    runtime = FakeRknnLite(init_result=-1)

    with pytest.raises(InferenceError, match="init_runtime"):
        RknnMobileFaceNetEmbedder(
            tmp_path / "model.rknn", runtime_factory=lambda: runtime
        )

    assert runtime.release_calls == 1


def test_runtime_factory_failure_is_reported_as_inference_error(tmp_path):
    def fail_to_create_runtime():
        raise OSError("missing NPU runtime library")

    with pytest.raises(InferenceError, match="runtime creation"):
        RknnMobileFaceNetEmbedder(
            tmp_path / "model.rknn",
            runtime_factory=fail_to_create_runtime,
        )


def test_backend_factory_builds_and_releases_rknn_pair(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "retinaface.rknn").write_bytes(b"detector")
    (model_dir / "mobilefacenet.rknn").write_bytes(b"embedder")
    config = parse_config(_config_data(), tmp_path)
    count = 16800
    runtimes = [
        FakeRknnLite(
            outputs=[
                np.zeros((1, count, 4), dtype=np.float32),
                np.zeros((1, count, 2), dtype=np.float32),
                np.zeros((1, count, 10), dtype=np.float32),
            ]
        ),
        FakeRknnLite(),
    ]

    pair = BackendFactory(rknn_factory=lambda: runtimes.pop(0)).create(config)

    assert isinstance(pair.detector, RknnRetinaFaceDetector)
    assert isinstance(pair.embedder, RknnMobileFaceNetEmbedder)
    detector_runtime = pair.detector._rknn_session._runtime
    embedder_runtime = pair.embedder._rknn_session._runtime
    pair.release()
    assert detector_runtime.release_calls == 1
    assert embedder_runtime.release_calls == 1
