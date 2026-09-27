from pathlib import Path

import cv2
import numpy as np

from tools.rknn.contracts import MOBILEFACENET_CONTRACT, RETINAFACE_CONTRACT
from tools.rknn.validate_outputs import validate_images


def _image_file(tmp_path, name="face.jpg"):
    path = tmp_path / name
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_mobilefacenet_validation_writes_passing_report(tmp_path):
    image_path = _image_file(tmp_path)
    report_path = tmp_path / "report.json"
    onnx_path = tmp_path / "mobilefacenet.onnx"
    rknn_path = tmp_path / "mobilefacenet.rknn"
    onnx_path.write_bytes(b"onnx")
    rknn_path.write_bytes(b"rknn")
    vector = np.arange(1, 513, dtype=np.float32)

    report = validate_images(
        MOBILEFACENET_CONTRACT,
        [image_path],
        onnx_runner=lambda image: [vector.reshape(1, -1)],
        rknn_runner=lambda image: [vector.copy()],
        onnx_path=onnx_path,
        rknn_path=rknn_path,
        report_path=report_path,
    )

    assert report["passed"] is True
    assert report["model_id"] == "mobilefacenet"
    assert report["thresholds"]["minimum_cosine"] == 0.99
    assert len(report["onnx_sha256"]) == 64
    assert len(report["rknn_sha256"]) == 64
    assert report["samples"][0]["metrics"]["embedding_size"] == 512
    assert report_path.is_file()


def test_mobilefacenet_low_cosine_and_nan_are_reported_per_sample(tmp_path):
    first = _image_file(tmp_path, "first.jpg")
    second = _image_file(tmp_path, "second.jpg")
    reference = np.arange(1, 513, dtype=np.float32)
    calls = {"count": 0}

    def rknn_runner(image):
        calls["count"] += 1
        if calls["count"] == 1:
            return [-reference]
        return [np.full(512, np.nan)]

    report = validate_images(
        MOBILEFACENET_CONTRACT,
        [first, second],
        onnx_runner=lambda image: [reference],
        rknn_runner=rknn_runner,
    )

    assert report["passed"] is False
    assert report["samples"][0]["passed"] is False
    assert "cosine" in report["samples"][0]["failure"]
    assert report["samples"][1]["passed"] is False
    assert "finite" in report["samples"][1]["failure"]


def test_retinaface_validation_accepts_shuffled_equal_outputs(tmp_path):
    image_path = _image_file(tmp_path)
    anchor_count = 16800
    boxes = np.zeros((1, anchor_count, 4), dtype=np.float32)
    scores = np.zeros((1, anchor_count, 2), dtype=np.float32)
    scores[0, 0] = (0.01, 0.99)
    landmarks = np.zeros((1, anchor_count, 10), dtype=np.float32)

    report = validate_images(
        RETINAFACE_CONTRACT,
        [image_path],
        onnx_runner=lambda image: [boxes, scores, landmarks],
        rknn_runner=lambda image: [landmarks.copy(), boxes.copy(), scores.copy()],
        box_tolerance=0.01,
        landmark_tolerance=0.01,
    )

    assert report["passed"] is True
    metrics = report["samples"][0]["metrics"]
    assert metrics["reference_count"] == 1
    assert metrics["candidate_count"] == 1
    assert metrics["max_box_error"] == 0.0
    assert metrics["max_landmark_error"] == 0.0


def test_retinaface_wrong_shape_is_captured_as_failure(tmp_path):
    image_path = _image_file(tmp_path)

    report = validate_images(
        RETINAFACE_CONTRACT,
        [image_path],
        onnx_runner=lambda image: [np.zeros((1, 1, 3), dtype=np.float32)],
        rknn_runner=lambda image: [np.zeros((1, 1, 3), dtype=np.float32)],
    )

    assert report["passed"] is False
    assert "4, 2 and 10" in report["samples"][0]["failure"]
