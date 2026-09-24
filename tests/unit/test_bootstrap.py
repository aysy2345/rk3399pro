from dataclasses import replace

import pytest

from face_recognition_app.app.bootstrap import (
    BackendFactory,
    BootstrapError,
    apply_overrides,
    build_application,
)
from face_recognition_app.app.config import ConfigError, parse_config
from face_recognition_app.inference.fake import FakeFaceDetector, FakeFaceEmbedder


def config_data(backend="fake"):
    return {
        "runtime": {"backend": backend, "inference_interval_ms": 0},
        "camera": {
            "index": 0,
            "width": 640,
            "height": 480,
            "target_fps": 30,
            "retry_count": 1,
        },
        "models": {
            "detector_path": "models/detector.onnx",
            "recognizer_path": "models/recognizer.onnx",
        },
        "recognition": {
            "detection_threshold": 0.8,
            "recognition_threshold": 0.6,
            "min_face_size": 80,
            "min_sharpness": 100.0,
            "window_size": 5,
            "votes_required": 3,
            "enrollment_samples": 15,
            "enrollment_interval_ms": 300,
            "save_photos": False,
        },
        "storage": {"data_dir": "face_data"},
    }


def test_backend_factory_creates_deterministic_fake_pair(tmp_path):
    config = parse_config(config_data(), tmp_path)

    pair = BackendFactory().create(config)

    assert isinstance(pair.detector, FakeFaceDetector)
    assert isinstance(pair.embedder, FakeFaceEmbedder)


def test_backend_factory_reports_missing_onnx_models(tmp_path):
    config = parse_config(config_data("onnx"), tmp_path)

    with pytest.raises(ConfigError, match="model file not found"):
        BackendFactory().create(config)


def test_backend_factory_rejects_reserved_rknn_backend(tmp_path):
    config = parse_config(config_data("rknn"), tmp_path)

    with pytest.raises(BootstrapError, match="RKNN"):
        BackendFactory().create(config)


def test_apply_overrides_returns_new_config(tmp_path):
    config = parse_config(config_data(), tmp_path)

    overridden = apply_overrides(config, backend="onnx", camera_index=2)

    assert overridden.runtime.backend == "onnx"
    assert overridden.camera.index == 2
    assert config.runtime.backend == "fake"
    assert config.camera.index == 0


@pytest.mark.parametrize("camera_index", [-1, True])
def test_apply_overrides_rejects_invalid_camera_index(tmp_path, camera_index):
    config = parse_config(config_data(), tmp_path)

    with pytest.raises(BootstrapError, match="camera index"):
        apply_overrides(config, camera_index=camera_index)


def test_default_fake_runtime_explains_that_real_enrollment_needs_onnx(
    qtbot, tmp_path, monkeypatch
):
    config = parse_config(config_data(), tmp_path)
    bundle = build_application(config, camera_factory=lambda: object())
    qtbot.addWidget(bundle.window)
    messages = []
    monkeypatch.setattr(
        "face_recognition_app.app.bootstrap.QMessageBox.warning",
        lambda parent, title, message: messages.append((title, message)),
    )

    bundle.coordinator.open_enrollment_wizard()

    assert messages
    assert "Fake" in messages[0][1]
    assert "ONNX" in messages[0][1]
    bundle.window.close()


def test_preview_flip_config_reaches_main_and_enrollment_windows(qtbot, tmp_path):
    data = config_data()
    data["camera"]["preview_flip_horizontal"] = True
    config = parse_config(data, tmp_path)
    bundle = build_application(
        config,
        camera_factory=lambda: object(),
        enrollment_session_factory=lambda: object(),
    )
    qtbot.addWidget(bundle.window)
    wizard = bundle.coordinator.create_enrollment_wizard()
    qtbot.addWidget(wizard)

    assert bundle.window.video_widget.flip_horizontal is True
    assert wizard.video_widget.flip_horizontal is True
    wizard.close()
    bundle.window.close()
