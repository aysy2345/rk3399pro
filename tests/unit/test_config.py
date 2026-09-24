from pathlib import Path

import pytest

from face_recognition_app.app.config import (
    ConfigError,
    load_config,
    parse_config,
    validate_runtime_paths,
)


def valid_config():
    return {
        "runtime": {"backend": "onnx", "inference_interval_ms": 100},
        "camera": {
            "index": 0,
            "width": 640,
            "height": 480,
            "target_fps": 30,
            "retry_count": 3,
            "preview_flip_horizontal": True,
        },
        "models": {
            "detector_path": "models/detector.rknn",
            "recognizer_path": "models/recognizer.rknn",
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


def test_parse_config_resolves_relative_paths(tmp_path):
    config = parse_config(valid_config(), tmp_path)

    assert config.runtime.backend == "onnx"
    assert config.runtime.inference_interval_ms == 100
    assert config.camera.width == 640
    assert config.camera.target_fps == 30
    assert config.camera.preview_flip_horizontal is True
    assert config.recognition.min_sharpness == pytest.approx(100.0)
    assert config.recognition.enrollment_interval_ms == 300
    assert config.models.detector_path == (tmp_path / "models/detector.rknn").resolve()
    assert config.storage.data_dir == (tmp_path / "face_data").resolve()


def test_example_config_uses_camera_friendly_sharpness_threshold():
    project_root = Path(__file__).resolve().parents[2]

    config = load_config(project_root / "configs" / "app.example.json")

    assert config.recognition.min_sharpness == pytest.approx(40.0)
    assert config.camera.preview_flip_horizontal is True


def test_camera_preview_flip_defaults_to_false(tmp_path):
    data = valid_config()
    data["camera"].pop("preview_flip_horizontal")

    config = parse_config(data, tmp_path)

    assert config.camera.preview_flip_horizontal is False


def test_camera_preview_flip_must_be_boolean(tmp_path):
    data = valid_config()
    data["camera"]["preview_flip_horizontal"] = 1

    with pytest.raises(ConfigError, match="preview_flip_horizontal"):
        parse_config(data, tmp_path)


@pytest.mark.parametrize("backend", ["fake", "onnx", "rknn"])
def test_parse_config_accepts_supported_backends(tmp_path, backend):
    data = valid_config()
    data["runtime"]["backend"] = backend

    assert parse_config(data, tmp_path).runtime.backend == backend


def test_parse_config_rejects_unknown_backend(tmp_path):
    data = valid_config()
    data["runtime"]["backend"] = "cuda"

    with pytest.raises(ConfigError, match="backend"):
        parse_config(data, tmp_path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("runtime", "inference_interval_ms", -1),
        ("camera", "target_fps", 0),
        ("recognition", "min_sharpness", -0.1),
        ("recognition", "enrollment_interval_ms", -1),
    ],
)
def test_runtime_timing_and_quality_values_have_valid_ranges(
    tmp_path, section, field, value
):
    data = valid_config()
    data[section][field] = value

    with pytest.raises(ConfigError, match=field):
        parse_config(data, tmp_path)


def test_votes_cannot_exceed_window(tmp_path):
    data = valid_config()
    data["recognition"]["votes_required"] = 6

    with pytest.raises(ConfigError, match="cannot exceed"):
        parse_config(data, tmp_path)


def test_enrollment_count_must_be_between_ten_and_twenty(tmp_path):
    data = valid_config()
    data["recognition"]["enrollment_samples"] = 9

    with pytest.raises(ConfigError, match="between 10 and 20"):
        parse_config(data, tmp_path)


def test_load_config_rejects_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{bad", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)


def test_runtime_validation_reports_missing_models(tmp_path):
    config = parse_config(valid_config(), tmp_path)

    with pytest.raises(ConfigError, match="model file not found"):
        validate_runtime_paths(config)


def test_fake_backend_does_not_require_model_files(tmp_path):
    data = valid_config()
    data["runtime"]["backend"] = "fake"
    config = parse_config(data, tmp_path)

    validate_runtime_paths(config)
