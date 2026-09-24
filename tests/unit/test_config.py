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
        "camera": {"index": 0, "width": 640, "height": 480, "retry_count": 3},
        "models": {
            "detector_path": "models/detector.rknn",
            "recognizer_path": "models/recognizer.rknn",
        },
        "recognition": {
            "detection_threshold": 0.8,
            "recognition_threshold": 0.6,
            "min_face_size": 80,
            "window_size": 5,
            "votes_required": 3,
            "enrollment_samples": 15,
            "save_photos": False,
        },
        "storage": {"data_dir": "face_data"},
    }


def test_parse_config_resolves_relative_paths(tmp_path):
    config = parse_config(valid_config(), tmp_path)

    assert config.camera.width == 640
    assert config.models.detector_path == (tmp_path / "models/detector.rknn").resolve()
    assert config.storage.data_dir == (tmp_path / "face_data").resolve()


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
