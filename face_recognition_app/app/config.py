"""Load and validate application configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


class ConfigError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    inference_interval_ms: int


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    target_fps: int
    retry_count: int
    preview_flip_horizontal: bool


@dataclass(frozen=True)
class ModelConfig:
    detector_path: Path
    recognizer_path: Path


@dataclass(frozen=True)
class RecognitionConfig:
    detection_threshold: float
    recognition_threshold: float
    min_face_size: int
    min_sharpness: float
    window_size: int
    votes_required: int
    enrollment_samples: int
    enrollment_interval_ms: int
    save_photos: bool


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    camera: CameraConfig
    models: ModelConfig
    recognition: RecognitionConfig
    storage: StorageConfig


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be an object".format(field))
    return value


def _integer(data: Mapping[str, Any], field: str, minimum: int) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("{} must be an integer".format(field))
    if value < minimum:
        raise ConfigError("{} must be at least {}".format(field, minimum))
    return value


def _number(
    data: Mapping[str, Any], field: str, minimum: float, maximum: float
) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{} must be a number".format(field))
    result = float(value)
    if result < minimum or result > maximum:
        raise ConfigError(
            "{} must be between {} and {}".format(field, minimum, maximum)
        )
    return result


def _boolean(data: Mapping[str, Any], field: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise ConfigError("{} must be true or false".format(field))
    return value


def _optional_boolean(
    data: Mapping[str, Any], field: str, default: bool
) -> bool:
    if field not in data:
        return default
    return _boolean(data, field)


def _choice(
    data: Mapping[str, Any], field: str, choices: tuple
) -> str:
    value = data.get(field)
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(
            "{} must be one of {}".format(field, ", ".join(choices))
        )
    return value


def _non_negative_number(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{} must be a number".format(field))
    result = float(value)
    if result < 0.0:
        raise ConfigError("{} must be at least 0".format(field))
    return result


def _path(data: Mapping[str, Any], field: str, base_dir: Path) -> Path:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("{} must be a non-empty path".format(field))
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_config(data: Mapping[str, Any], base_dir: Path) -> AppConfig:
    runtime_data = _mapping(data.get("runtime"), "runtime")
    camera_data = _mapping(data.get("camera"), "camera")
    model_data = _mapping(data.get("models"), "models")
    recognition_data = _mapping(data.get("recognition"), "recognition")
    storage_data = _mapping(data.get("storage"), "storage")

    runtime = RuntimeConfig(
        backend=_choice(
            runtime_data, "backend", ("fake", "onnx", "rknn")
        ),
        inference_interval_ms=_integer(
            runtime_data, "inference_interval_ms", 0
        ),
    )
    camera = CameraConfig(
        index=_integer(camera_data, "index", 0),
        width=_integer(camera_data, "width", 1),
        height=_integer(camera_data, "height", 1),
        target_fps=_integer(camera_data, "target_fps", 1),
        retry_count=_integer(camera_data, "retry_count", 0),
        preview_flip_horizontal=_optional_boolean(
            camera_data, "preview_flip_horizontal", False
        ),
    )
    models = ModelConfig(
        detector_path=_path(model_data, "detector_path", base_dir),
        recognizer_path=_path(model_data, "recognizer_path", base_dir),
    )
    recognition = RecognitionConfig(
        detection_threshold=_number(
            recognition_data, "detection_threshold", 0.0, 1.0
        ),
        recognition_threshold=_number(
            recognition_data, "recognition_threshold", -1.0, 1.0
        ),
        min_face_size=_integer(recognition_data, "min_face_size", 1),
        min_sharpness=_non_negative_number(
            recognition_data, "min_sharpness"
        ),
        window_size=_integer(recognition_data, "window_size", 1),
        votes_required=_integer(recognition_data, "votes_required", 1),
        enrollment_samples=_integer(recognition_data, "enrollment_samples", 1),
        enrollment_interval_ms=_integer(
            recognition_data, "enrollment_interval_ms", 0
        ),
        save_photos=_boolean(recognition_data, "save_photos"),
    )
    if recognition.votes_required > recognition.window_size:
        raise ConfigError("votes_required cannot exceed window_size")
    if not 10 <= recognition.enrollment_samples <= 20:
        raise ConfigError("enrollment_samples must be between 10 and 20")

    storage = StorageConfig(
        data_dir=_path(storage_data, "data_dir", base_dir),
    )
    return AppConfig(
        runtime=runtime,
        camera=camera,
        models=models,
        recognition=recognition,
        storage=storage,
    )


def load_config(path: Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError("configuration file not found: {}".format(config_path))
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError("invalid JSON in {}: {}".format(config_path, exc)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    return parse_config(raw, config_path.parent.resolve())


def validate_runtime_paths(config: AppConfig) -> None:
    if config.runtime.backend == "fake":
        return
    missing = [
        path
        for path in (
            config.models.detector_path,
            config.models.recognizer_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise ConfigError(
            "model file not found: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
