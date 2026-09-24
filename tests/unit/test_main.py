from pathlib import Path

import pytest

from face_recognition_app.app.bootstrap import BootstrapError
from face_recognition_app.main import _preload_onnx_runtime, parse_args


def test_parse_args_supports_runtime_overrides():
    args = parse_args(
        [
            "--config",
            "custom.json",
            "--backend",
            "fake",
            "--camera-index",
            "2",
        ]
    )

    assert args.config == Path("custom.json")
    assert args.backend == "fake"
    assert args.camera_index == 2


def test_preload_onnx_runtime_uses_importlib(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "face_recognition_app.main.importlib.import_module",
        lambda name: sentinel if name == "onnxruntime" else None,
    )

    assert _preload_onnx_runtime() is sentinel


def test_preload_onnx_runtime_reports_dll_failure(monkeypatch):
    def fail(name):
        raise ImportError("DLL load failed")

    monkeypatch.setattr(
        "face_recognition_app.main.importlib.import_module", fail
    )

    with pytest.raises(BootstrapError, match="ONNX Runtime 加载失败"):
        _preload_onnx_runtime()
