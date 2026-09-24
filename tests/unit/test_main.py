from pathlib import Path

from face_recognition_app.main import parse_args


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
