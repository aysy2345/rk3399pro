from pathlib import Path

from face_recognition_app.app.config import load_config


def test_rk3399pro_config_selects_rknn_models():
    project_root = Path(__file__).resolve().parents[2]

    config = load_config(project_root / "configs" / "rk3399pro.json")

    assert config.runtime.backend == "rknn"
    assert config.models.detector_path == (
        project_root / "models" / "retinaface_mobilenet025.rknn"
    )
    assert config.models.recognizer_path == (
        project_root / "models" / "mobilefacenet.rknn"
    )
