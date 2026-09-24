"""Command-line entry point for the desktop application."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional

from face_recognition_app.app.config import ConfigError, load_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3399Pro 本地人脸识别")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/app.json"),
        help="应用配置文件路径",
    )
    parser.add_argument(
        "--backend",
        choices=("fake", "onnx", "rknn"),
        help="覆盖配置中的推理后端",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        help="覆盖配置中的 USB 摄像头编号",
    )
    return parser.parse_args(argv)


def _preload_onnx_runtime():
    try:
        return importlib.import_module("onnxruntime")
    except (ImportError, OSError) as exc:
        from face_recognition_app.app.bootstrap import BootstrapError

        raise BootstrapError(
            "ONNX Runtime 加载失败：{}。"
            "请检查 onnxruntime 安装及 Microsoft Visual C++ 运行库。".format(
                exc
            )
        ) from exc


def _show_startup_error(exc: Exception) -> int:
    from PyQt5.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication([sys.argv[0]])
    message = "启动失败：{}".format(exc)
    print(message, file=sys.stderr)
    QMessageBox.critical(None, "启动失败", message)
    return 2


def run(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        selected_backend = args.backend or config.runtime.backend
        if selected_backend == "onnx":
            _preload_onnx_runtime()
        from face_recognition_app.app.bootstrap import (
            BootstrapError,
            apply_overrides,
            build_application,
        )

        config = apply_overrides(
            config,
            backend=args.backend,
            camera_index=args.camera_index,
        )
    except (ConfigError, RuntimeError, ValueError) as exc:
        return _show_startup_error(exc)
    from PyQt5.QtWidgets import QApplication
    from face_recognition_app.storage.face_store import FaceStoreError

    app = QApplication.instance() or QApplication([sys.argv[0]])
    try:
        bundle = build_application(config)
    except (BootstrapError, ConfigError, FaceStoreError, ValueError) as exc:
        return _show_startup_error(exc)
    bundle.window.showMaximized()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(run())
