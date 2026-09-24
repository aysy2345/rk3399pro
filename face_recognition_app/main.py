"""Command-line entry point for the desktop application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import QApplication, QMessageBox

from face_recognition_app.app.bootstrap import (
    BootstrapError,
    apply_overrides,
    build_application,
)
from face_recognition_app.app.config import ConfigError, load_config
from face_recognition_app.storage.face_store import FaceStoreError


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


def run(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    try:
        config = load_config(args.config)
        config = apply_overrides(
            config,
            backend=args.backend,
            camera_index=args.camera_index,
        )
        bundle = build_application(config)
    except (BootstrapError, ConfigError, FaceStoreError, ValueError) as exc:
        message = "启动失败：{}".format(exc)
        print(message, file=sys.stderr)
        QMessageBox.critical(None, "启动失败", message)
        return 2
    bundle.window.showMaximized()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(run())
