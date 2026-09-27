"""Build a reproducible RKNN INT8 calibration image list."""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from tools.rknn.contracts import get_contract


SUPPORTED_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".bmp"))


def _discover_images(image_dir):
    unique = {}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = path.resolve()
        unique[os.path.normcase(str(resolved))] = resolved
    return sorted(unique.values(), key=lambda item: os.path.normcase(str(item)))


def _image_is_usable(path, input_size):
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return False
        resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
        return resized.shape[:2] == (input_size[1], input_size[0])
    except (OSError, ValueError, cv2.error):
        return False


def _write_lines(path, paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            for image_path in paths:
                handle.write(str(image_path))
                handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def build_calibration_list(
    image_dir,
    output_path,
    model_id,
    max_images=100,
    seed=20260927,
    force=False,
):
    """Validate, sample and write absolute calibration image paths."""

    image_dir = Path(image_dir)
    output_path = Path(output_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(
            "calibration image directory not found: {}".format(image_dir)
        )
    if (
        not isinstance(max_images, int)
        or isinstance(max_images, bool)
        or max_images <= 0
    ):
        raise ValueError("max_images must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(force, bool):
        raise TypeError("force must be a boolean")
    if output_path.exists() and not force:
        raise FileExistsError(
            "calibration list already exists: {}".format(output_path)
        )

    contract = get_contract(model_id)
    candidates = [
        path
        for path in _discover_images(image_dir.resolve())
        if _image_is_usable(path, contract.input_size)
    ]
    if not candidates:
        raise ValueError("no valid calibration images found")
    if len(candidates) > max_images:
        generator = random.Random(seed)
        candidates = generator.sample(candidates, max_images)
        candidates.sort(key=lambda item: os.path.normcase(str(item)))
    _write_lines(output_path, candidates)
    return candidates


def _build_parser():
    parser = argparse.ArgumentParser(
        description="生成 RKNN INT8 校准图片清单"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("retinaface", "mobilefacenet"),
        help="目标模型",
    )
    parser.add_argument(
        "--images", type=Path, required=True, help="本地校准图片目录"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="输出清单路径"
    )
    parser.add_argument(
        "--max-images", type=int, default=100, help="最多写入的图片数量"
    )
    parser.add_argument(
        "--seed", type=int, default=20260927, help="固定抽样种子"
    )
    parser.add_argument(
        "--force", action="store_true", help="显式覆盖已有清单"
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        selected = build_calibration_list(
            image_dir=arguments.images,
            output_path=arguments.output,
            model_id=arguments.model,
            max_images=arguments.max_images,
            seed=arguments.seed,
            force=arguments.force,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        parser.exit(1, "生成校准清单失败：{}\n".format(exc))
    print(
        "已写入 {} 张校准图片：{}".format(
            len(selected), arguments.output.resolve()
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
