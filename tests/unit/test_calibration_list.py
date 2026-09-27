from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.rknn.build_calibration_list import build_calibration_list, main


def _write_image(path, value=127, size=(20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[0], size[1], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_builder_recurses_filters_and_writes_absolute_paths(tmp_path):
    image_root = tmp_path / "images"
    first = _write_image(image_root / "a.jpg", value=10)
    second = _write_image(image_root / "nested" / "b.PNG", value=20)
    _write_image(image_root / "ignored.tiff", value=30)
    (image_root / "broken.jpeg").write_bytes(b"not-an-image")
    (image_root / "notes.txt").write_text("ignore", encoding="utf-8")
    output_path = tmp_path / "retinaface.rknn-dataset.txt"

    selected = build_calibration_list(
        image_dir=image_root,
        output_path=output_path,
        model_id="retinaface",
        max_images=10,
        seed=7,
    )

    assert set(selected) == {first.resolve(), second.resolve()}
    written = [Path(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written == selected
    assert all(path.is_absolute() for path in written)


def test_sampling_is_reproducible_for_fixed_seed(tmp_path):
    image_root = tmp_path / "images"
    for index in range(10):
        _write_image(image_root / "{}.jpg".format(index), value=index)

    first = build_calibration_list(
        image_dir=image_root,
        output_path=tmp_path / "first.rknn-dataset.txt",
        model_id="retinaface",
        max_images=4,
        seed=20260927,
    )
    second = build_calibration_list(
        image_dir=image_root,
        output_path=tmp_path / "second.rknn-dataset.txt",
        model_id="retinaface",
        max_images=4,
        seed=20260927,
    )
    different = build_calibration_list(
        image_dir=image_root,
        output_path=tmp_path / "different.rknn-dataset.txt",
        model_id="retinaface",
        max_images=4,
        seed=8,
    )

    assert first == second
    assert first != different
    assert len(first) == 4


def test_mobilefacenet_accepts_decodable_images_that_can_resize(tmp_path):
    image_root = tmp_path / "aligned"
    image = _write_image(image_root / "face.bmp", size=(13, 17))

    selected = build_calibration_list(
        image_dir=image_root,
        output_path=tmp_path / "mobilefacenet.rknn-dataset.txt",
        model_id="mobilefacenet",
    )

    assert selected == [image.resolve()]


def test_builder_rejects_directory_without_valid_images(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "broken.png").write_bytes(b"broken")

    with pytest.raises(ValueError, match="no valid calibration images"):
        build_calibration_list(
            image_dir=image_root,
            output_path=tmp_path / "dataset.rknn-dataset.txt",
            model_id="retinaface",
        )


def test_builder_refuses_overwrite_unless_force_is_set(tmp_path):
    image_root = tmp_path / "images"
    image = _write_image(image_root / "face.jpg")
    output_path = tmp_path / "dataset.rknn-dataset.txt"
    output_path.write_text("keep-me\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        build_calibration_list(
            image_dir=image_root,
            output_path=output_path,
            model_id="retinaface",
        )

    selected = build_calibration_list(
        image_dir=image_root,
        output_path=output_path,
        model_id="retinaface",
        force=True,
    )
    assert selected == [image.resolve()]
    assert "keep-me" not in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("max_images", [0, -1, True])
def test_builder_rejects_invalid_max_images(tmp_path, max_images):
    image_root = tmp_path / "images"
    image_root.mkdir()

    with pytest.raises(ValueError, match="max_images"):
        build_calibration_list(
            image_dir=image_root,
            output_path=tmp_path / "dataset.rknn-dataset.txt",
            model_id="retinaface",
            max_images=max_images,
        )


def test_cli_reports_written_count(tmp_path, capsys):
    image_root = tmp_path / "images"
    _write_image(image_root / "face.jpg")
    output_path = tmp_path / "dataset.rknn-dataset.txt"

    result = main(
        [
            "--model",
            "mobilefacenet",
            "--images",
            str(image_root),
            "--output",
            str(output_path),
            "--max-images",
            "1",
            "--seed",
            "9",
        ]
    )

    assert result == 0
    assert "已写入 1 张校准图片" in capsys.readouterr().out
