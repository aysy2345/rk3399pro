"""Compare ONNX Runtime and RKNN Toolkit outputs on fixed images."""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from tools.rknn.build_calibration_list import _discover_images
from tools.rknn.contracts import get_contract
from tools.rknn.metrics import (
    MetricError,
    compare_mobilefacenet,
    compare_retinaface_detections,
    decode_retinaface,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("unable to decode image")
    return image


def _single_output(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise MetricError("MobileFaceNet must return exactly one output")
        return outputs[0]
    return outputs


def _write_json(path, payload):
    path = Path(path)
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def validate_images(
    contract,
    image_paths,
    onnx_runner,
    rknn_runner,
    onnx_path=None,
    rknn_path=None,
    report_path=None,
    minimum_cosine=0.99,
    box_tolerance=3.0,
    landmark_tolerance=3.0,
    confidence_threshold=0.8,
    nms_threshold=0.4,
):
    image_paths = [Path(path) for path in image_paths]
    if not image_paths:
        raise ValueError("at least one validation image is required")
    report = {
        "model_id": contract.model_id,
        "onnx_path": None if onnx_path is None else str(Path(onnx_path).resolve()),
        "rknn_path": None if rknn_path is None else str(Path(rknn_path).resolve()),
        "onnx_sha256": None if onnx_path is None else _sha256(onnx_path),
        "rknn_sha256": None if rknn_path is None else _sha256(rknn_path),
        "thresholds": {
            "minimum_cosine": float(minimum_cosine),
            "box_tolerance": float(box_tolerance),
            "landmark_tolerance": float(landmark_tolerance),
            "confidence_threshold": float(confidence_threshold),
            "nms_threshold": float(nms_threshold),
        },
        "samples": [],
    }
    for image_path in image_paths:
        sample = {"image": str(image_path.resolve())}
        try:
            image = _read_image(image_path)
            onnx_outputs = onnx_runner(image)
            rknn_outputs = rknn_runner(image)
            if contract.model_id == "mobilefacenet":
                metrics = compare_mobilefacenet(
                    _single_output(onnx_outputs),
                    _single_output(rknn_outputs),
                    minimum_cosine=minimum_cosine,
                )
                failure = None
                if not metrics["passed"]:
                    failure = "cosine similarity below minimum"
            elif contract.model_id == "retinaface":
                source_shape = image.shape[:2]
                reference_boxes, reference_points = decode_retinaface(
                    onnx_outputs,
                    source_shape=source_shape,
                    input_size=contract.input_size,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )
                candidate_boxes, candidate_points = decode_retinaface(
                    rknn_outputs,
                    source_shape=source_shape,
                    input_size=contract.input_size,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )
                metrics = compare_retinaface_detections(
                    reference_boxes,
                    reference_points,
                    candidate_boxes,
                    candidate_points,
                    box_tolerance=box_tolerance,
                    landmark_tolerance=landmark_tolerance,
                )
                failure = None
                if not metrics["passed"]:
                    failure = "detection count or coordinate tolerance failed"
            else:
                raise ValueError("unsupported validation model: {}".format(contract.model_id))
            sample.update(
                {"metrics": metrics, "passed": metrics["passed"], "failure": failure}
            )
        except Exception as exc:
            sample.update({"metrics": None, "passed": False, "failure": str(exc)})
        report["samples"].append(sample)
    report["passed"] = all(sample["passed"] for sample in report["samples"])
    report["failure_count"] = sum(
        1 for sample in report["samples"] if not sample["passed"]
    )
    if report_path is not None:
        _write_json(report_path, report)
    return report


class OnnxRunner(object):
    def __init__(self, contract, model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for validation") from exc
        self.contract = contract
        self.session = ort.InferenceSession(
            str(Path(model_path)), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]

    def __call__(self, image):
        resized = cv2.resize(image, self.contract.input_size).astype(np.float32)
        means = np.asarray(self.contract.mean_values, dtype=np.float32)
        normalized = (resized - means) / self.contract.scale_value
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return self.session.run(self.output_names, {self.input_name: tensor})


class RknnRunner(object):
    def __init__(self, contract, model_path):
        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise RuntimeError("RKNN Toolkit is required for validation") from exc
        self.contract = contract
        self.rknn = RKNN(verbose=True)
        result = self.rknn.load_rknn(str(Path(model_path)))
        if result not in (None, 0):
            self.rknn.release()
            raise RuntimeError("RKNN load_rknn failed with code {}".format(result))
        result = self.rknn.init_runtime()
        if result not in (None, 0):
            self.rknn.release()
            raise RuntimeError("RKNN init_runtime failed with code {}".format(result))

    def __call__(self, image):
        resized = cv2.resize(image, self.contract.input_size)
        outputs = self.rknn.inference(inputs=[resized])
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")
        return outputs

    def release(self):
        self.rknn.release()


def _build_parser():
    parser = argparse.ArgumentParser(description="比较 ONNX 与 RKNN 模型输出")
    parser.add_argument(
        "--model", required=True, choices=("retinaface", "mobilefacenet")
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--minimum-cosine", type=float, default=0.99)
    parser.add_argument("--box-tolerance", type=float, default=3.0)
    parser.add_argument("--landmark-tolerance", type=float, default=3.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    return parser


def main(argv=None):
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.onnx.is_file():
        parser.error("找不到 ONNX 模型：{}".format(arguments.onnx))
    if not arguments.rknn.is_file():
        parser.error("找不到 RKNN 模型：{}".format(arguments.rknn))
    if not arguments.images.is_dir():
        parser.error("找不到验证图片目录：{}".format(arguments.images))
    images = _discover_images(arguments.images.resolve())
    if not images:
        parser.error("验证图片目录中没有支持的图片")
    contract = get_contract(arguments.model)
    report_path = arguments.report or Path(
        "rknn-results/{}_validation.json".format(contract.model_id)
    )
    rknn_runner = None
    try:
        onnx_runner = OnnxRunner(contract, arguments.onnx)
        rknn_runner = RknnRunner(contract, arguments.rknn)
        report = validate_images(
            contract,
            images,
            onnx_runner,
            rknn_runner,
            onnx_path=arguments.onnx,
            rknn_path=arguments.rknn,
            report_path=report_path,
            minimum_cosine=arguments.minimum_cosine,
            box_tolerance=arguments.box_tolerance,
            landmark_tolerance=arguments.landmark_tolerance,
            confidence_threshold=arguments.confidence_threshold,
            nms_threshold=arguments.nms_threshold,
        )
    except (MetricError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, "验证失败：{}\n".format(exc))
    finally:
        if rknn_runner is not None:
            rknn_runner.release()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
