"""Reusable RKNN Toolkit 1.x conversion core."""

import hashlib
import json
import tempfile
from pathlib import Path

from .contracts import ModelContract


class ConversionError(RuntimeError):
    """Raised when an RKNN conversion stage fails."""


class ConversionRequest(object):
    """Inputs required to build one RKNN model."""

    __slots__ = (
        "contract",
        "onnx_path",
        "output_path",
        "quantize",
        "dataset_path",
        "precompile",
    )

    def __init__(
        self,
        contract,
        onnx_path,
        output_path,
        quantize=False,
        dataset_path=None,
        precompile=True,
    ):
        self.contract = contract
        self.onnx_path = Path(onnx_path)
        self.output_path = Path(output_path)
        self.quantize = quantize
        self.dataset_path = (
            None if dataset_path is None else Path(dataset_path)
        )
        self.precompile = precompile


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_rknn_factory():
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise ConversionError(
            "RKNN Toolkit is not installed; run this conversion in the "
            "Ubuntu 18.04 Python 3.6 environment"
        ) from exc
    return RKNN(verbose=True)


def _detect_toolkit_version():
    try:
        import rknn
    except ImportError as exc:
        raise ConversionError("unable to read RKNN Toolkit version") from exc
    return str(getattr(rknn, "__version__", "unknown"))


def _validate_request(request, summary_path):
    if not isinstance(request, ConversionRequest):
        raise TypeError("request must be a ConversionRequest")
    if not isinstance(request.contract, ModelContract):
        raise TypeError("contract must be a ModelContract")
    if not isinstance(request.quantize, bool):
        raise TypeError("quantize must be a boolean")
    if not isinstance(request.precompile, bool):
        raise TypeError("precompile must be a boolean")
    if request.onnx_path.suffix.lower() != ".onnx":
        raise ValueError("ONNX model path must end with .onnx")
    if not request.onnx_path.is_file():
        raise FileNotFoundError("ONNX model not found: {}".format(request.onnx_path))
    if request.onnx_path.stat().st_size <= 0:
        raise ValueError("ONNX model must not be empty")
    if request.output_path.suffix.lower() != ".rknn":
        raise ValueError("RKNN output path must end with .rknn")
    if request.output_path.exists():
        raise FileExistsError(
            "RKNN output already exists: {}".format(request.output_path)
        )
    if summary_path.exists():
        raise FileExistsError(
            "conversion summary already exists: {}".format(summary_path)
        )
    if request.quantize:
        if request.dataset_path is None:
            raise ValueError("quantized conversion requires a dataset")
        if not request.dataset_path.is_file():
            raise FileNotFoundError(
                "quantization dataset not found: {}".format(request.dataset_path)
            )
        if not any(
            line.strip()
            for line in request.dataset_path.read_text(encoding="utf-8").splitlines()
        ):
            raise ValueError("quantization dataset must not be empty")
    elif request.dataset_path is not None:
        raise ValueError("dataset is only valid for quantized conversion")


def _invoke(rknn, stage, *args, **kwargs):
    method = getattr(rknn, stage)
    try:
        result = method(*args, **kwargs)
    except Exception as exc:
        raise ConversionError("RKNN {} raised an error: {}".format(stage, exc)) from exc
    if result not in (None, 0):
        raise ConversionError(
            "RKNN {} failed with code {}".format(stage, result)
        )
    return result


def _release_after_failure(rknn):
    try:
        rknn.release()
    except Exception:
        pass


def _execute_conversion(request, rknn):
    contract = request.contract
    reorder_channel = "0 1 2" if contract.channel_order == "BGR" else "2 1 0"
    _invoke(
        rknn,
        "config",
        channel_mean_value=contract.channel_mean_value,
        reorder_channel=reorder_channel,
        target_platform=list(contract.target_platforms),
    )
    _invoke(rknn, "load_onnx", model=str(request.onnx_path.resolve()))
    build_options = {
        "do_quantization": request.quantize,
        "pre_compile": request.precompile,
    }
    if request.quantize:
        build_options["dataset"] = str(request.dataset_path.resolve())
    _invoke(rknn, "build", **build_options)
    _invoke(rknn, "export_rknn", str(request.output_path.resolve()))
    if (
        not request.output_path.is_file()
        or request.output_path.stat().st_size <= 0
    ):
        raise ConversionError(
            "RKNN export_rknn did not create a non-empty output file"
        )


def _write_json(path, payload):
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


def convert_model(
    request,
    rknn_factory=None,
    toolkit_version=None,
    summary_path=None,
):
    """Convert one ONNX model and write a reproducibility summary."""

    if summary_path is None:
        summary_path = Path(str(request.output_path) + ".json")
    else:
        summary_path = Path(summary_path)
    _validate_request(request, summary_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    factory = rknn_factory or _default_rknn_factory
    if toolkit_version is None:
        toolkit_version = _detect_toolkit_version()
    try:
        rknn = factory()
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError("unable to initialize RKNN Toolkit: {}".format(exc)) from exc

    try:
        _execute_conversion(request, rknn)
    except Exception:
        _release_after_failure(rknn)
        raise
    _invoke(rknn, "release")

    contract = request.contract
    summary = {
        "model_id": contract.model_id,
        "toolkit_version": str(toolkit_version),
        "onnx_path": str(request.onnx_path.resolve()),
        "output_path": str(request.output_path.resolve()),
        "input_sha256": _sha256(request.onnx_path),
        "output_sha256": _sha256(request.output_path),
        "input_size": list(contract.input_size),
        "channel_order": contract.channel_order,
        "channel_mean_value": contract.channel_mean_value,
        "output_semantics": list(contract.output_semantics),
        "target_platforms": list(contract.target_platforms),
        "quantized": request.quantize,
        "dataset": (
            None
            if request.dataset_path is None
            else str(request.dataset_path.resolve())
        ),
        "precompiled": request.precompile,
    }
    _write_json(summary_path, summary)
    return summary
