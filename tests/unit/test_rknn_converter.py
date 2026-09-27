import json
from pathlib import Path

import pytest

from tools.rknn.contracts import MOBILEFACENET_CONTRACT, RETINAFACE_CONTRACT
from tools.rknn.converter import (
    ConversionError,
    ConversionRequest,
    convert_model,
)


class FakeRKNN(object):
    def __init__(self, failures=None, create_output=True):
        self.failures = failures or {}
        self.create_output = create_output
        self.calls = []

    def config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return self.failures.get("config", 0)

    def load_onnx(self, **kwargs):
        self.calls.append(("load_onnx", kwargs))
        return self.failures.get("load_onnx", 0)

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        return self.failures.get("build", 0)

    def export_rknn(self, path):
        self.calls.append(("export_rknn", path))
        result = self.failures.get("export_rknn", 0)
        if result == 0 and self.create_output:
            Path(path).write_bytes(b"fake-rknn")
        return result

    def release(self):
        self.calls.append(("release", None))
        return self.failures.get("release", 0)


def _onnx_file(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-onnx")
    return path


def test_non_quantized_conversion_calls_rknn_in_order_and_writes_summary(tmp_path):
    onnx_path = _onnx_file(tmp_path)
    output_path = tmp_path / "model.rknn"
    fake = FakeRKNN()
    request = ConversionRequest(
        contract=RETINAFACE_CONTRACT,
        onnx_path=onnx_path,
        output_path=output_path,
        quantize=False,
        precompile=True,
    )

    summary = convert_model(
        request,
        rknn_factory=lambda: fake,
        toolkit_version="1.7.1",
    )

    assert [name for name, _ in fake.calls] == [
        "config",
        "load_onnx",
        "build",
        "export_rknn",
        "release",
    ]
    assert fake.calls[0][1] == {
        "channel_mean_value": "104 117 123 1",
        "reorder_channel": "0 1 2",
        "target_platform": ["rk3399pro"],
    }
    assert fake.calls[1][1] == {"model": str(onnx_path.resolve())}
    assert fake.calls[2][1] == {
        "do_quantization": False,
        "pre_compile": True,
    }
    assert fake.calls[3][1] == str(output_path.resolve())
    assert summary["model_id"] == "retinaface"
    assert summary["toolkit_version"] == "1.7.1"
    assert summary["quantized"] is False
    assert summary["precompiled"] is True
    assert len(summary["input_sha256"]) == 64
    assert len(summary["output_sha256"]) == 64
    summary_path = Path(str(output_path) + ".json")
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_quantized_conversion_passes_nonempty_dataset(tmp_path):
    onnx_path = _onnx_file(tmp_path)
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("/data/face.jpg\n", encoding="utf-8")
    output_path = tmp_path / "model_int8.rknn"
    fake = FakeRKNN()
    request = ConversionRequest(
        contract=MOBILEFACENET_CONTRACT,
        onnx_path=onnx_path,
        output_path=output_path,
        quantize=True,
        dataset_path=dataset_path,
        precompile=False,
    )

    summary = convert_model(
        request,
        rknn_factory=lambda: fake,
        toolkit_version="1.7.1",
    )

    assert fake.calls[2] == (
        "build",
        {
            "do_quantization": True,
            "dataset": str(dataset_path.resolve()),
            "pre_compile": False,
        },
    )
    assert summary["dataset"] == str(dataset_path.resolve())
    assert summary["quantized"] is True


@pytest.mark.parametrize(
    ("quantize", "dataset_contents", "message"),
    [
        (True, None, "requires a dataset"),
        (True, "   \n", "must not be empty"),
        (False, "/data/face.jpg\n", "only valid for quantized"),
    ],
)
def test_request_rejects_invalid_dataset_combinations(
    tmp_path, quantize, dataset_contents, message
):
    onnx_path = _onnx_file(tmp_path)
    dataset_path = None
    if dataset_contents is not None:
        dataset_path = tmp_path / "dataset.txt"
        dataset_path.write_text(dataset_contents, encoding="utf-8")
    request = ConversionRequest(
        contract=RETINAFACE_CONTRACT,
        onnx_path=onnx_path,
        output_path=tmp_path / "model.rknn",
        quantize=quantize,
        dataset_path=dataset_path,
    )

    with pytest.raises(ValueError, match=message):
        convert_model(
            request,
            rknn_factory=FakeRKNN,
            toolkit_version="1.7.1",
        )


def test_request_rejects_missing_onnx_file(tmp_path):
    request = ConversionRequest(
        contract=RETINAFACE_CONTRACT,
        onnx_path=tmp_path / "missing.onnx",
        output_path=tmp_path / "model.rknn",
    )

    with pytest.raises(FileNotFoundError, match="ONNX model"):
        convert_model(
            request,
            rknn_factory=FakeRKNN,
            toolkit_version="1.7.1",
        )


def test_request_refuses_to_overwrite_existing_output(tmp_path):
    output_path = tmp_path / "model.rknn"
    output_path.write_bytes(b"verified-model")
    request = ConversionRequest(
        contract=RETINAFACE_CONTRACT,
        onnx_path=_onnx_file(tmp_path),
        output_path=output_path,
    )

    with pytest.raises(FileExistsError, match="RKNN output"):
        convert_model(
            request,
            rknn_factory=FakeRKNN,
            toolkit_version="1.7.1",
        )


@pytest.mark.parametrize(
    "stage", ["config", "load_onnx", "build", "export_rknn", "release"]
)
def test_rknn_api_failure_names_stage_and_releases_resources(tmp_path, stage):
    fake = FakeRKNN(failures={stage: 7})
    request = ConversionRequest(
        contract=RETINAFACE_CONTRACT,
        onnx_path=_onnx_file(tmp_path),
        output_path=tmp_path / "model.rknn",
    )

    with pytest.raises(ConversionError, match=stage):
        convert_model(
            request,
            rknn_factory=lambda: fake,
            toolkit_version="1.7.1",
        )

    assert fake.calls[-1] == ("release", None)


def test_success_without_output_file_is_rejected_and_released(tmp_path):
    fake = FakeRKNN(create_output=False)
    request = ConversionRequest(
        contract=RETINAFACE_CONTRACT,
        onnx_path=_onnx_file(tmp_path),
        output_path=tmp_path / "model.rknn",
    )

    with pytest.raises(ConversionError, match="did not create"):
        convert_model(
            request,
            rknn_factory=lambda: fake,
            toolkit_version="1.7.1",
        )

    assert fake.calls[-1] == ("release", None)
