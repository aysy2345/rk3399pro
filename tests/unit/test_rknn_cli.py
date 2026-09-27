from pathlib import Path

import pytest

from tools.rknn import convert_mobilefacenet, convert_retinaface
from tools.rknn.cli import run_conversion_cli
from tools.rknn.contracts import MOBILEFACENET_CONTRACT, RETINAFACE_CONTRACT
from tools.rknn.converter import ConversionError


def _onnx_file(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-onnx")
    return path


def test_retinaface_cli_maps_non_quantized_arguments(tmp_path, capsys):
    calls = []
    onnx_path = _onnx_file(tmp_path)
    output_path = tmp_path / "retinaface.rknn"
    summary_path = tmp_path / "summary.json"

    def fake_convert(request, summary_path=None):
        calls.append((request, summary_path))
        return {"model_id": request.contract.model_id, "ok": True}

    result = run_conversion_cli(
        RETINAFACE_CONTRACT,
        [
            "--onnx",
            str(onnx_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ],
        convert_function=fake_convert,
    )

    request, actual_summary = calls[0]
    assert result == 0
    assert request.contract is RETINAFACE_CONTRACT
    assert request.onnx_path == onnx_path
    assert request.output_path == output_path
    assert request.quantize is False
    assert request.dataset_path is None
    assert request.precompile is True
    assert actual_summary == summary_path
    assert '"ok": true' in capsys.readouterr().out


def test_mobilefacenet_cli_maps_quantized_arguments(tmp_path):
    calls = []
    onnx_path = _onnx_file(tmp_path)
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("/data/face.jpg\n", encoding="utf-8")
    output_path = tmp_path / "mobilefacenet_int8.rknn"

    def fake_convert(request, summary_path=None):
        calls.append((request, summary_path))
        return {"ok": True}

    result = run_conversion_cli(
        MOBILEFACENET_CONTRACT,
        [
            "--onnx",
            str(onnx_path),
            "--output",
            str(output_path),
            "--quantize",
            "--dataset",
            str(dataset_path),
            "--no-precompile",
        ],
        convert_function=fake_convert,
    )

    request, actual_summary = calls[0]
    assert result == 0
    assert request.contract is MOBILEFACENET_CONTRACT
    assert request.quantize is True
    assert request.dataset_path == dataset_path
    assert request.precompile is False
    assert actual_summary is None


def test_quantized_default_output_uses_int8_suffix(tmp_path):
    calls = []
    onnx_path = _onnx_file(tmp_path)
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("/data/face.jpg\n", encoding="utf-8")

    def fake_convert(request, summary_path=None):
        calls.append(request)
        return {"ok": True}

    result = run_conversion_cli(
        RETINAFACE_CONTRACT,
        [
            "--onnx",
            str(onnx_path),
            "--quantize",
            "--dataset",
            str(dataset_path),
        ],
        convert_function=fake_convert,
    )

    assert result == 0
    assert calls[0].output_path == Path(
        "models/retinaface_mobilenet025_int8.rknn"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--quantize"], "量化转换必须提供 --dataset"),
        (["--dataset", "dataset.txt"], "--dataset 只能与 --quantize"),
        (
            [
                "--quantize",
                "--dataset",
                "dataset.txt",
                "--output",
                "model.rknn",
            ],
            "INT8 输出文件名必须以 _int8.rknn 结尾",
        ),
        (["--output", "model.bin"], "输出文件必须使用 .rknn 扩展名"),
    ],
)
def test_cli_rejects_invalid_argument_combinations(
    tmp_path, capsys, arguments, message
):
    onnx_path = _onnx_file(tmp_path)

    with pytest.raises(SystemExit) as error:
        run_conversion_cli(
            RETINAFACE_CONTRACT,
            ["--onnx", str(onnx_path)] + arguments,
            convert_function=lambda request, summary_path=None: {},
        )

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_rejects_missing_onnx_before_conversion(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        run_conversion_cli(
            RETINAFACE_CONTRACT,
            ["--onnx", str(tmp_path / "missing.onnx")],
            convert_function=lambda request, summary_path=None: {},
        )

    assert error.value.code == 2
    assert "找不到 ONNX 模型" in capsys.readouterr().err


def test_cli_reports_conversion_failure_with_exit_code_one(tmp_path, capsys):
    onnx_path = _onnx_file(tmp_path)

    def fail_conversion(request, summary_path=None):
        raise ConversionError("RKNN build failed with code 7")

    with pytest.raises(SystemExit) as error:
        run_conversion_cli(
            RETINAFACE_CONTRACT,
            ["--onnx", str(onnx_path)],
            convert_function=fail_conversion,
        )

    assert error.value.code == 1
    assert "转换失败" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("module", "contract"),
    [
        (convert_retinaface, RETINAFACE_CONTRACT),
        (convert_mobilefacenet, MOBILEFACENET_CONTRACT),
    ],
)
def test_entry_point_selects_its_model_contract(monkeypatch, module, contract):
    calls = []
    monkeypatch.setattr(
        module,
        "run_conversion_cli",
        lambda actual_contract, argv=None: calls.append((actual_contract, argv)) or 0,
    )

    assert module.main(["--help"]) == 0
    assert calls == [(contract, ["--help"])]
