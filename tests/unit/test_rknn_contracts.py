from pathlib import Path

import pytest

from tools.rknn.contracts import (
    MOBILEFACENET_CONTRACT,
    RETINAFACE_CONTRACT,
    ModelContract,
    get_contract,
)


def test_retinaface_contract_matches_verified_onnx_model():
    contract = RETINAFACE_CONTRACT

    assert contract.model_id == "retinaface"
    assert contract.onnx_path == Path("models/retinaface_mobilenet025.onnx")
    assert contract.rknn_path == Path("models/retinaface_mobilenet025.rknn")
    assert contract.input_size == (640, 640)
    assert contract.channel_order == "BGR"
    assert contract.mean_values == (104.0, 117.0, 123.0)
    assert contract.scale_value == 1.0
    assert contract.output_semantics == ("boxes", "scores", "landmarks")
    assert contract.target_platforms == ("rk3399pro",)
    assert contract.minimum_toolkit_version == "1.7.1"
    assert contract.channel_mean_value == "104 117 123 1"


def test_mobilefacenet_contract_matches_verified_onnx_model():
    contract = MOBILEFACENET_CONTRACT

    assert contract.model_id == "mobilefacenet"
    assert contract.onnx_path == Path("models/mobilefacenet.onnx")
    assert contract.rknn_path == Path("models/mobilefacenet.rknn")
    assert contract.input_size == (112, 112)
    assert contract.channel_order == "BGR"
    assert contract.mean_values == (127.5, 127.5, 127.5)
    assert contract.scale_value == 127.5
    assert contract.output_semantics == ("embedding",)
    assert contract.target_platforms == ("rk3399pro",)
    assert contract.minimum_toolkit_version == "1.7.1"
    assert contract.channel_mean_value == "127.5 127.5 127.5 127.5"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("retinaface", RETINAFACE_CONTRACT),
        ("RETINAFACE", RETINAFACE_CONTRACT),
        ("mobilefacenet", MOBILEFACENET_CONTRACT),
        (" MobileFaceNet ", MOBILEFACENET_CONTRACT),
    ],
)
def test_get_contract_is_case_insensitive(model_id, expected):
    assert get_contract(model_id) is expected


def test_get_contract_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown RKNN model"):
        get_contract("unknown")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_size": (0, 112)}, "input_size"),
        ({"channel_order": "GRAY"}, "channel_order"),
        ({"mean_values": (127.5,)}, "mean_values"),
        ({"scale_value": 0.0}, "scale_value"),
        ({"output_semantics": ()}, "output_semantics"),
        ({"output_semantics": ("embedding", "embedding")}, "unique"),
        ({"target_platforms": ()}, "target_platforms"),
        ({"minimum_toolkit_version": ""}, "minimum_toolkit_version"),
    ],
)
def test_contract_rejects_invalid_values(overrides, message):
    values = {
        "model_id": "test-model",
        "onnx_path": Path("models/test.onnx"),
        "rknn_path": Path("models/test.rknn"),
        "input_size": (112, 112),
        "channel_order": "BGR",
        "mean_values": (127.5, 127.5, 127.5),
        "scale_value": 127.5,
        "output_semantics": ("embedding",),
        "target_platforms": ("rk3399pro",),
        "minimum_toolkit_version": "1.7.1",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        ModelContract(**values)
