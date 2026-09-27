"""Stable model contracts shared by RKNN conversion and validation tools."""

import math
from pathlib import Path


class ModelContract(object):
    """Describe one model's stable conversion-facing interface."""

    __slots__ = (
        "model_id",
        "onnx_path",
        "rknn_path",
        "input_size",
        "channel_order",
        "mean_values",
        "scale_value",
        "output_semantics",
        "target_platforms",
        "minimum_toolkit_version",
    )

    def __init__(
        self,
        model_id,
        onnx_path,
        rknn_path,
        input_size,
        channel_order,
        mean_values,
        scale_value,
        output_semantics,
        target_platforms,
        minimum_toolkit_version,
    ):
        self.model_id = str(model_id).strip()
        self.onnx_path = Path(onnx_path)
        self.rknn_path = Path(rknn_path)
        self.input_size = tuple(input_size)
        self.channel_order = str(channel_order).upper()
        self.mean_values = tuple(float(value) for value in mean_values)
        self.scale_value = float(scale_value)
        self.output_semantics = tuple(output_semantics)
        self.target_platforms = tuple(target_platforms)
        self.minimum_toolkit_version = str(minimum_toolkit_version).strip()
        self._validate()

    def _validate(self):
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if (
            len(self.input_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in self.input_size
            )
        ):
            raise ValueError("input_size must contain two positive integers")
        if self.channel_order not in ("BGR", "RGB"):
            raise ValueError("channel_order must be BGR or RGB")
        if len(self.mean_values) != 3 or not all(
            math.isfinite(value) for value in self.mean_values
        ):
            raise ValueError("mean_values must contain three finite values")
        if not math.isfinite(self.scale_value) or self.scale_value <= 0.0:
            raise ValueError("scale_value must be a positive finite value")
        if not self.output_semantics:
            raise ValueError("output_semantics must not be empty")
        if len(set(self.output_semantics)) != len(self.output_semantics):
            raise ValueError("output_semantics values must be unique")
        if not self.target_platforms:
            raise ValueError("target_platforms must not be empty")
        if not self.minimum_toolkit_version:
            raise ValueError("minimum_toolkit_version must not be empty")

    @property
    def channel_mean_value(self):
        """Return the RKNN Toolkit 1.x mean/scale configuration string."""

        values = self.mean_values + (self.scale_value,)
        return " ".join("{:g}".format(value) for value in values)


RETINAFACE_CONTRACT = ModelContract(
    model_id="retinaface",
    onnx_path=Path("models/retinaface_mobilenet025.onnx"),
    rknn_path=Path("models/retinaface_mobilenet025.rknn"),
    input_size=(640, 640),
    channel_order="BGR",
    mean_values=(104.0, 117.0, 123.0),
    scale_value=1.0,
    output_semantics=("boxes", "scores", "landmarks"),
    target_platforms=("rk3399pro",),
    minimum_toolkit_version="1.7.1",
)


MOBILEFACENET_CONTRACT = ModelContract(
    model_id="mobilefacenet",
    onnx_path=Path("models/mobilefacenet.onnx"),
    rknn_path=Path("models/mobilefacenet.rknn"),
    input_size=(112, 112),
    channel_order="BGR",
    mean_values=(127.5, 127.5, 127.5),
    scale_value=127.5,
    output_semantics=("embedding",),
    target_platforms=("rk3399pro",),
    minimum_toolkit_version="1.7.1",
)


_CONTRACTS = {
    RETINAFACE_CONTRACT.model_id: RETINAFACE_CONTRACT,
    MOBILEFACENET_CONTRACT.model_id: MOBILEFACENET_CONTRACT,
}


def get_contract(model_id):
    """Return a built-in contract by case-insensitive model identifier."""

    key = str(model_id).strip().lower()
    try:
        return _CONTRACTS[key]
    except KeyError:
        raise ValueError("unknown RKNN model: {}".format(model_id))
