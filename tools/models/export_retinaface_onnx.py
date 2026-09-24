"""Export RetinaFace MobileNet0.25 and verify all three ONNX outputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> dict:
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.reshape(-1).astype(np.float64)
    right_vector = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator <= 1e-12:
        raise RuntimeError("cannot compare zero-length model outputs")
    return float(np.dot(left_vector, right_vector) / denominator)


def export_and_verify(source_dir: Path, weights: Path, output: Path) -> dict:
    source_dir = source_dir.resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from retinaface.data import cfg_mnet
    from retinaface.models.retinaface import RetinaFace

    config = copy.deepcopy(cfg_mnet)
    config["pretrain"] = False
    model = RetinaFace(cfg=config, phase="test")
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = remove_prefix(checkpoint, "module.")
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "missing model keys: {}".format(incompatible.missing_keys)
        )
    model.eval()

    torch.manual_seed(20260924)
    sample = torch.rand(1, 3, 640, 640, dtype=torch.float32) * 255.0
    sample -= torch.tensor((104.0, 117.0, 123.0)).view(1, 3, 1, 1)
    with torch.no_grad():
        torch_outputs = [item.detach().cpu().numpy() for item in model(sample)]

    output.parent.mkdir(parents=True, exist_ok=True)
    output_names = ["boxes", "scores", "landmarks"]
    torch.onnx.export(
        model,
        sample,
        str(output),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        opset_version=11,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    onnx_outputs = session.run(output_names, {"input": sample.numpy()})

    comparisons = {}
    for name, torch_output, onnx_output in zip(
        output_names, torch_outputs, onnx_outputs
    ):
        max_error = float(np.max(np.abs(torch_output - onnx_output)))
        cosine = cosine_similarity(torch_output, onnx_output)
        if max_error > 1e-4 or cosine < 0.99999:
            raise RuntimeError(
                "{} consistency failed: max_abs_error={}, cosine={}".format(
                    name, max_error, cosine
                )
            )
        comparisons[name] = {
            "shape": list(onnx_output.shape),
            "max_absolute_error": max_error,
            "cosine_similarity": cosine,
        }
    return {
        "source_revision": "2c720d6875488e94f4d4eb870936cb05613b74d5",
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "output": str(output),
        "output_sha256": sha256(output),
        "input_shape": list(sample.shape),
        "opset": 11,
        "outputs": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("models/.sources/foamliu-mobilefacenet"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "models/.sources/foamliu-mobilefacenet/"
            "retinaface/weights/mobilenet0.25_Final.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/retinaface_mobilenet025.onnx"),
    )
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        raise SystemExit("source directory not found: {}".format(args.source_dir))
    if not args.weights.is_file():
        raise SystemExit("weights not found: {}".format(args.weights))
    result = export_and_verify(args.source_dir, args.weights, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
