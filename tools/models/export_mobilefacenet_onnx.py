"""Export the selected MobileFaceNet TorchScript model and verify ONNX output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left.reshape(-1).astype(np.float64)
    right_vector = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator <= 1e-12:
        raise RuntimeError("cannot compare zero-length model outputs")
    return float(np.dot(left_vector, right_vector) / denominator)


def export_and_verify(source: Path, output: Path) -> dict:
    torch.manual_seed(20260924)
    model = torch.jit.load(str(source), map_location="cpu")
    model.eval()
    sample = torch.rand(1, 3, 112, 112, dtype=torch.float32) * 2.0 - 1.0
    with torch.no_grad():
        torch_output = model(sample).detach().cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(output),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=11,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(
        ["embedding"], {"input": sample.numpy()}
    )[0]
    max_absolute_error = float(np.max(np.abs(torch_output - onnx_output)))
    cosine = cosine_similarity(torch_output, onnx_output)
    if max_absolute_error > 1e-4 or cosine < 0.99999:
        raise RuntimeError(
            "ONNX consistency check failed: max_abs_error={}, cosine={}".format(
                max_absolute_error, cosine
            )
        )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "input_shape": list(sample.shape),
        "output_shape": list(onnx_output.shape),
        "opset": 11,
        "max_absolute_error": max_absolute_error,
        "cosine_similarity": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/mobilefacenet.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mobilefacenet.onnx"),
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit("source model not found: {}".format(args.source))
    result = export_and_verify(args.source, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
