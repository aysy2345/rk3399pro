"""Convert the verified MobileFaceNet ONNX model to RKNN."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from tools.rknn.cli import run_conversion_cli
from tools.rknn.contracts import MOBILEFACENET_CONTRACT


def main(argv=None):
    return run_conversion_cli(MOBILEFACENET_CONTRACT, argv)


if __name__ == "__main__":
    sys.exit(main())
