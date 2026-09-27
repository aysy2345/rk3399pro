"""Shared command-line handling for RKNN model conversion."""

import argparse
import json
from pathlib import Path

from .converter import ConversionError, ConversionRequest, convert_model


def _default_output(contract, quantize):
    output = contract.rknn_path
    if not quantize:
        return output
    return output.with_name(output.stem + "_int8" + output.suffix)


def _build_parser(contract):
    parser = argparse.ArgumentParser(
        prog="convert_{}".format(contract.model_id),
        description="将 {} ONNX 模型转换为 RK3399Pro RKNN 模型".format(
            contract.model_id
        ),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=contract.onnx_path,
        help="输入 ONNX 模型路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 RKNN 模型路径",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="启用 INT8 量化",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="INT8 校准图片清单",
    )
    parser.add_argument(
        "--no-precompile",
        action="store_false",
        dest="precompile",
        default=True,
        help="关闭课程默认启用的 RK3399Pro 预编译",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="转换摘要 JSON 路径",
    )
    return parser


def _validate_cli_arguments(parser, arguments, output_path):
    if output_path.suffix.lower() != ".rknn":
        parser.error("输出文件必须使用 .rknn 扩展名")
    if arguments.quantize and not output_path.stem.lower().endswith("_int8"):
        parser.error("INT8 输出文件名必须以 _int8.rknn 结尾")
    if arguments.quantize and arguments.dataset is None:
        parser.error("量化转换必须提供 --dataset")
    if not arguments.quantize and arguments.dataset is not None:
        parser.error("--dataset 只能与 --quantize 一起使用")
    if not arguments.onnx.is_file():
        parser.error("找不到 ONNX 模型：{}".format(arguments.onnx))
    if arguments.dataset is not None and not arguments.dataset.is_file():
        parser.error("找不到校准清单：{}".format(arguments.dataset))


def run_conversion_cli(contract, argv=None, convert_function=convert_model):
    """Parse arguments, run one conversion and return a process exit code."""

    parser = _build_parser(contract)
    arguments = parser.parse_args(argv)
    output_path = arguments.output or _default_output(
        contract, arguments.quantize
    )
    _validate_cli_arguments(parser, arguments, output_path)
    request = ConversionRequest(
        contract=contract,
        onnx_path=arguments.onnx,
        output_path=output_path,
        quantize=arguments.quantize,
        dataset_path=arguments.dataset,
        precompile=arguments.precompile,
    )
    try:
        result = convert_function(request, summary_path=arguments.summary)
    except (
        ConversionError,
        FileExistsError,
        FileNotFoundError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(1, "转换失败：{}\n".format(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
