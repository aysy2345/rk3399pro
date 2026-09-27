# 模型目录

此目录只保存本地模型，不将模型二进制文件提交到 Git。

预期文件：

- `retinaface_mobilenet025.onnx` 和转换后的 `retinaface_mobilenet025.rknn`
- `mobilefacenet.onnx` 和转换后的 `mobilefacenet.rknn`

接入模型时必须记录来源、许可证、SHA-256、输入尺寸、颜色通道、归一化参数和输出节点。

`model-manifest.example.json` 记录当前候选来源和预期输入输出。只有完成以下检查后，才可复制为本地 `model-manifest.json` 并将状态改为 `verified`：

1. 固定源仓库 revision 与权重下载地址；
2. 计算 ONNX 和 RKNN 文件的 SHA-256；
3. 对同一组测试图验证原框架、ONNX 与 RKNN 输出的一致性；
4. 用现场采集数据校准检测阈值与识别阈值。

MobileFaceNet 的本机导出与一致性检查：

    python tools/models/export_mobilefacenet_onnx.py

脚本固定使用 ONNX opset 11，并要求 TorchScript 与 ONNX 的最大绝对误差不超过 1e-4、余弦相似度不低于 0.99999。

RetinaFace 的候选源码先缓存在 models/.sources/foamliu-mobilefacenet，再执行：

    python tools/models/export_retinaface_onnx.py

该脚本分别核对边框、分类分数和五点关键点三个输出。

## RKNN 转换

RKNN 转换工具已经完成，真实模型仍需在 Ubuntu 18.04 x86_64、Python 3.6.9 和 RKNN Toolkit 1.7.1 中生成并验证。完整步骤见 [RKNN 模型转换虚拟机操作手册](../docs/rknn-conversion-setup.md)。

非量化转换入口：

    python tools/rknn/convert_retinaface.py
    python tools/rknn/convert_mobilefacenet.py

INT8 转换必须提供校准清单，输出文件必须带 `_int8.rknn` 后缀。转换后使用：

    python tools/rknn/validate_outputs.py --help

验证报告必须包含模型 SHA-256、逐图片结果和整体通过状态。`.rknn`、校准图片、清单和本地报告不会提交到 Git。
