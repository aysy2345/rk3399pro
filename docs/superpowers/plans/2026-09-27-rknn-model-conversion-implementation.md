# RKNN 模型转换实施计划

## 实施原则

本计划落实 `docs/superpowers/specs/2026-09-27-rknn-model-conversion-design.md`。转换工具兼容 Ubuntu 18.04、Python 3.6.9 和 RKNN Toolkit 1.7.1。Windows 测试环境不安装 RKNN Toolkit，通过依赖注入和测试替身验证参数、流程与错误处理。真实 `.rknn` 生成和模拟器推理只在 VMware 虚拟机中执行。

每个任务先添加失败测试，再做最小实现并运行定向测试。完成一个可独立验证的任务后提交一次 Git。

## Task 1：固定模型转换契约

### 文件

- 新增 `tools/rknn/__init__.py`
- 新增 `tools/rknn/contracts.py`
- 新增 `tests/unit/test_rknn_contracts.py`

### 步骤

1. 为 RetinaFace 和 MobileFaceNet 定义唯一模型标识、ONNX 默认路径、RKNN 输出路径、输入宽高、通道顺序、均值、缩放、输出语义和最低 Toolkit 版本。
2. 使用 Python 3.6 可运行的普通类或命名元组，避免依赖 Python 3.7 才提供的标准库功能。
3. 添加契约校验，拒绝未知模型、非正数输入尺寸、错误通道数和缺失输出定义。
4. 测试两个内置契约与设计文档完全一致。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_rknn_contracts.py -q

预期：测试通过，且不需要导入 `rknn`、ONNX Runtime 或模型文件。

### 提交

    git commit -m "feat: define RKNN conversion contracts"

## Task 2：实现可测试的 RKNN 转换核心

### 文件

- 新增 `tools/rknn/converter.py`
- 新增 `tests/unit/test_rknn_converter.py`

### 步骤

1. 定义 `ConversionRequest`，包含模型契约、ONNX 输入、RKNN 输出、是否量化、校准清单和是否预编译。
2. 通过工厂函数延迟导入 `rknn.api.RKNN`，保证 Windows 单元测试可以导入模块。
3. 按 RKNN Toolkit 1.7.1 顺序调用 `config`、`load_onnx`、`build`、`export_rknn` 和 `release`。
4. 配置目标平台为 RK3399Pro，并按照模型契约设置 BGR 顺序、均值和缩放。
5. 非量化请求禁止携带校准清单；INT8 请求必须提供存在且非空的清单。
6. 每个 RKNN API 返回非零值时抛出包含阶段名称的错误，并保证 `release` 在失败路径也执行。
7. 导出成功后检查输出文件存在且非空，再生成包含输入 SHA-256、输出 SHA-256、参数和 Toolkit 版本的 JSON 摘要。
8. 使用 Fake RKNN 对象验证调用顺序、参数、失败传播和资源释放。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_rknn_converter.py -q

预期：覆盖非量化、INT8、空清单、API 失败、输出缺失和正常释放。

### 提交

    git commit -m "feat: add reusable RKNN conversion core"

## Task 3：增加两个转换命令入口

### 文件

- 新增 `tools/rknn/convert_retinaface.py`
- 新增 `tools/rknn/convert_mobilefacenet.py`
- 新增 `tests/unit/test_rknn_cli.py`

### 步骤

1. 两个入口只负责解析命令行、选择契约并调用公共转换核心。
2. 支持 `--onnx`、`--output`、`--quantize`、`--dataset`、`--no-precompile` 和 `--summary`；默认启用课程要求的预编译，仅在诊断时显式关闭。
3. 默认先生成不同名的非量化模型；INT8 输出文件必须带有 `_int8` 后缀，避免覆盖。
4. 对无效组合给出中文错误，例如量化但未传清单、输入模型不存在或输出扩展名不是 `.rknn`。
5. 单元测试替换转换核心，验证参数映射和退出码。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_rknn_cli.py -q
    .\.test-venv\Scripts\python.exe tools/rknn/convert_retinaface.py --help
    .\.test-venv\Scripts\python.exe tools/rknn/convert_mobilefacenet.py --help

### 提交

    git commit -m "feat: add RKNN conversion commands"

## Task 4：生成安全且可复现的校准清单

### 文件

- 新增 `tools/rknn/build_calibration_list.py`
- 新增 `tests/unit/test_calibration_list.py`
- 更新 `.gitignore`

### 步骤

1. 接收本地图片目录、输出清单、最大数量和固定随机种子。
2. 只接收 `.jpg`、`.jpeg`、`.png` 和 `.bmp`，递归扫描后按规范化路径去重。
3. RetinaFace 清单引用原始摄像头帧；MobileFaceNet 清单要求图片可解码并能缩放为 112 × 112。
4. 清单使用绝对路径，确保从不同工作目录运行 RKNN Toolkit 时仍可读取图片。
5. 输出前检查至少有一张有效图片，且不覆盖未显式指定的文件。
6. 将本地清单和校准输出目录加入忽略规则；测试使用 pytest 临时目录，不接触真实人脸数据。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_calibration_list.py -q

### 提交

    git commit -m "feat: add RKNN calibration list builder"

## Task 5：实现 ONNX 与 RKNN 一致性校验

### 文件

- 新增 `tools/rknn/metrics.py`
- 新增 `tools/rknn/validate_outputs.py`
- 新增 `tests/unit/test_rknn_metrics.py`
- 新增 `tests/unit/test_rknn_validation.py`

### 步骤

1. 将余弦相似度、有限值检查、输出形状检查、框匹配和关键点误差计算放入不依赖 RKNN 的 `metrics.py`。
2. MobileFaceNet 校验先展平为 512 维并 L2 归一化，再计算 ONNX/RKNN 余弦相似度；非量化阈值固定为 0.99。
3. RetinaFace 根据最后一维 `4/2/10` 映射三个输出，不依赖 RKNN 返回顺序。
4. RetinaFace 分别比较输出形状、检测数量、匹配框坐标和五点关键点误差；误差容差由命令行参数控制并写入报告。
5. 验证入口延迟导入 ONNX Runtime 与 RKNN Toolkit，在虚拟机内分别执行两个运行时；单元测试注入假运行器。
6. 输出 JSON 报告，包含模型哈希、样本、阈值、逐样本指标、总体通过状态和失败原因。
7. 测试完全一致、轻微容差、维度错误、NaN、低余弦相似度和 RetinaFace 输出乱序。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_rknn_metrics.py tests/unit/test_rknn_validation.py -q

### 提交

    git commit -m "feat: validate ONNX and RKNN outputs"

## Task 6：编写 Ubuntu 18.04 虚拟机操作手册

### 文件

- 新增 `docs/rknn-conversion-setup.md`
- 新增 `tools/rknn/check_environment.py`
- 新增 `tests/unit/test_rknn_environment_check.py`
- 更新 `models/README.md`
- 更新 `README.md`

### 步骤

1. 记录 VMware 资源配置：4 核、6 GB 内存、40 GB 动态磁盘、NAT。
2. 记录 Ubuntu 18.04.6、Anaconda、Python 3.6.9 和课程提供的 `rknn_toolkit-1.7.1-cp36-cp36m-linux_x86_64.whl` 安装顺序。
3. 明确 RKNN Toolkit 的 wheel 和课程安装包不提交 Git；文档统一使用 `/opt/rknn/packages` 保存本地安装包，不依赖用户主目录。
4. 环境检查脚本打印 Python、平台、RKNN Toolkit、NumPy 与关键动态库状态，版本不符时返回非零退出码。
5. 文档给出从 Git 克隆、放置 ONNX、生成清单、非量化转换、INT8 转换和运行验证的完整命令。
6. 环境通过后要求创建 VMware 快照，并记录快照名称建议。
7. 更新项目 README 的阶段状态和最终上板路线，但在真实转换完成前仍标注“工具已准备，模型待验证”。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_rknn_environment_check.py -q
    rg -n "Python 3.6.9|RKNN Toolkit 1.7.1|6 GB|非量化|INT8" docs/rknn-conversion-setup.md README.md

### 提交

    git commit -m "docs: add RKNN conversion environment guide"

## Task 7：运行主机侧完整回归

### 文件

- 更新 `task_plan.md`
- 更新 `findings.md`
- 更新 `progress.md`

### 步骤

1. 运行所有 RKNN 工具单元测试。
2. 运行项目完整 pytest，确认新工具未影响桌面应用。
3. 运行 compileall、Git diff 检查和 CodeGraph 同步。
4. 检查 Git 暂存区，确认没有 `.onnx`、`.rknn`、校准图片、真实清单或日志。

### 验证

    .\.test-venv\Scripts\python.exe -m pytest -q
    .\.test-venv\Scripts\python.exe -m compileall face_recognition_app tools tests
    git diff --check
    codegraph.cmd sync D:\rk3399pro
    codegraph.cmd status D:\rk3399pro

### 提交

    git commit -m "test: verify RKNN conversion tooling"

## Task 8：在 VMware 中生成并验收真实模型

### 前置条件

- 用户已创建并启动 Ubuntu 18.04.6 x86_64 虚拟机。
- RKNN Toolkit 1.7.1 wheel 已从课程资料复制进虚拟机。
- 两个 ONNX 模型和本地校准、验证图片可在虚拟机内访问。

### 步骤

1. 按操作手册建立 `onnx2rknn` Python 3.6.9 环境并运行环境检查。
2. 先生成两个非量化 RKNN 模型，保存转换摘要和完整日志。
3. 使用固定验证图片运行 ONNX/RKNN 对比；MobileFaceNet 非量化余弦相似度必须不低于 0.99。
4. 非量化模型通过后生成两个 INT8 模型并重复验证。
5. 记录四个模型的大小、转换状态和一致性指标；分别选择检测器与特征模型的默认产物。
6. 若出现不支持算子，保存完整日志并定位具体节点；RetinaFace 无法解决时按设计评估替换轻量检测器，不改动其余接口。
7. 将验证指标和模型 SHA-256 更新到本地模型清单；只提交脱敏后的结果摘要，不提交模型或人脸图片。
8. 环境与模型通过后关闭虚拟机并创建可恢复快照。

### 虚拟机内验收命令

    python tools/rknn/check_environment.py
    python tools/rknn/convert_retinaface.py --onnx models/retinaface_mobilenet025.onnx --output models/retinaface_mobilenet025.rknn
    python tools/rknn/convert_mobilefacenet.py --onnx models/mobilefacenet.onnx --output models/mobilefacenet.rknn
    python tools/rknn/validate_outputs.py --model retinaface --onnx models/retinaface_mobilenet025.onnx --rknn models/retinaface_mobilenet025.rknn --images /data/rknn-validation/retinaface
    python tools/rknn/validate_outputs.py --model mobilefacenet --onnx models/mobilefacenet.onnx --rknn models/mobilefacenet.rknn --images /data/rknn-validation/mobilefacenet

预期：两个非量化模型均通过；INT8 是否作为默认模型由实测指标决定。

### 提交

    git commit -m "docs: record RKNN conversion validation"

## 完成条件

- 仓库内转换工具可在无 RKNN Toolkit 的 Windows 测试环境通过单元测试。
- Ubuntu 18.04 虚拟机能够重复生成两个非量化 RKNN 模型。
- RetinaFace 检测输出通过框与关键点一致性检查。
- MobileFaceNet 非量化特征余弦相似度不低于 0.99。
- INT8 评估结果可追溯，未通过精度标准的模型不会成为默认产物。
- Git 历史不包含模型二进制、真实人脸图片、校准清单、转换日志或虚拟机文件。
