# RK3399Pro 本地人脸识别

本项目面向 RK3399Pro，计划使用 USB UVC 摄像头实现 50 人以内的离线 1:N 人脸识别，并通过 PyQt5 提供添加成员、改名、重新采集和删除成员等本地管理功能。

识别方案采用 RetinaFace MobileNet0.25 检测人脸和五点关键点，经过人脸对齐后由 MobileFaceNet 提取特征，最后使用本地特征库进行余弦相似度匹配。

## 当前状态

项目当前已完成 **Phase 5：摄像头、工作线程与桌面界面** 的主机侧实现。

> 当前版本可在 Windows/Linux 主机使用 USB 摄像头运行 Fake 或 ONNX 桌面应用。RKNN 转换与 RK3399Pro NPU 后端仍属于后续阶段，因此暂时不能直接在板端使用 NPU 运行。

| 阶段 | 状态 |
|---|---|
| 需求与资料梳理 | 已完成 |
| 方案与实施规划 | 已完成 |
| 项目骨架与纯算法核心 | 已完成 |
| ONNX 推理与模型验证 | 已完成 |
| 摄像头、工作线程与桌面界面 | 已完成（主机侧） |
| RKNN 模型转换 | 待实现 |
| RK3399Pro 板端集成 | 待实现 |
| 阈值校准与现场验收 | 待实现 |

详细进度见 [task_plan.md](task_plan.md) 和 [progress.md](progress.md)。

## 已完成功能

- 配置加载、字段校验和运行路径检查。
- 成员信息与特征矩阵的本地存储、原子更新和故障回滚。
- 新增成员、改名、替换特征和删除成员所需的数据层能力。
- 五点人脸对齐、人脸质量检查和登记样本聚合。
- L2 特征归一化、余弦相似度匹配和陌生人判断。
- 连续 5 帧至少 3 帧一致的身份稳定策略。
- 可用于无模型测试的 Fake 检测器和特征提取器。
- RetinaFace MobileNet0.25 和 MobileFaceNet ONNX Runtime 后端。
- Torch/PyTorch、ONNX 和 ONNX Runtime 输出一致性验证。
- USB UVC 摄像头封装、最新帧工作线程和可恢复错误处理。
- PyQt5 主窗口、三步添加成员向导和可搜索成员管理表格。
- 成员改名、重新采集、删除及识别库即时刷新。
- Fake/ONNX 后端工厂、命令行启动入口和 Fake 端到端集成测试。

当前自动化测试覆盖配置、成员、人脸库、核心算法、推理层、摄像头、工作线程、桌面界面和 Fake 端到端流程。

## 识别流程

```text
USB UVC 摄像头
  -> RetinaFace 检测人脸框和五点关键点
  -> 五点仿射对齐为 112x112 人脸
  -> MobileFaceNet 提取 512 维归一化特征
  -> 与 50 人以内本地特征库计算余弦相似度
  -> 多帧稳定确认
  -> 显示成员姓名、相似度或“陌生人”
```

添加成员不需要重新训练模型。程序会采集多张有效人脸，过滤重复或异常样本，生成平均特征模板并立即写入本地人脸库。

## 模型

| 模型 | 输入 | 输出 | 当前状态 |
|---|---|---|---|
| RetinaFace MobileNet0.25 | `1x3x640x640` BGR float32，减去 `[104,117,123]` | 16800 个边框、二分类分数和五点关键点 | ONNX 已导出并验证 |
| MobileFaceNet | `1x3x112x112` BGR float32，`(pixel-127.5)/127.5` | 512 维特征 | ONNX 已导出并验证 |

模型来源、固定 revision、许可证、输入输出和 SHA-256 记录在 [模型清单](models/model-manifest.example.json)。模型二进制文件、登记照片和人脸特征库不会提交到普通 Git 历史。

模型目录说明见 [models/README.md](models/README.md)。

## 主机安装与运行

### 1. 自动化测试环境

推荐使用标准 Windows Python 3.11 创建隔离测试环境：

```powershell
py -3.11 -m venv .test-venv
.\.test-venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m pytest -q
```

该环境用于自动测试，不要求 USB 摄像头或 ONNX 模型。

### 2. 实际运行环境

实际运行 ONNX 桌面应用推荐使用 Python 3.7，并安装运行依赖：

    py -3.7 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-dev.txt
    Copy-Item configs/app.example.json configs/app.json

如果只想验证界面、摄像头和线程，不加载模型，可使用 Fake 后端：

    python -m face_recognition_app.main --config configs/app.json --backend fake

Fake 后端不需要模型文件，但仍会打开配置中的 USB 摄像头。默认摄像头不可用时，可覆盖编号：

    python -m face_recognition_app.main --config configs/app.json --backend fake --camera-index 1

准备好两个 ONNX 模型后，使用实际识别后端：

    python -m face_recognition_app.main --config configs/app.json --backend onnx

程序启动后处于待机状态，需要点击“开始识别”。添加成员时会暂停识别并自动采集 15 个有效样本，保存成功后无需重启即可参与识别。

### 3. USB 摄像头检查

可先用 OpenCV 检查摄像头编号 0：

    python -c "import cv2; c=cv2.VideoCapture(0); print('opened=', c.isOpened()); c.release()"

输出 opened=True 后再启动应用。如果为 False，请关闭占用摄像头的软件，并尝试 --camera-index 1。

### 4. 重新导出模型

需要重新导出并验证 ONNX 时：

```powershell
python -m pip install -r requirements-model-export.txt
python tools/models/export_mobilefacenet_onnx.py
python tools/models/export_retinaface_onnx.py
```

导出脚本需要先按 [模型目录说明](models/README.md) 准备本地权重和候选源码；这些文件由 `.gitignore` 排除。

## 常见问题

- “configuration file not found”：先复制 configs/app.example.json 为 configs/app.json。
- “model file not found”：确认两个 ONNX 文件位于配置指定位置，或先使用 --backend fake。
- “unable to open camera”：检查摄像头连接、系统权限、占用程序和摄像头编号。
- 画面正常但一直显示陌生人：先添加成员；实际阈值仍需使用现场验证集校准。
- RKNN 后端提示尚未提供：这是预期行为，RKNN 转换和板端后端将在 Phase 6、Phase 7 完成。

## 最终上板路线

完整部署分为三个环境：

| 环境 | 作用 | 主要产物或任务 |
|---|---|---|
| Windows 开发机 | 开发、单元测试和 ONNX 验证 | `.onnx` 模型和应用源码 |
| Ubuntu 18.04 x86_64 | 使用 RKNN Toolkit 1.7.1 转换模型 | `.rknn` 模型 |
| RK3399Pro Ubuntu 18.04 ARM64 | 使用 RKNN Toolkit Lite 1.7.1 运行 | USB 摄像头、NPU 推理和 PyQt5 界面 |

板端计划使用 Python 3.7、OpenCV 4.5.4.60、NumPy 1.16.3 和 PyQt5。最终还需要完成：

1. 两个 ONNX 模型到 RKNN 的转换及输出一致性验证。
2. RKNN Toolkit Lite 推理后端。
3. 板端 PyQt5、USB 摄像头和 NPU 的联合运行。
4. 板端性能测试、阈值校准和两小时稳定运行验证。

板端依赖基线见 [requirements-rk3399pro.txt](requirements-rk3399pro.txt)。

## 配置

复制示例配置后，根据模型和数据目录修改路径：

```powershell
Copy-Item configs/app.example.json configs/app.json
```

示例配置默认使用 ONNX 后端并指向：

```text
models/retinaface_mobilenet025.onnx
models/mobilefacenet.onnx
```

检测阈值和识别阈值只是初始值，最终必须使用独立的现场验证集校准，不能直接把示例值视为最终参数。

## 项目文档

- [系统设计](docs/superpowers/specs/2026-09-24-rk3399pro-face-recognition-design.md)
- [详细实施计划](docs/superpowers/plans/2026-09-24-rk3399pro-face-recognition-implementation.md)
- [README 刷新设计](docs/superpowers/specs/2026-09-24-readme-refresh-design.md)
- [Phase 5 桌面应用设计](docs/superpowers/specs/2026-09-24-phase5-camera-ui-design.md)
- [Phase 5 实施计划](docs/superpowers/plans/2026-09-24-phase5-camera-ui-implementation.md)
- [当前任务计划](task_plan.md)
- [研究结论](findings.md)
- [进度记录](progress.md)
