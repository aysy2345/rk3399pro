# RK3399Pro 本地人脸识别实施计划

## 1 实施原则

本项目从零搭建。先完成可在普通电脑上测试的业务核心和界面，再接入 ONNX 验证后端，最后接入 RK3399Pro 的 RKNN Toolkit Lite 1.7.1。所有硬件、模型和界面代码通过明确接口隔离，避免无法连接开发板时阻塞其余开发。

每个阶段都必须通过对应测试后再进入下一阶段。预训练模型文件、登记照片和人脸特征库不得提交到 Git。

## 2 目标目录结构

```text
rk3399pro/
├─ face_recognition_app/
│  ├─ main.py
│  ├─ app/
│  │  ├─ config.py
│  │  └─ logging_config.py
│  ├─ core/
│  │  ├─ alignment.py
│  │  ├─ enrollment.py
│  │  ├─ matcher.py
│  │  ├─ quality.py
│  │  └─ stabilizer.py
│  ├─ domain/
│  │  ├─ member.py
│  │  └─ recognition.py
│  ├─ hardware/
│  │  └─ camera.py
│  ├─ inference/
│  │  ├─ interfaces.py
│  │  ├─ onnx_backend.py
│  │  ├─ rknn_backend.py
│  │  └─ fake_backend.py
│  ├─ storage/
│  │  └─ face_store.py
│  ├─ ui/
│  │  ├─ main_window.py
│  │  ├─ enrollment_dialog.py
│  │  └─ member_manager_dialog.py
│  └─ workers/
│     └─ recognition_worker.py
├─ configs/
│  └─ app.example.json
├─ models/
│  └─ README.md
├─ tools/
│  └─ model_conversion/
│     ├─ verify_onnx.py
│     ├─ convert_retinaface.py
│     └─ convert_mobilefacenet.py
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ fixtures/
├─ requirements-dev.txt
├─ requirements-rk3399pro.txt
└─ README.md
```

## 3 阶段一 项目骨架与配置

### 任务

1. 创建上述 Python 包结构和入口文件。
2. 定义 `configs/app.example.json`，包含摄像头、模型、人脸质量、匹配、连续帧确认及数据路径配置。
3. 实现配置加载、类型检查、默认值和错误提示。
4. 更新 `.gitignore`，排除 `models/*.onnx`、`models/*.rknn`、`face_data/`、登记照片、测试输出和运行日志。
5. 编写开发环境和板端环境依赖文件。

### 验收

- 缺少配置文件、字段类型错误或模型路径不存在时给出明确错误。
- 默认配置可被程序加载。
- `pytest` 能发现测试目录。

## 4 阶段二 领域模型与本地人脸库

### 任务

1. 在 `domain/member.py` 定义成员编号、姓名、登记时间和状态。
2. 在 `storage/face_store.py` 实现成员元数据与特征矩阵的加载、校验和保存。
3. 使用临时文件加原子替换同时更新 `members.json` 和 `embeddings.npy`。
4. 实现新增、改名、替换特征和删除成员。
5. 启动时验证成员数量、特征行数、特征维度、索引唯一性和数值有效性。

### 测试

- 空人脸库初始化。
- 新增成员后重新加载一致。
- 重复人员编号被拒绝。
- 同名不同编号允许保存。
- 改名、替换特征和删除操作保持索引一致。
- 模拟写入失败时原有人脸库不受损坏。
- JSON 与 NPY 数量不一致时停止加载并报告错误。

## 5 阶段三 人脸识别核心算法

### 任务

1. 在 `core/alignment.py` 实现基于五点关键点的人脸仿射对齐。
2. 在 `core/quality.py` 实现最小人脸尺寸、模糊度、姿态范围和单人登记检查。
3. 在 `core/matcher.py` 实现特征 L2 归一化、矩阵化余弦相似度和陌生人阈值判断。
4. 在 `core/stabilizer.py` 实现“连续 5 帧至少 3 帧一致”的身份稳定策略。
5. 在 `core/enrollment.py` 实现样本去重、异常特征过滤和平均模板生成。

### 测试

- 相同向量相似度接近 1。
- 正交或低相似度向量返回陌生人。
- 空人脸库始终返回陌生人。
- 多成员情况下返回最高相似度成员。
- 阈值边界行为确定且可配置。
- 连续帧不足时不确认身份，满足窗口规则后确认。
- 异常特征不会污染最终成员模板。

## 6 阶段四 推理接口与电脑端验证

### 任务

1. 在 `inference/interfaces.py` 定义检测器和特征提取器协议。
2. 实现 `fake_backend.py`，为界面和业务测试提供确定性结果。
3. 实现 `onnx_backend.py`，分别加载 RetinaFace MobileNet0.25 与 MobileFaceNet ONNX。
4. 固化两个模型的颜色通道、输入尺寸、归一化、输出顺序和后处理参数。
5. 建立固定测试图片，记录 ONNX 输出形状和归一化特征。

### 模型准入检查

- 记录模型来源、许可证、SHA-256 和输入输出节点。
- 确认模型允许项目使用和分发；模型文件默认不进入 Git。
- 使用正面、轻微转头、多人和无人脸样本验证检测结果。
- 同一人的不同照片相似度应显著高于不同人员。

### 验收

- Fake 后端完整通过集成测试。
- ONNX 后端能输出人脸框、五点关键点和归一化特征。
- 预处理与后处理具有固定测试，防止转换前后口径漂移。

## 7 阶段五 摄像头与识别工作线程

### 任务

1. 在 `hardware/camera.py` 封装 OpenCV `VideoCapture`。
2. 支持摄像头编号、分辨率、打开重试和断开恢复。
3. 在 `workers/recognition_worker.py` 中实现只保留最新帧的处理循环。
4. 将检测、对齐、特征提取、匹配和稳定输出串联。
5. 定义线程安全的帧结果、状态和错误信号。

### 测试

- 使用录制视频或假摄像头测试，不依赖真实 USB 设备。
- 推理速度低于摄像头帧率时不会形成帧队列。
- 摄像头读取失败会有限重试并上报错误。
- 停止识别后线程和摄像头资源正确释放。

## 8 阶段六 PyQt5 桌面界面

### 任务

1. 实现主窗口的视频画面、状态栏、人数和识别结果显示。
2. 实现开始识别与停止识别状态切换。
3. 实现添加成员对话框，包括信息填写、采集引导、有效样本进度和保存确认。
4. 实现成员管理对话框，包括列表、改名、重新采集和二次确认删除。
5. 将工作线程异常转换为可理解的界面提示。
6. 确保关闭窗口时停止线程并释放摄像头。

### 测试

- 使用 Fake 后端运行界面冒烟测试。
- 重复编号、空姓名和无效采集不能保存。
- 添加成功后成员人数和识别库立即刷新。
- 删除或重新采集后界面与磁盘数据保持一致。
- 推理期间窗口可移动、按钮可响应且不会冻结。

## 9 阶段七 RKNN 模型转换

### 任务

1. 在 Ubuntu 18.04 x86_64 环境安装 RKNN Toolkit 1.7.1。
2. 为两个 ONNX 模型分别编写可重复执行的转换脚本。
3. 明确目标平台、输入尺寸、预处理参数、量化策略和校准数据目录。
4. 先尝试非量化转换并比较 ONNX 与 RKNN 输出。
5. 再评估 INT8 量化对速度、模型大小和相似度分布的影响。
6. 如果 RetinaFace 存在不支持算子，仅替换检测器，不改变对齐、特征和存储接口。

### 验收

- 转换脚本可从干净环境重复生成 RKNN 文件。
- 固定图片上的检测框和关键点与 ONNX 结果在容许误差内一致。
- MobileFaceNet 的 RKNN 与 ONNX 特征余弦相似度达到预设一致性要求。
- 转换日志中无未处理的不支持算子错误。

## 10 阶段八 RK3399Pro 集成

### 任务

1. 安装 NPU 驱动、Python 3.7 虚拟环境和 RKNN Toolkit Lite 1.7.1。
2. 按资料要求安装 OpenCV 4.5.4.60 后重新固定 NumPy 1.16.3。
3. 实现 `rknn_backend.py`，加载两个 RKNN 模型并复用已有后处理。
4. 接入 USB UVC 摄像头并确认设备编号。
5. 测量检测、特征提取和完整帧处理耗时。
6. 验证 PyQt5 在板端桌面或 VNC 会话中正常显示。

### 验收

- 板端可加载两个模型并持续读取摄像头。
- 新成员可以现场登记并立即被识别。
- 50 人规模下匹配耗时不成为主要瓶颈。
- 程序停止后 NPU、线程和摄像头资源正确释放。
- 连续运行两小时无界面冻结或持续内存增长。

## 11 阶段九 阈值校准与现场验收

### 任务

1. 至少选取若干测试人员，分别采集登记集和独立验证集。
2. 验证集覆盖正面、轻微转头、表情变化、眼镜和正常室内光线变化。
3. 统计同人相似度与异人相似度分布。
4. 选择兼顾误识别和拒绝识别的阈值，写入配置而非代码。
5. 测试陌生人、多人、逆光、模糊和不同距离。
6. 记录摄像头型号、安装距离、光照和最终阈值。

### 最终验收

- 50 人以内可稳定增删改查。
- 登记照片与测试照片分离。
- 未登记人员不会轻易被错误识别。
- 已登记人员在规定距离和光线下能稳定确认。
- 程序重启后成员库完整可用。
- 摄像头断开、模型缺失和人脸库损坏均有明确提示。

## 12 建议提交顺序

1. `chore: scaffold face recognition application`
2. `feat: add atomic local face store`
3. `feat: add matching enrollment and stabilization core`
4. `feat: add inference interfaces and fake backend`
5. `feat: add ONNX face inference backend`
6. `feat: add camera and recognition worker`
7. `feat: add PyQt member management interface`
8. `feat: add RKNN model conversion tools`
9. `feat: add RK3399Pro inference backend`
10. `test: add device integration and threshold calibration`

## 13 第一轮实施范围

第一轮只完成阶段一至阶段三：项目骨架、配置、本地人脸库和纯算法核心。这些内容不依赖模型、摄像头或开发板，能够先建立稳定的测试基础。完成并通过测试后，再进入 ONNX 模型接入。
