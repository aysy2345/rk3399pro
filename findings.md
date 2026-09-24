# Findings & Decisions

## Requirements

- 在 RK3399Pro 上运行离线 1:N 人脸识别。
- 使用 USB UVC 摄像头。
- 本地人脸库不超过 50 人。
- 已登记成员显示姓名和相似度，未登记人员显示陌生人。
- 提供 PyQt5 桌面界面。
- 支持添加新成员、修改姓名、重新采集和删除成员。
- 新成员保存后立即参与识别，无需重启。
- 第一版不包含活体检测、考勤报表、云端同步和网页管理。

## Research Findings

- 原始课程资料以目标检测为主，没有完整的人脸特征识别流程。
- 资料中的 PC 基础环境采用 Windows、Anaconda、PyCharm 和 Python 3.6。
- 训练资料采用 PyTorch 1.5.1、Torchvision 0.6.1、CUDA 10.1 和 cuDNN 8.0.5。
- 模型转换资料将流程拆为 Windows Python 3.7.13 的 ONNX 环境，以及 Ubuntu 18.04 x86_64、Python 3.6.9、RKNN Toolkit 1.7.1 的 RKNN 环境。
- RK3399Pro 板端资料采用 Ubuntu 18.04、Python 3.7、RKNN Toolkit Lite 1.7.1、OpenCV 4.5.4.60 和 NumPy 1.16.3。
- 板端必须在安装 OpenCV 后重新固定 NumPy 1.16.3，以避免 RKNN 1.7.1 兼容问题。
- USB 摄像头可通过 OpenCV `VideoCapture` 读取，比 IPC 路线更适合第一版。
- 50 人以内可以直接在内存中对特征矩阵计算余弦相似度，无需向量数据库。
- 新成员登记不需要重新训练模型，只需要采集多张人脸并生成平均特征模板。
- 仓库原始目录约 9.9 GB，已通过 `.gitignore` 排除 ISO、IMG、EXE、大型压缩包、数据集和 CodeGraph 索引。
- GitHub 仓库为 `https://github.com/aysy2345/rk3399pro`，当前 `main` 基线提交为 `f695481`。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 首选 RetinaFace MobileNet0.25 | 可同时提供人脸框与五点关键点，便于人脸对齐；转换兼容性需实测 |
| 使用 MobileFaceNet | 轻量，适合嵌入式特征提取和小规模本地人脸库 |
| 使用余弦相似度和可配置阈值 | 不同模型与现场条件的相似度分布不同，阈值必须通过验证集校准 |
| 连续 5 帧中至少 3 帧一致才确认 | 降低单帧误识别和界面姓名跳动 |
| 登记 10 至 20 个有效样本 | 覆盖轻微姿态和表情变化，又不增加过多登记时间 |
| JSON 保存成员信息，NPY 保存特征矩阵 | 便于人工检查元数据并进行高效矩阵计算 |
| 使用临时文件加原子替换保存 | 降低程序中断导致成员信息与特征不同步的风险 |
| 工作线程只保留最新帧 | 推理慢于摄像头时避免视频帧积压和延迟不断增加 |
| 先提供 Fake 和 ONNX 后端，再接 RKNN | 可在没有开发板时完成大部分功能和自动测试 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 当前仓库只有课程资料和设计文档，没有可复用源码 | 按绿地项目规划包结构和测试边界 |
| `writing-plans` 技能不存在 | 使用已经安装的 planning-with-files 维护根目录计划文件 |
| 原始资料包含多个 GB 级镜像和安装包 | 使用 `.gitignore` 排除，不放入普通 Git 历史 |
| RetinaFace 的 RKNN 1.7.1 算子兼容性尚未验证 | 保持检测器接口可替换，并将转换验证设为独立阶段 |

## Resources

- `docs/superpowers/specs/2026-09-24-rk3399pro-face-recognition-design.md`
- `docs/superpowers/plans/2026-09-24-rk3399pro-face-recognition-implementation.md`
- `边缘人工智能应用开发(IPC Camera 口罩识别)/02 第二模块：目标检测的实现/02 目标检测模型编译的准备/`
- `边缘人工智能应用开发(IPC Camera 口罩识别)/03 第三模块：嵌入式设备实现目标检测/02 目标检测模型的转化/`
- `边缘人工智能应用开发(IPC Camera 口罩识别)/03 第三模块：嵌入式设备实现目标检测/03 嵌入式开发板环境配置/`
- `边缘人工智能应用开发(USB Camera 水果识别)/01 第一模块：目标检测的准备/05 摄像头采集图片/`

## Visual/Browser Findings

- 本轮未使用网页或视觉检查；环境信息来自仓库内 Word 和 PPT 文本内容。

