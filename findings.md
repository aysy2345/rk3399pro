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
- 当前工作站的 `python` 来自 MSYS2，虚拟环境使用 `.venv/bin` 而不是 Windows 常见的 `.venv/Scripts`。
- 系统同时安装了标准 Windows Python 3.11，可用 `C:\Users\27162\AppData\Local\Programs\Python\Python311\python.exe` 创建本机测试环境并安装预编译 wheel。
- Phase 3 已形成 22 项单元测试，覆盖配置、成员模型、人脸库原子保存与回滚、匹配、登记聚合、质量检查、多帧稳定和五点对齐。
- CodeGraph 同步后索引了 18 个新增源码文件和 155 个节点。
- `biubug6/Pytorch_Retinaface` 是 MIT 许可，仓库自带 `convert_to_onnx.py`，适合作为 RetinaFace MobileNet0.25 的结构与 ONNX 导出基线；预训练权重仍需单独核验下载来源和授权。
- `Xiaoccer/MobileFaceNet_Pytorch` 未声明 GitHub 可识别的许可证，不能直接作为可再分发模型来源。
- `TreB1eN/InsightFace_Pytorch` 使用 MIT 许可，但当前仓库树中没有现成 MobileFaceNet ONNX 文件，需要继续核验其网络结构、权重来源及导出方式。
- `rockchip-linux/rknn-toolkit` 使用 BSD-3-Clause，官方仓库包含 RKNN Toolkit/Toolkit Lite 1.7.5 文档及 ONNX 转换示例，可作为 RK3399Pro 转换和部署的主依据，但未发现官方 RetinaFace/MobileFaceNet 示例。
- `rockchip-linux/rknn_model_zoo` 当前 GitHub API 返回 404，不能据此作为 RK3399Pro 的官方模型来源。
- RetinaFace 候选实现的推理约定已核实：输入为 BGR `NCHW` float32，逐通道减 `(104,117,123)`；MobileNet0.25 配置采用三层特征步长 `8/16/32`、每层两个 anchor、variance `(0.1,0.2)`，输出为位置、二分类置信度和五点关键点，需要在主机端解码与 NMS。
- `biubug6/Pytorch_Retinaface` 的现有 ONNX 导出脚本把三个网络输出错误地只命名为一个 `output0`，接入时应改为显式的 `loc/conf/landms` 三输出导出或按输出形状映射。
- `wujiyang/Face_Pytorch` 和 `yeyupiaoling/Pytorch-MobileFaceNet` 均为 Apache-2.0；后者在仓库内直接提供 MobileFaceNet `.pth` 权重，当前是更容易复现的识别模型候选，但仍要检查权重的数据来源说明和导出后的数值一致性。
- `ZhaoJ9014/face.evoLVe` 为 MIT，但仓库树中没有 MobileFaceNet 预训练权重；`JDAI-CV/FaceX-Zoo` 带 ONNX 转换器但许可证被 GitHub 标记为 `NOASSERTION`，优先级较低。
- 已固定 `yeyupiaoling/Pytorch-MobileFaceNet` 候选 revision 为 `080aab37323b2736122aa49b6a4b634549714dde`。该版本默认权重是 TorchScript，网络默认输出 512 维；预测代码直接使用 OpenCV BGR 顺序，并执行 `(pixel-127.5)/127.5`，不能套用常见的 RGB、除以 128 或 128 维假设。
- 已从上述固定 revision 下载 `save_model/mobilefacenet.pth`，文件大小 5,051,471 字节，SHA-256 为 `330787c19d95745f7c882c2883a30e0ca75661952953680bffc95405084f9064`；归档内容确认是 TorchScript，而非普通 state_dict。
- 使用 PyTorch 2.11.0 将该 TorchScript 导出为 ONNX opset 11，ONNX SHA-256 为 `be53e4bc6a2af3bef44254ef4f3ef9bd4d9f55ce25abc01359816c68eb25de8a`。固定随机输入下，ONNX Runtime 对 TorchScript 的最大绝对误差为 `1.6298145055770874e-08`，余弦相似度为 `0.9999999999974729`。
- `foamliu/MobileFaceNet-PyTorch` 当前 HEAD 固定为 `2c720d6875488e94f4d4eb870936cb05613b74d5`，仓库顶层许可证为 Apache-2.0，且仓库树检索曾显示包含 `retinaface/weights/mobilenet0.25_Final.pth`；其 RetinaFace 子目录没有独立 README，因此仍需核对代码与权重是否完整继承自 `biubug6/Pytorch_Retinaface`。
- 已浅克隆并检查该固定 commit：RetinaFace 权重大小 1,789,735 字节，SHA-256 为 `2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220`。网络输出确为 `loc/conf/landmarks` 三张量，测试阶段对分类输出执行 softmax；预处理与 anchor 配置和当前 ONNX 适配器一致。
- 候选源码的 `cfg_mnet.pretrain=True` 会在构造网络时额外读取一个未随子目录提供的 ImageNet backbone 权重。导出时应复制配置并改为 `pretrain=False`，再加载完整的 `mobilenet0.25_Final.pth`，避免无关文件依赖。
- RetinaFace 已使用 PyTorch 2.11.0、torchvision 0.26.0 导出为 ONNX opset 11，ONNX SHA-256 为 `34274686d588a0c0936ad2b851513c5a0d97cd69a3244129525d405509bed0b3`。边框、分数、关键点相对 PyTorch 的最大绝对误差分别为 `1.0550e-05`、`1.1921e-07`、`1.6492e-05`，三者余弦相似度均高于 `0.999999999998`。
- Phase 5 规划核对发现：`FaceStore` 已完整提供 add、rename、replace_embedding 和 delete，可由成员服务直接编排，无需重写存储层。
- 当前 `AppConfig` 尚无 backend、目标帧率、清晰度门槛和采样间隔字段；Phase 5 必须先扩展配置和示例，再建立启动工厂。
- 当前质量检查只覆盖单人、人脸尺寸和清晰度，尚未根据五点关键点判断正视、左转和右转；自动采样前需要补充姿态估计与姿态配额。
- `IdentityStabilizer` 需要 track_id，但当前没有跟踪器；Phase 5 可使用基于检测框 IoU 的轻量关联生成短时轨迹编号，不引入重型跟踪依赖。
- 当前测试依赖没有 PyQt5 或 pytest-qt；Phase 5 的离屏 Qt 测试需要补充主机测试依赖，同时继续通过 Fake Camera 和 Fake 推理隔离真实硬件。

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
| Phase 5 主界面采用上下分区布局 | 用户在视觉伴侣中选择 B：视频在上，操作与识别结果在下 |
| 主窗口默认最大化但保留桌面窗口控制 | 便于板端调试和日常最小化、关闭，不采用强制全屏终端模式 |
| 添加成员时暂停识别并独占摄像头 | 避免两个流程争用摄像头；登记窗口关闭后自动恢复原识别状态 |
| 添加成员采用三步向导 | 用户选择 A：基本信息 → 采集样本 → 确认并保存，减少一次性界面的操作负担 |
| Fake 与 ONNX 后端可切换 | 用户选择 C：自动测试使用 Fake，实际运行默认使用 ONNX，兼顾可测性与真实运行 |
| 成员管理采用可搜索表格 | 用户选择 A：按姓名或编号查找，并在行内提供编辑、重新采集和删除操作 |
| 程序启动后手动开始识别 | 用户选择 B：主界面先进入待机状态，由用户点击开始，便于先处理成员与设备错误 |
| 登记时自动采集有效样本 | 用户选择 A：按质量门槛、采集间隔和姿态引导自动收集 15 个样本，避免逐张点击 |
| Phase 5 采用分层加单工作线程 | 用户选择 A：Qt 主线程负责 UI，最新帧工作线程负责采集与推理，服务和后端保持接口化 |
| Phase 5 架构与数据流获确认 | 主窗口状态机、摄像头线程所有权、最新帧策略、后端工厂和成员仓储边界通过用户确认 |
| Phase 5 界面与成员流程获确认 | 上下分区主界面、三步登记、自动采样、可搜索成员表格及原子重采/删除规则通过用户确认 |
| Phase 5 配置、异常与验收获确认 | Fake/ONNX 配置、可恢复错误、线程释放、自动化测试和板端人工验收边界通过用户确认 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 当前仓库只有课程资料和设计文档，没有可复用源码 | 按绿地项目规划包结构和测试边界 |
| `writing-plans` 技能不存在 | 使用已经安装的 planning-with-files 维护根目录计划文件 |
| 原始资料包含多个 GB 级镜像和安装包 | 使用 `.gitignore` 排除，不放入普通 Git 历史 |
| RetinaFace 的 RKNN 1.7.1 算子兼容性尚未验证 | 保持检测器接口可替换，并将转换验证设为独立阶段 |
| 工作站默认 MSYS2 Python 不适合直接安装 PyPI Windows wheel | 使用标准 Windows Python 3.11 的 `.test-venv` 运行主机测试 |
| GitHub 搜索首次返回 `unexpected EOF` | 改用 GitHub API 直接检查候选仓库元数据和文件树 |
| 固定 revision 的 GitHub Contents API 一次连接超时 | 改用 raw.githubusercontent.com 固定 commit 地址下载，并在本地计算 SHA-256 |
| `foamliu/MobileFaceNet-PyTorch` 的 `retinaface/README.md` 返回 404 | 不假定存在子目录说明，改为逐项核对源码文件与权重 |
| Windows 的 `bash.exe` 指向未配置发行版的 WSL，无法启动视觉伴侣脚本 | 直接使用脚本内部的 Node 服务，并以隐藏后台进程运行 |
| 首个视觉伴侣后台进程退出并清理了 state 目录 | 新建独立会话并重新启动 Node 服务，不复用已失效的 server-info |

## Resources

- `docs/superpowers/specs/2026-09-24-rk3399pro-face-recognition-design.md`
- `docs/superpowers/plans/2026-09-24-rk3399pro-face-recognition-implementation.md`
- `边缘人工智能应用开发(IPC Camera 口罩识别)/02 第二模块：目标检测的实现/02 目标检测模型编译的准备/`
- `边缘人工智能应用开发(IPC Camera 口罩识别)/03 第三模块：嵌入式设备实现目标检测/02 目标检测模型的转化/`
- `边缘人工智能应用开发(IPC Camera 口罩识别)/03 第三模块：嵌入式设备实现目标检测/03 嵌入式开发板环境配置/`
- `边缘人工智能应用开发(USB Camera 水果识别)/01 第一模块：目标检测的准备/05 摄像头采集图片/`
- https://github.com/biubug6/Pytorch_Retinaface
- https://github.com/TreB1eN/InsightFace_Pytorch
- https://github.com/rockchip-linux/rknn-toolkit
- https://github.com/wujiyang/Face_Pytorch
- https://github.com/yeyupiaoling/Pytorch-MobileFaceNet

## Visual/Browser Findings

- Phase 5 使用本地视觉伴侣对比三种主界面布局；用户选择 B“上下分区”，即横向视频区域位于上方，开始/停止、添加成员、成员管理和识别状态位于下方。
- 添加成员流程对比后，用户选择 A“三步向导”：基本信息、采集样本、确认保存。
- 成员管理布局对比后，用户选择 A“表格管理”，优先保证 50 人以内成员的浏览和定位效率。
