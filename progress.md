# Progress Log

## Session: 2026-09-24

### Phase 1: 需求与资料梳理

- **Status:** complete
- Actions taken:
  - 只检查 Word 和 PPT 资料，未细读原始代码。
  - 整理 Windows、训练、ONNX、RKNN 转换和 RK3399Pro 板端环境。
  - 确认目标为 50 人以内、USB 摄像头、本地 1:N 人脸识别。
- Files created/modified:
  - `findings.md`（创建）

### Phase 2: 方案与实施规划

- **Status:** complete
- Actions taken:
  - 确认 RetinaFace MobileNet0.25 加 MobileFaceNet 方案。
  - 确认 PyQt5 添加成员和成员管理界面。
  - 编写完整系统设计和详细实施计划。
  - 初始化 Git 仓库，排除大型安装包与数据集后提交并推送至 GitHub。
  - 切换到 planning-with-files 持久化计划管理。
- Files created/modified:
  - `.gitignore`
  - `docs/superpowers/specs/2026-09-24-rk3399pro-face-recognition-design.md`
  - `docs/superpowers/plans/2026-09-24-rk3399pro-face-recognition-implementation.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 3: 项目骨架与纯算法核心

- **Status:** complete
- Actions taken:
  - 定义本阶段范围为配置、人脸库、对齐、质量、匹配、稳定器、登记聚合及单元测试。
  - 创建配置、领域模型、人脸库与纯算法核心。
  - 创建标准 Windows Python 3.11 测试环境和 20 项单元测试。
  - 补充五点人脸对齐测试，最终 22 项测试全部通过。
  - 完成 Python 语法编译检查并同步 CodeGraph。
- Files created/modified:
  - `face_recognition_app/app/config.py`
  - `face_recognition_app/domain/`
  - `face_recognition_app/core/`
  - `face_recognition_app/storage/face_store.py`
  - `configs/app.example.json`
  - `requirements-dev.txt`
  - `requirements-rk3399pro.txt`
  - `requirements-test.txt`
  - `tests/unit/`

### Phase 4: ONNX 推理与模型验证

- **Status:** complete
- Actions taken:
  - 核验 RetinaFace、MobileFaceNet 和 RKNN Toolkit 候选来源及许可证。
  - 定义检测器、特征提取器和检测结果的后端无关接口。
  - 实现可重复的 Fake 检测与特征后端。
  - 实现 RetinaFace MobileNet0.25 的 ONNX 预处理、anchor 解码、五点解码和 NMS。
  - 实现 MobileFaceNet 的 ONNX BGR 预处理和 L2 特征归一化。
  - 新增模型来源与输入输出清单，并写入真实权重/ONNX SHA-256 和跨框架一致性结果。
  - 下载固定 revision 的 MobileFaceNet TorchScript，导出 ONNX opset 11，并完成 TorchScript/ONNX 数值一致性验证。
  - 固定 RetinaFace 源码与权重 revision，导出三输出 ONNX opset 11，并完成 PyTorch/ONNX 数值一致性验证。
- Files created/modified:
  - `face_recognition_app/inference/`
  - `tests/unit/test_inference.py`
  - `models/model-manifest.example.json`
  - `models/README.md`
  - `requirements-dev.txt`
  - `requirements-model-export.txt`
  - `tools/models/export_mobilefacenet_onnx.py`
  - `tools/models/export_retinaface_onnx.py`

### 文档维护：GitHub README 刷新

- **Status:** complete
- Actions taken:
  - 用户确认方案 B：GitHub 首页展示真实进度、已完成功能、模型、验证命令和上板路线。
  - 编写并提交 README 刷新设计说明。
  - 重写根目录 README，明确 Phase 1 至 Phase 4 已完成、Phase 5 及后续待实现。
  - 验证 README 全部相对链接存在，并运行完整单元测试。
  - 提交根 README、计划与进度更新并推送到 GitHub `main`。
- Files created/modified:
  - `README.md`
  - `docs/superpowers/specs/2026-09-24-readme-refresh-design.md`
  - `task_plan.md`
  - `progress.md`

## Test Results

### Phase 5 设计确认

- **Status:** complete
- Actions taken:
  - 通过视觉原型确认上下分区主界面、三步登记向导和表格式成员管理。
  - 确认默认最大化窗口、手动开始识别、登记独占摄像头和自动采集 15 个样本。
  - 确认 Fake/ONNX 双后端、分层单工作线程、错误处理和验收边界。
  - 编写 Phase 5 摄像头、工作线程与桌面界面设计文档。
- Files created/modified:
  - `docs/superpowers/specs/2026-09-24-phase5-camera-ui-design.md`
  - `findings.md`
  - `task_plan.md`
  - `progress.md`

### Phase 5 实施计划

- **Status:** complete
- Actions taken:
  - 核对现有 FaceStore、配置、质量检查、稳定器以及 Fake/ONNX 后端接口。
  - 识别出配置字段、姿态判断、短时人脸轨迹和 Qt 测试依赖缺口。
  - 将 Phase 5 拆分为八个按 TDD 执行的任务，并固定文件、测试命令、验收条件和提交顺序。
- Files created/modified:
  - `docs/superpowers/plans/2026-09-24-phase5-camera-ui-implementation.md`
  - `findings.md`
  - `task_plan.md`
  - `progress.md`

### Phase 5 Task 1：运行配置与测试依赖

- **Status:** complete
- Actions taken:
  - 先新增 backend、推理间隔、目标帧率、清晰度和登记采样间隔的失败测试。
  - 扩展不可变 AppConfig 数据类和字段校验；Fake 后端不再要求本地模型文件存在。
  - 将示例运行后端和模型路径切换为 ONNX。
  - 在测试虚拟环境安装锁定版本的 PyQt5 与 pytest-qt。
- Files created/modified:
  - `face_recognition_app/app/config.py`
  - `configs/app.example.json`
  - `requirements-test.txt`
  - `requirements-rk3399pro.txt`
  - `tests/unit/test_config.py`

### Phase 5 Task 2：USB 摄像头抽象

- **Status:** complete
- Actions taken:
  - 先用 StubCapture 编写打开、失败重试、读取错误、幂等释放和上下文管理测试。
  - 定义 Camera 抽象、CameraError、CameraReadError 和 OpenCVCamera。
  - 支持摄像头编号、分辨率、目标帧率、有限打开重试和 BGR 帧复制。
  - 所有自动测试均通过 capture_factory 注入，不访问真实 USB 设备。
- Files created/modified:
  - `face_recognition_app/hardware/__init__.py`
  - `face_recognition_app/hardware/camera.py`
  - `tests/unit/test_camera.py`

### Phase 5 Task 3：识别与登记流水线

- **Status:** complete
- Actions taken:
  - 定义 AppState、FaceOverlay、FrameResult 和 WorkerErrorInfo 运行值对象。
  - 实现基于检测框 IoU 的轻量短时人脸轨迹，不增加外部跟踪依赖。
  - 串联检测、五点对齐、特征提取、余弦匹配和连续帧稳定。
  - 使用五点关键点的归一化鼻尖偏移实现 front/left/right 粗粒度姿态分类。
  - 实现自动登记会话的姿态计划、质量门槛、采样间隔、重复样本过滤和模板聚合。
- Files created/modified:
  - `face_recognition_app/domain/runtime.py`
  - `face_recognition_app/core/tracking.py`
  - `face_recognition_app/core/pipeline.py`
  - `face_recognition_app/core/enrollment_session.py`
  - `face_recognition_app/core/quality.py`
  - `tests/unit/test_tracking.py`
  - `tests/unit/test_pipeline.py`
  - `tests/unit/test_enrollment_session.py`

### Phase 5 Task 4：Qt 工作线程与应用控制器

- **Status:** complete
- Actions taken:
  - 实现 RecognitionWorker，以单循环读取当前帧并按推理间隔处理，不建立帧队列。
  - 使用 threading.Event 和锁接收停止、登记模式和人脸库刷新命令，避免阻塞 Qt 事件队列。
  - 实现 WorkerThreadHost，负责 QObject 移入 QThread、信号转发、停止等待和资源回收。
  - 实现 AppController，统一管理待机、识别、登记和错误状态，并支持可恢复错误重试。
  - 使用真实 QThread 和 Fake Camera 验证启动、停止、线程结束和摄像头释放。
- Files created/modified:
  - `face_recognition_app/workers/__init__.py`
  - `face_recognition_app/workers/recognition_worker.py`
  - `face_recognition_app/app/controller.py`
  - `tests/unit/test_worker.py`
  - `tests/unit/test_controller.py`

### Phase 5 Task 5：PyQt5 主窗口

- **Status:** complete
- Actions taken:
  - 实现上下分区主窗口：上方视频画面，下方状态、识别结果、成员数和操作按钮。
  - 实现 BGR 到 QImage 的深复制、等比例缩放、人脸框、姓名和相似度绘制。
  - 根据待机、识别、登记和错误状态统一更新按钮与提示。
  - 支持开始、停止、添加成员、成员管理、错误重试和关闭资源释放。
  - 使用 Qt 离屏测试和 Windows 平台截图完成逻辑与视觉检查。
- Files created/modified:
  - `face_recognition_app/ui/__init__.py`
  - `face_recognition_app/ui/video_widget.py`
  - `face_recognition_app/ui/main_window.py`
  - `tests/ui/__init__.py`
- `tests/ui/test_main_window.py`

### Phase 5 Task 6：三步成员登记向导

- **Status:** complete
- Actions taken:
  - 为成员信息校验、重复编号、同名提示、保存通知与失败路径编写单元测试。
  - 为三步向导的输入校验、采集门槛、最终确认、取消和保存失败编写离屏界面测试。
  - TDD 红灯确认两个目标模块尚不存在，符合预期。
  - 实现 MemberService，统一执行中文业务校验、同名提示、原子新增和成功后的单次快照通知。
  - 实现三步 EnrollmentWizard；采集完成前不可确认，最终保存前不写成员库，取消和失败均保持无半成品。
- Files created/modified:
  - `face_recognition_app/app/member_service.py`
  - `face_recognition_app/ui/enrollment_wizard.py`
  - `face_recognition_app/ui/__init__.py`
  - `tests/unit/test_member_service.py`
  - `tests/ui/test_enrollment_wizard.py`

### Phase 5 Task 7：表格式成员管理

- **Status:** complete
- Actions taken:
  - 恢复已批准的 Phase 5 设计与实施边界。
  - 刷新 CodeGraph，确认 Task 6 提交后的索引最新。
  - 为成员列表、改名、删除、重新采集与搜索过滤编写失败测试。
  - TDD 红灯确认成员管理对话框尚不存在，符合预期。
  - 扩展 MemberService，统一提供列表、改名、特征替换、删除和成功后的单次快照通知。
  - 实现可搜索表格、行内姓名编辑、删除确认和重新采集入口。
  - 扩展 EnrollmentWizard 的重新采集模式，取消或失败时保留旧特征，最终确认后才替换。
- Files created/modified:
  - face_recognition_app/app/member_service.py
  - face_recognition_app/ui/enrollment_wizard.py
  - face_recognition_app/ui/member_manager_dialog.py
  - face_recognition_app/ui/__init__.py
  - tests/unit/test_member_service.py
  - tests/ui/test_enrollment_wizard.py
  - tests/ui/test_member_manager_dialog.py

### Phase 5 Task 8：后端工厂、启动入口与完整集成

- **Status:** complete
- Actions taken:
  - 恢复已批准的 Phase 5 启动流程、后端切换和集成测试边界。
  - 刷新 CodeGraph，确认 Task 7 提交后的索引最新。
  - 为后端工厂、命令行覆盖和 Fake 摄像头端到端流程编写失败测试。
  - TDD 红灯确认 bootstrap.py 与 main.py 尚不存在，符合预期。
  - 实现 BackendFactory、依赖组装、桌面协调器和命令行启动入口。
  - 使用注入的 Fake 摄像头与 Fake 后端跑通待机、识别、登记、刷新、管理和关闭。
  - 更新 README 的主机安装、Fake/ONNX 启动、USB 摄像头检查和常见问题。
  - 用户实际运行发现 Fake 后端无法采集；诊断确认 Fake 检测器按设计不返回真实人脸。
  - 进一步发现 Windows 下若先创建 QApplication，再首次导入 ONNX Runtime，会发生 DLL 初始化失败；在 QApplication 前预加载即可正常创建两个 ONNX 后端。
  - 先新增回归测试，确认预加载函数与 Fake 采集提示在实现前失败。
  - 生产 Fake 模式现在会明确提示真实采集必须改用 ONNX，成员管理中的重新采集按钮同步禁用。
  - ONNX 启动现在在创建 QApplication 前预加载 onnxruntime，避免 Windows DLL 初始化顺序冲突。
- Files created/modified:
  - face_recognition_app/app/bootstrap.py
  - face_recognition_app/main.py
  - tests/unit/test_bootstrap.py
  - tests/unit/test_main.py
  - tests/integration/__init__.py
  - tests/integration/test_fake_ui_flow.py
  - README.md
  - findings.md
  - progress.md
  - task_plan.md

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Git 本地与远程哈希核对 | `HEAD` 与 `origin/main` | 两者一致 | 均为 `f6954817c371381e4cbd078d488c7f8f8dad07d5` | 通过 |
| Git 工作区状态 | `git status --short` | 无未提交文件 | 在新增计划文件前为空 | 通过 |
| CodeGraph 状态 | 当前仓库 | 索引可用 | 索引存在，当前无源码节点 | 通过 |
| Phase 3 首轮单元测试 | 20 项 | 全部通过 | 19 通过，1 个测试断言类型错误 | 待复测 |
| Phase 3 完整单元测试 | 22 项 | 全部通过 | 22 项通过，耗时 0.63 秒 | 通过 |
| Python 语法编译 | `face_recognition_app` | 无语法错误 | `compileall` 退出码 0 | 通过 |
| CodeGraph 同步 | 新增源码 | 索引新增源码 | 18 个文件、155 个节点 | 通过 |
| Phase 4 推理层单元测试 | 28 项 | 全部通过 | 28 项通过，耗时 0.59 秒 | 通过 |
| Phase 4 Python 语法编译 | `face_recognition_app` 和 `tests` | 无语法错误 | `compileall` 退出码 0 | 通过 |
| MobileFaceNet ONNX 一致性 | 固定随机输入 | 最大绝对误差小于 1e-4，余弦不低于 0.99999 | 最大误差 1.63e-08，余弦 0.999999999997 | 通过 |
| 真实 MobileFaceNet 适配器 | 112x112 全零 BGR 图 | 512 维单位向量 | shape=(512,), norm=0.99999994 | 通过 |
| 导出环境依赖检查 | `.test-venv` | 无破损依赖 | `pip check` 无错误 | 通过 |
| RetinaFace ONNX 一致性 | 固定随机输入 | 三输出最大绝对误差小于 1e-4，余弦不低于 0.99999 | 最大误差不超过 1.65e-05，余弦均高于 0.999999999998 | 通过 |
| 真实 RetinaFace 适配器 | 640x480 全零 BGR 图 | 可完成推理和后处理 | 完成，阈值 0.8 下 0 个检测结果 | 通过 |
| Phase 4 CodeGraph 同步 | 新增推理与导出源码 | 索引无待处理变化 | 25 个文件、254 个节点，索引最新 | 通过 |
| README 相对链接 | 根 README | 所有本地链接目标存在 | 全部存在 | 通过 |
| README 更新后单元测试 | 28 项 | 全部通过 | 28 项通过，耗时 0.75 秒 | 通过 |
| Phase 5 设计一致性自审 | 用户确认的架构、界面、登记、配置与异常决策 | 正式文档全部覆盖 | 关键决策全部可定位 | 通过 |
| Phase 5 文档差异检查 | 当前待提交改动 | `git diff --check` 无错误 | 退出码 0，仅有 Windows 换行提示 | 通过 |
| Phase 5 Task 1 TDD 红灯 | 新增配置测试 | 新字段实现前测试失败 | 10 失败、4 通过，原因均为待实现字段 | 通过 |
| Phase 5 配置测试 | `tests/unit/test_config.py` | 全部通过 | 14 项通过，耗时 0.14 秒 | 通过 |
| Phase 5 Task 1 完整回归 | 全部测试 | 全部通过 | 37 项通过，耗时 0.84 秒 | 通过 |
| Phase 5 Qt 测试依赖 | 测试虚拟环境 | 无依赖冲突 | PyQt5/pytest-qt 可导入，pip check 无错误 | 通过 |
| Phase 5 Task 2 TDD 红灯 | 摄像头测试 | 实现前导入失败 | Camera 模块不存在，符合预期 | 通过 |
| Phase 5 摄像头测试 | `tests/unit/test_camera.py` | 全部通过 | 5 项通过，耗时 0.17 秒 | 通过 |
| Phase 5 Task 2 完整回归 | 全部测试 | 全部通过 | 42 项通过，耗时 0.60 秒 | 通过 |
| Phase 5 Task 3 TDD 红灯 | 三个新测试文件 | 实现前导入失败 | tracking、pipeline、enrollment_session 均不存在，符合预期 | 通过 |
| Phase 5 识别与登记流水线 | Task 3 定向测试 | 全部通过 | 7 项通过，耗时 0.25 秒 | 通过 |
| Phase 5 Task 3 完整回归 | 全部测试 | 全部通过 | 49 项通过，耗时 0.58 秒 | 通过 |
| Phase 5 Task 4 TDD 红灯 | worker/controller 测试 | 实现前导入失败 | 两个模块不存在，符合预期 | 通过 |
| Phase 5 Task 4 首轮实现 | 线程与状态测试 | 全部通过 | 8 通过、1 失败，发现首次启动多发取消登记命令 | 待修复 |
| Phase 5 Qt 工作线程与控制器 | Task 4 定向测试 | 全部通过 | 10 项通过，耗时 0.23 秒 | 通过 |
| Phase 5 Task 4 完整回归 | 全部离屏测试 | 全部通过 | 59 项通过，耗时 0.67 秒 | 通过 |
| Phase 5 Task 5 TDD 红灯 | 主窗口离屏测试 | 实现前导入失败 | ui 包不存在，符合预期 | 通过 |
| Phase 5 主窗口测试 | `tests/ui/test_main_window.py` | 全部通过 | 6 项通过，耗时 0.27 秒 | 通过 |
| Phase 5 主窗口视觉检查 | 1100×760 Windows Qt 截图 | 上下分区、中文和四个主按钮正常 | 布局与已批准原型一致 | 通过 |
| Phase 5 Task 5 完整回归 | 全部离屏测试 | 全部通过 | 65 项通过，耗时 0.71 秒 | 通过 |
| Phase 5 Task 6 TDD 红灯 | 成员服务与登记向导测试 | 实现前导入失败 | 两个目标模块不存在，符合预期 | 通过 |
| Phase 5 Task 6 定向测试 | 成员服务与登记向导 | 全部通过 | 13 项通过，耗时 0.46 秒 | 通过 |
| Phase 5 Task 6 完整回归 | 全部离屏测试 | 全部通过 | 78 项通过，耗时 1.02 秒 | 通过 |
| Phase 5 Task 6 Python 语法编译 | `face_recognition_app` 和 `tests` | 无语法错误 | `compileall` 退出码 0 | 通过 |
| Phase 5 Task 6 CodeGraph 同步 | 新增服务、向导与测试 | 索引无待处理变化 | 49 个文件、612 个节点，索引最新 | 通过 |
| Phase 5 Task 7 TDD 红灯 | 成员服务、管理对话框与重新采集测试 | 实现前导入失败 | member_manager_dialog 模块不存在，符合预期 | 通过 |
| Phase 5 Task 7 定向测试 | 成员服务、管理表格与重新采集 | 全部通过 | 20 项通过，耗时 0.73 秒 | 通过 |
| Phase 5 Task 7 完整回归 | 全部离屏测试 | 全部通过 | 85 项通过，耗时 1.21 秒 | 通过 |
| Phase 5 Task 7 Python 语法编译 | face_recognition_app 和 tests | 无语法错误 | compileall 退出码 0 | 通过 |
| Phase 5 Task 7 CodeGraph 同步 | 新增成员管理源码与测试 | 索引无待处理变化 | 51 个文件、673 个节点，索引最新 | 通过 |
| Phase 5 Task 8 TDD 红灯 | 后端工厂、命令行与 Fake UI 集成测试 | 实现前导入失败 | bootstrap.py 与 main.py 不存在，符合预期 | 通过 |
| Phase 5 Task 8 定向测试 | 后端工厂、命令行与 Fake UI 集成测试 | 全部通过 | 8 项通过，耗时 0.54 秒 | 通过 |
| Phase 5 Task 8 完整回归 | 全部离屏测试 | 全部通过且无残留 QThread | 93 项通过，耗时 1.31 秒 | 通过 |
| Phase 5 Task 8 Python 语法编译 | face_recognition_app 和 tests | 无语法错误 | compileall 退出码 0 | 通过 |
| Phase 5 Task 8 CLI 检查 | python -m face_recognition_app.main --help | 参数完整可解析 | config、backend、camera-index 均正常 | 通过 |
| Phase 5 Task 8 README 链接 | README 本地相对链接 | 全部目标存在 | 无缺失链接 | 通过 |
| Phase 5 Task 8 CodeGraph 同步 | 启动层、入口与集成测试 | 索引无待处理变化 | 57 个文件、760 个节点，索引最新 | 通过 |
| Fake 采集与 ONNX 预加载 TDD 红灯 | 新增回归测试 | 修复前失败 | preload 函数不存在，符合预期 | 通过 |
| Fake 采集提示首轮回归 | 提示内容断言 | 应明确包含 ONNX | 统一为“ONNX 后端（--backend onnx）” | 通过 |
| Fake 采集与 ONNX 预加载定向回归 | 后端、入口与 Fake 集成测试 | 全部通过 | 11 项通过，耗时 0.37 秒 | 通过 |
| Windows ONNX/PyQt5 实际加载 | 先预加载 ONNX Runtime，再创建 QApplication 与两个模型会话 | 两个后端均可创建 | OnnxRetinaFaceDetector 与 OnnxMobileFaceNetEmbedder 加载成功 | 通过 |
| 采集问题修复完整回归 | 全部离屏测试 | 全部通过 | 96 项通过，耗时 1.09 秒 | 通过 |
| 采集问题修复语法编译 | face_recognition_app 和 tests | 无语法错误 | compileall 退出码 0 | 通过 |

### Phase 5.1：成员采集质量与文字可读性修复

- **Status:** complete
- Actions taken:
  - 根据用户实拍截图确认 ONNX 检测正常，采集被清晰度门槛持续拒绝。
  - 截图登记人脸区域估算清晰度为 58.3，当前配置门槛为 100.0。
  - 确认姿态、原因、进度数字和主界面结果因嵌套控件未显式着色而回落为黑色。
  - 完成设计、自检、用户确认及详细实施计划。
  - 按 TDD 补充配置、质量提示和高对比度样式回归测试。
  - 将示例配置和本地运行配置的清晰度门槛调整为 40.0。
  - 模糊原因现在显示实际清晰度与门槛。
  - 为登记动作、拒绝原因、进度条和主界面识别结果增加显式高对比度样式。
  - 生成两张 Windows Qt 截图并完成视觉检查，文字和背景对比度清晰。
  - CodeGraph 同步后索引为 57 个 Python 文件、771 个节点和 1598 条边，状态最新。
  - git diff --check 通过，本地忽略配置已确认使用 40.0。
  - 修复提交 `cae7cc0` 已推送到 GitHub main，远程哈希与本地一致。

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Phase 5.1 TDD 红灯 | 4 项新增定向测试 | 修复前全部失败 | 4 项失败，分别命中阈值、提示、登记样式和主界面样式 | 通过 |
| Phase 5.1 定向回归 | 4 项新增定向测试 | 全部通过 | 4 项通过，耗时 0.30 秒 | 通过 |
| Phase 5.1 完整回归 | 全部离屏测试 | 全部通过 | 100 项通过，耗时 1.06 秒 | 通过 |
| Phase 5.1 语法编译 | face_recognition_app 和 tests | 无语法错误 | compileall 退出码 0 | 通过 |
| Phase 5.1 Windows 视觉检查 | 登记页和主界面截图 | 关键文字高对比度 | 白色动作、黄色原因、白色进度数字、白色识别结果均清晰 | 通过 |

### 识别结果未更新诊断

- **Status:** in_progress
- 用户截图确认状态为“识别中”、实时预览正常、成员数为 1，但结果仍是初始文字。
- 检查 worker、pipeline、controller 和 main window 信号链路；空检测结果同样应更新为“未检测到人脸”。
- 真实 ONNX 截图诊断：检测 0.093 秒、嵌入 0.017 秒、检测置信度 0.996、模板相似度 0.9314，排除模型慢或匹配阈值问题。
- 已定位 WorkerThreadHost 的 pending enrollment 在“从待机登记并保存”后未清空，导致下一次开始识别仍运行登记分支。
- **Status:** root_cause_confirmed

### Phase 5.2：登记状态残留修复

- **Status:** complete
- Actions taken:
  - 完成根因设计、自检和用户确认。
  - 创建测试驱动实施计划，限定为 AppController 的最小状态清理。
  - 新增控制器与完整 Fake UI 回归测试；修复前控制器 cancel_calls 为 0，完整流程无法收到识别结果。
  - AppController 现在在登记退出时无条件先调用 host.cancel_enrollment，再恢复识别或停止线程。
  - 定向测试进一步发现旧线程 finished 的延迟回调可能清空新线程引用；WorkerThreadHost 现在按信号发送线程校验代际后再清理。
  - 改为在连接 finished 时用闭包显式传递 QThread；完整复现测试通过且进程退出码为 0。
  - controller、worker、Fake UI 共 11 项定向测试通过，耗时 0.40 秒。
  - 完整测试 100 项通过，耗时 1.17 秒；compileall 和 git diff --check 通过。
  - CodeGraph 同步完成：57 个 Python 文件、771 个节点、1561 条边，索引最新。
  - 修复提交 `28c7643` 已推送到 GitHub main，本地与远程哈希一致。

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Phase 5.2 控制器红灯 | 待机登记后取消/完成 | 必须清除宿主会话 | 修复前 cancel_calls=0 | 通过 |
| Phase 5.2 完整流程红灯 | 登记保存后立即开始识别 | 收到识别结果 | 修复前无法更新结果 | 通过 |
| Phase 5.2 Fake UI 复测 | 登记保存后立即开始识别 | 显示未检测到人脸且正常停止 | 通过，退出码 0 | 通过 |
| Phase 5.2 定向回归 | controller、worker、Fake UI | 全部通过 | 11 项通过，耗时 0.40 秒 | 通过 |
| Phase 5.2 完整回归 | 全部测试 | 全部通过 | 100 项通过，耗时 1.17 秒 | 通过 |
| Phase 5.2 语法与索引 | compileall、CodeGraph、diff check | 全部正常 | 均通过，索引最新 | 通过 |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-09-24 | `writing-plans` 技能路径不存在 | 1 | 使用 planning-with-files 作为持久化计划替代方案 |
| 2026-09-24 | Git 初始化后命令执行器报告 `setup refresh had errors` | 1 | 经用户批准后使用沙箱外 Git 命令 |
| 2026-09-24 | 首次 Git 推送未立即建立远程跟踪 | 1 | 改用 HTTP/1.1、提高 postBuffer，重试后核对本地与远程哈希 |
| 2026-09-24 | Phase 3 普通环境检查再次报告 `setup refresh had errors` | 1 | 不重复普通执行，改用沙箱外只读检查 |
| 2026-09-24 | 本机 Python 3.11 缺少 pytest、NumPy 和 OpenCV | 1 | 先创建依赖清单，随后在项目虚拟环境安装测试依赖 |
| 2026-09-24 | PowerShell 未直接执行 `.\.venv\Scripts\python.exe` | 1 | 改用 PowerShell 调用运算符 `&`，保留已创建的虚拟环境 |
| 2026-09-24 | 加调用运算符后仍找不到 `.venv\Scripts\python.exe` | 2 | 确认解释器为 MSYS2 Python，虚拟环境使用 Unix 风格 `.venv\bin\python.exe` |
| 2026-09-24 | MSYS2 Python 安装 NumPy 时只取得源码包且未完成安装 | 1 | 发现系统另有标准 Windows Python 3.11，改建 `.test-venv` 使用官方 wheel |
| 2026-09-24 | 20 项测试中 1 项因 `pytest.approx` 不支持嵌套列表失败 | 1 | 改用 `numpy.testing.assert_allclose`；人脸库回滚内容本身正确 |
| 2026-09-24 | GitHub 搜索请求返回 `unexpected EOF` | 1 | 改用 GitHub API 直接读取候选仓库元数据和文件树 |
| 2026-09-24 | apply_patch 包装脚本不存在 `btoa` 和 `TextEncoder` | 2 | 改用纯 JavaScript UTF-8 与 Base64 编码函数，继续通过 apply_patch 模式编辑 |
| 2026-09-24 | GitHub Contents API 查询固定权重时连接超时 | 1 | 改用固定 commit 的 raw 地址下载并本地计算 SHA-256 |
| 2026-09-24 | README 刷新时自动审批额度耗尽且沙箱初始化失败 | 2 | 未绕过审批；等待额度恢复后继续使用 apply_patch |
| 2026-09-24 | README 进度补丁包含多余空 hunk | 1 | 删除空 hunk 后重新应用补丁 |
| 2026-09-24 | Phase 5 设计补丁中的 Markdown 围栏与 JavaScript 模板字符串冲突 | 1 | 改用缩进代码块后重新应用补丁 |
| 2026-09-24 | Phase 5 测试环境缺少 PyQt5 | 1 | 按 requirements-test.txt 锁定版本安装 PyQt5 5.15.7 与 pytest-qt 4.4.0 |
| 2026-09-24 | AppController 首次开始识别时多发一次取消登记命令 | 1 | 仅在 ERROR 恢复路径清理登记状态，定向测试恢复全绿 |
| 2026-09-24 | 主窗口长补丁的 JavaScript 包装字符串出现语法错误 | 1 | 拆分补丁并直接调用 apply_patch，未产生半成品源码 |
| 2026-09-24 | view_image 因 Windows 沙箱刷新失败无法读取截图 | 2 | 经 PowerShell 读取 PNG Base64 后以内联图片完成检查 |
| 2026-09-24 | 直接 apply_patch 更新 Task 5 进度时沙箱刷新失败 | 1 | 改用已验证的 UTF-8 Base64 apply-patch 包装命令 |
| 2026-09-24 | Phase 5 Task 6 首次刷新 CodeGraph 时执行器初始化失败 | 1 | 按既定安全流程重试沙箱外只读索引检查，确认索引最新 |
| 2026-09-24 | Task 7 进度补丁先后发生模板解析和跨文件锚点不匹配 | 3 | 分开更新计划文件并改用实际存在的稳定锚点，未产生源码改动 |
| 2026-09-25 | Task 7 首次 GitHub 推送因自动审批额度到期未执行 | 1 | 未绕过审批；额度恢复后继续执行原推送并核对哈希 |
| 2026-09-25 | Task 8 首次读取时假定 ONNX 检测器和识别器分属两个文件 | 1 | 使用 rg 定位到统一的 inference/onnx_backend.py，并读取真实构造接口 |
| 2026-09-25 | Task 8 README 长补丁包含未转义的 Markdown 代码围栏，导致 JavaScript 解析失败 | 1 | 改用缩进代码块并拆除反引号后重新应用，未产生文件改动 |
| 2026-09-25 | Fake 模式登记始终为 0/15 | 1 | 确认 Fake 检测器不处理真实人脸；改为启动时明确提示使用 ONNX 后端 |
| 2026-09-25 | QApplication 初始化后首次加载 ONNX Runtime 出现 DLL 初始化失败 | 1 | 验证在创建 QApplication 前预加载 onnxruntime 可正常创建 RetinaFace 与 MobileFaceNet 会话 |
| 2026-09-25 | Fake 采集提示测试因 ONNX 大小写不一致失败 | 1 | 统一用户提示为“ONNX 后端（--backend onnx）”后复测 |
| 2026-09-25 | 仅在 QApplication 前预加载仍出现 ONNX Runtime DLL 初始化失败 | 2 | 定位到 main.py 顶层已导入 PyQt/Bootstrap；改为 ONNX 成功预加载后再延迟导入全部桌面组件 |
| 2026-09-25 | 修复验证结果补丁遗漏跨文件 Update File 标记 | 1 | 根据 rg 定位后拆分为正确的多文件补丁，未产生文件改动 |
| 2026-09-25 | 补丁封装脚本使用运行器不支持的 TextEncoder/btoa | 2 | 改用直接补丁文本，未产生文件改动 |
| 2026-09-25 | 原生 apply_patch 遇到 Windows sandbox helper 错误 | 1 | 改用已知可用的 Codex apply-patch 入口，未产生文件改动 |
| 2026-09-25 | 多文件补丁的进度日志锚点不匹配 | 1 | 读取文件尾部后按真实上下文拆分应用；首个 .gitignore 修改已生效 |
| 2026-09-25 | 高对比度定向复测中进度条选择器断言过于宽泛 | 1 | 实现使用更精确的 QProgressBar#captureProgress::chunk，修正测试断言后复测 |
| 2026-09-25 | view_image 读取高对比度 QA 截图时 Windows 沙箱刷新失败 | 2 | 截图已生成且文件大小正常，改用 PowerShell 读取 PNG 数据进行视觉检查 |
| 2026-09-25 | 识别诊断命令可能完整输出 members.json 中的人脸特征 | 1 | 安全机制拒绝且未读取数据；改为仅统计成员数量、矩阵形状和向量范数 |
| 2026-09-25 | 两项 TDD 红灯测试合并执行时仅输出 F，集成测试未给出摘要 | 1 | 分开以 verbose 模式运行；控制器失败原因明确，集成流程卡在预期结果等待，随后实施最小修复 |
| 2026-09-25 | 修复登记残留后 Fake UI 立即重启触发 Qt 进程退出码 -1073740791 | 2 | sender() 在延迟槽中未可靠标识旧线程；改为连接信号时通过闭包显式传递对应 QThread |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 5，摄像头、工作线程与桌面界面 |
| Where am I going? | 摄像头与 UI、RKNN 转换、板端集成、阈值校准和交付 |
| What's the goal? | 在 RK3399Pro 上交付支持 50 人以内和成员管理的本地人脸识别应用 |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已完成需求、设计、Git 基线、纯算法核心，以及两个真实 ONNX 模型的导出和一致性验证 |
