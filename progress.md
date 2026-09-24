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

## Test Results

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

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 5，摄像头、工作线程与桌面界面 |
| Where am I going? | 摄像头与 UI、RKNN 转换、板端集成、阈值校准和交付 |
| What's the goal? | 在 RK3399Pro 上交付支持 50 人以内和成员管理的本地人脸识别应用 |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已完成需求、设计、Git 基线、纯算法核心，以及两个真实 ONNX 模型的导出和一致性验证 |
