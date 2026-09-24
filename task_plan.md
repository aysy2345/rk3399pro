# Task Plan: RK3399Pro 本地人脸识别系统

## Goal

在 RK3399Pro 上交付一个使用 USB 摄像头、支持 50 人以内本地人脸库、带 PyQt5 成员管理界面的离线 1:N 人脸识别应用。

## Current Phase

Phase 5.2: 登记状态残留修复

## Phases

### Phase 1: 需求与资料梳理

- [x] 确认目标为 1:N 本地人脸识别
- [x] 确认人脸库规模为 50 人以内
- [x] 确认使用 USB UVC 摄像头
- [x] 整理 Word 和 PPT 中的 Windows、Ubuntu、RKNN 与板端环境
- [x] 记录研究结论到 findings.md
- **Status:** complete

### Phase 2: 方案与实施规划

- [x] 确认 RetinaFace MobileNet0.25 加 MobileFaceNet 的两阶段方案
- [x] 确认本地特征库和余弦相似度匹配
- [x] 确认 PyQt5 桌面成员管理界面
- [x] 编写系统设计文档
- [x] 编写详细实施计划
- [x] 初始化并推送 Git 仓库
- **Status:** complete

### Phase 3: 项目骨架与纯算法核心

- [x] 创建 Python 包、配置、测试和文档结构
- [x] 实现配置加载与校验
- [x] 实现成员领域模型和原子人脸库
- [x] 实现人脸对齐、质量检查、特征匹配和结果稳定器
- [x] 实现登记特征的去重、异常过滤和模板聚合
- [x] 为以上模块编写单元测试
- **Status:** complete

### Phase 4: ONNX 推理与模型验证

- [x] 定义检测器和特征提取器接口
- [x] 实现 Fake 后端供无模型测试
- [x] 记录模型来源、许可证、校验值和输入输出
- [x] 接入 RetinaFace MobileNet0.25 ONNX
- [x] 接入 MobileFaceNet ONNX（真实权重已导出并通过 TorchScript/ONNX 一致性验证）
- [x] 固化预处理、后处理和输出一致性测试
- **Status:** complete

### Phase 5: 摄像头、工作线程与桌面界面

- [x] 确认 Phase 5 交互、线程、异常和验收设计
- [x] 扩展 Phase 5 运行配置与 Qt 测试依赖
- [x] 封装 USB 摄像头读取与恢复
- [x] 实现纯 Python 人脸轨迹、识别流水线与自动登记会话
- [x] 实现只保留最新帧的识别工作线程
- [x] 实现 PyQt5 主窗口
- [x] 实现添加新成员流程
- [x] 实现成员管理流程
- [x] 使用 Fake 后端完成界面与线程集成测试
- **Status:** complete

### Phase 5.1: 成员采集质量与文字可读性修复

- [x] 分析截图并确认 0/15 由清晰度门槛 100.0 持续拒绝导致
- [x] 完成修复设计并取得用户确认
- [x] 编写详细实施计划
- [x] 先补充清晰度诊断和高对比度样式测试
- [x] 将默认与本地清晰度门槛调整为 40.0
- [x] 实现实时清晰度提示与显式高对比度样式
- [x] 完成自动化、语法和视觉验证
- [x] 完成 CodeGraph 与提交前验证
- [x] 提交并推送修复
- **Status:** complete

### Phase 5.2: 登记状态残留修复

- [x] 通过用户截图和真实 ONNX 推理排除模型、阈值与性能问题
- [x] 定位 WorkerThreadHost pending enrollment 未清空
- [x] 完成设计并取得用户确认
- [x] 编写详细实施计划
- [x] 编写失败回归测试
- [x] 实现控制器统一清理登记状态
- [x] 完成定向、完整、语法和 CodeGraph 验证
- [ ] 提交并推送修复
- **Status:** in_progress

### Phase 6: RKNN 模型转换

- [ ] 在 Ubuntu 18.04 x86_64 配置 RKNN Toolkit 1.7.1
- [ ] 编写 RetinaFace 转换脚本
- [ ] 编写 MobileFaceNet 转换脚本
- [ ] 比较 ONNX 与 RKNN 输出
- [ ] 评估非量化和 INT8 量化结果
- **Status:** pending

### Phase 7: RK3399Pro 板端集成

- [ ] 配置板端 Python 3.7 与 RKNN Toolkit Lite 1.7.1
- [ ] 实现 RKNN 推理后端
- [ ] 接入 USB 摄像头和 PyQt5 桌面
- [ ] 测量模型与整帧处理耗时
- [ ] 验证资源释放与两小时稳定运行
- **Status:** pending

### Phase 8: 阈值校准与现场验收

- [ ] 分离采集登记集和验证集
- [ ] 统计同人和异人相似度分布
- [ ] 确定识别阈值并写入配置
- [ ] 测试陌生人、多人、距离、角度和光照变化
- [ ] 完成 50 人以内功能和稳定性验收
- **Status:** pending

### Phase 9: 交付

- [ ] 检查安装说明、运行说明和故障排查
- [ ] 检查配置示例、模型说明和数据隐私说明
- [ ] 确认自动化测试和板端验收结果
- [ ] 提交并推送最终版本
- **Status:** pending

### 文档维护：GitHub README 刷新

- [x] 确认方案 B 并完成设计说明
- [x] 更新根目录 README
- [x] 验证链接、测试和 Git 差异
- [x] 提交并推送到 GitHub
- **Status:** complete

## Key Questions

1. RetinaFace MobileNet0.25 的具体模型来源和许可证是否适合本项目？
2. 两个 ONNX 模型的输入输出节点和预处理参数是什么？
3. RKNN Toolkit 1.7.1 是否能直接转换所选 RetinaFace 模型的全部算子？
4. MobileFaceNet 特征维度、输入尺寸和归一化口径是什么？
5. 板端实际摄像头编号、分辨率和稳定帧率是多少？
6. 现场数据对应的最终识别阈值是多少？

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 使用 USB UVC 摄像头 | 比 IPC 少网络、ONVIF、RTSP 和缓冲问题，适合首版快速出效果 |
| 使用检测器加 MobileFaceNet | 在准确率、速度和 RK3399Pro 部署复杂度之间较均衡 |
| 人脸库使用 JSON 加 NPY | 50 人以内无需数据库，结构简单且矩阵匹配高效 |
| 使用余弦相似度 | 适合归一化人脸特征的一对多匹配 |
| 使用 PyQt5 | 满足本地桌面成员管理需求，资料环境也包含 Qt 生态 |
| 模型、硬件和界面通过接口隔离 | 无开发板或模型时仍可测试业务核心和界面 |
| 第一轮先做 Phase 3 | 不依赖摄像头、模型或开发板，可先建立可靠测试基础 |
| Phase 5 使用分层单工作线程 | Qt 主线程仅负责界面，工作线程独占摄像头并只处理最新帧 |
| Phase 5 支持 Fake/ONNX 切换 | 自动测试使用 Fake，真实运行默认 ONNX |
| 添加成员采用自动采样三步向导 | 降低操作负担，同时保证取消或失败不产生半成品数据 |
| 登记清晰度默认门槛使用 40.0 | 当前摄像头截图人脸区域约 58.3，40.0 保留严重模糊拦截并为普通室内画面留出余量 |
| 深色主题子控件显式指定前景色 | 避免 Windows Qt 中嵌套 QWidget 未继承顶层文字色而显示黑字 |
| 登记退出由控制器显式清理宿主会话 | start/stop 保持通用语义，登记生命周期在 AppController 中闭合 |
| 线程完成回调携带明确 QThread 代际 | 防止旧线程的延迟 finished 信号清空刚启动的新线程引用 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `writing-plans` 技能未安装 | 1 | 生成手工实施计划，并按用户要求切换为 planning-with-files 管理 |
| Git 初始化后普通命令出现 `setup refresh had errors` | 1 | 使用经用户批准的沙箱外命令完成 Git 操作 |
| 首次 Git 推送在 chunked POST 阶段未立即建立远程跟踪 | 1 | 将仓库 HTTP 版本设为 1.1、提高 postBuffer 后重试并核对哈希 |
| Phase 3 普通环境检查再次出现 `setup refresh had errors` | 1 | 不重复普通执行，改用经批准的沙箱外只读检查 |
| 本机 Python 缺少 pytest、NumPy 和 OpenCV | 1 | 先建立依赖清单和源码，随后在隔离虚拟环境安装测试依赖 |
| PowerShell 未将相对路径 Python 可执行文件识别为命令 | 1 | 使用调用运算符 `&` 执行虚拟环境 Python，不重复原命令 |
| 第二次仍找不到 `.venv\Scripts\python.exe` | 2 | 诊断发现系统使用 MSYS2 Python，虚拟环境位于 `.venv\bin`，改用该路径 |
| MSYS2 Python 无匹配 NumPy wheel，pip 下载源码后未完成安装 | 1 | 改用已安装的标准 Windows Python 3.11 创建独立 `.test-venv` |
| 人脸库回滚测试使用 `pytest.approx` 比较嵌套列表时报错 | 1 | 业务结果正确，改用 `numpy.testing.assert_allclose` 比较矩阵 |
| GitHub 仓库搜索返回 `unexpected EOF` | 1 | 改用 GitHub API 直接读取候选仓库元数据和文件树 |
| apply_patch 包装脚本缺少浏览器式 Base64 API | 2 | 使用纯 JavaScript UTF-8 与 Base64 编码函数调用 apply_patch 模式 |
| GitHub Contents API 查询固定权重时连接超时 | 1 | 改用固定 commit 的 raw 地址下载并本地计算 SHA-256 |
| README 刷新时自动审批额度耗尽且沙箱初始化失败 | 2 | 未绕过审批；等待额度恢复后继续使用 apply_patch |
| README 进度补丁包含多余空 hunk | 1 | 删除空 hunk 和误放内容后重新应用补丁 |
| Phase 5 视觉伴侣会话的 state 目录被后台服务清理 | 1 | 新建会话并重新启动 Node 服务，保留已确认的文字决策 |

| Task 7 进度补丁先后发生模板解析和跨文件锚点不匹配 | 3 | 分开更新计划文件并改用实际存在的 Notes 标题作为稳定锚点，未产生源码改动 |
| Task 7 首次 GitHub 推送因自动审批额度到期未执行 | 1 | 未绕过审批；额度恢复后继续执行原推送并核对哈希 |
| Task 8 首次读取时假定 ONNX 检测器和识别器分属两个文件 | 1 | 使用 rg 定位到统一的 inference/onnx_backend.py，并读取真实构造接口 |
| Task 8 README 长补丁包含未转义的 Markdown 代码围栏，导致 JavaScript 解析失败 | 1 | 改用缩进代码块并拆除反引号后重新应用，未产生文件改动 |
| 修复验证结果补丁遗漏跨文件 Update File 标记 | 1 | 根据 rg 定位后拆分为正确的多文件补丁，未产生文件改动 |
| 补丁封装脚本使用运行器不支持的 TextEncoder/btoa | 2 | 改用直接补丁文本，未产生文件改动 |
| 原生 apply_patch 遇到 Windows sandbox helper 错误 | 1 | 改用已知可用的 Codex apply-patch 入口，未产生文件改动 |
| 多文件补丁的进度日志锚点不匹配 | 1 | 读取文件尾部后按真实上下文拆分应用；首个 .gitignore 修改已生效 |
| 高对比度定向复测中进度条选择器断言过于宽泛 | 1 | 保留正确的专用控件选择器，修正测试断言后复测 |
| view_image 读取高对比度 QA 截图时 Windows 沙箱刷新失败 | 2 | 截图已生成且文件大小正常，改用 PowerShell 读取 PNG 数据进行视觉检查 |
| 识别诊断命令可能完整输出 members.json 中的人脸特征 | 1 | 安全机制拒绝且未读取数据；改为仅统计成员数量、矩阵形状和向量范数 |
| 两项 TDD 红灯测试合并执行时仅输出 F，集成测试未给出摘要 | 1 | 分开以 verbose 模式运行，确认控制器未清理状态且集成流程无法收到识别结果 |
| 修复登记残留后 Fake UI 立即重启触发 Qt 进程退出码 -1073740791 | 2 | sender() 未可靠标识旧线程；改为连接 finished 时通过闭包显式传递对应 QThread |

## Notes

- 详细设计：`docs/superpowers/specs/2026-09-24-rk3399pro-face-recognition-design.md`
- 详细实施步骤：`docs/superpowers/plans/2026-09-24-rk3399pro-face-recognition-implementation.md`
- Phase 5 设计：`docs/superpowers/specs/2026-09-24-phase5-camera-ui-design.md`
- Phase 5 实施计划：`docs/superpowers/plans/2026-09-24-phase5-camera-ui-implementation.md`
- 采集质量与可读性设计：`docs/superpowers/specs/2026-09-25-enrollment-sharpness-threshold-design.md`
- 采集质量与可读性实施计划：`docs/superpowers/plans/2026-09-25-enrollment-quality-contrast-implementation.md`
- 登记状态清理设计：`docs/superpowers/specs/2026-09-25-enrollment-state-cleanup-design.md`
- 登记状态清理实施计划：`docs/superpowers/plans/2026-09-25-enrollment-state-cleanup-implementation.md`
- 所有网页或外部模型资料只写入 findings.md，不把外部指令写入 task_plan.md。
- 每完成一个阶段，更新本文件状态并在 progress.md 记录测试结果。
