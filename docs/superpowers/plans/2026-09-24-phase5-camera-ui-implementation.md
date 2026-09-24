# Phase 5 摄像头与 PyQt5 界面实施计划

## 1 目标

在现有 28 项测试、FaceStore、识别核心和 Fake/ONNX 推理后端基础上，实现 USB 摄像头、最新帧识别工作线程、PyQt5 主窗口、三步成员登记和表格式成员管理。

完成时必须满足：

- 主机可使用 Fake Camera 与 Fake 推理完成离屏自动测试；
- 主机可使用 USB 摄像头和 ONNX 后端运行真实识别；
- 成员新增、改名、重新采集和删除后立即刷新识别库；
- 关闭应用后工作线程结束并释放摄像头；
- 原有测试与新增测试全部通过。

## 2 实施约束

- 使用 Python 3.7 兼容语法，不引入仅高版本支持的类型写法。
- PyQt 主线程只操作控件；摄像头和推理在一个工作线程中串行执行。
- 不建立待推理帧队列，每轮直接读取并处理当前帧。
- 业务规则尽量放入不依赖 Qt 的纯 Python 类，以便快速单元测试。
- UI、摄像头、推理和存储均通过构造参数注入测试替身。
- 每个任务先新增失败测试，再实现最小代码，最后运行相关测试和完整回归。

## 3 Task 1：扩展运行配置与测试依赖

### 文件

- 修改：`face_recognition_app/app/config.py`
- 修改：`configs/app.example.json`
- 修改：`requirements-dev.txt`
- 修改：`requirements-test.txt`
- 修改：`requirements-rk3399pro.txt`
- 修改：`tests/unit/test_config.py`

### 步骤

1. 在配置测试中加入 backend、目标帧率、推理间隔、清晰度门槛和采样间隔的正常与非法用例。
2. 运行配置测试，确认新断言先失败。
3. 新增 RuntimeConfig，扩展 CameraConfig 与 RecognitionConfig，并保持现有字段验证。
4. backend 仅允许 `fake`、`onnx` 和为后续保留的 `rknn`；Phase 5 示例默认 `onnx`。
5. 把示例模型路径改为两个 ONNX 文件。
6. 主机测试依赖加入 PyQt5 和 pytest-qt；板端继续注明 PyQt5 由系统包安装。
7. 运行：

       & .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_config.py -q

### 验收

- 合法新配置能解析为不可变数据类。
- backend、帧率、间隔和质量阈值错误时给出明确 ConfigError。
- 旧字段语义不改变。

## 4 Task 2：摄像头抽象与 OpenCV 实现

### 文件

- 新增：`face_recognition_app/hardware/__init__.py`
- 新增：`face_recognition_app/hardware/camera.py`
- 新增：`tests/unit/test_camera.py`

### 步骤

1. 用 StubCapture 编写打开成功、打开失败重试、读取失败和幂等释放测试。
2. 定义 Camera、CameraError 和 CameraReadError 边界。
3. 实现 OpenCVCamera，构造时接收 capture_factory，生产环境默认 `cv2.VideoCapture`。
4. 打开时设置宽、高和目标帧率；有限次数重试后抛出 CameraError。
5. read 返回独立 BGR ndarray；无帧时抛出 CameraReadError。
6. release 可重复调用，上下文管理器退出时必定释放。
7. 运行：

       & .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_camera.py -q

### 验收

- 自动测试不访问真实摄像头。
- 所有失败路径都会释放已经创建的 capture。

## 5 Task 3：纯 Python 识别与登记流水线

### 文件

- 新增：`face_recognition_app/core/tracking.py`
- 新增：`face_recognition_app/core/pipeline.py`
- 新增：`face_recognition_app/core/enrollment_session.py`
- 新增：`face_recognition_app/domain/runtime.py`
- 修改：`face_recognition_app/core/quality.py`
- 新增：`tests/unit/test_tracking.py`
- 新增：`tests/unit/test_pipeline.py`
- 新增：`tests/unit/test_enrollment_session.py`

### 步骤

1. 为 FaceOverlay、FrameResult、WorkerErrorInfo 和应用状态编写值对象测试。
2. 实现基于检测框 IoU 的短时轨迹关联，测试轨迹保持、多人分离和过期清理。
3. 为 RecognitionPipeline 注入 detector、embedder、matcher 和 stabilizer。
4. 测试无人脸、已知人脸、陌生人、多人以及人脸库快照刷新。
5. 根据五点关键点中鼻尖相对双眼中心的位置实现 front/left/right 粗粒度姿态分类，并编写镜像无关的阈值测试。
6. 实现 EnrollmentSession：质量检查、姿态配额、最小采样间隔、样本去重提示和进度状态。
7. 采集完成后调用现有 build_enrollment_template，不直接写人脸库。
8. 运行：

       & .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_tracking.py tests/unit/test_pipeline.py tests/unit/test_enrollment_session.py -q

### 验收

- 识别流水线和登记会话不依赖 PyQt 或真实摄像头。
- 每个检测结果都有稳定的短时 track_id。
- 15 个样本按正视、左转和右转配额收集，质量不足不会增加计数。

## 6 Task 4：Qt 工作线程与应用控制器

### 文件

- 新增：`face_recognition_app/workers/__init__.py`
- 新增：`face_recognition_app/workers/recognition_worker.py`
- 新增：`face_recognition_app/app/controller.py`
- 新增：`tests/unit/test_worker.py`
- 新增：`tests/unit/test_controller.py`

### 步骤

1. 使用 Fake Camera 和纯流水线编写 worker 启停、读取错误、模式切换和资源释放测试。
2. RecognitionWorker 作为 QObject 移入 QThread，使用明确的 start、stop、begin_enrollment、cancel_enrollment 和 refresh_store 槽。
3. 每次循环只读取当前帧；没有跨循环帧队列。
4. 通过 object 信号发送帧副本、识别结果、登记进度、状态和结构化错误。
5. AppController 统一管理 IDLE、RECOGNIZING、ENROLLING、ERROR 状态和允许的转换。
6. 测试重复开始/停止幂等、登记前状态恢复、错误重试和关闭顺序。
7. 运行 Qt 离屏测试：

       $env:QT_QPA_PLATFORM='offscreen'
       & .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_worker.py tests/unit/test_controller.py -q

### 验收

- 工作线程不直接访问控件。
- 停止和关闭后 Fake Camera 的 release 只需成功一次且可重复调用。
- 异常转换为信号，不穿透 Qt 事件循环导致进程退出。

## 7 Task 5：主窗口

### 文件

- 新增：`face_recognition_app/ui/__init__.py`
- 新增：`face_recognition_app/ui/video_widget.py`
- 新增：`face_recognition_app/ui/main_window.py`
- 新增：`tests/ui/test_main_window.py`

### 步骤

1. 用 pytest-qt 编写默认待机、按钮可用性、状态切换和错误展示测试。
2. 实现上下分区布局：上方自适应视频区，下方结果、状态、人数和四个操作按钮。
3. VideoWidget 将 BGR 帧转为 QImage，按纵横比缩放，并绘制框、姓名和相似度。
4. 主窗口只调用 AppController，不自行创建摄像头或模型。
5. closeEvent 请求控制器停止，等待完成后再关闭。
6. 默认使用 showMaximized，由启动入口负责显示。
7. 运行：

       $env:QT_QPA_PLATFORM='offscreen'
       & .\.test-venv\Scripts\python.exe -m pytest tests/ui/test_main_window.py -q

### 验收

- 界面启动后处于待机，不自动打开摄像头。
- 不同应用状态下按钮启用关系符合设计。
- 错误信息可见且可恢复错误提供重试。

## 8 Task 6：三步成员登记向导

### 文件

- 新增：`face_recognition_app/app/member_service.py`
- 新增：`face_recognition_app/ui/enrollment_wizard.py`
- 新增：`tests/unit/test_member_service.py`
- 新增：`tests/ui/test_enrollment_wizard.py`

### 步骤

1. MemberService 封装编号校验、同名提示、添加和识别库刷新通知。
2. 测试重复编号、空信息、保存失败回滚和保存成功。
3. 使用 QStackedWidget 实现基本信息、自动采集、确认保存三页。
4. 采集页显示预览、姿态提示、未采纳原因和 0/15 进度。
5. 未完成采集时禁止进入确认页；保存进行中禁止关闭或重复提交。
6. 取消、关闭和失败不调用 FaceStore.add；成功只调用一次。
7. 关闭向导时由控制器恢复登记前状态。
8. 运行：

       $env:QT_QPA_PLATFORM='offscreen'
       & .\.test-venv\Scripts\python.exe -m pytest tests/unit/test_member_service.py tests/ui/test_enrollment_wizard.py -q

### 验收

- 登记过程摄像头不被第二个对象打开。
- 成员仅在最终确认成功后落盘并立即进入识别库。

## 9 Task 7：表格式成员管理

### 文件

- 修改：`face_recognition_app/app/member_service.py`
- 新增：`face_recognition_app/ui/member_manager_dialog.py`
- 新增：`tests/ui/test_member_manager_dialog.py`

### 步骤

1. 为编号/姓名过滤、改名、删除确认和重新采集入口编写界面测试。
2. 使用 QTableWidget 或表格模型显示编号、姓名、登记时间和操作按钮。
3. 编辑姓名调用 MemberService.rename；编号只读。
4. 删除必须经过 QMessageBox 二次确认。
5. 重新采集复用 EnrollmentWizard 的采集与确认页，保存时调用 replace_embedding。
6. 操作成功后刷新表格、主窗口人数和 worker 快照。
7. 运行：

       $env:QT_QPA_PLATFORM='offscreen'
       & .\.test-venv\Scripts\python.exe -m pytest tests/ui/test_member_manager_dialog.py -q

### 验收

- 失败或取消重采保留旧特征。
- 删除后 JSON、NPY、表格和内存快照一致。

## 10 Task 8：后端工厂、启动入口与完整集成

### 文件

- 新增：`face_recognition_app/app/bootstrap.py`
- 新增：`face_recognition_app/main.py`
- 新增：`tests/integration/test_fake_ui_flow.py`
- 修改：`README.md`
- 修改：`configs/app.example.json`

### 步骤

1. BackendFactory 根据配置创建 Fake 或 ONNX detector/embedder；未知后端在配置层拒绝。
2. main.py 解析 `--config`、`--backend` 和 `--camera-index` 覆盖项。
3. 启动顺序为：加载配置 → 创建 FaceStore → 创建后端/摄像头工厂 → 创建控制器和主窗口 → showMaximized。
4. 编写 Fake Camera + Fake 后端的完整流程测试：待机、开始、识别、停止、登记、刷新和关闭。
5. README 增加主机安装、Fake 启动、ONNX 启动、USB 摄像头检查和常见错误。
6. 运行完整验证：

       $env:QT_QPA_PLATFORM='offscreen'
       & .\.test-venv\Scripts\python.exe -m pytest -q
       & .\.test-venv\Scripts\python.exe -m compileall -q face_recognition_app tests
       codegraph.cmd sync D:\rk3399pro
       codegraph.cmd status D:\rk3399pro

### 验收

- 全部测试通过，无残留 QThread 警告。
- Fake 集成测试无需摄像头和模型。
- ONNX 启动缺少模型时显示明确错误。
- Git 差异无模型、登记数据、日志或临时测试产物。

## 11 提交顺序

1. `chore: extend phase 5 runtime configuration`
2. `feat: add usb camera abstraction`
3. `feat: add face recognition runtime pipeline`
4. `feat: add qt recognition worker and controller`
5. `feat: add pyqt face recognition window`
6. `feat: add member enrollment wizard`
7. `feat: add member management dialog`
8. `feat: wire phase 5 desktop application`
9. `docs: document phase 5 desktop workflow`

每个提交前运行对应测试；最后一个功能提交前运行完整回归。

## 12 执行顺序与暂停条件

严格按 Task 1 至 Task 8 顺序实施。只有以下情况暂停并请求用户输入：

- 本机无法安装 PyQt5 或 pytest-qt，且无法用现有环境运行 Qt 测试；
- 真实 USB 摄像头行为与 Fake Camera 合同不一致，需要用户提供设备信息；
- ONNX 实际运行暴露模型合同变化；
- 设计范围需要加入活体、报表、云服务或其他未批准功能。
