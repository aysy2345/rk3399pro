# RK3399Pro 板端部署与验收

本文用于把已经转换并通过无板一致性验证的模型部署到 RK3399Pro。
当前仓库已经包含 RKNN Lite 应用后端和板端配置；NPU、摄像头、性能与
稳定性仍需在真实开发板上完成最终验收。

## 1. 部署前提

- RK3399Pro 系统能够进入本地图形桌面或 VNC 桌面。
- 系统提供与 RKNN Toolkit 1.7.1 模型兼容的 NPU 驱动和运行库。
- Python 3.7、`pip`、USB 摄像头和 PyQt5 可用。
- 两个 RKNN 模型已经从转换虚拟机复制到项目的 `models/` 目录。

板端 Python 依赖见 `requirements-rk3399pro.txt`。PyQt5 使用系统软件源
安装；`rknn_toolkit_lite-1.7.1-cp37-cp37m-linux_aarch64.whl` 必须使用
Rockchip 提供的、与板端 Python 和架构匹配的安装包，不从 PyPI 猜测安装。

## 2. 模型文件

模型二进制文件被 `.gitignore` 排除，不会随 `git clone` 下载。部署时必须
另外复制以下文件：

```text
models/retinaface_mobilenet025.rknn
models/mobilefacenet.rknn
```

已验证的非量化预编译模型 SHA-256：

```text
dee5f01f9a1b730f099daf8220b6a8223634d29f7402f9d46bff3ccd4a930d09  models/retinaface_mobilenet025.rknn
453aada409bfa6072efabc6a1e21b017d52d9d3679bf7bc51aabadbe7a669847  models/mobilefacenet.rknn
```

复制后在板端项目根目录执行：

```bash
sha256sum models/retinaface_mobilenet025.rknn models/mobilefacenet.rknn
```

哈希不一致时停止部署并重新复制，不能继续加载模型。

## 3. 安装依赖

以下命令需要根据板端发行版调整软件源，但不要升级 NPU 驱动或 RKNN
运行库的大版本：

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-pyqt5
python3 -m venv --system-site-packages .venv-rk3399pro
source .venv-rk3399pro/bin/activate
python -m pip install -r requirements-rk3399pro.txt
python -m pip install /path/to/rknn_toolkit_lite-1.7.1-cp37-cp37m-linux_aarch64.whl
```

验证 RKNN Lite 可以导入：

```bash
python -c "from rknnlite.api import RKNNLite; print('RKNN Lite import OK')"
```

## 4. 手动启动

首次部署必须在可见桌面会话中手动启动，便于观察摄像头、模型加载和 Qt
错误。不要在首次验收前配置无人值守自启动。

```bash
source .venv-rk3399pro/bin/activate
python -m face_recognition_app.main --config configs/rk3399pro.json
```

需要切换摄像头时可临时覆盖编号：

```bash
python -m face_recognition_app.main \
  --config configs/rk3399pro.json \
  --camera-index 1
```

## 5. 板端验收顺序

按以下顺序验证，前一项失败时不要继续：

1. RKNN Lite 导入成功，两个模型文件哈希正确。
2. RetinaFace 和 MobileFaceNet 均能完成 `load_rknn` 与 `init_runtime`。
3. USB 摄像头可以持续读取，关闭窗口后摄像头和 NPU 资源均被释放。
4. 单人正脸能够稳定检测、录入并再次识别。
5. 多人、侧脸、远距离和不同光照场景不崩溃。
6. 连续运行至少 30 分钟，记录 FPS、单帧延迟、内存和温度。
7. 根据实测结果调整检测、识别、清晰度和时间窗口阈值。

当前转换模型为非 INT8 量化版本。INT8 转换必须等待代表真实摄像头场景的
校准照片准备完成后再进行，不能使用单张样例或无关公开图片代替。

## 6. 无板阶段已经完成的验证

- Windows 完整测试套件通过。
- RKNN 转换工具专项测试通过。
- RetinaFace ONNX 与非预编译 RKNN 模拟输出对比通过。
- MobileFaceNet ONNX/RKNN 特征余弦相似度为 `0.999762`。
- 两个板端预编译 RKNN 模型的复制哈希已经校验。

上述结果证明接口、转换配置和模拟输出一致，但不能替代真实 NPU 性能与
稳定性测试。
