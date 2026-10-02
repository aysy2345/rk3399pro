# RKNN 模型转换虚拟机操作手册

## 1 当前结论

仓库已经具备 RetinaFace 和 MobileFaceNet 的 RKNN 转换、校准清单生成及 ONNX/RKNN 输出对比工具。当前还没有在 RKNN Toolkit 1.7.1 中生成真实 `.rknn` 文件，因此两个模型仍处于“工具已准备，模型待验证”状态。

模型转换固定使用 VMware 中的 Ubuntu 18.04.6 x86_64、Python 3.6.9 和 RKNN Toolkit 1.7.1。RK3399Pro 开发板不负责转换，只在后续阶段使用 RKNN Toolkit Lite 加载已验证模型。

## 2 VMware 配置

宿主机内存为 16 GB，虚拟机使用以下配置：

- 4 个虚拟 CPU 核心
- 6 GB 内存
- 40 GB 动态扩展磁盘
- NAT 网络
- Ubuntu 18.04.6 64 位 ISO
- 默认显示设置，不依赖 3D 加速

虚拟机运行时会占用约 6 GB 宿主机内存。模型转换结束后应正常关闭虚拟机，不要长期挂起。

安装完成后先确认系统架构：

    lsb_release -a
    uname -m

预期为 Ubuntu 18.04 和 `x86_64`。ARM 虚拟机、WSL 和 Windows Python 不能安装课程提供的 `cp36-cp36m-linux_x86_64` wheel。

## 3 Ubuntu 基础环境

安装编译工具、Git、动态库和 UTF-8 locale：

    sudo apt update
    sudo apt install -y build-essential cmake gcc g++ git locales \
      libprotobuf-dev protobuf-compiler libglib2.0-0 libsm6 libxrender1 libxext6
    sudo locale-gen en_US.UTF-8
    sudo update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

重新登录终端后运行 `locale`，确认 `LANG` 和 `LC_ALL` 使用 UTF-8，避免中文路径和 JSON 报告乱码。

建立固定目录：

    sudo mkdir -p /opt/rknn/packages /data/rknn-calibration/retinaface \
      /data/rknn-calibration/mobilefacenet /data/rknn-validation/retinaface \
      /data/rknn-validation/mobilefacenet
    sudo chown -R "$USER":"$USER" /opt/rknn /data/rknn-calibration /data/rknn-validation

将课程提供的 Anaconda Linux x86_64 安装程序、`requirements-cpu.txt` 和 `rknn_toolkit-1.7.1-cp36-cp36m-linux_x86_64.whl` 复制到 `/opt/rknn/packages`。这些安装包不得放入 Git 仓库。

## 4 Python 3.6.9 与 RKNN Toolkit 1.7.1

先确认课程提供的 Anaconda 安装文件：

    find /opt/rknn/packages -maxdepth 1 -name 'Anaconda3-*-Linux-x86_64.sh' -print

确认只显示一个安装文件后，直接执行该文件，安装时采用默认目录 `~/anaconda3`：

    ANACONDA_INSTALLER="$(find /opt/rknn/packages -maxdepth 1 -name 'Anaconda3-*-Linux-x86_64.sh' -print -quit)"
    test -n "$ANACONDA_INSTALLER"
    bash "$ANACONDA_INSTALLER"

重新打开终端后执行：

    source ~/anaconda3/etc/profile.d/conda.sh
    conda create -n onnx2rknn pip python=3.6.9 -y
    conda activate onnx2rknn
    python --version

预期输出 `Python 3.6.9`。

课程要求 PyTorch 与 torchvision 单独安装。复制并调整 CPU requirements，避免重复安装这两个包，同时加入课程补充的 tqdm 版本：

    cd /opt/rknn/packages
    cp requirements-cpu.txt requirements-cpu-rk3399pro.txt
    sed -i '/^[[:space:]]*torch[=<>]/d;/^[[:space:]]*torchvision[=<>]/d' requirements-cpu-rk3399pro.txt
    grep -qxF 'tqdm==4.64.1' requirements-cpu-rk3399pro.txt || echo 'tqdm==4.64.1' >> requirements-cpu-rk3399pro.txt
    pip install torch==1.5.1+cpu torchvision==0.6.1+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements-cpu-rk3399pro.txt
    pip install rknn_toolkit-1.7.1-cp36-cp36m-linux_x86_64.whl \
      -i https://pypi.mirrors.ustc.edu.cn/simple/

安装项目验证工具需要的固定版本，并最后重新固定 NumPy：

pip install onnxruntime==1.10.0 opencv-python==4.5.4.60
    pip install numpy==1.16.3
    pip check

如果 `requirements-cpu.txt` 已包含 TensorFlow、SciPy、Pillow、MXNet 和 PyYAML，应保留课程版本。课程基线分别为 TensorFlow CPU 1.14.0、SciPy 1.2.1、Pillow 5.3.0、MXNet 1.5.0 和 PyYAML 6.0。

## 5 获取项目和本地模型

克隆项目：

    mkdir -p ~/workspace
    cd ~/workspace
    git clone https://github.com/aysy2345/rk3399pro.git
    cd rk3399pro

将 Windows 已验证的两个 ONNX 文件复制到虚拟机：

    models/retinaface_mobilenet025.onnx
    models/mobilefacenet.onnx

ONNX 和 RKNN 二进制已被 `.gitignore` 排除，不得使用 `git add -f` 提交。复制后按照 `models/model-manifest.example.json` 核对文件 SHA-256：

    sha256sum models/retinaface_mobilenet025.onnx models/mobilefacenet.onnx

## 6 环境检查

在每个新终端先激活环境：

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate onnx2rknn
    cd ~/workspace/rk3399pro

运行只读环境检查：

    python tools/rknn/check_environment.py

程序检查 Python 3.6.9、Linux x86_64、UTF-8、RKNN Toolkit 1.7.1、NumPy 1.16.3、OpenCV 4.5.4、ONNX Runtime 1.10.0、libstdc++ 和 libprotobuf。只有 JSON 中 `passed` 为 `true` 才进入模型转换。

环境通过后关闭虚拟机并创建 VMware 快照，建议名称：

    rk3399pro-rknn-1.7.1-ready

## 7 准备校准和验证图片

RetinaFace 使用摄像头原始帧，图片应覆盖正脸、侧脸、远近距离和常见室内光照。MobileFaceNet 使用经过五点对齐的人脸图片。分别放入：

    /data/rknn-calibration/retinaface
    /data/rknn-calibration/mobilefacenet
    /data/rknn-validation/retinaface
    /data/rknn-validation/mobilefacenet

生成固定种子的 INT8 校准清单：

    mkdir -p rknn-data rknn-results
    python tools/rknn/build_calibration_list.py \
      --model retinaface \
      --images /data/rknn-calibration/retinaface \
      --output rknn-data/retinaface.rknn-dataset.txt \
      --max-images 100 --seed 20260927
    python tools/rknn/build_calibration_list.py \
      --model mobilefacenet \
      --images /data/rknn-calibration/mobilefacenet \
      --output rknn-data/mobilefacenet.rknn-dataset.txt \
      --max-images 100 --seed 20260927

清单使用绝对路径。清单、图片和转换报告均为本地文件，不提交 Git。

## 8 先转换非量化模型

先生成非量化模型，用于排除算子兼容和输入输出问题：

    python tools/rknn/convert_retinaface.py \
      --onnx models/retinaface_mobilenet025.onnx \
      --output models/retinaface_mobilenet025.rknn \
      --summary rknn-results/retinaface-fp-source.json
    python tools/rknn/convert_mobilefacenet.py \
      --onnx models/mobilefacenet.onnx \
      --output models/mobilefacenet.rknn \
      --summary rknn-results/mobilefacenet-fp-source.json

转换命令默认启用课程要求的 `pre_compile=True`。只有诊断问题时才使用 `--no-precompile`，并给诊断产物使用新的文件名。

工具拒绝覆盖已存在的 `.rknn`。需要重新转换时，先把已有模型和摘要移动到带日期的备份目录，避免误删已通过验证的产物。

## 9 验证非量化模型

使用与校准集分开的固定图片验证：

    python tools/rknn/validate_outputs.py \
      --model retinaface \
      --onnx models/retinaface_mobilenet025.onnx \
      --rknn models/retinaface_mobilenet025.rknn \
      --images /data/rknn-validation/retinaface \
      --report rknn-results/retinaface-fp-validation.json \
      --box-tolerance 3 --landmark-tolerance 3
    python tools/rknn/validate_outputs.py \
      --model mobilefacenet \
      --onnx models/mobilefacenet.onnx \
      --rknn models/mobilefacenet.rknn \
      --images /data/rknn-validation/mobilefacenet \
      --report rknn-results/mobilefacenet-fp-validation.json \
      --minimum-cosine 0.99

RetinaFace 报告必须显示检测数量一致，框和关键点误差不超过指定容差。MobileFaceNet 每张图片的余弦相似度必须不低于 0.99。

## 10 转换和验证 INT8 模型

只有两个非量化模型均通过后才开始 INT8：

    python tools/rknn/convert_retinaface.py \
      --onnx models/retinaface_mobilenet025.onnx \
      --output models/retinaface_mobilenet025_int8.rknn \
      --quantize --dataset rknn-data/retinaface.rknn-dataset.txt \
      --summary rknn-results/retinaface-int8-source.json
    python tools/rknn/convert_mobilefacenet.py \
      --onnx models/mobilefacenet.onnx \
      --output models/mobilefacenet_int8.rknn \
      --quantize --dataset rknn-data/mobilefacenet.rknn-dataset.txt \
      --summary rknn-results/mobilefacenet-int8-source.json

重复运行第 9 节验证命令，将 `--rknn` 和 `--report` 分别改为 INT8 文件和 INT8 报告路径。记录四个模型的文件大小和验证指标。RetinaFace 与 MobileFaceNet 可以分别选择非量化或 INT8，不要求使用同一种量化策略。

## 11 成功标准和故障处理

转换成功需要同时满足：

- `export_rknn` 返回成功并生成非空模型
- 转换摘要中输入和输出 SHA-256 完整
- 固定验证图片的报告整体通过
- 日志中没有未处理的不支持算子错误

课程资料指出，没有连接开发板时日志可能出现设备连接 Error。只有模型导出和模拟器验证均通过时，才可以忽略纯设备连接提示。不支持算子、输出形状错误、NaN、检测数量不一致和余弦相似度不达标都不能忽略。

如果 RetinaFace 确认存在 RKNN Toolkit 1.7.1 不支持的算子，应保存完整日志并评估替换轻量检测器，不修改人脸对齐、MobileFaceNet、成员库和界面接口。

真实模型通过后，再创建第二个 VMware 快照：

    rk3399pro-rknn-models-verified

至此才能进入 RK3399Pro 板端 RKNN Lite 后端集成阶段。板端安装、模型复制、
启动和验收步骤见 [`rk3399pro-deployment.md`](rk3399pro-deployment.md)。
