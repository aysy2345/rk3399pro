# RK3399Pro 本地人脸识别

本项目面向 RK3399Pro，使用 USB UVC 摄像头实现 50 人以内的离线 1:N 人脸识别，并通过 PyQt5 提供添加和管理成员的桌面界面。

## 当前状态

项目处于 Phase 3，正在实现不依赖模型和开发板的配置、人脸库与识别算法核心。

## 文档

- 系统设计：`docs/superpowers/specs/2026-09-24-rk3399pro-face-recognition-design.md`
- 实施计划：`docs/superpowers/plans/2026-09-24-rk3399pro-face-recognition-implementation.md`
- 当前任务：`task_plan.md`
- 研究结论：`findings.md`
- 进度记录：`progress.md`

## 配置

复制 `configs/app.example.json` 为自己的运行配置，并根据实际模型和数据目录修改路径。识别阈值只是初始值，必须使用现场验证集校准。
