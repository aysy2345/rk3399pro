# 摄像头预览水平翻转实施计划

## Task 1：配置回归测试

- 扩展 CameraConfig，新增 `preview_flip_horizontal`。
- 先添加 true、false、缺省值与非布尔值测试。
- 更新示例配置与测试配置工厂。

## Task 2：VideoWidget 显示与坐标测试

- 添加左右颜色不同的帧，验证启用翻转后左右像素交换。
- 添加偏左 FaceOverlay，验证启用翻转后叠加框移动到右侧。
- 验证关闭翻转时保持原始方向。
- 先运行定向测试确认实现前失败。

## Task 3：实现并接线

- 为 VideoWidget 增加 `flip_horizontal` 参数。
- 仅对内部显示图像做水平翻转。
- 绘制时按源图宽度变换人脸框横坐标。
- MainWindow 与 EnrollmentWizard 接收并传递相同配置。
- Bootstrap 从 CameraConfig 注入两个窗口。
- 本地与示例配置设为 true。

## Task 4：验证与交付

- 运行配置、UI、Bootstrap 和集成定向测试。
- 运行完整 pytest、compileall、CodeGraph 和 diff check。
- 生成 Windows Qt 非对称画面截图，检查方向与叠加框。
- 更新计划、发现和进度。
- 提交并推送修复。

## 验收标准

- 当前笔记本摄像头预览恢复为非镜像方向。
- 推理与登记仍接收原始帧。
- 翻转后人脸框与标签位置正确。
- 配置设为 false 时恢复原始显示。
- 全部自动化测试通过。
