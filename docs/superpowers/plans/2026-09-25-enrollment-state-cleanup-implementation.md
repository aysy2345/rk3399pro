# 登记结束后识别状态清理实施计划

## Task 1：建立回归测试

- 在控制器测试中复现：待机进入登记，完成登记，再开始识别。
- 断言登记完成同时清除宿主登记会话并停止旧线程。
- 断言重新开始识别时宿主不再保留旧登记会话。
- 保留“识别中进入登记再返回识别”的行为。
- 先运行定向测试，确认修复前失败。

## Task 2：实现最小状态清理

- 修改 `AppController.cancel_enrollment`。
- 进入取消逻辑后统一调用 `host.cancel_enrollment()`。
- 恢复识别分支不停止线程；待机登记分支在清理后停止线程。
- 不修改 WorkerThreadHost 的通用 start/stop 语义。

## Task 3：验证与交付

- 运行 controller、worker 和 Fake UI 定向测试。
- 运行完整 pytest 与 compileall。
- 同步 CodeGraph 并执行 git diff --check。
- 更新 task_plan.md、findings.md 和 progress.md。
- 提交并推送，提供重新启动与复测步骤。

## 验收标准

- 从待机完成登记后，WorkerThreadHost 不保留 pending enrollment。
- 再次点击开始识别会进入 recognition 分支并发送 FrameResult。
- 用户截图对应的人脸可按实测 0.9314 相似度识别为已登记成员。
- 全部自动化测试通过。
