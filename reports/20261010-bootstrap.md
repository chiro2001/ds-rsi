# 工程初始化与训练恢复基础验证

日期：2026-10-10。

远期目标已创建，状态 active。GitHub 仓库 `chiro2001/ds-rsi` 已创建并核验 `isPrivate=true`，初始设计和路线已推送。

## 已完成

- 独立 Python CPU 开发环境：Python 3.12、torch 2.10.0+cpu。
- 可恢复训练的 rank payload：可训练权重、buffers、优化器、调度器/scaler、累计梯度、进度、数据游标、life state、Python/NumPy/torch/所选设备 RNG。
- 分 rank 原子文件、全部文件校验后提交 manifest、latest 指针原子替换。
- 恢复校验 identity、world size、参数形状/类型与优化器参数顺序，不默默重置。
- 控制请求与信号仅记录意图，完成 checkpoint 后确认，旧暂停请求不会在恢复时再次暂停。

验证：`python -m pytest -q tests/test_recovery.py tests/test_control.py`，**12 passed**。

核心对照在梯度累积 4 个微批、完成第 6 个微批时中断，恢复后继续至第 11 个微批。样本、loss、参数、待累计梯度、优化器、调度器和数值流状态与连续运行逐位一致。包含 dropout 与三种 CPU 随机源。

这是 CPU 恢复基础验证；多进程协调、真实 Qwen/GDN 状态和 NPU 中断延迟尚待集成验证。

## a3-21 资源核验

授权 chip8–15 当前没有 npu-smi 中的运行进程。其所在卡 4–7 返回：

```text
Health Status: Alarm
Error Code: 80C98001
Error Information: node type=AIC, sensor type=RAS State,
                   event state=module error can not be fixed
```

已将实际故障告知用户，询问设备是否已修复或有替代资源。没有在该告警条件下启动正式训练，没有复位设备或改动其他人的负载。继续本地 CPU 实现与验证，远期目标保持 active。

## 下一步

实现回答 CE、支撑组寻址与 top-k+tail 保持损失；完成数据编译器和微批训练执行器，再接入实际模型。设备恢复后先做小负载与多 rank 控制验证，通过后启用正式训练。
