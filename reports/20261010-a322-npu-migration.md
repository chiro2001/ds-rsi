# a3-22 迁移与真实 NPU 恢复验证

用户通知 a3-21 故障，授权改用 a3-22 空闲芯片。此次迁移已完成：选用 host22 物理 chip8–11，完整 Qwen3.5-0.8B 的 A/B 入口、工具训练、持续上下文均已在四 rank HCCL 上实际执行，暂停恢复的全部训练态与连续运行逐位一致。

## 资源与隔离

启动前 npu-smi 逐颗检查 chip8–11 健康 OK、无进程。0–7、12–13 有其他进程，未使用。其他运行容器的物理映射和显式可见设备另行检查；广泛可见所有设备的开发容器依实际进程占用判断，专用设备容器的保留映射优先排除。

新建本项目容器 `ds-rsi-a322-npu`，非 privileged，项目目录 `/home/l00886679/ds-rsi`，底座 `/home/l00886679/models/Qwen3.5-0.8B` 只读。仅映射原名 `/dev/davinci8`–`11` 及管理设备，驱动和 firmware 只读。原始镜像内 PyTorch/torch_npu 没有被替换。

物理节点在隔离容器内会被运行时重新编号：`DSRSI_PHYSICAL_CHIPS=8,9,10,11`，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`，逻辑 npu:0–3 对应物理 chip8–11。直接将物理编号8–11设为容器运行时可见编号会使 device_count 返回0。新 guard 同时检查主机、原名物理节点集合、逻辑可见设备数、rank 范围与健康，拒绝旧 a3-21 八芯片配置。

环境：Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post4、Transformers5.14.1、CANN9.1、npu-smi26.1.1。`+cpu` 是镜像原有 torch 构建标签；实际 NPU 张量、BF16 矩阵反向和完整模型梯度已验证。底座权重 SHA256 与此前 CPU 验证一致。映射和镜像 ID 见 [环境清单](a322-resource-environment.json)。

## 实际验证

| 路径 | 最终微批/优化器步数 | 全局学生/教师 token | 验证 |
|---|---:|---:|---|
| A/B 执行器的 A 对照 | 5 / 1 | 2376 / 358 | 第3微批末 rank 触发暂停；累计梯度、优化器、RNG、游标完整恢复 |
| C_tools | 5 / 1 | 11643 / 1200 | 1,302,528 LoRA 参数；真实工具观察与 journal；恢复逐位一致 |
| C_context | 16 / 2 | 998 / 998 | 窗口16、unroll16；消息中途及待写 buffer 恢复逐位一致 |

三个对照都是四 rank、确定性算法和完整0.8B 权重。C_context 第3微批暂停时各 life 时钟48，消息 offset27，待写 Writer27 token；最终 student/teacher 时钟一致，每个 cache 19,955,712 bytes，M7条共35,840 bytes。不同 rank 的数据流和查询数量不同，最终进度按全局 token/查询规范核对。

额外用正式 `train_ab.py` CLI 和外部控制文件测试实际进程退出：第1微批有1份待累计梯度时请求 pause，约5.723秒内保存并退出整个四 rank launcher。用新进程 `--resume` 运行到第5微批，再与相同 CLI 的独立连续训练比较，四份完整 checkpoint 包括 Python/NumPy/CPU/NPU RNG 全部逐位一致。

新进程对照必须保持同一入口。验证脚本和训练 CLI 的导入顺序不同，会影响初始 Python RNG；跨入口直接比较时模型、优化器和梯度一致，但初始 Python RNG 不同。本次最终对照使用同一 CLI，未忽略任何 RNG 字段。

`torchrun --standalone` 在当前旧容器的 UTS hostname=host22、容器 hosts 未含 host22 的情况下无法建立 rendezvous。实际 CLI 验证使用 `--master-addr=127.0.0.1 --master-port=29583`；新容器启动脚本同时添加 host22 的本机 hosts 映射。验证结束后 chip8–11 健康、无残留进程，项目容器仅待命。

原始结果：[A/B](pretrained-npu-hccl-recovery.json)、[工具](pretrained-npu-tools-recovery.json)、[持续状态](pretrained-npu-context-recovery.json)、[外部控制和新进程](pretrained-npu-external-control.json)。本地全套144项 CPU 测试通过；后续增加的资源节点/拓扑/容器保留检查共19项专项测试通过，最终完整集合由 GitHub CI 再核验。

## 能力与下一步

这些是 NPU 工程验证与短负载，不构成记忆语义、工具能力或无限上下文能力达标。正式 A/B 训练、长期持续任务、递归压缩 D 和最终交付仍未完成，远期 goal 保持 active。当前四芯片环境可作为后续训练执行环境。

正式 A/B 前仍需版本化强化样本：取消支撑与负例的固定实体前缀、支撑固定在尾部等捷径，并接入开发集驱动的 oracle 退火。C 的标准逐 token 路径在此次小窗口 smoke 中开销显著；须以更长运行 profile 决定预算与优化，不能用5/16微批直接推算正式机时。
