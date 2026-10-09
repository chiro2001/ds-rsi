# 快速中断与完整训练状态恢复

日期：2026-10-10。实现和测试将在本约定下推进，性能目标以实测记录更新。

基础实现位于 `src/dsrsi/training/checkpoint.py`、`control.py` 与 `distributed.py`，现已接入多 rank A/B 执行器，完成完整 0.8B 双 rank Gloo 暂停/恢复对照，见 [分布式报告](../reports/20261010-distributed-and-evaluation.md)。控制命令使用纯标准库；HCCL、NPU 延迟尚待验证。具体入口见 [执行指南](running-ab.md)。

外部请求接口：

```bash
python -m dsrsi.training.control --run-dir runs/<run-id> --action pause
```

该命令写控制请求，执行器在所有 rank 协调的可恢复边界处理；当前没有正式训练负载在运行。

## 控制语义

- checkpoint：保存完整状态后继续。
- pause：在最近可恢复边界保存，停止本次负载；远期目标是否暂停按用户指令更新。
- stop：安全保存后停止当前运行，不将其标为训练完成。
- completed：只有达到配置规定的训练预算与收尾条件才使用。

信号 handler 只设置标志，不在 handler 中调用 torch.save 或集合通信。控制文件支持在训练尚未就绪时预先发请求；多 rank 用明确的同步检查点传播请求，防止部分 rank 退出而其他 rank 挂在通信上。

## 可恢复边界

首版每个微批完成 backward、提交数据消费和状态变化后可以中断，梯度累积到一半仍能保存累计梯度和已完成微批计数。正式流式训练以完成 backward 的短 unroll 为边界。

autograd 图不直接序列化。跨段训练在边界完成反传并 detach 数值状态；如必须中断未完成 unroll，应保存其起始状态与重放日志并从该边界重做，不能只保存当前 latent 数值就假装恢复了计算图。

## Checkpoint 内容

1. trainable 权重与必要 buffers、冻结底座身份和内容指纹。
2. 优化器、调度器、AMP scaler、累计梯度、梯度累积微批计数。
3. 全局 optimizer step、micro step、stage、实际学生/教师 token 账本。
4. 已消费数据游标、sampler/生成器版本、数据与 tokenizer 指纹。
5. 每 rank Python/NumPy/torch/对应 NPU RNG 状态。
6. life state：KV、GDN/conv、位置原点、逻辑时钟、待写缓冲、M 与索引、harness 引用/版本。
7. world size、rank 设备映射、训练配置、代码 commit 与 checkpoint 版本。

当前 A/B 和 C 工具实现另存冻结的增量模块：它们不属于底座权重文件，不能只引用底座指纹。记忆结构的 heads/读写位置等也须核验，即使 tensor shape 相同仍可能改变执行语义。C 工具阶段已有执行文件/包版本身份与持久 journal 校验，见 [C 工具阶段](stage-c-tools.md)；完整持续状态 unroll 的训练接入仍待完成。

## 原子提交

每个 rank 写独立状态文件到临时位置，flush/fsync 后原子改名。全部 rank 文件校验完成后 rank0 才提交 manifest，再原子更新 latest 指针。恢复仅接受有完整 manifest 且文件大小/摘要相符的 checkpoint；临时或部分写入不作为最新状态。

保存失败不能覆盖上一个已提交 checkpoint。停止时写 status=paused/stopped；不能以 filename=final 充当完成判据。

普通计算/反传/优化器或读盘异常会协调全体退出，不将失败微批写成新的有效 checkpoint。进程突然死亡或设备/集合通信故障仍需依靠进程组超时与 launcher 清理，恢复从旧 latest 重做其后工作。

## 恢复校验

恢复先校验底座、数据、配置、优化器和并行拓扑，再加载状态。默认完整恢复；不自动跳过不兼容优化器或清空数据游标。阶段切换、权重迁移等有意重置要用不同命令和明确记录。

首版保持相同 world size 与采样规则做精确恢复。后续如需要变更设备数量，必须显式转换梯度、游标与每 rank 流状态，并重新验证，不能直接沿用旧 offset。

## 验证

至少比较连续 N 个微批与中途保存/退出/恢复后的 N 个微批：样本 id、token 账本、权重、优化器、累计梯度、RNG 和最终流状态一致。覆盖梯度累积中途暂停、NumPy 随机采样、CPU/NPU 随机状态、损坏/部分 checkpoint、配置与拓扑不匹配。

对真实模型记录请求到退出的时间、保存耗时和恢复后第一步时间；先验证小负载，再启用长期多 chip 训练。
