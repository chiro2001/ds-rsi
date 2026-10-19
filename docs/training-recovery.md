# 快速中断与完整训练状态恢复

日期：2026-10-10。实现和测试将在本约定下推进，性能目标以实测记录更新。

基础实现位于 `src/dsrsi/training/checkpoint.py`、`control.py` 与 `distributed.py`，现已接入多 rank A/B、C_tools 和 C_context 执行器。完整0.8B 的双 rank Gloo 与 a3-22 chip8–11 四 rank HCCL 暂停恢复对照均逐位一致。NPU A/B 外部暂停请求到全部 launcher 退出本次实测约5.7秒；另起进程恢复后的完整状态（含全部 RNG）与同一 CLI 连续训练逐位一致，见 [外部控制结果](../reports/pretrained-npu-external-control.json)。这是短负载的单次测量，不能代替不同窗口和干扰长度下的响应时间 profile。控制命令使用纯标准库，具体入口见 [执行指南](running-ab.md)。

外部请求接口：

```bash
python -m dsrsi.training.control --run-dir runs/<run-id> --action pause
```

该命令写控制请求，执行器在所有 rank 协调的可恢复边界处理；运行状态以对应run的提交manifest和进程核验为准。

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

当前 A/B 和 C 实现另存冻结的增量模块：它们不属于底座权重文件，不能只引用底座指纹。记忆结构的 heads/读写位置等也须核验，即使 tensor shape 相同仍可能改变执行语义。C 已有执行文件/包版本身份与持久 journal 校验；[持续短 unroll](stage-c-context.md)还保存学生/教师 life、部分消息和剩余 tape。长期语义及 NPU 验收仍待完成。

## 原子提交

每个 rank 写独立状态文件到临时位置，flush/fsync 后原子改名。全部 rank 文件校验完成后 rank0 才提交 manifest，再原子更新 latest 指针。恢复仅接受有完整 manifest 且文件大小/摘要相符的 checkpoint；临时或部分写入不作为最新状态。

保存失败不能覆盖上一个已提交 checkpoint。停止时写 status=paused/stopped；不能以 filename=final 充当完成判据。

普通计算/反传/优化器或读盘异常会协调全体退出，不将失败微批写成新的有效 checkpoint。进程突然死亡或设备/集合通信故障仍需依靠进程组超时与 launcher 清理，恢复从旧 latest 重做其后工作。

## 恢复校验

恢复先校验底座、数据、配置、优化器和并行拓扑，再加载状态。默认完整恢复；不自动跳过不兼容优化器或清空数据游标。阶段切换、权重迁移等有意重置要用不同命令和明确记录。

显式A→A/A→B权重初始化可用 `--init-from RUN --init-checkpoint GENERATION` 选择来源的较早完整代次；结构、权重和来源manifest均从同一固定代次读取并校验。它必须创建新run，重置优化器/RNG/游标，不能与resume混用；来源latest、较早和较晚完整状态均不改变。恢复新run只用其原参数加 `--resume`，不再次选择来源。成本账本仍应计入来源链所有实际消费，不能因选择较早权重抹去之后的训练成本。

相同 world size 与采样规则支持精确恢复。A/B新增显式减少副本的离线转换，要求来源已停止/暂停、位于优化器边界、无累计梯度、全部rank共享权重/buffers/优化器完全一致，且life与harness均为空。以新run保存相同优化器、进度与原rank RNG，将未消费的连续ordinal区间按新world size分配；不参与的新rank状态保留在旧checkpoint，来源文件和latest不改动。

转换不会重置学生/教师token账本、优化器步数或来源链。它会把未来每个优化器步的episode数由16改为12，不保证与原四rank的未来数值轨迹一致。当前仅支持保留至少两rank的A/B副本缩减，拒绝C持续状态、非连续游标、累计中途以及不一致的优化器。新拓扑自己的中断/恢复需做连续对照。

2026-10-11资源拆分时，将chip12留给性能实验，chip14/15用于训练。三rank的2000微批已完整停止，可在新run上显式 `--world-size 2 --grad-accum 6` 转换；3×4和2×6均为12个episode/优化器更新。累积数更改仅限此离线、空梯度边界，写入目标配置及迁移报告，普通resume仍禁止改配置。所有保留rank的完整payload只改变world_size，旧run与未参与rank保持原样。保持全局批量仍会改变梯度加法顺序与rank分配，不承诺原拓扑的未来数值轨迹。

原run在停止后留下的暂停请求不复制到转换run；新run的control从空开始，避免恢复即暂停。原请求和完整checkpoint保留，用于审计及未来显式恢复原拓扑。

## 验证

至少比较连续 N 个微批与中途保存/退出/恢复后的 N 个微批：样本 id、token 账本、权重、优化器、累计梯度、RNG 和最终流状态一致。覆盖梯度累积中途暂停、NumPy 随机采样、CPU/NPU 随机状态、损坏/部分 checkpoint、配置与拓扑不匹配。

对真实模型记录请求到退出的时间、保存耗时和恢复后第一步时间；先验证小负载，再启用长期多 chip 训练。

2026-10-19已新增独立的`python -m dsrsi.training.hardware`：仅在已停止、空累计、stateless A/B且共享状态一致的NPU来源创建单CUDA新run，保持优化器和账本，显式初始化CUDA RNG；旧副本缩减入口的至少两rank限制不变。真实188个episode的独立进程外部暂停/恢复已逐位一致，暂停到退出约3.52秒、重启到首微批约90.49秒。见[5080训练与恢复](cuda-training.md)。

CUDA学习率变更另用`python -m dsrsi.training.learning_rate`，仅在停止、空累积的单卡reference A/B边界创建新run，严格保留所有其它payload，不作为普通resume或权重初始化。已从1434保留全态迁至lr0.000125，真实188个episode在1438/累积4外部暂停、独立恢复到1622/累积8，与连续轨迹逐位一致；暂停退出3.74秒、重启首微批79.31秒。当前正式lr125分支最近已核验/评测边界2634，已完整resume上限3594，使用新run原参数加resume/累积12/lr0.000125；详情及当前暂停命令见[5080训练与恢复](cuda-training.md)。
