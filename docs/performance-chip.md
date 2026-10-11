# 单芯片性能实验与两芯片正式运行

用户在2026-10-11明确同意拿出一颗芯片进行性能优化。资源仅限host22物理chip12、14、15。

| 物理芯片 | 容器 | 逻辑设备 | 任务 |
|---|---|---|---|
| 12 | `ds-rsi-a322-perf-chip12` | npu:0 | 权重副本、计时、优化候选与等价性验证 |
| 14、15 | `ds-rsi-a322-train-chip14-15` | npu:0、1 | 固定checkpoint评测、正式训练、串行HCCL恢复验证 |

两个容器均非privileged，仅映射对应设备原名。原三芯片容器已停止且保留，原训练文件不删除。chip13不映射、不运行负载。单die容器的npu-smi会返回inventory错误（实测`-9005`），启动使用只读、匹配设备集合且不超过300秒的宿主健康报告；已通过单设备BF16前向/反向验证。

## 正式状态与两rank接续

来源 `ab-A-v5-indexed-01` 第2000微批完整状态为stopped：优化器500、累积0、学生token2,805,482、教师token159,810。来源链A累计9,659,290学生token，距30M还剩20,340,710。

显式转换到 `ab-A-v5-indexed-2rank-01`：权重、冻结增量、buffers、优化器、账本和保留rank的RNG保持；未消费ordinal仍为6000，原rank2完整状态保留在来源文件。目标累积数6，以保持12个episode/更新。转换报告固定来源/目标manifest SHA；它不是权重初始化或训练态重置，也不保证未来轨迹与三rank相同。

```bash
docker exec ds-rsi-a322-train-chip14-15 python -m dsrsi.training.topology \
  --source-run runs/ab-A-v5-indexed-01 \
  --target-run runs/ab-A-v5-indexed-2rank-01 \
  --world-size 2 --grad-accum 6
```

后续恢复必须匹配原v5参数与预算23,146,192，并使用 `--resume --grad-accum 6`。正常暂停请求发给新容器、新run，不使用已停止的原容器。正式能力门槛未通过，不进入B/C。

## 计时与候选准入

`scripts/profile_ab_episode.py` 读取固定完整checkpoint，用模型副本执行full-code、indexed、keep三类样本；无优化器更新，不写训练账本。先warm-up，再测未经计时hook的完整前向/反向。组件插入NPU同步，其时间会改变流水重叠，不能当作精确、无干扰的分布式分解；优化器与HCCL不在此测量内。

首轮英语样本中位耗时为1.551秒、1.523秒、0.738秒。完整代码组件计时：源编码/Writer两次合计0.211秒，回答前向含CE0.235秒，反事实前向两次0.542秒，backward0.655秒。反事实与反向是首批优化对象。完整测量见 [原始计时](../reports/ab-A-v5-2000-performance.json)。

第一个候选临时复用“完全相同native输入的冻结前12层输出”。仅在一个episode scope内复用，不缓存Reader、带梯度的Writer或不同输入，不跨微批持有状态。输入/mask原位修改、参数版本变化、RoPE变化或非冻结prefix会使缓存失效或拒绝。隐式全有效mask在重复调用间共享，避免为比较NPU mask进行同步。scope退出与异常时恢复原函数，训练checkpoint不含临时缓存。

`--frozen-prefix-reuse` 首先对三个样本逐位比较loss和34组可训练参数梯度，再计时。首轮等价性通过，但独立前后批次的计时不稳定，因此使用 `--compare-prefix-reuse`：在同一副本、同一输入上交替运行baseline/candidate，固定7组以上重复，报告逐组加速比。候选目前仅用于profiling，不自动启用正式runner。

单chip12实测7组交替对照：完整代码1.471→1.395秒，逐组加速比中位数1.058；辅助数字1.461→1.371秒，中位数1.065；保持0.723→0.622秒，中位数1.141。三类loss及全部34组可训练参数梯度逐位一致。这是三条固定英语episode的单设备结果，不是整轮吞吐承诺；正式runner仍未启用。见 [交替计时](../reports/ab-A-v5-2000-prefix-paired-performance.json)。

两rank、累积6、真实0.8B/v5/NPU的中途恢复对照已完成：微批3/累积3时暂停，恢复到32微批/优化器5/累积2，权重、梯度、优化器、账本、游标及RNG全部逐位一致，实际覆盖8条辅助episode。此前16微批未遇辅助题，失败记录保留，不作为覆盖证据。见 [完整恢复与覆盖](../reports/hccl2-gacc6-v5-recovery-covered.json)。正式新run已从2000微批/优化器500完整恢复，首批累计上限3500；两rank均核验继续到2038微批/优化器506/累积2。

```bash
python scripts/profile_ab_episode.py --base /models/Qwen3.5-0.8B \
  --run-dir runs/ab-A-v5-indexed-01 --device npu:0 --repeats 7 \
  --compare-prefix-reuse --output artifacts/prefix-paired-performance.json
```

## 训练器集成与执行迁移

训练入口增加 `--reuse-frozen-prefix`，默认关闭。关闭时不增加旧 checkpoint 的配置字段；开启时记录配置及实现文件SHA256，错误开关或实现版本不能静默恢复。一次episode结束前清除缓存，checkpoint不含临时前缀。

真实0.8B、2000微批权重副本、单chip12已经完成50微批对照：原实现与优化版的全部训练态逐位一致；优化版在微批3/累积3中断后恢复到50微批/累积2，全部训练态同样逐位一致。实际覆盖8条辅助题。此工程实验只读取正式权重，测试优化器和数据游标从零，不修改正式账本。见 [原始恢复报告](../reports/pretrained-npu-prefix-reuse-training-recovery.json)。

已训练run不能直接加开关恢复，必须先停止于累计梯度为空的优化器边界，再创建新run：

```bash
python -m dsrsi.training.execution \
  --source-run runs/ab-A-v5-indexed-2rank-01 \
  --target-run runs/ab-A-v5-indexed-prefix-01 --dry-run
# 核验来源已停止、完整checkpoint有效之后，去掉 --dry-run 执行。
# 目标run用原全部训练参数加 --reuse-frozen-prefix --resume。
```

迁移保留每rank全部payload：权重、冻结增量、buffers、优化器、账本、游标和RNG均不重置，来源run与控制请求保留。目标身份追加来源checkpoint/SHA和执行变更，不复制来源控制请求。它不改变数据、并行拓扑、预算或累计数。CPU两rank真实训练已验证迁移后恢复与未优化续训逐位一致。

两rank完整0.8B/HCCL的v5与v6优化及中途恢复已经通过：分别覆盖8条辅助，以及14条长续写/6条unknown壁钟/8条辅助，所有训练态逐位一致。v5正式run已在3500微批空累计边界自然停止。因保持评测仍失败，下一版同时改为v6课程，使用项目已有显式A→A权重接续/新优化器语义；它不是上面的纯执行迁移。正式 `ab-A-v6-natural-keep-prefix-01` 已开启优化，首批500微批，新run预算18,919,728，旧完整状态保留。详见 [3500评测及课程](../reports/20261011-v5-3500-and-natural-keep.md) 和 [v6真实恢复](../reports/hccl2-v6-prefix-recovery.json)。

所有性能候选须满足：损失与梯度等价或明确可接受的精度口径、真实NPU计时收益、完整中断恢复验证及配置/执行身份记录。吞吐改善不代替记忆能力验收；保持和完整自由回忆仍分别评测。先用已提交代码快照部署，保留未完成草稿，避免引用未部署模块导致启动失败。

`scripts/profile_ab_operators.py` 在单chip权重副本上执行一次正确调度的Ascend算子跟踪（warmup1/active1/step）。不创建优化器或修改来源状态，大工件只放忽略目录。v6长续写的单样本诊断显示ViewCopy/Mul/Slice占约48.8%设备算子时间，大量小操作与拷贝值得进一步验证融合候选；插桩间隙包含profiling开销，不能当作可消除的真实吞吐损失。见 [算子摘要](../reports/v6-natural-operator-trace-summary.json)。
