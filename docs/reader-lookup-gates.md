# 检索期间保持读取决策

Writer位置池化后，后缀变化仍能显著改变slots，但首批500微批的完整代码EM为0。第二个Reader的门控从查询处约0.97降为后缀处约0.005–0.012，内容反事实优势也接近0。该相关性不是因果证明，因此采用显式结构干预及独立新run验证。

新模式 `reader_gate_mode=lookup` 保留各Reader原来的gate网络。在检索位置用该Reader当前的因果hidden计算一个sigmoid标量，之后本次候选集合的生成位置复用该标量。每个位置仍用自己的hidden生成attention query，不固定整个注意力。新查询或周期检索重新计算读取决策，不使用未来答案状态。

native训练按显式answer-free retrieval index获取门控；该索引之后才能读M，索引之前仍旁路。cached执行在lookup的当前可见位置计算门控，放入该life的 `StreamingCache.read_gates`，并记录 `read_gate_position`。继续同次读取时复用；缺失过去决策、未来位置或不合法快照明确拒绝，不能拿较晚token重算过去意图。

cache fork只复制映射，不原地修改父life。detached切断门控的计算图；快照保存数值及位置，恢复核验概率、形状、有限性和逻辑范围。门控不存入模型全局buffer，避免不同life串扰。每个Reader只增加一个标量持久状态，没有新增参数。

`token`是旧模式，架构身份省略新字段、缓存不添加门控快照字段；老A/B的数值路径保留。`lookup`在架构中记录模式及 `reader_gate_version=lookup_v1`。结构变更必须新run、显式 `--reader-gate-mode lookup --init-from ...`；普通 `--resume`从checkpoint继承并禁止更改。

CPU验证覆盖native/cached对齐、答案未来扰动不改变查询决策、检索刷新、父cache不变、快照/缺失决策/坏概率拒绝、detach与gate梯度、连续life外部分块和pending Writer恢复。完整213项测试通过。完整0.8B三rank HCCL的A/B累计中途恢复逐位一致；C短unroll在持有两个门控和14/15个pending token时暂停恢复，也逐位一致。C/B仅用工程fixture，未通过任何正式能力阶段。

直接将旧权重切到新模式的未训练干预使后缀及结束token NLL变坏，说明更强读取会破坏原有校准，不能视为修复或能力提升。需联合训练Reader及Writer，并独立评测语言保持。具体证据见 [训练报告](../reports/20261010-reader-lookup-gates.md)。

当前正式实验 `ab-A-anchor1-lookup-01` 从Writer位置池化第500微批权重接续，原完整状态保留，新run重启优化器/游标/RNG并记录架构来源。v4数据、损失和40%保持比例不变；A预算沿来源链累计，剩余23,849,322学生token，首批500微批，每100微批完整保存。后续恢复只用 `--resume`。

同一芯片上不要并行启动两个HCCL进程组。torchrun rendezvous端口不同也不能避免NPU端16666和host端60000冲突；当前HCCL验证与训练串行，单设备只读诊断可并行。
