# Qwen3.5-0.8B 的记忆结构改造 v0

日期：2026-10-09，实施更新于 2026-10-10。状态：记忆模块和无缓存文本主干接入已实现并经 CPU 验证，流式缓存与正式训练尚待完成。见 [接入报告](../reports/20261010-memory-data-integration.md)。
已确定首个底座为 Qwen3.5-0.8B。这里把 [整体记忆设计](memory-design-v0.md) 收敛为可执行的模块、位置、尺寸和状态接口。

训练单元、各模块的梯度路径、监督掩码及课程预算见 [训练协议](training-protocol-v0.md)。

## 1. 已核验的底座

a3-22 的 `/home/l00886679/models/Qwen3.5-0.8B/config.json`：

- 文本 hidden=1024，24 个 decoder block，词表 248,320，输入输出 embedding 绑定。
- 每 4 层为 3 层 GDN + 1 层 gated full attention，共 18 个 GDN、6 个 full attention。
- 全注意力层索引为 `[3,7,11,15,19,23]`；8 个 Q 头、2 个 KV 头、head_dim=256。
- GDN 为 16 个 QK/V 头、head_dim=128，卷积核长度 4。
- 原始模型类为 `Qwen3_5ForConditionalGeneration`；首版只运行文本主干。

实际运行容器中的 Transformers 为 **5.14.1**，已只读核验模型和缓存源码。模型 config 中记载的 `4.57.0.dev0` 是生成配置时的版本，不能据此决定当前执行接口。

保留原有 24 层及其顺序、FFN、GDN、attention 投影、norm、词表与聊天协议。新增 slot 是中间层张量，不需要新增特殊 token。以组合 wrapper 调用原文本主干和独立模块，原权重与记忆模块分别保存。

## 2. 首版接入位置

下文层索引从 0 开始；`writer_after=11` 是第 12 个 block 之后，`reader_before=[12,20]` 是第 13、21 个 block 之前。

```text
tokens -> embedding
    -> blocks 0..11
    -> H11 -----------------------> Query/Address -> 检索候选
       |                             |
       +-> Writer，积累已完成片段      +-> Reader A -> 残差注入
    -> blocks 12..19
    -> Reader B，复用同一批候选 -> 残差注入
    -> blocks 20..23 -> final norm -> 原 lm_head

Writer 产物在片段处理完成后提交 M，供后续位置读取。
```

选择 H11 的原因：已有 12 层上下文处理，同时它位于两个新增 Reader 之前。查询与写入来源不直接包含这两个模块刚取回的内容，有利于控制检索反馈；原有早期 GDN 仍会携带历史影响。模型自身输出也受记忆影响，仍需用来源元数据区分新观测和派生内容。

Reader A 给后半段 GDN 和注意力处理记忆的机会，Reader B 在最后一组前进一步选择并使用候选中的细节。这是可复现的首版选择，不是已经验证的最佳深度；对照单 Reader B，以及不同写入深度。

真正跨记录多跳可作为后续扩展：第一轮读出新的寻址线索后，用 QueryUpdate 发起第二次检索，再供 Reader B 使用。首版共享候选的结构只声明一次检索可取齐的证据任务；相关样本与诊断见训练协议。

## 3. 新增模块及初始尺寸

| 模块 | 首版结构 | 输出 / 作用 |
|---|---|---|
| Writer | 8 个 learnable query、4 头 cross-attention、内部宽度 256、SwiGLU intermediate 512 | 每完成 256 token，写 `[8,256]` 的经历 slots |
| Query/Address | 当前 H11 投影到 128；2 个 learnable query 从经历 slots 池化并投影到 128 | 每条记录 2 个归一化语义键，当前查询为 128 维 |
| Reader A/B | hidden 1024 投影到 256；4 头；记忆 K/V 为 256；输出回 1024 | 对少量记录做 cross-attention，门控加性注入 |
| 时间与来源编码 | 相对时间等数值特征经小 MLP 到 256 | 读取时提供时间与来源条件 |
| Consolidator | 与 Writer 同类，输入宽度 256，16 个 learnable query | 起步把 4×8 个 slots 合成 16 个 summary slots |

初始检索 top-8 条叶记录，共 64 个 memory slots；候选每次刷新后复用于两个 Reader。记录与摘要 slot 数可变，Reader 使用有效性 mask。首轮小池用精确检索，能力成立后再升级近似索引。

Reader 计算：

```text
Q = Wq * RMSNorm(h)
K,V = Wk,Wv * RMSNorm(Z + time_and_source_features)
r = Wo * CrossAttention(Q,K,V,validity_mask)
h_out = h + alpha * sigmoid(gate_proj(RMSNorm(h))) * r
```

`alpha` 是每个 Reader 的可学习增益，起步可为 0.05。记忆为空或禁用时直接旁路；记忆开启时不把增益硬置零，以便联合训练时 Writer 有梯度。记录和决策尺度必须实际测量，不能只看有没有梯度或模块调用次数。

按上述线性投影、无 bias attention/FFN、RMSNorm、标量 gate 与时间 MLP 的一种实现计算，新增参数约 **3.23M**：Writer 1.05M、两个 Reader 合计 1.32M、地址模块 0.17M、Consolidator 0.66M、时间模块 0.034M。实际代码的 bias、norm 和额外目标会略改变该数，不含后续 LoRA。

Writer 默认每满 256 token 提交；在明确、已观察到的源段结束事件可提交不足 256 token 的有效片段。它不因任意物理 chunk/batch 结束自动提交，padding 不参与池化。

## 4. M 的记录与存储预算

```text
MemoryRecord {
    id, stream_id,
    slots: [n_slots,256],
    keys: [2,128],
    source_begin, source_end, visible_after,
    logical_time, wall_time,
    source_role, source_kind,
    history_ref, codec_version,
    level, child_refs, retention_stats
}
```

叶记录 fp16 内容为 4096 bytes，两条键为 512 bytes，合计 4608 bytes，另加元数据和索引。记录可引用 harness 历史区间；隐空间底层记录删除后，必要时仍能显式查询该历史。

M 首轮可给 64MiB 的实验预算，之后做容量消融；接近预算时 Consolidator 合并记录并按策略释放细粒度内容。压缩器的 32→16 slot 是起步倍率，后续摘要之间的合并也需训练并测量保留率。

读到的记录必须来自已经完成的过去：可见性依据 `source_end` 和 `visible_after`，不依据本次物理 batch 的顺序。训练的 padding、不同 stream 和未来片段都不可被池化或检索到。

## 5. 有界全注意力与持续 GDN 状态

原 6 个 full attention 改为窗口可配置的局部 gated attention，首版目标 W=2048。保留原投影和权重，改变可见范围并进行适配训练。单独冻结底座训练 M 时，先保留原短序列注意力行为；随后联合训练窗口限制，分开判断两种变化造成的损失。

18 层 GDN 的 recurrent state 和 conv state 在真实会话间持续存在；6 层局部 attention 的 KV 使用 ring/sliding buffer。snapshot 一并保存状态、pending Writer 内容、M 版本、逻辑时钟与索引引用。

每序列推理状态的形状预算：

- 6 层 KV，W=2048、2 KV heads、head_dim=256、bf16：约 **24MiB**。
- 18 层 GDN matrix state，按 `[16,128,128]` 的 fp32 保守预算：约 **18MiB**。
- 卷积状态 bf16：约 0.84MiB；256 token Writer buffer bf16：约 0.5MiB。

这只统计推理持久状态，不含模型权重、M、算子 workspace、prefill 中间量和训练激活，不能据此估算训练峰值显存。

## 6. 必须重做的 Cache 语义

已核验 Transformers 5.14.1 的实际行为：

- `DynamicLayer.crop(W)` 保留的是 **前 W 个** KV，而不是末 W 个。
- 顶层 `Cache.crop` 会遍历全部层，linear attention 的 crop 是回滚语义，不能拿来滑动上下文。
- `Cache.get_seq_length()` 会取第一个 attention 层的缓存长度；裁剪后它不再等于累计经历位置。
- GDN 缓存更新使用 `copy_`，训练阶段不应直接假定这一推理缓存接口适合跨段反传。

因此需要独立 `LifeState/StreamingCache`：分别管理 attention 尾部 KV、GDN 状态、每层 KV 的逻辑起点、全局 token 计数和位置原点，显式传入位置与 mask。

分块计算时，先保留本块各 query 所需的历史 KV，计算各位置窗口 mask；整块计算完成后再留下最后 W 个 KV。不能先把拼接后的缓存截到末 W 个再算整块，否则段首位置缺失历史，分块与逐 token 结果会不一致。

长期 RoPE 位置需要独立解决：保存全局时间，但为最近窗口使用有界位置坐标。若移动位置原点，缓存中已旋转的 K 必须用相同 M-RoPE 配对规则重定位；仅重置 `position_ids` 会错误改变相对关系。该路径先在纯文本模式下做数值验证，并测量反复重定位误差；不能宣称原生 262K 位置配置自动支持无限位置。

## 7. 因果检索、写入与批处理

检索以固定逻辑位置节奏刷新，例如每 64 token，遇到已经可见的查询结束事件可提前刷新。候选预算有界，不在每层扫描全 M。

模型执行器在检索、写入和事件边界切分内部执行跨度；切分依据 stream 的全局时钟。查询只使用当前位置已产生的 H11，不能使用物理 chunk 末位或 GDN 的块末最终状态为块首检索。外部调用者改变输入分块方式，不应改变内部事件节奏和提交顺序。

Writer 在完成过去片段后产生待提交记录；Reader 只消费该位置可见的记录。Writer、Reader 和 Consolidator 的 forward 返回张量与辅助结果，档案提交、索引更新由执行器完成，避免梯度检查点重算再次写入记录。

## 8. 训练时的计算图与目标

源经历过底座得到 H11 -> Writer 得到 Z -> 稍后查询读取 Z -> 经底座后半段得到回答 loss。冻结底座参数时，Reader 之后仍要允许对输入求梯度，不能把整个底座 forward 放入 `no_grad`。

训练记忆池保存带图的张量，不能先序列化落盘或量化 detach 再指望 Writer 学会。跨段 GDN 用功能式 state-in/state-out 接口，在截断边界明确 detach；快照、外存、原地静态地址更新用于运行阶段。

训练阶段：

1. **读写适配**：底座冻结；先给正确记录与干扰记录，训练 Writer、Reader、门控；使用回答 CE 和语言保持目标。
2. **寻址适配**：训练 query/key 的对比排序与实际检索；分别测实际检索和正确记录条件下的回答。
3. **连续状态适配**：窗口逐步限制至目标 W，保留 GDN；按需要在记忆接入层附近加 LoRA，联合训练语言、记忆与工具轨迹。
4. **压缩遗忘**：加入多次 Consolidator 合并、记忆预算压力、更新与长干扰；harness 显式记忆和历史查找由既有工具协议承接。

测试 M 是否真的写入时，清空底座全部 GDN/conv/KV，只保留 M 提问；正常连续运行则保留这些状态。另设禁用工具测试，防止明文历史查询替代了需要验证的 latent 记忆功能。

## 9. 接口与代码责任边界

建议后续实现文件：

```text
src/dsrsi/model/qwen35_memory.py      # 原模型接入和显式 forward 编排
src/dsrsi/state/life_state.py        # 状态、时钟、快照及函数式训练状态
src/dsrsi/state/streaming_cache.py   # 有界 KV、GDN 与位置坐标
src/dsrsi/memory/writer.py
src/dsrsi/memory/reader.py
src/dsrsi/memory/address.py
src/dsrsi/memory/consolidator.py
src/dsrsi/memory/store.py            # 张量训练池与运行存储的共同语义
src/dsrsi/harness/memory_tools.py    # 明文记忆与历史查询
```

采用显式模型接入，保持原层索引。开发用 hook 可捕获状态做诊断，正式写入不依赖带副作用的 forward hook。导出记录 base checkpoint 指纹、Transformers 版本、结构配置、记忆模块和 codec 版本，避免重现旧项目的运行时实验目录依赖。

首批必须通过：M 禁用且窗口不受限时与原模型一致；无未来泄漏和无跨 stream 泄漏；分块/逐 token 对照；snapshot 恢复对照；未来回答能改变 Writer 参数；关闭 M 后随机新事实回忆下降；缓存长期增长有界。GDN/Reader 的 NPU 前反向正确性与性能需要实际验证，本轮只做了源码核验和形状计算。
