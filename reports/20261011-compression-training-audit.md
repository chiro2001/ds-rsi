# D 压缩训练就绪度审计

日期：2026-10-11。结论：**D 阶段的压缩内容训练目前完全没有开始，也不在任何阶段间接发生。**
现有代码已实现压缩原语及其安全约束，距离“训练压缩与遗忘”还差数据、损失、训练器、摘要寻址和评测五条链路。
本报告的机制结论可由 `scripts/audit_compression_status.py` 在任何时刻重跑复核，原始输出见
[ab-D-compression-status.json](ab-D-compression-status.json)。

## 已实现且仍然成立的机制

`MemoryQwen.consolidate_records` 的输入是子记录 slots 加上每个 slot 的 8 维时间/来源特征
（逻辑年龄、壁钟年龄、层级、有效性、角色、来源类别、壁钟是否已知、是否为摘要）。
审计在 tiny 模型上重跑了同一代码路径，结果与 docs 描述一致：

- 改变未来提交时刻不改变摘要内容（`future_commit_time_changes_content=false`）；同一子集在 visible_after=20
  与 4000 产生的 slots/keys 逐位相同。
- 改变子记录的来源元数据会改变内容（`source_metadata_changes_content=true`）。
- 部分子记录壁钟未知时，摘要不会被升级为“已知壁钟”（`unknown_wall_time_reported_known=false`）。
- 子记录张量顺序不是可用通道：正序与逆序压缩的 slots 相对差约 1.35e-7，属于浮点求和顺序噪声
  （`child_order_material_delta=false`）。**因此“第一组四叶”这类位置语义必须靠时间/来源特征表达，
  不能靠 slots 排列顺序。**
- 非法压缩被拒绝：未来可见性、跨 stream、少于两条子记录均抛错。
- 摘要回答 CE 的梯度只到达 Consolidator（`non_consolidator_parameter_grads=[]`，非零梯度范数平方 0.041），
  冻结语义可训练。
- 检索使用写入时冻结的 keys：把 AddressEncoder 的 key 权重清零后，`MemoryStore.retrieve` 的选择完全不变。

## 3d+1 叶链在当前原语上可真实展开

按 docs 约定的“支撑在第一组四叶，其后每次旧摘要+三条新叶”构造，深度 1/2/4/8 分别消耗
4/7/13/25 条叶记录，summary level 与 source 范围正确累积，字节数单调下降（tiny 模型实测
0.69/0.82/0.90/0.95 压缩后保留比例的对偶，即 reduction）。

生产尺寸下每叶记录 8×256 slots + 2×128 keys，BF16 共 4608 字节；摘要 16×256 + 2×128 共 8704 字节。
**这意味着一次压缩只有把至少三条叶合并才真正减少字节：四条叶压缩减少 52.8%，而两条叶压缩仅减少 5.6%。**
紧预算路径（`StreamExecutor._fit_record`）已经遵循这一算术，并在合并无法减小占用时明确报
`BudgetExceeded`，不做静默丢弃。

## 未实现的部分

程序扫描 `src/`、`scripts/`、`configs/` 当前状态：

| 需求 | 现状 |
|---|---|
| D 数据模块（3d+1 叶链、支撑限定第一组四叶） | 不存在；`src/dsrsi/data/` 无压缩相关模块 |
| D 目标函数（压缩后回答 CE + 筛查后的叶教师 top32+tail KL） | 不存在；`EpisodeObjective` 明确拒绝 `merge_events` |
| D 训练器（stage D、恢复、控制、账本） | 不存在；`train_ab.py --stage` 只接受 A/B |
| 摘要寻址（叶/摘要混合池 coverage 训练） | 不存在 |
| 压缩评测（禁用/反事实 payload、递归深度分项、保留率） | 不存在；`m_only.py` 明确拒绝 `merge_events` |
| D token 预算 | 未定；`configs/training_protocol_v0.yaml` 仍写 `pending_C_results` |

`consolidator.requires_grad_(False)` 出现在 A/B、C_context、C_tools 目标与 A/B 评测加载器中，
四处一致；因此**此前所有训练都没有更新压缩器**，不存在“无意中已经训练了压缩”的情况。

## 风险与顺序约束

1. 摘要寻址必须与内容训练分开验收。`MemoryStore.retrieve` 用写入时 keys 做精确检索，
   摘要 keys 由冻结的 address 模块在压缩时生成；若地址模块保持冻结，摘要很可能永远选不中，
   压缩内容质量就无法通过实际检索体现。
2. 生产维度下摘要占 8704 字节、叶占 4608 字节，64MiB 预算约可容纳 14000 条叶记录。
   训练用的字节压力必须显式构造，不能指望自然写满。
3. 顺序语义不能依赖 slots 排列（见上文噪声差异）；D 数据必须把“第一组四叶”编码为
   时间/来源特征上的真实差别，否则支撑定位在数学上不可学。

## 状态

`capability_gate_passed=false`。本审计只确认机制与缺口，不构成压缩能力证据，也不改变
ROADMAP 中 M5–M7 未通过的结论。
