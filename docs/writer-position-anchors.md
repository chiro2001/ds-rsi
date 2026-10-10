# Writer 的位置池化通道

v4第2000微批的48个自由查询中，字母和单个数字均91.67%，三位后缀及完整代码仍为0%。已有字段反事实监督未解决这一问题。逐位置诊断进一步显示：三个后缀位置的teacher-forced NLL约2.3，即接近十种数字均匀分布；更换对应payload几乎不改变正确token概率。native与标准单token路径都失败，不是仅由执行路径差异造成。

源端注意力诊断中，后缀变化使冻结H11改变约8.5%–14.8%，但Writer slots仅改变0.04%–0.09%。Writer的32个head/query分布平均熵仅约0.01，三个数字的平均注意力质量约10^-6–10^-5，分数跨度约64–68。注意力已经集中于前部字符，未把足够后缀信息传给Reader。详见 [机制报告](../reports/20261010-writer-position-anchors.md)。

新增结构保留原有learnable-query attention，并把有效源token按顺序分成8个连续自适应区间。每个区间平均池化同一个已有V投影，再加入对应slot的attention结果：

```text
h = RMSNorm(H11[valid])
v = Wv h
pooled[i] = mean(v[adaptive_valid_token_bin(i)])
z = query + Wo(attention(Q,K,v) + anchor_weight * pooled)
slots = RMSNorm(z + SwiGLU(RMSNorm(z)))
```

不增加参数或slot数量，原attention仍可选择重要内容；池化通道让所有已观察位置有一条不依赖注意力softmax的数值与梯度路径。它不保证语义或细节可精确重建，仍需训练与自由生成验证。区间按有效token累计序号划分，忽略padding及masked NaN；源短于8token时使用重叠非空区间。只处理已完成源段，不读取当前问题、答案或未来事件。

`writer_anchor_weight=0`完全调用旧Writer路径，架构身份省略新字段，保留旧A/B checkpoint语义。非零时记忆架构记录权重与 `writer_anchor_version=position_pool_v1`，默认恢复不能更改。首个实验用权重1，训练分布、损失及保持比例沿用v4，以观察结构变更本身。

结构改变使用新run与显式 `--init-from ... --writer-anchor-weight 1`。只允许Writer这一项变更，其他记忆尺寸、层位置和heads必须一致；记录来源manifest、进度及两份架构，导入已有权重并重置新run优化器/游标/RNG。原完整状态保留。这是架构接续，不是精确恢复；A的实际token预算沿来源链累计。

训练、只读评测及C工具/持续状态入口从checkpoint继承架构，不能把新slots放入默认旧结构推理。旧C checkpoint有执行文件指纹，需要在其原代码版本恢复；此次改动不绕过该保护。

目前仅证实：饱和attention下晚期输入和梯度仍能通过新通道；短源、NaN padding、有界流式分块、pending Writer快照及源→未来loss梯度均通过CPU测试。完整Qwen3.5-0.8B三rank HCCL累计中途暂停/恢复逐位一致。未训练的结构挑战使三位后缀变化引起的slots变化达到5%–14%；这不是回忆能力验收。

正式实验 `ab-A-anchor1-fields-01` 从v4第2000微批接续，学生预算为剩余24,552,452，首批上限500微批，每100微批完整保存。先看完整代码和逐字段自由生成、payload替换、语言保持，再决定续训或其他调整；M5–M7保持未通过。
