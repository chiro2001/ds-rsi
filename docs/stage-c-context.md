# C：跨段持续状态训练

日期：2026-10-10。子阶段 `C_context` 已有执行器与恢复对照；完整0.8B 的 a3-22 chip8–11 四 rank HCCL 暂停恢复与连续训练逐位一致，见 [NPU 原始结果](../reports/pretrained-npu-context-recovery.json)。长历史语义能力和正式训练验收尚未完成。

## 追加消息与 TBPTT

Qwen 的整段聊天模板会在后来用户消息出现时去掉旧助手的空思考块，造成旧 token 前缀改写。持续执行使用 `FrameCompiler` 单独生成原生消息帧并追加到已执行 tape；保留真正已经输入的 token，不重新渲染历史。助手帧的 header/思考标记没有标签，正文、函数标签与 EOS 有标签，尾部换行没有标签；工具返回是确定性 JSON 的观察。

每 rank 有一条持续 life。案例之间不清空 KV、GDN、M 或 E。案例由程序生成的事实/任务、真实逐条处理的干扰、后续查询、保存到 E 或核实原始历史的工具行为，以及当前窗口足够完成的普通保持任务组成。`interference_tokens` 按实际原生 token 累计，不将跳过背景当作已经处理。

默认每 32 个输入 token 做一个短 unroll，内部仍用单 token 数值路径。分块监督预测 **当前位置的下一个 token**，因此跨边界答案首 token 不依赖上一块未保存的 logits，也不会丢失首 token 或 EOS。每个完整助手事件按其总目标长度归一化；各块只贡献该事件中实际覆盖的损失份额，再按全局有效事件片段归一化。

只有全局存在监督事件的微批才累计梯度/计入累积组；纯观察微批仍推进、detach 并可保存数值状态，但不产生假 optimizer step。每个完成的 backward 后截断图，保留所有数值状态，不能把 detach 当作 reset。

## 更新模块与保持教师

沿用 B 或 C-tools 权重及后半主干 LoRA。Writer、key pooling/key 投影与 Consolidator 暂时冻结，保留旧记录的内容/缓存 key 兼容性；更新 Reader、时间/来源编码、LoRA 和 query 投影。

寻址辅助损失只标注仍存在、完整覆盖原观察的叶记录。仅在其之前、不能知道该新事实的记录作为负样本；其后记录可能隐含旧事实，不默认标负。分块叶记录或已压缩摘要不被假定充分，缺完整标注时跳过辅助项。连续状态下检索没命中直接记录不代表所有信息都已丢失，因此回答 CE 不据此改成“不知道”。

保持教师有独立的 GDN/KV life，关闭 M 与 LoRA，从出生开始处理同一条输入 tape。不能把学生已经受 M/适配影响的 state 当成原底座教师。教师只在普通保持事件蒸馏，当前数据为算术控制，需要扩展为广泛的语言/指令回归。

## harness 与完整 checkpoint

规划器用真实工具执行生成 teacher 观察，但后续物理 journal 行并不自动可见。每帧有完成后的 revision 引用；只有模型消费完该帧才推进可见 harness 引用，途中暂停不暴露未消费的动作、返回或观察。

checkpoint 包括学生与教师 KV/GDN/conv、时钟/位置原点、M 与活动记录、待写 buffer、部分消息位置、剩余 tape、可见 E/历史 revision，以及模型/冻结增量、优化器、累计梯度与 RNG。剩余训练标签可作为恢复数据存储，但只能进入未来损失，不进入当前模型状态、寻址 query 或 Writer。journal 必须随 checkpoint 保留。

```bash
PYTHONPATH=src python scripts/train_context.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/C-context \
  --init-from runs/C-tools --student-tokens 1000000 \
  --unroll-tokens 32 --interference-tokens 2048 --cpu-threads 4
```

1M 为实验默认预算，正式预算仍需开发集/吞吐确定。`--init-from` 从 B/C-tools 只接续权重，重置阶段训练状态并记录来源；恢复使用相同参数加 `--resume`。`--no-tools --interference-tokens 0` 仅用于明确标识的短工程 smoke。NPU 使用同一授权检查，目前不能通过健康门槛。

```bash
PYTHONPATH=src python -m dsrsi.training.control --run-dir runs/C-context --action pause
PYTHONPATH=src python scripts/verify_pretrained_context.py \
  --base /path/to/Qwen3.5-0.8B --init-from runs/C-tools \
  --run-root runs/context-proof-new --output artifacts/context-proof.json
```

后者是双 CPU rank、窗口16、unroll16、短无干扰的完整权重对照，覆盖消息中途待写 buffer、学生/教师状态、有效梯度与参数更新；不能替代 W=2048、2K/8K/32K 干扰或长期质量验收。状态内存、暂停延迟与 throughput 均须分项实测。

## 长历史工具的工作内存

历史 open/search 不再把全部祖先文本或所有命中行读入 Python。可见性在 SQL 判断，search 在过滤分支后才 LIMIT8，并只返回最多160字符摘要；E 重放只提取其变更记录。旧分支与未来观察仍不可见。

合成 1,000/12,000 条历史的 Python allocation profile 见报告。该改动限制了 Python 层历史文本工作集；SQLite 的祖先遍历、E 重放和检索成本仍可能随历史增长，tracemalloc 也不测 SQLite C allocator。它没有证明任意长历史的常数时间或总体 RAM 上限。后续需要分支索引、E 快照与更长历史 profile。
