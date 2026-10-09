# C 阶段：工具行为与记忆回放

日期：2026-10-10。已实现子阶段 `C_tools`，尚未完成 C 的连续状态训练或能力验收。A/B、C 工具事件和 M-only 回放分别提供证据，不能把工具 SFT 的整段前缀重放当作持续上下文训练。

## 模型接入

沿用已经验证的 Writer/Reader/Address，LoRA 只加在 Writer 后的 block 12–23：每层 MLP 的 up/down 投影，GDN 的 out_proj，以及 full attention 的 q/o_proj。rank=8、alpha=16，共 **1,302,528** 个 LoRA 参数。A 为 FP32 随机初始化，B 为零；原投影冻结，初始输出与未安装 LoRA 一致。FP32 增量计算后转回原 BF16 hidden dtype。

H11 与源编码不经过 LoRA，防止工具微调直接改变 Writer 的冻结输入。M 为空时助手损失仍能更新主干适配参数。Writer/Reader/Address/时间编码在回放中继续训练；Consolidator 暂时冻结。语言保持教师同时关闭 M 和 LoRA，使用原底座。

## 数据与损失

`ToolEpisodeGenerator` 按 seed/split/world 生成独立中文/英文世界，所有工具返回通过真实 `HarnessStore` 执行产生；固定测试数据库 namespace 保证 ID 可复现。每 rank 使用独立、持久的 journal。重复生成从指定 life 的 revision 0 开始，既有未来分支不会变成当前输入。

任务包含显式 write/read/update/delete、通过 search/open 找回过去事实、暂停任务时主动保存待办、无需工具的对照，以及普通任务保持。更新/删除先 read，record_id 来自真实返回值；历史查询使用问题里的线索，summary 不含隐藏校验码，打开记录后才得到真值。

工具轨迹用原 Qwen 模板编译，仅监督当前助手动作或回答，包含 EOS；工具返回、源观察、过去助手事件与其他前缀均无当前损失。每个微批选择一个助手事件，避免整条多步轨迹的全部图阻碍快速中断。此前已发生的示范动作可以作为 teacher-forcing 前缀，独立评测必须用模型自己的动作执行。

```text
工具/回答事件：L = CE(当前助手事件)
普通保持事件：L = 0.2 * (CE + KL_top32_and_tail(原底座 || 当前模型))
M-only 回放：L = L_answer + 0.05 * L_route
```

普通保持事件带一条不相关 M，slots/keys detach，主要训练 Reader 门控与 LoRA，避免保持目标抹掉 Writer。学生账本计入实际编码源段和本次前缀/目标；教师输入另计。

默认按世界采样 25% A/B 的 M-only 任务回放，75% 工具/普通事件；回放当前用 oracle 支撑，避免主干适配时丢失已训练的读写映射。工具分支内部按示范族混合，保持与无需工具的样本起步为算术控制，**尚不是完整语言质量保持数据集**。采样比例不等于实际 token 比例，日志保留族、事件数与消费账本。

## 完整恢复

```bash
PYTHONPATH=src python scripts/train_tools.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/C-tools \
  --init-from runs/B --student-tokens 1000000 --cpu-threads 4
```

1M 是本子阶段的实验默认上限，最终课程预算需开发集和吞吐验证。正式路径从已验证的 B 初始化；`--allow-untrained-memory` 仅用于明确标识的工具独立实验，不构成 A/B 达标证据。NPU 入口复用授权映射和健康检查，a3-22 chip8–11 的完整0.8B 四 rank HCCL 暂停恢复已逐位一致，见 [NPU 原始结果](../reports/pretrained-npu-tools-recovery.json)。

用相同参数追加 `--resume` 完整恢复，不重新提供 `--init-from`。暂停与 A/B 共用纯标准库控制 CLI，全体 rank 在可恢复事件边界保存。checkpoint 记录 LoRA 布局、实际记忆结构、训练/数据配置、执行文件及包版本指纹、累计梯度、优化器、RNG、游标和 journal 引用；journal 必须一起保留。相关执行文件改变时，C 的精确恢复会拒绝不一致身份。

冻结的 Address/Consolidator 等增量模块另存 `frozen_auxiliary`，不假定它们属于原底座权重。新 checkpoint 不再依赖构造时的随机值还原这些模块。早期 A/B format=1 缺该字段的 CPU smoke 兼容读取时，仍依赖原构造 seed 和执行实现；它不是新阶段完整状态的交付格式。

```bash
PYTHONPATH=src python scripts/verify_pretrained_tools.py \
  --base /path/to/Qwen3.5-0.8B --init-from runs/B \
  --run-root runs/C-recovery-proof-new --output artifacts/C-recovery-proof.json
```

验证包括 rank1 第三微批暂停、恢复后第五微批的权重、冻结增量、buffers、优化器、梯度、RNG、游标以及可见 journal revision；实验目录路径不同的 journal 用数据库身份/life/revision 比较。

## 自由执行与归因

```bash
PYTHONPATH=src python scripts/evaluate_tools.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/C-tools \
  --journal artifacts/tools-dev.sqlite --output artifacts/tools-dev.json \
  --worlds 100 --languages zh en --max-new-tokens 128 --max-turns 4
```

评测从初始 E/历史快照开始，仅向模型提供当前用户请求。生成并验证原生函数标签，实际执行调用，把确定性 JSON 返回作为后续观察。正确动作、答案和预期 E 状态只在生成结束后评分；未生成 EOS 不执行截断动作。各助手回合重新编码可见前缀，因此这是工具策略的自由执行，不证明跨回合隐状态保持。

分别报告调用/参数精确匹配、最终 E 状态、回答文本匹配、严格示范端到端匹配和请求操作是否完成。保存后的额外验证读取不会改变 E 操作成功；严格轨迹匹配仍会区分它。开放式确认回复可能正确但与示范措辞不同，不能只看文本 EM 断定操作失败。预算或语法失败有明确终止原因。

默认单 token 路径；`--parallel-prefill-diagnostic` 只用于明确标识的 CPU 预填充诊断，不能替代默认持续执行验收。追加 `--disable-lora` 做同输入、同 harness 的原底座对照；已有工具行为不能归因于新增微调。

```bash
PYTHONPATH=src python scripts/evaluate_m_only.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/C-tools \
  --output artifacts/memory-after-tools.json --worlds 100 \
  --distractors 63 --max-new-tokens 64 --include-foreign
```

`foreign` 保留 actual 的地址、选择与元数据，但将每条选中记录的所有 slots 替换为另一反事实 life 的内容，避免池内置乱仍保留原内容。报告原答案 EM 和跟随被替换内容的答案比例；M、工作状态和工具能力分开归因。当前这些诊断仍未实现完整正式验收，统一 `capability_gate_passed=false`。

下一步是 C 的持续短 unroll、跨边界状态与更广语言保持，以及 D 的压缩训练。D 还需给合并器输入子记录的时间/来源信息；当前纯 slots 合并不能假定保留了它们的独立时间元数据。
