# 记忆模块、数据链路与实际 Qwen CPU 接入

日期：2026-10-10。目标保持 active，正式 NPU 训练尚未开始。

## 已实现

- Writer、两个 Reader、Query/Address、Consolidator 与时间/来源编码；默认尺寸实际新增 **3,228,036** 参数。
- Tensor MemoryStore：按 stream 和逻辑时间读取、独立数值快照、明确容量检查、父记录验证后提交压缩。
- 语义 episode 编译器：原生 Qwen 模板、源观察/助手目标分离、结束标记监督、padding、回答前检索索引和 Reader 激活位置。
- 可按 seed/split/ordinal 恢复的控制样本生成器，涵盖事实、改写、经历理由、时间更新、缺证据和语言保持；支持反事实配对。
- Qwen 文本主干的无缓存 M-only 接入，不改变原层与权重；从实际 checkpoint 只加载文本部分，保留输出头绑定。

本地全套测试 **51 passed**。包含原生 Qwen 类的禁用 M 对照、冻结 GDN 中的 Writer 反传、较早前缀不受检索影响，以及独立模块的随机位存储/读取测试。随机位测试只是模块学习检查，不作为语言模型回忆能力结论。

## 真实 tokenizer 与数据

100 个随机世界生成 200 个反事实 episode，实际 tokenizer 编译为 222 个监督事件。数据池含源输入 45,150 token、响应输入 16,218 token、监督目标 2,076 token；它们是池统计，未计作训练消费。

11 个语义预览也编译成功。当前生成和编译缓存位于被 Git 忽略的 artifacts，代码与数据指纹可复现。

新检验暴露并落实的约束：

1. 检索 query 使用回答前的状态，不能用含 teacher-forcing 答案的最后状态。
2. 按完整问题取回的新候选，只从检索位置起注入，不能提前影响问题前缀。
3. 后来的连续隐状态可能携带早先事实，删除原记录不证明证据消失。A/B 控制样本明确使用 independent_events；连续状态阶段保留真实上下文，并另处理证据副本/标注歧义。
4. 连续任务编译只提供缓存历史之后的新输入，不重插已经处理的源原文。

## 实际 Qwen3.5-0.8B 权重

在 a3-21 创建了本项目 CPU 验证容器 `ds-rsi-cpu-dev`，限制 8 CPU/8GiB 内存，非 privileged、无设备映射，未占用 NPU。使用服务器原有 Qwen3.5-0.8B 权重，torch 2.10.0+cpu、Transformers 5.14.1。

复现命令：

```bash
PYTHONPATH=src python scripts/verify_pretrained_bridge.py \
  --base /models/Qwen3.5-0.8B \
  --output artifacts/pretrained_cpu_integration.json --threads 8
```

结果见 [原始报告](pretrained-cpu-integration.json)：

- 文本底座 **752,393,024 参数**，bf16。
- M 禁用 vs 原生文本主干的最大差异 **0**。
- 新检索发生前的问题前缀最大差异 **0**。
- 未来回答 CE 能反传到 Writer，平方梯度范数约 **12.009**；底座参数无梯度。
- 单次完整权重 CPU 核验约 **8 秒**。

底座语言能力、训练后的回忆、寻址、持续上下文、压缩保留率与工具执行效果没有被这个核验证明。当前接入是无缓存 M-only 路径；流式有界缓存、功能式 GDN 状态和完整训练循环仍需完成。

## 资源状态与下一步

授权设备此前的 RAS 故障仍存在，本轮复查 card4/chip8 返回 80C98001。未启动 NPU 训练，未复位硬件或停止其他人的进程。

接下来完成流式状态/位置管理、课程执行与微批中断恢复，接入实际检索与语义压缩训练。有健康的授权设备后，验证 NPU 前反向与多 rank 恢复，正式训练 A/B，再推进持续状态和工具/压缩阶段。
