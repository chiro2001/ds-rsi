# Norizon／无涯智远与SHINE权重记忆调研

核验日期：2026-10-26。结论：作为第二条记忆路线有较高参考价值；亚秒级写入的公开技术依据主要是一次前向生成LoRA。它没有直接证明通用持续记忆、任意LoRA组合或我们0.8B/5080上的性能，不能据宣传时延直接替换当前正式分支。

## 1. 实体、论文和公开产物

与用户线索最吻合的公开方案是张牧涵参与的 **SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass**，arXiv:2602.06358。作者仓库记录2026-05-01获ICML 2026接收，2026-05-23更新camera-ready。仓库MuLabPKU/SHINE及作者Hugging Face发布了代码、数据入口和检查点。

2026-10-11的 **Knowledge Interaction in Hypernetwork-based Continual Learning: Conflict and Composition**，arXiv:2610.15677v1，在张牧涵作者信息中明确列出Norizon AI。这提供了公司与该研究路线的直接学术关联。SHINE早于这份公司署名论文；本次未获得用户所见商业演示的完整基准说明，因此不能断言商业版本与公开SHINE完全相同。

官方公开检查点已通过模型目录API核验：

- Yewei-Liu/SHINE-Pretrain
- Yewei-Liu/SHINE-ift_mqa
- Yewei-Liu/SHINE-ift_mqa_1qa

最后一项元数据revision为`79707b4b2e3dd37bd01e534bdaab3e5faec9486a`。公开主示例基座为Qwen3-8B；仓库同时存在Qwen3-0.6B、1.7B的配置。配置存在不代表对应完整训练权重已发布，也不代表与Qwen3.5-0.8B兼容。

## 2. 500ms究竟测了什么

SHINE论文Table 1报告约0.3秒的LoRA生成准备开销，量级符合亚秒级记忆写入。新上下文的处理没有反向传播或优化器更新；昂贵训练发生在学习生成器的离线阶段。原文实验为8×A100服务器，尚未将这个时延核验为5080、单卡/多卡具体拓扑或包含冷启动及磁盘I/O的端到端协议。回答生成时间在表中另列。

同一表中MS MARCO MQA F1为SHINE 55.6、完整上下文69.4。该实验支持“较低准备和后续上下文开销”，不支持“去掉原文后能力无损”。这些指标与我们的随机完整码EM口径不同。

原文配置包括Qwen3-8B、生成LoRA rank8、MetaLoRA rank128、148个memory embeddings、4层M2P、最长1150-token上下文。生成器先用6B-token数据学习重建和补全，再做多问题及单问题指令微调。不能把复用已训练生成器的写入延迟，当作从零获得通用写入能力的成本。

## 3. 架构与我们的位置

```mermaid
flowchart LR
  C[新经历文本 c] --> E[冻结LLM与共享MetaLoRA]
  E --> M[逐层memory hidden states]
  M --> H[M2P参数生成网络]
  H --> D[经历对应的LoRA增量 ΔW(c)]
  D --> B[底座W加ΔW(c)]
  Q[未来问题q] --> B
  B --> A[回答]
```

公开代码`Metanetwork.generate_lora_dict`先编码evidence，读取memory states，经M2P映射成LoRA因子；随后学生回答输入仅含问题。其核心增量为`ΔW_l(c)=B_l(c) A_l(c)`，冻结底座W保留，生成器学习“如何把一段经历编成参数”。代码中共享的MetaLoRA和每段经历生成的LoRA是两个不同角色。

| 维度 | 我们当前分支 | SHINE公开路线 |
|---|---|---|
| 写入产物 | 每条事件的latent slots | 随上下文生成的低秩权重增量 |
| 查询时读取 | 两处Reader从候选slots交叉注意力读取 | LoRA改变原LLM内部变换 |
| 原文是否重入查询 | M-only时不重入 | 生成LoRA后回答可不重入 |
| 通用写入器学习 | 训练Writer／Reader等新增模块 | 离线训练MetaLoRA、memory embeddings及M2P |
| 多条记忆的组织 | 可区分记录、候选、时间等元数据 | 一段上下文可整体编码；独立adapter的管理仍需设计 |

对本项目的启发是改变内容注入位置：从额外Reader通道转向参数增量通道。这有机会绕开当前16个slots共同attention时的一类竞争，但不会自动消除编码损失、问题绑定失败或参数干扰。

## 4. 公司最新论文对能力边界的提醒

arXiv:2610.15677研究的是底座知识、一个生成的LoRA更新、显式输入上下文三个知识通道的冲突与多跳组合。其受控实验观察到新知识可能不能可靠覆盖旧知识；涉及LoRA通道的组合常失败，双链查询中还会出现问题辨别不足、无关关系或Unknown答案。底座与显式上下文的对照在相应任务上较好。

这不是我们“两个候选slots”的同一个实验，也不是任意两个独立LoRA直接相加的完整验收。它说明参数中出现知识与模型能按问题正确运用知识之间仍有距离；不能据该论文推断所有权重记忆都不成立。

因此，单份LoRA内的一条事实、单份LoRA内的多条事实、多份独立LoRA的选择、多份LoRA的组合，需要分别验证。两个低秩更新的和是`B1 A1 + B2 A2`；把A和B分别相加会产生交叉项，不等价。精确保留多份增量可拼接rank，但容量与应用成本随之增长；固定rank融合需另外验证。

## 5. 5080与存储的实际边界

只读取公开元数据，未下载模型权重、未启动GPU任务：上述IFT检查点中的`metalora.pth`为1,396,856,126字节，`metanetwork.pth`为2,151,238,040字节，另有约2.43MB的memory tokens；这些是通用生成器产物，不是每段新经历都存一份。其加载显存还取决于保存/加载dtype，不能把文件字节数直接当作显存。

按Qwen3-8B官方config的36层、hidden4096、FFN12288、Q/KV宽4096/1024计算：若rank8增量覆盖所有层的q/k/v/o和FFN gate/up/down，A/B因子共21,823,488个参数；BF16约43,646,976字节，即41.625MiB／adapter。此值是明确目标模块假设下的计算，不是实测SHINE输出文件大小，也不与我们每条短事件4.5KiB slots+keys作等容量比较。它说明“写进权重”不自动意味着占用更小。

8B底座连同生成器不能按原公开全精度方案直接视为16GB卡的轻量任务。量化、CPU卸载或小模型重训都需要独立对照。现有8B生成器不能直接套到0.8B上；Qwen3与Qwen3.5混合GDN/attention的层结构也不同。本次未核实到可直接加载到当前底座的完整SHINE检查点。

## 6. 推荐的独立副本实验

优先把它列为“可学习权重写入器”备选，先做小实验，保留当前正式训练态与预算，不在旧run上混改架构：

1. **单份adapter、双事实、多问题**：同一段上下文包含A和B两条独立事实，只生成一次LoRA，分别问A与B；再加入同实体不同属性、改写、字段位置变化和无证据问题。和底座看原文、当前slots记忆对照，先测问题绑定与编码能力。
2. **分开测试两种多记忆语义**：一次把两条事实编进一份LoRA，和分别生成两份LoRA后选择／组合，是两个实验。不能用前者代替后者的证明。
3. **知识冲突与更新**：同实体旧值、新值、指定过去时点的问题；检验底座已有知识、权重记忆和当前上下文谁影响答案，区分覆盖、拒答与串扰。
4. **跨通道组合**：一跳事实在记忆、一跳在问题上下文；检查完整答案与中间实体，不把只复述某一跳当成组合成功。
5. **工程与能力同时验收**：测写入计算、参数生成/装载/可选融合、首token及完整回答延迟；记录上下文长度、rank、dtype、warmup、GPU和峰值显存。保留现有语言保持阈值，验证adapter卸载、删除、快照与恢复一致性。

若小模型验证，先评估FFN或少数层的低rank输出以限制生成器规模；这是本项目的新提案，不是原论文已经证明的等价简化。需要明确新增训练数据、教师计算和预算，先验证有效性再扩展。SHINE的重建预训练可作为编解码热身，但要靠多问题和反事实验收排除固定区域复制。

## 7. 公开来源

- SHINE原论文：arXiv:2602.06358，核对v1正文与Table 1；作者仓库另提供camera-ready信息。
- 官方代码：`https://github.com/MuLabPKU/SHINE`；核对README、`metanetwork_family.py`、`configs/Qwen3-{0.6B,1.7B,8B}.yaml`。
- 官方模型目录：`https://huggingface.co/Yewei-Liu`下SHINE系列；IFT revision及文件大小来自公开模型API。
- 基座维度：`https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json`。
- Norizon AI署名的新论文：arXiv:2610.15677v1，2026-10-11。
- 对照路线LIFT：**LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning**，arXiv:2502.14644；属于对新输入做梯度适配，与SHINE的前向参数生成需要区分计时口径。

本次完成的是公开技术核验、代码阅读和实验设计；没有复现论文的时延或能力，没有更改正式checkpoint，也没有新增正式训练消费。
