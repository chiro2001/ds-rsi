# ds-rsi

基于 **Qwen3.5-0.8B** 的可写隐空间记忆与持续上下文研究项目。

模型沿用预训练语言主干，新增记忆写入、语义寻址、读取与压缩模块。隐空间记忆允许逐渐遗失细节；harness 提供可读写、修改、删除的明文记忆与上下文历史查询。运行状态跨交互接续，并能保存和恢复。

当前状态：A/B、C 工具与持续短 unroll 执行器、LoRA、harness、完整恢复和独立评测已有实现。因 a3-21 故障，已按用户授权迁到 a3-22 健康空闲 chip8–11；完整 0.8B 的四 rank HCCL 已验证 A/B 入口、工具训练和持续上下文，暂停恢复的完整训练态与连续执行逐位一致。正式训练、长期语义与能力验收未完成。原 CPU 证据见 [连续训练与历史内存报告](reports/20261010-continuous-training-and-history-memory.md)，NPU 原始结果见 [A/B](reports/pretrained-npu-hccl-recovery.json)、[工具](reports/pretrained-npu-tools-recovery.json) 与 [持续状态](reports/pretrained-npu-context-recovery.json)。

真实A/v2首1,000微批已完成，消费116.5万学生输入token；独立自由回忆EM仅9.4%、校验码0%、反事实成对0%，M5未通过。新v3取消实体编号最小值捷径，并加入答案首次分歧前的内容反事实损失与单条记忆译码课程，完整四rank NPU恢复再次逐位一致。本地170项CPU测试通过；当前继续A机制训练，具体见 [译码课程与诊断](docs/counterfactual-codec-curriculum.md)。时间/来源条件化压缩与紧预算持续写入已有实现及实际NPU梯度证据，D训练仍未接入。

v3进一步训练103万token后，仅字母达到91.7%，整条代码仍0%；无关M还使小型语言探针从75%降至50%。新默认v4将反事实变更覆盖代码所有字符并扩展六类保持任务，当前完整177项CPU测试及新配置四rank NPU恢复通过。详细证据与下一阶段见 [字段/语言诊断](reports/20261010-codec-field-coverage-and-language-retention.md)，M5–M7保持未通过。

## 入口

- [远期目标、里程碑与验收](ROADMAP.md)
- [Qwen3.5-0.8B 模型接入](docs/qwen35-08b-memory-architecture-v0.md)
- [训练样本、损失与课程](docs/training-protocol-v0.md)
- [预训练与微调预算](docs/training-budget-v0.md)
- [旧项目审计](docs/legacy-audit.md)
- [训练中断与恢复约定](docs/training-recovery.md)
- [A/B 执行、阶段转换与独立评测](docs/running-ab.md)
- [C 工具训练、LoRA 与记忆回放](docs/stage-c-tools.md)
- [C 跨段持续状态训练](docs/stage-c-context.md)

首轮学生输入预算为 50M token，分读写与寻址两个阶段，实际吞吐和后续阶段预算通过验证确定。数据、模型权重、训练状态和大日志存放在仓库外；仓库保存代码、配置、数据清单、语义样例与评测摘要。

## 资源与运行约束

当前使用：**a3-22（host22），空闲 chip8–11**。独立非 privileged 容器保留物理节点 `/dev/davinci8`–`11`，运行时按可见设备压缩为逻辑 npu:0–3，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`，`DSRSI_PHYSICAL_CHIPS=8,9,10,11`。启动前逐芯片核验健康、进程和运行容器的设备映射；其他人的容器和进程不在本项目操作范围内。接到暂停或设备变化通知后，在最近可恢复边界保存并停止本项目负载。

所有长期训练须先通过中断/恢复对照、因果性、分块一致性、Writer 梯度与真实记忆贡献检查。NPU 短负载和 HCCL 完整恢复已验证；实际记忆能力、语言保持与正式课程验收仍需训练和独立评测。
