# 流式状态、harness 与实际训练中断验证

日期：2026-10-10。远期目标仍 active；正式 50M 训练、多 rank 与能力验收尚未完成。

## 流式状态

实现了函数式 GDN/卷积状态、有界未旋转 KV、独立逻辑时钟、局部 RoPE 原点、分支与快照。当前流式缓存是一条 life 一个状态，避免通过 padding 推进或衰减状态。

StreamExecutor 按逻辑位置管理检索、写入、显式 flush 与压缩。Writer 256-token 缓冲、当前活动候选、M、时钟与位置原点都可存储恢复。M 超过 payload 预算时使用可学习 Consolidator 合并，并明确保留有界的任务活动缓存。这些接口已实现；压缩器的语义保留能力仍需训练与评测。

完整 Qwen3.5-0.8B BF16 权重核验见 [原始报告](pretrained-stream-canonical.json)：冷启动的原生诊断差异为 0；默认连续执行的外部分块/逐 token 与快照恢复差异为 0。

这个保证有具体执行前提：**默认 execution_quantum=1**，任意外部输入分块都执行相同的单 token 状态转移。原生 GDN 的并行块和 recurrent 路径有浮点差异，实际权重的诊断最大隐状态差为 0.5625，不能用很高的余弦相似度声称行为完全一致。并行诊断显式使用 execution_quantum=None，不作为相同数值路径；后续融合优化须与参考路径比较。

这只验证了 26-token 完整权重集成及更长的小配置测试，没有证明真实长期语义能力或端侧速度。

## harness

SQLite journal 实现明文 memory.write/read/update/delete 与 history.search/open。历史搜索支持中文 trigram，查询限定当前 life 和可见版本。删除明文记录不会声称抹除历史、M 或参数知识。

checkpoint 保存数据库身份和 revision 引用。旧版本恢复排除后续事件，后续写入形成独立分支；同样的恢复后操作具有稳定的工具记录 id。journal 需与 checkpoint 一起保存/备份，不能只有引用而丢掉数据库。

原生 Qwen 函数标签可以解析为共享工具 schema，实际模型是否会正确调用工具仍需 LoRA 轨迹训练。

## A/B 训练执行器与中断

EpisodeObjective 执行真实源段编码、Writer、候选与回答损失、B 阶段地址损失和不相关 M 的语言保持，记录全部学生/教师输入与 padding 位置。A/B 受控源事件独立编码，连续状态阶段使用不同数据与执行语义。

当前 CLI 为单 rank，明确拒绝 WORLD_SIZE>1，不能把它称为已经验证了 8-chip 训练。B 阶段 oracle 目前固定 0.8，必须完成开发集实际检索 coverage gate 后再退火。

实际 0.8B CPU 短训练完成了 21 个微批、5 个 optimizer step、3054 个学生输入 token；随后保存了累计梯度第 1/4 批的状态。它是训练与恢复 smoke，不是完成了能力训练。见 [训练恢复报告](pretrained-cpu-training-recovery.json)。

首次 CPU 运行遇到镜像 torch_npu 的优化器自动 backend 查询，在无设备映射的 CPU 容器内导致 aclInit 报错。已显式禁用 AdamW foreach/fused 自动选择，恢复重试通过。没有在 RAS 故障设备上启动训练负载。

真实控制测试还暴露出控制 CLI 导入 Torch 的开销。控制路径已拆成纯标准库 atomic JSON，新的请求命令约 0.22 秒返回，训练在下一微批完成后保存并退出。一次 checkpoint 保存约 0.09 秒；端到端延迟见原始报告，仍受正在计算的微批影响。

本地全套 **77 项 CPU 测试通过**，包括原生 Qwen 小配置的真实 episode 优化中途中断恢复逐位对照、跨段梯度、缓存与未完成 Writer 缓冲、harness 分支和轻量控制命令。

## 仍待完成

多 rank/Gloo/HCCL 控制与精确恢复、正式数据消费和性能 gate、真实自由生成及消融评测、阶段 B 的开发集检索 gate、持续上下文 LoRA 和工具轨迹训练、Consolidator 语义训练、长期稳定性与部署速度。a3-21 chip8–15 仍存在 RAS 故障，需要健康的授权资源才能验证并开展正式 NPU 训练。
