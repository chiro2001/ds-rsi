# 持续读取门控实现与验证

新模式在每次检索时计算Reader的门控强度，在候选复用期间保持；attention query仍逐位置变化。Writer结构、参数数量、v4数据和损失配置不变。架构身份、显式权重接续以及持续life的门控快照/detach均已实现，见 [设计说明](../docs/reader-lookup-gates.md)。

本地213项CPU测试通过。真实Qwen3.5-0.8B三rank NPU/HCCL的A/B训练在第3微批、累计3个梯度时暂停，恢复到第5微批后权重、优化器、累计梯度、RNG、游标、账本全部逐位一致，见 [A/B恢复](pretrained-npu-lookup-gate-ab-recovery.json)。

持续短unroll使用专用B工程fixture初始化C，包含窗口16、位置rebase32、unroll16和检索周期8。暂停时clock64、两Reader门控对应位置56、pending Writer14/15 token；恢复到16微批后三rank各自与连续运行逐位一致。最终保留2个门控，cache只比teacher多4字节；3个监督优化器更新确实发生。见 [C恢复](pretrained-npu-lookup-gate-context-recovery.json)。这不证明寻址、压缩或长期语义能力。

验证启动曾出现两个配置问题：C接口要求B/C-tools来源，已创建专用B fixture；两HCCL组并行占用同一芯片出现端口冲突，已确认全部退出后改为串行，失败日志保留。失败启动未提交checkpoint、未消费正式训练数据，来源状态不变；没有修改其他项目进程或系统端口设置。

对原Writer位置池化第500微批的未训练门控干预，20个英语teacher-forced查询的三位后缀平均NLL约4.81、6.43、5.81，EOS约12.46；比旧模式的约2.3和0.03明显变坏。native/cached最大目标NLL差为3.82。见 [干预记录](ab-A-anchor1-500-lookup-intervention.json)。这些结果不支持仅改开关即可改善回忆，而是要求重新校准整个读写链并监测执行路径敏感性和语言保持。

`ab-A-anchor1-lookup-01` 已开始正式A训练。只导入来源 `ab-A-anchor1-fields-01` 第500微批权重，显式记录token→lookup的架构接续，新run优化器和游标重启，原完整状态保留。A来源链此前已用6,150,678学生token，新run预算23,849,322；首批上限500微批，每100微批完整保存。不进入B，不标记目标完成。
