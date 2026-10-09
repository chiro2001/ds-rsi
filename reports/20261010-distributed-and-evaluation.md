# 分布式训练恢复、阶段接续与独立评测

日期：2026-10-10。远期目标仍 active。正式 50M A/B、HCCL、持续任务/工具 LoRA、压缩语义训练和最终能力验收尚未完成。本报告更新此前单 rank 执行器记录的实施状态。

## 恢复与失败处理

实现 Gloo/HCCL collective 接口：rank 间参数初始化、全局 token 账本、按有效查询数归一化、条件梯度同步、暂停传播以及全体写盘后原子提交。JSON 用设备 uint8 广播，避免依赖 HCCL object collective 的隐式 CPU 路径。HCCL 接口尚无实际硬件验证。

普通初始化、前向、反向、优化器、恢复、控制和日志异常在下一通信之前协调失败。保存失败不更新 latest；计算失败不提交失败微批。设备或通信本身故障仍依靠进程组超时与 launcher 清理，不能声称能从损坏的当前进程继续。

**97 项 CPU 测试通过**。新增覆盖双 rank 连续/恢复对照、仅 worker1 暂停、worker1 IO 失败、前向/反向/优化器/恢复异常、非有限梯度、不同查询数与未使用参数的梯度同步，以及评测权重一致性。最终运行使用 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`，耗时约 35 秒。

完整 **Qwen3.5-0.8B BF16** 双 rank Gloo 对照见 [恢复数据](pretrained-cpu-distributed-recovery.json)：

- 连续执行 5 微批，与 worker1 在第 3 微批暂停、加载 checkpoint 后继续到第 5 微批比较。
- 两个 rank 的权重、buffers、优化器、累计梯度、RNG、数据游标和 A/B life state 均逐位一致。
- 最终 1 个 optimizer step，累计梯度 1/4；全局 1248 个学生输入 token、244 个教师 token；两种执行合计约 17.6 秒。

这是完整权重的工程 smoke。A/B life state 只含 M-only 模式的游标/引用，不能用该对照替代后续连续多 life 训练状态验证。CPU 时间不能推算 HCCL/NPU 吞吐。

## A→B 接续

新增 `--init-from`：将 A 的已训练记忆模块和 buffers 导入新 B run，记录来源 manifest 摘要、identity 与进度；显式重置优化器、累计梯度、游标、RNG 和 token 账本，并启用地址参数训练。B 的精确恢复用 `--resume`，不重新跟随 A 的 latest。

完整权重双 rank 验证见 [接续数据](pretrained-cpu-stage-transition.json)：连续执行 B 的 2 个微批，与执行 1 微批后退出、恢复到 2 微批比较，全部训练态逐位一致。全局学生输入 2538 token，累计梯度 2/4，尚无 B optimizer step；原 A 已训练权重保持，地址存在非零待累计梯度。它验证接续与反传路径，尚未证明寻址能力改善。

## 独立评测

评测只读取固定的已提交 checkpoint，验证底座、manifest、所有 rank 文件和模型参数/buffers 一致；不恢复训练优化器、游标和 RNG。只使用未见答案的问题前缀，按默认单 token 持续执行路径进行问题编码与生成。更改监督目标不会改变生成或检索地址。

[自由生成数据](pretrained-cpu-distributed-free-eval.json) 使用上述 A smoke：1 个 dev 世界、2 个反事实变体、每题最多 32 个新 token。实际检索、oracle、关闭 M 和池内置乱四模式的 EM 与成对 EM 均为 **0**，结束标记命中率也为 0。实际检索覆盖率为 50%，oracle 的这个字段仍表示注入前实际检索的覆盖率。输出中的校验码与真值不符；不能把 loss、非零梯度或输出变化当作学会回忆。

[检索诊断数据](pretrained-cpu-distributed-retrieval.json) 使用同一 A smoke、15 个干扰记录，覆盖事实、改写指代、当前/历史时点及中文、英文、跨语言查询，2 个 dev 世界×2 个反事实变体，共 48 个有效查询。所有必要支撑组的 coverage@8 为 **12.5%**，分项保留在摘要中。A 地址参数尚未训练，这个小样本结果是基线，不能替代 B 的开发集检索验收。各模式各查询都清空工作状态，未消费 teacher-forcing 答案。

池内置乱保留 actual 的地址与选中顺序，仅交换 payload。事实内容可能仍落在其他被选中的记录里，因此这种扰动不能保证清除了事实；正式归因需要额外的跨世界替换或完整支撑移除。经历解释的文本 EM 也不能单独作为开放语义正确性的判断。当前报告统一标记 `capability_gate_passed=false`。

## 资源与下一步

本轮仅 CPU 执行，无 NPU 训练。a3-21 chip8–15 八颗芯片再次全部返回 `Health Status: Alarm`、`80C98001`、`module error can not be fixed`。未复位设备、未改用 chip0–7 或其他资源。

实现与运行入口见 [A/B 指南](../docs/running-ab.md)。健康授权设备恢复后，须先验证 8-rank 映射、HCCL 短负载与暂停恢复，再测吞吐、进行机制小训练和正式 A/B。软件侧仍需开发集检索驱动的 oracle 退火、更强的记忆内容消融、C 的持续状态与工具 LoRA、D 的正确教师筛查与压缩训练、长期稳定性及语言保持评测。
