# 连续训练状态与长历史工作内存

日期：2026-10-10。目标仍 active；正式 A/B/C/D、长期语义和最终交付未完成。本轮将运行态接入短 unroll 训练，而非重放独立整段前缀。

实现追加原生帧、跨案例学生/教师状态、固定 key/Writer 的 M、query/Reader/LoRA 学习、纯观察微批与有监督微批处理，以及包含部分消息与可见 journal 的 checkpoint。[运行与损失细节](../docs/stage-c-context.md)。

最终本地 **131 项 CPU 测试通过**，运行 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`，约54秒。

## 完整权重恢复

[原始数据](pretrained-cpu-context-recovery.json) 使用完整 0.8B BF16、2 个 CPU rank、窗口16、unroll16，连续到第16微批，与 rank1 在第3微批暂停后继续到第16微批比较。

- 暂停时逻辑时钟48、消息偏移27，待写 buffer 有27 token；不是只在消息结束后保存。
- 权重/冻结增量、buffers、优化器、待累计梯度、RNG、数据游标、学生/教师 GDN/conv/KV/M/活动候选，以及可见 journal 身份/revision 全部数值一致。
- 最终各 life 240 token；全局学生480、教师480 token，1 optimizer step，累计梯度1/2。
- 每个学生/教师 cache 19,955,712 bytes；M 有7条、35,840 bytes。位置原点224，窗口始终有界。

这是短无干扰工程验证，尚未证明 W2048 和2K/8K/32K干扰下的语义能力。冻结 Writer 不意味着不写 M，它使用已有编码参数持续提交已观察源段；其能力需要先通过正式 A/B。

## 历史工作内存

先测量旧实现，再在同一批 journal 上复测。原实现 open 单条记录也会加载全部祖先文本。新实现用 SQL 可见性判断、限制返回行和摘要，并只把 E 变更读入 Python。

| 操作 | 12,000 条历史原 Python 峰值 | 修改后 Python 峰值 |
|---|---:|---:|
| 恢复 | 54,969,210 B | 8,514 B |
| open | 55,606,532 B | 20,204 B |
| search | 55,606,297 B | 10,222 B |

1,000 与12,000条的修改后 Python峰值基本相同。原始 [修改前](harness-history-python-memory-before.json) 与 [修改后](harness-history-python-memory-after.json) 包含时间和测量口径；只有 Python tracemalloc，不包含 SQLite C allocator、系统页缓存或模型张量。祖先遍历时间仍随历史增长，不能宣称无限历史已获常数时间/总体内存保证。

测试覆盖新用户导致原模板重写旧助手的事实、tape保留旧token、答案首 token/EOS恰好一次、纯观察不更新优化器、深层 GDN/LoRA 跨段反传、单/双 rank 中途恢复、同模型只读导出、分支过滤先于 LIMIT，以及 Unicode短词检索。

本轮仍仅CPU。授权 a3-21 chip8–15八颗芯片均报 `80C98001`，未启动 NPU。后续需要 HCCL与暂停延迟/吞吐验证、实际干扰课程、语言回归和自由生成，修正/训练时间来源感知压缩器，以及正式机制验收。
