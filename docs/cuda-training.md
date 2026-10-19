# 5080训练与完整恢复

2026-10-19起后续负载使用Windows `192.168.101.5` 的RTX5080，WSL负责执行，模型、run、评测、日志、临时文件和增长型缓存放`E:\ds-rsi`。GPU任务通过`scripts/run_5080_checked.py`启动：同进程检查设备、显存、温度、E盘实际余量，获取项目GPU锁并继承到训练进程；评测与训练串行。控制请求使用纯标准库入口，不获取GPU锁。

## 当前run与迁移

- run：`/mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-01`。
- 起点：714微批/优化器119/累积0，学生696,760、教师57,508，未消费ordinal1428。
- 初始CUDA代次：`micro_000000000714_88210fc1bb`，manifest SHA256 `e9cba530f44a3deb9c6c41afc403e783b1762d2dc2ca676594350baeb2fe4994`。
- 来源NPU代次：`micro_000000000714_5559865da6`，SHA256 `d7a6d252be6105173e0d4f7df9b2cbf69c942f337c797de0fa2e2f2f4c3f8f4b`；来源全状态和失活rank均保留。
- 使用独立环境`/mnt/e/ds-rsi/envs/train-cu130-torch210`，Python3.12.3、torch2.10.0+cu130、Transformers5.14.1、NumPy1.26.4；完整依赖固定在`configs/cuda210-requirements.txt`。

迁移是新run内的显式硬件/拓扑转换，不是`init-from`，也不宣称与旧NPU未来轨迹一致。完整payload只改变world_size及accelerator RNG；优化器、权重、buffers、冻结附加模块、Python/NumPy/CPU RNG、进度、游标和账本逐项保持。CUDA RNG显式使用seed20261010，原NPU RNG保留在来源。配置由2rank×累积6改为1rank×累积12，保持12 episode/更新。

`micro_step=714`保留历史计数，随后每个CUDA微批消耗一个episode；不能用微批数乘当前world_size重构历史游标或账本。后续240个episode到954，游标到1668，优化器139，累积0。

A来源链已有14,191,050/30M学生token；配置预算仍16,505,710，run内已消费696,760，剩余15,808,950。报告只把当前run新增消费追加到来源链，不能再次加初始化来源，也不能把性能/恢复副本当作正式训练成果。

## 执行身份与恢复

CUDA默认显式绑定原PyTorch GDN、卷积、gated norm和eager attention，避免自动安装的FLA库改变正式路径。checkpoint固定CUDA运行时/包版本、GPU、CPU线程数、确定性/TF32设置、HF参考函数与关键实现SHA。普通resume不能切换后端、配置、代码语义或环境；不兼容即拒绝。FLA仅有数值变化试验证据，未用于正式续训。

正式恢复使用完整已提交快照和本run原参数加`--resume`：累积12、lr0.00025、keep-weight1、student-tokens16505710、ab-v6-natural-keep/facts、seed20261010、distractors7、memory-candidates1、keep-fraction0.4、counterfactual-weight1/margin2、save-every-micro100、reuse-frozen-prefix。`scripts/verify_cuda_resume.py`中的`train_arguments`从已提交manifest生成相同完整CLI参数，拒绝不支持的优化器默认值。

安全暂停可从WSL调用；`CODE`为经验证的完整快照路径：

```bash
PYTHONPATH="$CODE/src" /usr/bin/python3 -m dsrsi.training.control \
  --run-dir /mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-01 --action pause
```

收到请求后完成当前微批的反向和数据提交，保存全部累计梯度并退出。必须核对ack、完整manifest、校验和及实际进程退出，再恢复；不能据观察超时重启。SIGINT/SIGUSR2为pause，SIGTERM为stop，中途均不能写为completed。`checkpoint`动作则完整保存后继续。

真实独立进程对照已通过：连续188个episode与第718微批/累积4外部暂停、新进程恢复到902的全部payload逐位一致，样本、loss及账本相同；覆盖20条长续写、6 unknown、14 known壁钟、32辅助题。暂停请求到进程退出3.52秒，完整重启到首个新微批90.49秒，后者包含环境/底座加载。两个诊断轨迹各另消费92,800学生/6,090教师token，未修改正式714起点。

## 评测与续训门槛

先固定714的96题完整代码回忆/反事实/关闭与替换M、48篇native中英保持，再按证据运行240个episode的小批。回忆和保持均固定checkpoint/manifest与数据口径；保持语料不进入训练，原阈值不变。M5–M7仍需真实能力证据。

回忆评测新增逐episode原子journal，保持评测沿用逐前向journal。SIGINT/SIGTERM/SIGUSR2在当前任务完成后保存并退出；`--resume`只接续原固定代次及相同代码、配置、数据和运行时，拒绝跟随新latest或混合环境。完成所有任务前不发布完整结果。

迁移和独立恢复证据见`reports/cuda5080-hardware-migration-20261019.json`、`reports/cuda5080-separate-process-recovery-20261019.json`。
