# 5080训练与完整恢复

2026-10-19起后续负载使用Windows `192.168.101.5` 的RTX5080，WSL负责执行，模型、run、评测、日志、临时文件和增长型缓存放`E:\ds-rsi`。GPU任务通过`scripts/run_5080_checked.py`启动：同进程检查设备、显存、温度、E盘实际余量，获取项目GPU锁并继承到训练进程；评测与训练串行。控制请求使用纯标准库入口，不获取GPU锁。

## 当前状态与后续课程

当前A run为`/mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-lr125-01`，完整GPU执行快照`/mnt/e/ds-rsi/code/92657521bdd5`。已完整stopped/核验/评测4554，全局优化器439、累积0、学生2542884/教师212524、游标5268，checkpoint `micro_000000004554_fceac90b62`、manifest SHA256 `d89910a3b02d2f1c2d36a9e0e70699e7d8c24029ad8e5bde5e26478561d05c4c`。单条oracle完整代码79/96、成对36/48，中英native保持过原线；这只是总分译码中间门槛，M5–M7未通过。

没有提交旧A继续到5514或B正式训练。64池实际coverage@8基线已完成17/96，未达90%，PID12407已退出；已串行启动同4554、原9池/2候选的完整96题oracle/off/oracle_foreign挑战，PID13246、`control/cuda-lr125-4554-oracle2-submit.json`。结束按`docs/counterfactual-codec-curriculum.md`继续2/4/8与混合任务课程证据，再进入正式实际寻址训练，不直接跳到B。

原A若恢复，仍仅原全部参数加`--resume --lr 0.000125 --grad-accum 12`及前缀复用，预算16505710、keep-weight1不变，不能在旧run里改候选数。新课程须显式A→A新run、保留旧全态、按合同声明新优化器/RNG/游标；新A预算13962826、链前缀16037174。正式新课前验证真实CUDA完整恢复和前缀优化等价，辅助诊断不替代能力。原B20M额度保持，未挪用A余额。

Writer12个Adam437、其它22个439的历史差异须原样恢复；是保持题组按既有协议detach Writer所致，不能按全局步数归一。当前全态及全部旧来源保留，细节见`reports/20261019-cuda-lr125-4554.md`。

## 首次硬件迁移与来源run

- run：`/mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-01`。
- 起点：714微批/优化器119/累积0，学生696,760、教师57,508，未消费ordinal1428。
- 初始CUDA代次：`micro_000000000714_88210fc1bb`，manifest SHA256 `e9cba530f44a3deb9c6c41afc403e783b1762d2dc2ca676594350baeb2fe4994`。
- 来源NPU代次：`micro_000000000714_5559865da6`，SHA256 `d7a6d252be6105173e0d4f7df9b2cbf69c942f337c797de0fa2e2f2f4c3f8f4b`；来源全状态和失活rank均保留。
- 使用独立环境`/mnt/e/ds-rsi/envs/train-cu130-torch210`，Python3.12.3、torch2.10.0+cu130、Transformers5.14.1、NumPy1.26.4；完整依赖固定在`configs/cuda210-requirements.txt`。

迁移是新run内的显式硬件/拓扑转换，不是`init-from`，也不宣称与旧NPU未来轨迹一致。完整payload只改变world_size及accelerator RNG；优化器、权重、buffers、冻结附加模块、Python/NumPy/CPU RNG、进度、游标和账本逐项保持。CUDA RNG显式使用seed20261010，原NPU RNG保留在来源。配置由2rank×累积6改为1rank×累积12，保持12 episode/更新。

`micro_step=714`保留历史计数，随后每个CUDA微批消耗一个episode；不能用微批数乘当前world_size重构历史游标或账本。后续240个episode到954，游标到1668，优化器139，累积0。

A来源链已有14,191,050/30M学生token；配置预算仍16,505,710，run内已消费696,760，剩余15,808,950。报告只把当前run新增消费追加到来源链，不能再次加初始化来源，也不能把性能/恢复副本当作正式训练成果。

首批240个episode已正式接续并stopped在954：`micro_000000000954_65721a0550`，SHA256 `dddef5ea8dc887bd756b533b21fcc81c7a4f89323190012d6bf9b64a4f06c6ca`；优化器139、累积0、学生813,974/教师65,946、游标1668。新增117,214学生/8,438教师，来源链累计14,308,264，余15,691,736；配置预算不变。实际使用兼容完整快照`ca3449f25833`，954完整复评通过保持后接续至1434，1434回忆回落后转下述lr125新分支。本轮回忆journal修复后使用`05b38a0bb84a`，固定714基线51/96、19/48；中英native保持过原线。

## 端到端计时

本机已实测UTC回拨，禁止用提交/manifest的`created_unix`差作为可靠elapsed耗时。历史401/404/436 token/s是UTC差值折算，现已标记为不可靠端到端估计；微批与中断恢复验证使用单调时钟，保持原口径。

只读工具`scripts/observe_cuda_process.py`使用`/proc/pid/stat`启动tick、CLOCK_BOOTTIME与pidfd退出通知测完整进程寿命，含启动/保存/退出及机器挂起，不发信号、不拿GPU锁、不改训练态。WSL实际工具在完整快照`/mnt/e/ds-rsi/code/b3858061a323`，六个训练核心文件SHA与92657521bdd5完全相同。运行时提供E盘内当前存活训练的`--submission`记录和全新`--output`路径；不要用旧的已退出PID。退出通知不表示成功返回码，仍须核验实际完整checkpoint和日志。4554实测1125.406秒、412.416学生token/s，源数据与更正见`reports/20261019-cuda-clock-timing-audit.md`。

## 执行身份与恢复

CUDA默认显式绑定原PyTorch GDN、卷积、gated norm和eager attention，避免自动安装的FLA库改变正式路径。checkpoint固定CUDA运行时/包版本、GPU、CPU线程数、确定性/TF32设置、HF参考函数与关键实现SHA。普通resume不能切换后端、配置、代码语义或环境；不兼容即拒绝。FLA仅有数值变化试验证据，未用于正式续训。

来源cuda-01恢复使用完整已提交快照和本run原参数加`--resume`（当前lr125恢复应使用0.000125）：累积12、lr0.00025、keep-weight1、student-tokens16505710、ab-v6-natural-keep/facts、seed20261010、distractors7、memory-candidates1、keep-fraction0.4、counterfactual-weight1/margin2、save-every-micro100、reuse-frozen-prefix。`scripts/verify_cuda_resume.py`中的`train_arguments`从已提交manifest生成相同完整CLI参数，拒绝不支持的优化器默认值。

安全暂停可从WSL调用；`CODE`为经验证的完整快照路径：

```bash
PYTHONPATH="$CODE/src" /usr/bin/python3 -m dsrsi.training.control \
  --run-dir /mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-lr125-01 --action pause
```

收到请求后完成当前微批的反向和数据提交，保存全部累计梯度并退出。必须核对ack、完整manifest、校验和及实际进程退出，再恢复；不能据观察超时重启。SIGINT/SIGUSR2为pause，SIGTERM为stop，中途均不能写为completed。`checkpoint`动作则完整保存后继续。

真实独立进程对照已通过：连续188个episode与第718微批/累积4外部暂停、新进程恢复到902的全部payload逐位一致，样本、loss及账本相同；覆盖20条长续写、6 unknown、14 known壁钟、32辅助题。暂停请求到进程退出3.52秒，完整重启到首个新微批90.49秒，后者包含环境/底座加载。两个诊断轨迹各另消费92,800学生/6,090教师token，未修改正式714起点。

## 评测与续训门槛

先固定714的96题完整代码回忆/反事实/关闭与替换M、48篇native中英保持，再按证据运行240个episode的小批。回忆和保持均固定checkpoint/manifest与数据口径；保持语料不进入训练，原阈值不变。M5–M7仍需真实能力证据。

回忆评测新增逐episode原子journal，保持评测沿用逐前向journal。SIGINT/SIGTERM/SIGUSR2在当前任务完成后保存并退出；`--resume`只接续原固定代次及相同代码、配置、数据和运行时，拒绝跟随新latest或混合环境。完成所有任务前不发布完整结果。

迁移和独立恢复证据见`reports/cuda5080-hardware-migration-20261019.json`、`reports/cuda5080-separate-process-recovery-20261019.json`。

## 保留完整状态的学习率调整

1434完整复评为oracle EM54/96、反事实成对20/48，较954的61/96、24/48回落；native中英配对平均ΔNLL0.01026035/0.01518519均过原线。原学习率不继续盲目扩展。已从最新1434完整态创建`ab-A-v6-natural-keep-w1-cuda-lr125-01`，学习率0.00025→0.000125，保留优化器179步及全部矩、RNG、游标2148和学生1041480账本，不回退旧954或重播数据。

`python -m dsrsi.training.learning_rate --source-run SOURCE --target-run TARGET --learning-rate 0.000125 --dry-run`先检查，再去掉`--dry-run`执行。仅允许已停止、单CUDA、空累计梯度的reference A/B、无scheduler/scaler的AdamW；来源及运行时/数据身份必须匹配。新run仅更改配置和所有optimizer param_group的lr，其它payload保存后重载逐项相同，来源不变。迁移记录追加`execution_migrations`并以kind标识数值优化策略变化，不声称未来轨迹与原run一致。所有调用仍经checked入口串行执行。

新分支已完成独立188个episode的外部pause/resume全态逐位对照（34长续写/18 unknown/16 known/34辅助），暂停退出3.739秒、重启首微批79.311秒。正式1434→1674小批已完整保存并按原96题回忆和48篇保持复评；结果见上方当前状态。恢复使用新run原全部参数加`--resume --lr 0.000125`，累积12、keep-weight1、预算16505710不变；不得init-from或重置账本。迁移1434起点A链14535770/30M；1674边界A链14650296/30M，剩余15349704。
