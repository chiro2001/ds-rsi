# 5080训练与完整恢复

2026-10-19起后续负载使用Windows `192.168.101.5` 的RTX5080，WSL负责执行，模型、run、评测、日志、临时文件和增长型缓存放`E:\ds-rsi`。GPU任务通过`scripts/run_5080_checked.py`启动：同进程检查设备、显存、温度、E盘实际余量，获取项目GPU锁并继承到训练进程；评测与训练串行。控制请求使用纯标准库入口，不获取GPU锁。

## 当前支撑记录监督试验（2026-10-20）

当前新run为`ab-A-v6-natural-keep-w1-cuda-k12-fg-rs1-lr125-01`，完整执行快照`90028447c53a`。已显式从4554权重新建A优化器/RNG/游标，正式stopped0，初态全payload与旧冻结课一致；CI411通过2跳过。保持单/双复习、6门控参数冻结与原训练参数，仅新增`reader-support-weight1`，不改变模型输入或评测。实际CUDA188三轨迹、100/188全态和全部日志逐位一致已通过，暂停2.873秒、首微批恢复77.139秒。正式已原参数resume完成240并完整stopped，优化器20/累积0/学生117828/教师10324，checkpoint `micro_000000000240_67eaeabdba`、SHA `5442e2ff2ecc80f697ac6c3c9196b910ecd690a4c85ca542ad5960c4420db046`。正式100与前188也匹配验证。固定240正串行评测oracle1/2各96题与48篇native，不自动扩展。

A链前缀16,390,658、首批后累计16,508,486/30M，余13,491,514，配置预算仍13,609,342；保留旧三个失败课353,484学生token成本。旧4554、纯双候选、混合复习及冻结课程的全部完整态保留。旧fg240单候选59/96、双候选1/96，native中英通过，尚无多记录读取通过证据。细节及恢复身份见[支撑记录监督协议](reader-support-supervision.md)，M5–M7仍未通过，B20M未使用。

## 历史纯双候选课程

当前run `/mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-k2-lr125-01`，GPU完整快照`/mnt/e/ds-rsi/code/92657521bdd5`。已正式原参数resume完成240并完整stopped，优化器20/累积0/学生117828/教师10324/游标240，checkpoint `micro_000000000240_37195f79f5`、SHA `9e6ec3cbd56e0ad533a2b7d56f061957294279833abdae9720314387376f0234`；34项Adam均20、空梯度和全部RNG已反序列化核验，100/200/240完整态保留。源4554权重通过明确A→A新课程初始化加载，旧4554完整训练态保留，不是旧run恢复。

新配置预算13962826、来源链前缀16037174；A2已用117828后总A16155002/30M、余13844998，不能沿用旧13494290前缀或重复加初始化。新run后续仅原全部参数加`--resume`：候选2、v6/facts、lr0.000125、累积12、keep-weight1、keep-fraction0.4、CF1/margin2、每100保存、前缀复用；不能再次init-from或默改参数。

正式训练前参考/优化188、外部pause4/累积4后独立resume188全部通过，100/188两点完整payload、191张量原始字节和步幅、34份累计梯度一致，loss/样本/账本同，覆盖38长续写/14 unknown/24 known/30辅助。暂停退出3.19秒、重启首微批80.15秒。三个诊断轨迹成本273318学生/23694教师单列，正式从初态0重跑；正式100也与验证轨迹全payload相同。可靠进程计时仍用b3858061a323纯标准库观察器，不拿GPU锁，训练六个核心文件/环境保持。

固定240完整复评已完成：双候选仅3/96、成对0，格式40/96、结束61/96，未见可接受读取能力；48篇native中英Δ0.008054/0.011236过原线。正式保持stopped240、未扩展。当前同240权重的候选1完整96题只读回读诊断PID16309，`control/cuda-k2-240-oracle1-submit.json`；仅改评测条件，训练配置仍2，源全态不改，诊断未完成前不定位内部原因。不以工程验证或保持改善替代能力。原4554的单候选79/96只是中间门槛，2候选初始仅2/96且真支撑全部包含，64池实际coverage@8只有17/96；M5–M7未过。课程继续2/4/8与位置/更新/经历混合任务后才进入B，B20M额度未使用，未挪A余额。详见`reports/20261019-cuda-k2-course-readiness.md`。

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
  --run-dir /mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-k2-lr125-01 --action pause
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
