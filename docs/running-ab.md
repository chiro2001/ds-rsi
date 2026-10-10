# A/B 执行、恢复与独立评测

日期：2026-10-10。Gloo 双 rank 已在完整 Qwen3.5-0.8B 权重上验证。a3-21 故障后已按用户授权改用 a3-22 健康空闲 chip8–11，完整0.8B 的四 rank HCCL A/B、C_tools、C_context 暂停恢复对照均已逐位一致，原始结果见 reports 中 pretrained-npu 开头的三份 JSON。

最新资源：用户暂停后明确恢复，并确认可用chip12、14、15以及原14/15预留已释放。当前容器为 `ds-rsi-a322-npu-3chip`，逻辑npu:0–2；下文chip8–11四rank指历史验证，不可据此占用现有忙碌芯片。

当前A实验为 `ab-A-anchor1-fields-01`，采用 [Writer位置池化](writer-position-anchors.md)。原v4已累计2000微批完整停止，完整代码EM仍0；新结构通过显式权重接续保留原run并另起训练态，首批上限500微批。恢复此新run使用下文配置，将run-dir改为 `runs/ab-A-anchor1-fields-01`、预算改为 `24552452`、累计上限改为 `500`，只用 `--resume`。架构从checkpoint继承，不再次加 `--init-from`，不能在恢复时改变anchor权重。

## 三芯片接续

原v4在第1000微批、250优化器步完整停止，累计梯度为空。显式转换保留其训练态并创建 `ab-A-v4-fields-keep-3chip-01`，原checkpoint、第四rank状态和暂停请求保留：

```bash
python -m dsrsi.training.topology \
  --source-run runs/ab-A-v4-fields-keep-01 \
  --target-run runs/ab-A-v4-fields-keep-3chip-01 --world-size 3
```

该命令只执行一次，可先加 `--dry-run` 核验。之后使用 `--resume`，不能使用会重置优化器的 `--init-from`：

```bash
torchrun --master-addr=127.0.0.1 --master-port=29583 --nproc-per-node=3 \
  scripts/train_ab.py --base /models/Qwen3.5-0.8B \
  --run-dir runs/ab-A-v4-fields-keep-3chip-01 --device npu --stage A \
  --student-tokens 27804578 --distractors 7 \
  --data-revision ab-v4-fields-and-language --curriculum facts \
  --memory-candidates 1 --counterfactual-weight 1 --counterfactual-margin 2 \
  --keep-fraction 0.4 --keep-weight 0.5 --save-every-micro 100 \
  --max-micro-steps 2000 --resume
```

每100微批完整保存，累计到第2000微批安全保存为stopped后重新评测；这不是A阶段完成。每步episode数由16降为12，学习率和四微批累积保持原配置，迁移后不声称四rank轨迹一致。

单颗chip12隔离时，npu-smi不会在容器内显示其双die卡；宿主启动前逐颗验证健康和空闲，将报告只读挂载到 `/run/dsrsi-host-health/`。运行时只对缺失兄弟节点且返回Invalid card id的情况接受300秒内的匹配报告，其他错误拒绝启动。每次后续启动前在宿主刷新：

```bash
python3 scripts/launch_a322.py --chips 12,14,15 \
  --name ds-rsi-a322-npu-3chip --refresh-health-only
```

刷新只读健康信息；设备使用授权和无其他负载检查仍须在真正启动前核验。不要映射正在被其他任务使用的chip13。

## CPU 小负载与中断

在项目根目录使用已安装本项目及固定 Transformers 版本的 Python 环境；模型目录必须含原始配置、tokenizer、索引和全部权重文件。

```bash
PYTHONPATH=src python scripts/train_ab.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-cpu-smoke \
  --stage A --student-tokens 100000 --distractors 1 --cpu-threads 4 \
  --max-micro-steps 5
PYTHONPATH=src python -m dsrsi.training.control \
  --run-dir runs/ab-cpu-smoke --action pause
```

暂停请求可提前写入。控制命令返回表示请求已写入，须等执行器提交 `paused` manifest 并退出后才释放负载。微批内计算失败、通信断开或硬退出不能保存未完成图，只能恢复到最后已提交的 checkpoint。

恢复使用相同参数并追加 `--resume`；`--max-micro-steps` 是该 run 的累计上限，可提高或去掉。恢复不能修改 stage、预算、学习率、累积数、seed、干扰数或 world size。

新 A/B run 默认使用 `--data-revision ab-v4-fields-and-language`：延续同前缀/异属性负例、随机支撑/编号顺序，并让反事实改动覆盖各代码字符、保持任务扩至六类。v1–v3的实施和指纹保持原样。数据、编译器与工具模板文件的 SHA256 纳入 checkpoint 身份，恢复与独立评测均校验。`--curriculum mixed|facts|locations`、`--memory-candidates`、`--counterfactual-weight`、`--keep-fraction`、`--keep-weight` 固定训练分布与损失，`--save-every-micro` 固定自动保存间隔。见 [内容译码课程](counterfactual-codec-curriculum.md)。

旧 A/B run 仍须显式 `--data-revision ab-v1 --curriculum mixed` 恢复；旧生成器与原配置身份保持不变。默认新版本会拒绝恢复旧版本，不能默默将旧 ordinal 替换成新数据。C 原有 replay 数据与执行身份不因 A/B 新默认而自动重解释；恢复旧 C 必须匹配其原实施文件指纹，权重接续与完整状态恢复仍是不同操作。

## 多 rank

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 scripts/train_ab.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-gloo-smoke \
  --device cpu --stage A --student-tokens 100000 --distractors 1 \
  --cpu-threads 2 --max-micro-steps 5
```

每 rank 处理不同 ordinal，全局学生/教师 token 求和。每个全局微批按全部有效查询数归一化，累积组平均这些微批损失；不同查询数的 rank 不获得相同权重。条件分支未参与计算的参数保持 `grad=None`，有效梯度在 optimizer step 前平均同步。暂停优先级在所有 rank 间传播；所有 rank 的状态文件完成后才提交 checkpoint。

前向、反向、优化器、恢复、控制和日志的可捕获异常在下一通信前协调退出；集合通信本身故障依靠进程组超时和 launcher 退出。失败时保留旧 latest，不能用已部分更新的当前进程继续训练。

实际双 rank 中断/恢复对照：

```bash
PYTHONPATH=src python scripts/verify_pretrained_distributed.py \
  --base /path/to/Qwen3.5-0.8B --run-root runs/recovery-proof-new \
  --output artifacts/recovery-proof.json --threads-per-rank 2
```

在 a3-22 的独立容器内执行；容器仅映射 `/dev/davinci8`–`11` 原名节点，运行时自动压缩为逻辑设备0–3。宿主机先检查空闲与健康并建立容器：

```bash
python3 scripts/launch_a322.py --project /home/l00886679/ds-rsi
```

容器内先做短负载：

```bash
DSRSI_PHYSICAL_CHIPS=8,9,10,11 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
  torchrun --master-addr=127.0.0.1 --master-port=29583 --nproc-per-node=4 scripts/train_ab.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-hccl-smoke \
  --device npu --stage A --student-tokens 100000 --distractors 1 \
  --max-micro-steps 5
```

CLI 检查 host22、物理节点集合、逻辑可见设备数与 LOCAL_RANK，逐颗核验对应物理芯片健康。不能把宿主机物理编号8–11直接传给隔离容器的运行时可见编号。启动前的无进程检查由宿主机执行；各 rank 启动后的健康检查允许同一任务的其他 rank 已占用设备。

实际 NPU 中断/恢复对照入口（必须使用新的 run-root）：

```bash
python scripts/verify_pretrained_distributed.py \
  --base /models/Qwen3.5-0.8B --device npu --world-size 4 \
  --run-root runs/hccl-recovery-new --output artifacts/hccl-recovery.json
```

所有 NPU 训练入口统一开启确定性算法。该验证入口由末 rank 第3微批提出暂停，检查全部 rank 的优化器、待累计梯度、RNG、游标等完整状态与连续执行到第5微批逐位一致。

## 阶段接续

正式 A 使用 30M 学生 token；B 使用 20M，进入前先检查 A 的独立回忆与语言保持。候选池应由小池逐渐扩大，B 至少在超过 top8 的池中学习寻址，例如 63 个干扰记录。不能在仅 8 条记录的池里用 coverage=100% 证明寻址有效。

```bash
PYTHONPATH=src python scripts/train_ab.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-B \
  --stage B --student-tokens 20000000 --distractors 63 \
  --init-from runs/ab-A
```

`--init-from` 明确导入 A 的读写/时间模块和 buffers，并记录来源 manifest 摘要与进度；新 run 的优化器、累计梯度、游标、RNG 和 token 账本重新开始，寻址模块开始训练。这是阶段转换，不是精确恢复。B 后续恢复只用 `--resume`，保留其已记录来源，不重新跟随 A 的 latest。即使 A 尚未达标，接口也可用于短负载验证，但不能据此声称通过阶段门槛。

同样的入口支持显式数据/课程变更后的 A→A 权重初始化，要求新run和不同数据、课程、候选数或CF配置；原A完整状态保留。预算在来源链中累计，不把新run重置当作此前训练未消费token。

## 自由生成与检索诊断

```bash
PYTHONPATH=src python scripts/evaluate_m_only.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-A \
  --output artifacts/free-eval.json --split dev --worlds 100 \
  --families single_fact paraphrase_coreference temporal_update \
  --languages zh en cross_language --distractors 63 --max-new-tokens 64
```

在已核验的隔离容器内可追加 `--device npu:0`；`--modes oracle off foreign` 可用于读写阶段的定向诊断。最终验收仍需 actual 模式、足量独立世界与所有必要的消融。各模式共享同一答案前的无记忆前缀缓存，每个生成分支都独立 fork，保持标准单 token 路径，未复用生成答案或标签。

每个世界生成两个反事实变体；每个查询清空工作状态，只有源 M 保留。按 actual、oracle、off、shuffled 四种模式自由生成；置乱模式保留 actual 的地址和选择顺序，只交换独立记录的 payload。问题按默认持续执行的单 token 数值路径处理，正确答案只在生成之后用于评分。当前严格文本匹配适合受控事实；经历说明的开放表达不能仅用文本 EM 作为语义判据。

池内置乱可能把事实内容移动到仍被选中的其他记录，因此它是地址/内容对应关系的扰动，不能保证清除了事实证据。正式归因还需要跨世界 payload 替换或移除全部支撑内容的对照，并报告实际被清除的内容；当前四模式诊断不能单独完成这一验收。

single/multi-rank checkpoint 都可以评测。读取时固定一个已提交 generation，核验底座、manifest、所有 rank 文件以及各 rank 权重/buffers 一致；不恢复训练优化器、游标或 RNG。摘要保留 checkpoint 指纹和运行口径。`capability_gate_passed=false` 表示这份诊断本身没有实现完整正式验收。

```bash
PYTHONPATH=src python scripts/evaluate_m_only.py \
  --base /path/to/Qwen3.5-0.8B --run-dir runs/ab-B \
  --output artifacts/retrieval-dev.json --split dev --worlds 100 \
  --families single_fact paraphrase_coreference temporal_update \
  --languages zh en cross_language --distractors 63 --retrieval-only
```

检索诊断不生成答案，报告所有必要支撑组是否落在 top8；无证据或无支撑需求的问题单独排除计数，不能虚增分母。实际寻址要在 16/64/256/1024 条池分项验证。B 当前仍以 0.8 的 oracle 概率训练；开发集 coverage≥90% 后的自动退火尚未接入，不能宣称已经完成课程闭环。
