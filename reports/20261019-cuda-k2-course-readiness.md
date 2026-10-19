# 5080双候选A课程：初态核验与执行对照

日期：2026-10-19。4554原权重的2候选完整96题oracle挑战仅2/96、反事实成对0/48，off和oracle_foreign按原答案均0；96题逐条再生成核对，真实支撑确实被选入且均列第一候选。因此单记录译码的79/96不代表多记录读取，M5–M7未过。按既有课程协议，先建立双候选A课程，不进入正式B。

## 明确的新课程，不是旧run恢复

新run `/mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-k2-lr125-01`已从原4554固定权重完成显式A→A初始化，原4554全态、优化器439及其逐参数计数/矩、RNG、游标与旧账本完整保留。按`docs/counterfactual-codec-curriculum.md`新建本课程优化器/RNG/游标，记录`A-to-A-reset-training-state`来源，不称为原run的resume。

- 新初态`micro_000000000000_84075f4752`，manifest SHA256 `a9c055371ea5c743bf12897d1da69feac1fb1fbaf51a7cd3156b63f2d069a9da`，stopped0/优化器0/累积0/学生0/教师0/游标0。
- 已实际反序列化核验：训练权重、冻结辅助模块、buffers、life state与4554逐项相同；新AdamW状态明确为空，四类RNG完整，seed20261010，所有累计梯度为空。
- 课程候选数2；v6/facts、lr0.000125、keep-weight1、keep-fraction0.4、CF1/margin2、单CUDA、累积12、每100微批保存、前缀复用保持。
- 新预算13962826、来源链前缀16037174，总A仍30M；不能再用旧13494290前缀，不能重复计算初始化来源。当前新课程正式学生token仍0，原B20M未使用，A余额未挪作B。

未启用前缀复用的参考副本位于`/mnt/e/ds-rsi/artifacts/cuda-k2-reference-unoptimized-01`，初态`micro_000000000000_bc2473c51b`、SHA `0858707ce61cf7727b064b789ddc86527efe18333ab020d935f6aaf7f902679f`。两套初态的全部payload已逐位相同，仅执行身份/复用配置不同。该初态比较还不是损失/梯度或恢复验证完成。

## 当前正在验证

参考副本已通过checked入口resume至188上限，PID13931，提交`control/cuda-k2-unoptimized-reference-188.json`，日志`logs/cuda-k2-unoptimized-reference-188.log`；是诊断副本，不计正式课程进度。先核验其实际进程/最终checkpoint，不据超时重复启动。标准库可靠进程观察器在完整b3858061a323快照，独立记录启动tick到退出，训练仍92657521bdd5完整快照。

参考188完成后，再从优化正式初态的副本运行真实独立进程外部pause/resume188对照，并比较优化连续轨迹与参考的完整payload、样本、loss、所有累计梯度、优化器/RNG/游标和账本。起始ordinal0的188样本预先生成覆盖38长续写、14 unknown、24 known壁钟、30辅助，实际覆盖仍须由运行日志证明。原正式初态必须保持0不变。损失/梯度等价及恢复均通过后，才用新run原参数加resume首批最多240个episode；不能再次init-from、不能用诊断成绩代替完整能力验收。

所有模型、环境、checkpoint、日志、评测与缓存保持E盘；GPU负载串行，用户要求暂停时先保存/停止本项目负载。M5–M7未通过，Goal保持active。

证据：[2候选完整评测](ab-A-v6-w1-cuda-lr125-4554-oracle2-dev.json)、[支撑包含核验](cuda5080-oracle2-support-inclusion-20261019.json)、[新课程初态](cuda5080-k2-course-initialization-20261019.json)、[参考/优化初态逐位比较](cuda5080-k2-initial-reference-equivalence-20261019.json)、[原4554完整记录](20261019-cuda-lr125-4554.md)。
