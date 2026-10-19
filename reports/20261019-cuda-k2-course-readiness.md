# 5080双候选A课程：执行验证通过、首批240完整保存

日期：2026-10-19。原4554权重的2候选完整96题oracle挑战只有2/96、成对0/48，逐条再生成核验真支撑始终在候选中且列第一。单条oracle79/96仅是译码中间门槛，多记录读取尚未掌握，M5–M7仍未通过。按课程协议明确建立双候选新A，完成原实现/前缀优化、真实外部暂停/独立进程恢复对照后，已正式resume首批240个episode，完整stopped并正在同口径能力复评；没有跳入B。

## 新课程与来源完整保存

新run `/mnt/e/ds-rsi/runs/ab-A-v6-natural-keep-w1-cuda-k2-lr125-01`从原4554固定权重显式A→A初始化，新建本课程优化器/RNG/游标，原4554完整训练态、439步优化器及其逐参数矩/计数、全部RNG/游标/账本保持。该转换按`docs/counterfactual-codec-curriculum.md`声明为`A-to-A-reset-training-state`，不是旧run的resume。

新初态`micro_000000000000_84075f4752`、SHA `a9c055371ea5c743bf12897d1da69feac1fb1fbaf51a7cd3156b63f2d069a9da`已实际反序列化核验，权重、冻结辅助、buffers、life与4554逐项相同，优化器状态明确为空、梯度为空、四类RNG完整，seed20261010。候选数2，v6/facts、lr0.000125、keep-weight1、keep-fraction0.4、CF1/margin2、单CUDA、累积12、每100微批保存、前缀复用保持。新预算13962826、来源链前缀16037174，总A仍30M；不能沿用旧13494290前缀或重复加初始化来源。B20M尚未使用，未挪A余额。

## 真实执行与恢复证据

未复用前缀参考副本初态`micro_000000000000_bc2473c51b`、SHA `0858707ce61cf7727b064b789ddc86527efe18333ab020d935f6aaf7f902679f`与优化初态全部payload逐位相同。参考188轨迹实际完整stopped188/优化器15/累积8/学生91106/教师7898，未修改正式初态。

优化连续188与参考在100/188微批的完整payload逐位相同；每个检查点191个张量叶的原始字节、步幅及34份累计梯度均一致，全部188条样本、loss、学生/教师和进度账本相同。真实覆盖38长续写、14 unknown、24 known壁钟、30辅助。单次匹配计时参考223.099秒、优化207.296秒；这是一对串行样本，不作广泛稳态性能声明。

另一优化轨迹在第4微批/累计梯度4外部pause保存，独立进程resume至188/累计梯度8，完整payload、全部样本/loss/账本与连续优化轨迹一致；100/188两点的191张量原始字节和步幅再次核对。暂停至保存退出3.1906秒，重启至首新微批80.1455秒，均为单调时钟。验证时正式初态0和原4554均未改，两个验证完成后才启动正式首批。

三条诊断轨迹各91106学生/7898教师，共273318学生/23694教师，单列诊断成本，不冒充正式训练进度或能力验收。正式起点仍0，通过原参数加resume实际重跑，而非采用探针训练态，也没有再次init-from。GPU执行均为完整已提交快照92657521bdd5，可靠只读进程观察器在完整b3858061a323；六个核心文件与训练环境保持，FLA和GDN候选未启用。

## 首批正式240与复评

已实际stopped240/优化器20/累积0，学生117828/教师10324、游标240，34项Adam计数均20、梯度全空、权重/矩有限且四类RNG完整，全部payload已反序列化核验。checkpoint `micro_000000000240_37195f79f5`，manifest SHA256 `9e6ec3cbd56e0ad533a2b7d56f061957294279833abdae9720314387376f0234`。100/200/240全态保留。

正式前188条样本/loss/账本与已验证优化轨迹相同，正式100全payload、191张量原始字节和步幅也相同。新增117828学生后A链16155002/30M，剩余13844998；本run预算仍13962826。计算262.075秒、449.597学生token/s，可靠进程寿命339.057秒、含启动/保存/退出347.517学生token/s，保存0.65–0.80秒。新小窗口的端到端速度不能与旧960样本窗口直接作受控比，UTC时间戳不用于elapsed。

训练进程14834已退出。固定240、2候选、原9池、完整96题oracle/off/oracle_foreign复评已启动，PID15248，提交`control/cuda-k2-240-recall-submit.json`、日志`logs/cuda-k2-240-recall.log`、输出`evaluations/ab-A-v6-w1-cuda-k2-240-dev.json`；随后仍需48篇native保持。两项结果完成前不扩展新课，不把原单候选成绩当双候选能力。新run后续仅原参数加resume，不能再次init-from、丢累计梯度、重置账本或改变候选数。M5–M7未通过，Goal保持active。

证据：[双候选初始完整评测](ab-A-v6-w1-cuda-lr125-4554-oracle2-dev.json)、[支撑包含](cuda5080-oracle2-support-inclusion-20261019.json)、[新初态](cuda5080-k2-course-initialization-20261019.json)、[初态全payload一致](cuda5080-k2-initial-reference-equivalence-20261019.json)、[参考188](cuda5080-k2-reference188-20261019.json)、[优化/参考轨迹](cuda5080-k2-prefix-reference-equivalence-20261019.json)、[独立暂停恢复](cuda5080-k2-separate-process-recovery-20261019.json)、[验证汇总](cuda5080-k2-validation-summary-20261019.json)、[正式240全态与计时](ab-A-v6-w1-cuda-k2-240-training-summary.json)、[正式进程寿命](cuda5080-k2-formal240-process-duration-20261019.json)。
