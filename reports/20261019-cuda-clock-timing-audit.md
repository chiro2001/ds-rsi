# 5080端到端计时校正

2026-10-19在实际训练仍运行时发现UTC时间戳存在校准。五次10秒sleep后，CLOCK_MONOTONIC累计50.188305秒，CLOCK_REALTIME仅累计47.996033秒，差-2.192272秒。因此此前用提交时间与最终manifest创建时间相减所得401/404/436学生token/s只能作为UTC差值折算，不能作为已验证的真实端到端吞吐。对应JSON字段已明确改名并标记无效用于elapsed测量，历史原始时间戳保留。训练token账本、模型评测和微批time.monotonic计算计时未改。

新增纯标准库只读工具scripts/observe_cuda_process.py：核验项目进程UID、命令行和启动tick，绑定pidfd监听该进程退出，以CLOCK_BOOTTIME减/proc/pid/stat字段22得到进程寿命。当前主机CLK_TCK=100，起始时间量化0.01秒。包含启动、保存和退出，也包含机器挂起；不发送信号、不获取GPU锁、不改训练态，且不从退出通知推断成功返回码。最终训练成功仍以完整checkpoint和日志核验。CLI只接受E盘项目内的提交记录与输出，重复结果拒绝覆盖。

本地conda Python缺os.pidfd_open：解析/禁止覆盖2项测试通过，2项真实Linux进程测试明确跳过，不能当作接口已在本地实测。WSL Python3.12.3具备pidfd和CLOCK_BOOTTIME，已在真实训练PID11120中途附加等价只读观察，等待该进程退出后收集全程数据。正式工具还需WSL实测，后续报告如实追加结果；未改现有训练执行快照92657521bdd5或任一核心文件。

接口依据为Linux man-pages的proc_pid_stat(5)字段22、pidfd_open(2)退出可读通知，以及Linux fs/proc/array.c中start_boottime到clock tick的转换；Python标准库os/time提供对应接口。原数据与方法见cuda5080-clock-probe-20261019.json。

WSL正式工具实测追加：完整已提交快照b3858061a323在WSL Python3.12.3通过4项检查（复杂comm字段解析、真实进程退出及UTC倒退、错误进程拒绝且不发信号、禁止结果覆盖），六个训练核心文件SHA与当前来源完全相同，训练环境未改。实测记录见cuda5080-process-observer-verification-20261019.json。

PID11120训练已实际退出，独立pidfd观察得到完整进程寿命1125.406115秒，新增464136学生token，可靠进程吞吐412.416455 token/s，含启动/保存/退出。观察期间单调时钟推进381.744176秒而UTC只推进368.238990秒，再次确认UTC发生校准；没有信号或训练态修改，最终4554全态另行核验为stopped。该原始观察由等价只读性能探针完成，正式工具已完成上述WSL检查；数据见cuda5080-4554-process-duration-20261019.json与ab-A-v6-w1-cuda-lr125-4554-training-summary.json。
