# 5080端到端计时校正与可靠进程寿命

2026-10-19实际观察到系统UTC校准：五次10秒sleep后，CLOCK_MONOTONIC累计50.188305秒，CLOCK_REALTIME仅47.996033秒，差-2.192272秒；训练观察后半段单调时间381.744176秒、UTC368.238990秒，再次确认该问题。

此前按提交/manifest UTC时间戳差算出的401/404/436学生token/s不作为已验证的真实端到端吞吐。对应JSON保留原始时间戳及差值，字段改名为UTC差值折算，并标明不能当elapsed；历史报告追加更正。所有checkpoint原始manifest、SHA和created_unix均保持原样。微批compute_seconds、暂停/恢复验证使用单调时钟，原计数与结果未改；训练token账本、模型和数据也未改。

新增纯标准库只读工具`scripts/observe_cuda_process.py`核验进程UID、命令行和启动tick，绑定pidfd监听该进程退出，用CLOCK_BOOTTIME减/proc/pid/stat字段22得到进程寿命。当前CLK_TCK=100、起始时间量化0.01秒，含启动/保存/退出及机器挂起，不发送信号、不拿GPU锁、不修改训练态。退出通知不代表成功返回码，仍须独立检查完整checkpoint与日志。CLI限E盘项目提交记录和全新输出，重复结果拒绝覆盖；不支持该内核接口时明确失败，不退回UTC计时。

完整已提交快照`/mnt/e/ds-rsi/code/b3858061a323`已在WSL Python3.12.3通过4项真实检查：复杂comm解析、真实进程退出且模拟UTC倒退、错误进程拒绝且不发信号、禁止覆盖。六个训练核心文件SHA与当前92657521bdd5完全相同，训练环境未改。GitHub CI为374通过、2项跳过；本地conda缺os.pidfd_open，明确记录2通过/2跳过，没有把跳过当实测。

实际训练PID11120用等价只读性能探针附加观察，从内核启动tick到退出为1125.406115秒，新增464136学生token，可靠进程吞吐412.416455 token/s。GPU正式训练始终使用92657521bdd5完整快照，完整stopped4554和全部训练态另行核验。该点与历史不同样本/不同计时定义不构成受控加速比。

接口依据：Linux man-pages proc_pid_stat(5)字段22、pidfd_open(2)退出可读通知，Linux fs/proc/array.c中start_boottime到clock tick转换，以及Python标准库os/time。原数据：[时钟对照](cuda5080-clock-probe-20261019.json)、[实际进程寿命](cuda5080-4554-process-duration-20261019.json)、[WSL工具检查](cuda5080-process-observer-verification-20261019.json)。
