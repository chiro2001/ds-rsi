# 支撑记录注意力监督试验

冻结门控首批240的完整双候选回忆为1/96，单候选59/96，native中英保持通过。固定18例诊断中，后段Reader的首位正确记录注意力占比0.70287→0.62044，所有预测位置平均0.41482→0.38860；双候选正确token仍68/126，NLL5.513→4.251不能代替读取改善。门控参数不变也未保护多记录选择。这里只能判断本批没有通过，不能断言更长训练必然失败；下一受控试验增加直接的支撑记录训练信号。

## 数值定义与边界

显式参数`--reader-support-weight 1`，版本`reader-support-mass-v1`。仅支持v6 A/facts、single-double-world-v1复习、freeze-reader-gates-v1。保留原CE、CF1/margin2、keep-weight1/比例0.4、lr0.000125、累积12、原GDN/eager和6门控参数冻结。

双候选记忆题的原实际回答前向中，支撑ID来自训练episode已有support_groups。每个Reader在因果内容预测位置，计算所有head对正确记录各slot的概率总和的head均值，新增其负对数；对预测位置和Reader取均值。EOS、padding、prompt位置不计此项。目标为：

`L = L_original + reader_support_weight * mean_reader,content_position(-log(mean_head(sum_support_slots P)))`

单候选、保持和缺失支撑题沿用原目标，CF额外前缀和保持教师前向不加此项。监督只使用支撑ID和因果位置mask，没有把正确答案或记录标签加进模型输入；评测不启用监督，继续原自由生成及内容消融。

辅助损失在临时forward hook内重算确定性的Reader q/k投影，匹配原FP32注意力，再用稳定logsumexp求支撑质量。学生原前向、原loss字段和token账本保持；没有增加模型token输入。scope在每个微批返回前解除所有hook，异常亦清理，不持久缓存图或临时状态。原EpisodeObjective、MemoryReader和v6生成器文件保持不变；新目标、配置和源文件SHA单独锁定，旧run不能用新代码普通resume。

## 来源和预算

计划新run `ab-A-v6-natural-keep-w1-cuda-k12-fg-rs1-lr125-01`，固定同4554权重显式A→A初始化新的优化器/RNG/游标，保留旧全部完整态。与冻结门控初态保持相同参数值、冻结分区和种子，仅增加上述目标。

当前A链16390658/30M，包含纯双候选、复习、冻结三课各117828的未采用消费；新预算13609342，不能因为选择旧权重抹掉成本。首批最多240微批，恢复只用新run原全部参数加resume，不再init-from、不改配置预算为余额。

## 验证和验收

30专项已通过，包括实际BF16 Reader概率一致、slot置换、梯度提高真支撑概率、内容预测位置前移且排除EOS、CF/保持分支不误加、原字段和输入账本不变、零权重/单候选/保持的全部梯度不变、scope异常清理、完整恢复/前缀参考和CLI身份。全套CPU和真实CUDA188三轨迹仍须完成后正式训练，核对100/188全payload、原始字节/步幅、28梯度、6冻结参数及全部样本/loss/新辅助统计。

正式240后固定同代次单/双候选96题、反事实成对和48篇native保持。训练attention loss或概率改善不是能力通过，M5–M7仍按原门槛验收，B未启动。小批续训须依据完整答案、反事实与保持的实际变化决定，不能只看辅助指标。

完整本地410项测试通过、2项跳过；随后扩充的14项目标专项全部通过，另覆盖time/source特征、全局无效NaN padding和BF16真实attention的一致性。正式GPU尚未初始化或训练，先经完整提交部署后验证。
