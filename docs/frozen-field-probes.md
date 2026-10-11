# 冻结表示中的字段信息诊断

完整代码回忆失败时，应先区分源表示、Writer编码和Reader读出。单纯测slots变化只能证实数值敏感性；它不能证明变化携带可解码的对应字符。

`scripts/diagnose_field_information.py` 读取一个固定的完整A checkpoint，冻结全部模型参数。分别从独立train/dev世界的已观察支撑事件抽取两种表示：

- Writer原始8×256 slots，未加入时间/来源特征。
- 冻结主干H11按有效源token位置平均池化到8个区间，作为源端对照。其维度较大，不是同容量压缩对照。

输入只包含源事实，不包含当前问题或助手回答。代码标签来自源事实，仅用于离线诊断头监督。两种反事实变体都归入同一split；检查train/dev的完整源token序列无重叠。v5显式采用完整代码视图，与正式课程的辅助问题不混合。

线性岭回归头预测五个变化字符，字符位置为0、1、3、4、5。每个特征的均值、标准差、标签先验与回归解只使用train；正则系数预先固定为0.1，不按dev成绩调参。报告train/dev逐字段与完整代码正确率、train多数类在dev的基线、以及对应改动位上反事实成对正确率。probe权重不写回模型，不占用正式训练账本，也不作为新模型模块或可写记忆验收。

dev成功给出“信息可以被此线性读出”的下界。失败可能由样本量、特征选择或非线性造成，不能证明信息不存在。主干对照成功而slots失败，是进一步检查编码路径的理由；slots成功而模型回答失败，则优先检查读取和主干注入。任何判断仍需自由生成和payload替换交叉验证。

```bash
python scripts/diagnose_field_information.py \
  --base /models/Qwen3.5-0.8B --run-dir runs/ab-A-v5-indexed-01 \
  --device npu:2 --train-worlds 256 --dev-worlds 128 --language en \
  --output artifacts/ab-A-v5-1000-field-information.json
```

`scripts/diagnose_codec.py` 同时支持固定v4/v5 checkpoint。v5默认完整代码，显式 `--probe-view indexed` 可诊断辅助单字符输出的正确payload概率优势；两种视图分开报告。它比较native与标准逐token路径，使用过去的正确答案前缀来分析后续字符，因此不是自由回忆成绩。

CPU测试覆盖已知线性可解码信号的独立集泛化、开发标签对拟合与预测无影响、随机独立标签接近机会水平、输入不变和坏输入拒绝。真实Qwen/NPU结果须另附带checkpoint、底座和数据指纹。
