# Qwen3-0.6B SHINE小模型复现协议 v1

2026-10-26，用户已确认“按此预算，先用Qwen3-0.6B复现”。本轮独立新增学生输入token上限300000（训练、评测、验证重放和保守失败预留合计），独立增长文件上限15GiB，全在`/mnt/e/ds-rsi/reproduction/shine-qwen3-06b`。旧Qwen3.5分支、旧149523/150000账本、正式A来源链及checkpoint均保留，本轮不借用它们的余额，不开其他方法分支。

## 固定来源

- 官方SHINE仓库提交`fd606798c5d0e0f7d2c82df1204a83f8a1104036`，Apache-2.0。原仓库已含0.6B配置和预训练脚本；此前“未找到小模型入口”的判断不完整，以本次源码核验为准。
- 底座`Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，下载流直接写E盘并校验LFS SHA256，不在WSL虚拟盘缓存大权重。
- 官方使用的预训练集`fxmeng/transmla_pretrain_6B_tokens` revision `92e313b9dd8eca6165f2ccd4d07e316712f26bd3`；只取前512条，datasets-server响应的`x-revision`必须与此相同。
- 官方MQA语料`Yewei-Liu/ift_mqa_collection` revision `6f07d80835a9c124ce5b56b197dd2aa137a188df`，从固定revision JSON流读取前512条，不下载完整数据集。
- vendored源码、来源SHA与唯一包相对导入调整见`src/dsrsi/vendor/shine/provenance.json`，官方训练chat template通过AST提取，原样使用。

## 模型与执行

沿用官方0.6B脚本的生成rank8、共享写入rank128、每层q/k/v/o/gate/up/down全部投影、4层轴向Transformer、宽1024/8头/FFN2048、GELU、post-norm、dropout0、layer轴先行、`rl`展开、scale0.001。每层180224个因子数，28层5046272个FP32数，按模型宽度精确对应176个memory tokens；不加低维瓶颈或A/B输出头。

官方`LoraQwen.py`和M2P数值代码保持原样。源编码用共享写入LoRA，源后追加memory tokens；M2P直接展开生成LoRA。问答时忽略memory tokens、关闭共享写入LoRA，仅使用生成载荷，源原文不放入问题。底座冻结，memory tokens、M2P和共享写入因子训练。

使用旧环境torch2.10+cu130，独立覆盖层固定官方README要求的Transformers4.57.1/HF Hub0.36.2，不替换旧环境。FP32、eager attention、禁用AMP/TF32/compile、确定性算法、单卡、累积4。官方torch2.5.1/cu124环境不能直接作为5080运行环境沿用，因此不声称原论文数值轨迹逐位复现。

官方非DDP优化器分组会包含冻结底座并触发其断言，DDP的`module.metamodel`排除还与“mem_tokens包含在优化器”的注释存在不一致。本轮只将requires_grad参数纳入优化器，逐项证明每个出现一次，memory tokens按作者注释参与更新。这是显式单卡适配，不隐藏成原脚本原样可执行。

## 数据与课程

预训练选64条不同TransMLA训练文本、16条独立文本开发集，训练片段以单词边界裁至最多96个Qwen token；再加入QA训练集的32条源文本作源内容热身，不加入问题答案。QA为16个官方自然上下文、8世界×2反事实的16个中文U上下文；开发QA为8个独立自然上下文、4世界×2反事实的8个中文上下文。自然上下文最多192 token且两问的短答案必须仍在保留的原文内。精确归一化源文本跨训练/开发去重；不承诺排除语义重复或底座原始预训练污染。

预训练按固定ordinal种子等概率使用RECON/COMP：RECON读全文，COMP只读前70%–90%的单词，**两种目标都恢复整段全文**，与官方collator一致。读入问题只有对应控制token；不用未来问题选择memory。QA阶段把该源的两轮问答作为一个会话，沿用官方assistant标签范围（含EOS后换行）；开发自由生成每问独立清空cache。

短程计划在CPU预先编译：预训练输入≤155000、QA输入≤80000，均在空累积边界结束；其余至少65000供资源、完整恢复、固定代次评测和保守失败预留。每个模型输入都计入，包括176内部memory向量；若启用激活重计算，保守额外计入源/读回放。本轮默认不启用。裸底座、原文对照也保守占新增学生额度，不留隐形教师开销。

lr5e-5/AdamW、weight decay0.01、clip1。短预算的warmup显式改为每阶段4个优化器步，预训练和QA各自线性调度。预训练→QA仅在空累积处，先保存旧阶段完整态，再保持权重/RNG/账本/游标，显式新建Adam与QA调度并保存；不得伪装成原阶段resume或丢弃来源状态。

## 验证与保护

先CPU验证关闭LoRA与原生HF logits一致、上游直接展开数值和梯度链、完整Adam与非空累积恢复。真实5080先推进4微批形成真实Adam矩，再从同一第4状态对照连续4→10与pause7（累积3）→独立进程resume10，比较整个payload、损失/样本/账本，不仅比较权重。验证计算与来源继承分开记账，原初态不改。通过后才推进本轮主run。

所有GPU进程经已提交完整快照中的`run_5080_checked.py`串行持锁启动；可用显存至少8GiB，预留峰值上限10GiB，E盘留20GiB绝对空闲。每次计算前保留完整中途梯度态的磁盘空间，目录总量按15GiB限制；不覆盖或自动删除checkpoint。验证副本仅以硬链接共享不可变的来源文件，后续新代次独立保存；磁盘统计按inode去重，并核对SHA和链接数。周期、阶段、pause/stop均保存完整态，达到小预算或诊断窗口只标stopped，不标论文训练completed。

训练身份绑定底座文件、语料/计划、实现、上游源码、环境、并行拓扑与RNG。恢复仅同一run、原参数加resume。评测固定完整代次，逐题journal支持暂停续测，包含memory/off/foreign/plaintext和gold NLL；后者不代替自由生成。评测重编码也计账，旧训练态不写回。

这是缩小语料和训练预算的**方法复现**，不是6B token训练或论文榜单复现。即使小规模失败，也不据此宣称SHINE无效、载体容量不足或M5–M7通过。
