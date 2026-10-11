# 广泛语言保持评测

日期：2026-10-11。目的：把原有 36 题小探针扩展为可复现、可给出置信区间的语言保持证据。它衡量的是**回归**（加入记忆模块后是否损害底座原有语言能力），不是新能力门槛，也不能当成 M5 通过。

## 两层证据

1. **自然文本 NLL**：48 篇固定 revision 和取得文本SHA的中英文维基百科条目（每篇 5–6k 字符），每篇切成 256 token 窗口，中英各取 80 个窗口。按文档轮转取窗口，报告逐篇实际覆盖，避免预算先被长文耗尽。每个窗口比较三种前向：
   - `base`：底座权重、关闭记忆（LoRA 禁用），等于原底座；
   - `off`：适配后的权重，关闭记忆；
   - `unrelated`：适配后的权重，M 中放入一条与该语言无关的固定记录。

   两种打分协议：
   - `raw`：直接以窗口为输入，只监督 token 1..n，检索位置为 0；
   - `native`：前置 Qwen 原生 chat 模板的非记忆指令，取窗口前 55% 解码文本作为可读前缀，后 45% 作为被监督续写，检索位置在用户回合结束处。这条协议测量指令格式下的teacher-forced续写NLL。

   NLL使用并行原生训练路径；生成探针使用规范单token流式路径。报告分别记录执行方式，不把teacher forcing写成自由生成。

2. **生成探针**：固定 36 题（`language-retention-dev-v1`）加 48 题确定性生成扩展（`language-retention-broad-v1`：算术 16、阅读 12、逻辑 8、代码 6、格式 6），中英各半，全部自由生成、答案不作为输入、带无关 M。

## 统计口径

NLL 按 **token 加权**聚合，报告 mean NLL 与 perplexity。与 `base` 的比较按**文档配对的 bootstrap**（2000 次重采样，固定 seed）：给出平均 ΔNLL、95% 区间和单篇最大 Δ。判定回归的保守阈值：

- 平均 ΔNLL ≤ 0.05；
- 95% 区间上界 ≤ 0.10；
- 单篇最大 Δ ≤ 0.25。

这三个阈值是工程回归线，不是“语言能力达标”线；报告里始终记录 `broad_language_capability_gate_claimed: false`。

## 语料固定与复现

- 语料本体不入 Git，只提交清单 `data/manifests/broad-language-corpus-v1.json`（标题、revision、URL、字符数、文本 SHA256、转换器版本、语料文件 SHA256、标题清单 SHA256）。
- 取文本使用 `action=parse&oldid=<revid>`；Wiki模板与渲染器仍可能随时间改变，所以清单同时固定取得文本SHA256。`--verify` 重新渲染全部revision并逐篇比对，不写文件；不一致就拒绝复现，不静默换语料。
- `data/broad-language/corpus-v1.jsonl` 由 `scripts/fetch_broad_language_corpus.py` 生成，脚本只访问 Wikipedia API，不在训练容器内运行。
- 这些文本不进入项目A/B/C/D适配训练；底座是否在预训练见过不作假设。此处测量相对底座的回归，而非独立知识泛化成绩。

## 运行

```bash
# 重新生成语料（需要外网；仅在开发主机）
python scripts/fetch_broad_language_corpus.py
# 只读复现校验
python scripts/fetch_broad_language_corpus.py --verify

# 评测（chip12专用容器，逻辑npu:0，不建立HCCL组）
python scripts/evaluate_broad_language.py \
  --base /models/Qwen3.5-0.8B \
  --run-dir runs/ab-A-v5-indexed-01 \
  --checkpoint micro_000000002000_6eabf825af \
  --corpus data/broad-language/corpus-v1.jsonl \
  --manifest data/manifests/broad-language-corpus-v1.json \
  --device npu:0 --output artifacts/broad-language.json
# 使用相同参数并追加 --resume 接续。
```

冻结的 36 题实现文件保持不变，新扩展放在 `src/dsrsi/data/broad_language.py`，两者指纹都写入报告；任何改动都会显式生效于新报告，不会静默改变旧报告口径。

## 中断与恢复

默认在 `OUTPUT.journal.json` 逐项原子保存，NLL每一次前向、生成每一种模式各为一个任务。SIGINT/SIGTERM/SIGUSR2记录暂停意图，当前任务完成并保存后退出，不写完整评测报告。硬退出只重算尚未提交的任务；生成任务最长受 `--max-new-tokens` 限制。

`--resume` 严格匹配固定checkpoint及manifest SHA、底座/语料/配置/实现文件指纹和任务顺序；未显式给checkpoint时使用journal原固定代次，不能跟随已前进的latest。已提交结果直接复用，只有全部任务完成才原子发布报告。`--stop-after-items N` 用于中断恢复验证，不属于评测口径。journal使用SHA256自校验，进度损坏或不一致时拒绝恢复。评测状态不修改任何训练态。
