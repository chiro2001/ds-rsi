# 当前可执行的数据与记忆检查

日期：2026-10-10。以下为数据和模型接入验证；正式训练命令将在执行器完成后提供。

## 本地环境

```bash
python -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
.venv/bin/python -m pip install -e '.[model,test]'
.venv/bin/python -m pytest -q
```

NPU 环境使用对应 torch_npu 与原有 torch 版本，不安装 CPU wheel 替换训练环境。

## 程序生成与原生编译

```bash
.venv/bin/python -m dsrsi.data.generate \
  --output artifacts/generated/pilot_train.jsonl --count 100 \
  --split train --counterfactual-pairs

.venv/bin/python -m dsrsi.data.compiler \
  --input artifacts/generated/pilot_train.jsonl \
  --output artifacts/compiled/pilot_train.jsonl \
  --tokenizer artifacts/tokenizer/Qwen3.5-0.8B
```

输出已存在时默认拒绝覆盖；只有明确需要重建生成缓存时使用 `--overwrite`。不同 split 使用独立 world 标识和问题模板，配对变体必须一起划分。

控制生成器默认独立编码源事件，用于 A/B 的明确证据标注。连续任务编译使用真实的持续源历史，并仅输出缓存之后的新 token；运行执行器仍需落实对应状态语义。

## 真实预训练权重检查

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_pretrained_bridge.py \
  --base /path/to/Qwen3.5-0.8B \
  --output artifacts/pretrained_cpu_integration.json --threads 8
```

需要原始 config、tokenizer 与 safetensors/index 文件。脚本只在 CPU 上检查加载、原生一致性、因果注入和 Writer 梯度，不报告训练后的回忆能力。

## 仍待实现

完整课程消费、实际支撑跨度映射、训练池重编码、寻址到候选集的编排、流式功能缓存、原生工具执行器、多 rank 中断恢复与训练评测。不能将当前的池 token 统计或 CPU 接入检查当作完成了训练。
